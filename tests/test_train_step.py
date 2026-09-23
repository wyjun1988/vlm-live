"""학습 한 스텝 — 순수 zero-init 에서 시작해 기울기가 프로젝터에만 흐르는지.

예전 테스트는 gate 를 0.5 로 강제로 열고 시작해서 **초기화 교착(모든 기울기 0)을 가리고 있었다.**
이제는 아무것도 손대지 않은 zero-init 그대로 돈다.
"""

import torch

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
from live3r.data.vision import geometry_frame, llm_grid, prepare_image, tokens_per_step
from live3r.model.live3r import Live3RModel
from live3r.testing import SMALL_IDS, tiny_model


def build(zero_init=True, pose=True):
    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(0, 1), hidden_size=64, image_size=128,
                                expose_pose_token=pose),
        fusion=FusionConfig(inject_layers=(0, 1), merge_size=2, zero_init=zero_init),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, tiny_model())


def image_batch(m, sizes=((72, 96), (64, 64)), seed=0):
    """크기가 서로 다른 이미지 2장 → 이미지 모드 입력 (Sensenova 와 같은 모양)."""
    g = torch.Generator().manual_seed(seed)
    imgs = [torch.randint(0, 255, (h, w, 3), dtype=torch.uint8, generator=g) for h, w in sizes]
    vlm = [prepare_image(im, m.spec) for im in imgs]
    segs = [torch.randint(0, 900, (1, 4), generator=g)]
    for _, grid in vlm:
        segs += [torch.tensor([[SMALL_IDS["vision_start"]]]),
                 torch.full((1, tokens_per_step(grid[0], m.spec)), SMALL_IDS["image"]),
                 torch.tensor([[SMALL_IDS["vision_end"]]])]
    segs.append(torch.randint(0, 900, (1, 3), generator=g))
    ids = torch.cat(segs, 1)
    labels = torch.full_like(ids, -100)
    labels[:, -3:] = ids[:, -3:]
    geom = [geometry_frame(im, m.cfg.geometry.image_size, getattr(m.geometry, "patch_size", 16))
            for im in imgs]
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=labels,
        mm_token_type_ids=(ids == SMALL_IDS["image"]).long(),
        pixel_values=torch.cat([pv for pv, _ in vlm], 0),
        image_grid_thw=torch.cat([gr for _, gr in vlm], 0),
        geom_outs=m.run_geometry(geom),
        llm_grids=[llm_grid(gr[0], m.spec) for _, gr in vlm],
        pool_temporal=False,
    )


def trainable_only_projectors(m):
    m.freeze_base()
    for p in m.projectors.parameters():
        p.requires_grad_(True)
    for p in m.pose_projector.parameters():
        p.requires_grad_(True)


def test_one_step_from_pure_zero_init_updates_only_projectors():
    m = build(zero_init=True)
    trainable_only_projectors(m)
    m.train()
    b = image_batch(m)
    out = m(**b, keep_primed=True)
    out.loss.backward()
    m.injector.clear()

    fc2 = sum(p.fc2.weight.grad.abs().sum().item() for p in m.projectors)
    pose = m.pose_projector.net[-1].weight.grad.abs().sum().item()
    assert fc2 > 0, "zero-init 프로젝터에 기울기가 안 흐른다 (초기화 교착 재발)"
    assert pose > 0, "포즈 프로젝터가 손실 경로에 없다 (DDP 미사용 파라미터가 된다)"
    assert all(p.grad is None for p in m.base.parameters()), "동결된 베이스에 기울기가 생겼다"
    assert all(p.grad is None for p in m.geometry.parameters()), "기하 인코더가 안 동결됐다"


def test_loss_decreases_from_zero_init_with_projectors_only():
    """기하 경로 전체(인코더→프로젝터→주입→손실)가 실제로 학습 가능한지 — 한 배치 과적합."""
    torch.manual_seed(0)
    m = build(zero_init=True)
    trainable_only_projectors(m)
    m.train()
    b = image_batch(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(15):
        out = m(**b, keep_primed=True)
        opt.zero_grad()
        out.loss.backward()
        m.injector.clear()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < losses[0] - 0.05, f"손실이 안 내려간다: {losses[0]:.3f} → {losses[-1]:.3f}"


def test_zero_init_forward_equals_base():
    m = build(zero_init=True)
    b = image_batch(m)
    with torch.no_grad():
        a = m(**b).logits
        base_kw = {k: v for k, v in b.items() if k not in ("geom_outs", "llm_grids", "pool_temporal")}
        c = m(**base_kw).logits
    assert torch.allclose(a, c, atol=1e-6)


def test_train_mode_keeps_frozen_geometry_in_eval():
    """model.train() 이 동결 인코더의 dropout 을 켜면 학습 때만 기하 토큰이 흔들린다."""
    m = build()
    m.train()
    assert m.training and not m.geometry.training


def test_supervised_only_logits_match_full_loss_and_grads():
    """감독 위치만 LM 헤드에 넣는 손실 == HF 전 위치 손실 (값과 기울기 모두).

    어휘 248k 에서 전 위치 로짓은 2k 토큰에 2GB — M2 파일럿이 여기서 OOM 났다.
    감독 구간이 흩어진 경우(멀티턴)와 배치마다 다른 경우도 본다.
    """
    torch.manual_seed(0)
    m = build(zero_init=False)
    trainable_only_projectors(m)
    b = image_batch(m)
    ids = b["input_ids"]
    labels = torch.full_like(ids, -100)
    labels[:, 3:5] = ids[:, 3:5]           # 앞쪽 구간
    labels[:, -3:] = ids[:, -3:]           # 뒤쪽 구간 (끝 토큰 포함)
    b["labels"] = labels

    def run(sparse: bool):
        m.zero_grad(set_to_none=True)
        if sparse:
            out = m(**b)
        else:  # HF 기준: 전 위치 로짓 + 내장 손실
            orig = m._run_base
            m._run_base = lambda i, a, lab, **kw: m.base(input_ids=i, attention_mask=a, labels=lab, **kw)
            try:
                out = m(**b)
            finally:
                m._run_base = orig
        out.loss.backward()
        grads = [p.grad.clone() for p in m.projectors.parameters() if p.grad is not None]
        return out, grads

    sparse, g_sparse = run(True)
    full, g_full = run(False)
    assert sparse.logits.shape[1] == 5 < full.logits.shape[1]   # 감독 5개 위치만
    assert torch.allclose(sparse.loss, full.loss, atol=1e-6), (sparse.loss, full.loss)
    assert g_sparse and all(torch.allclose(a, c, atol=1e-6) for a, c in zip(g_sparse, g_full))


def test_supervised_only_logits_batch_union():
    """배치 안에서 감독 위치가 다르면 합집합을 계산하고, 각자 아닌 위치는 -100 이 가린다."""
    m = build(zero_init=False)
    ids = torch.randint(0, 900, (2, 12), generator=torch.Generator().manual_seed(1))
    lab = torch.full_like(ids, -100)
    lab[0, -2:] = ids[0, -2:]
    lab[1, 4:6] = ids[1, 4:6]
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab)
        ref = m.base(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab)
    assert out.logits.shape[1] == 4
    assert torch.allclose(out.loss, ref.loss, atol=1e-6)
