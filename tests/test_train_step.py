"""학습 한 스텝이 실제로 도는지 — 그래디언트가 프로젝터에만 흐르는지 확인.

'동결했다'고 믿었는데 안 돼 있는 건 조용히 몇 시간을 태우는 종류의 버그다.
"""

import torch
from tiny_model import VIDEO_TOKEN_ID, tiny_model

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
from live3r.data.collate import pack_video_patches
from live3r.model.live3r import Live3RModel


def build():
    cfg = Live3RConfig(
        base_model="tiny",
        dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=(0, 1), hidden_size=64, image_size=224),
        fusion=FusionConfig(inject_layers=(0, 1), merge_size=2, zero_init=True),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, tiny_model())


def test_pack_video_patches_pads_odd_frame_count():
    pv, grid, hw = pack_video_patches(torch.randn(7, 3, 64, 64), patch=16, temporal_patch=2)
    assert grid.tolist() == [[4, 4, 4]]          # 7 -> 8 프레임 -> 4 temporal patch
    assert pv.shape == (4 * 4 * 4, 3 * 2 * 16 * 16)
    assert hw == (4, 4)


def test_one_training_step_updates_only_projectors():
    m = build()
    m.freeze_base()
    for p in m.projectors.parameters():
        p.requires_grad_(True)

    T, H, W = 4, 64, 64
    pv, grid, (gh, gw) = pack_video_patches(torch.randn(T, 3, H, W), m.vision_patch, m.temporal_patch)
    llm_grid = (gh // m.spatial_merge, gw // m.spatial_merge)
    n_vis = int(grid[0, 0]) * llm_grid[0] * llm_grid[1]

    # zero_init 이라 첫 스텝 기울기가 0이 되지 않도록 gate 를 먼저 연다.
    # (bundle 을 만든 뒤 gate 를 in-place 로 바꾸면 autograd 버전이 어긋난다)
    with torch.no_grad():
        for p in m.projectors:
            torch.nn.init.normal_(p.fc2.weight, std=0.02)
            p.gate.fill_(0.5)

    m.geometry.reset()
    outs = [m.geometry.ingest(torch.randn(1, 3, 224, 224)) for _ in range(T)]
    bundle = m.build_geometry_embeds(outs, llm_grid)
    assert bundle.embeds[0].shape[0] == n_vis

    # Qwen3.5 규약: 프레임마다 별도 vision 세그먼트 (vision_start/end 로 감싼다)
    from tiny_model import VISION_END_ID, VISION_START_ID

    per_patch = llm_grid[0] * llm_grid[1]
    segs = [torch.randint(0, 900, (1, 4))]
    for _ in range(int(grid[0, 0])):
        segs += [
            torch.tensor([[VISION_START_ID]]),
            torch.full((1, per_patch), VIDEO_TOKEN_ID),
            torch.tensor([[VISION_END_ID]]),
        ]
    segs.append(torch.randint(0, 900, (1, 3)))
    ids = torch.cat(segs, 1)
    labels = ids.clone()
    labels[:, :-3] = -100
    mm = (ids == VIDEO_TOKEN_ID).long() * 2

    out = m(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=labels,
        mm_token_type_ids=mm,
        pixel_values_videos=pv,
        video_grid_thw=grid,
        geometry=bundle,
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()

    proj_grad = sum(
        p.grad.abs().sum().item() for p in m.projectors.parameters() if p.grad is not None
    )
    assert proj_grad > 0, "프로젝터에 기울기가 안 흘렀다"
    assert all(p.grad is None for p in m.base.parameters()), "동결된 베이스에 기울기가 생겼다"
    assert all(p.grad is None for p in m.geometry.parameters()), "기하 인코더가 안 동결됐다"
