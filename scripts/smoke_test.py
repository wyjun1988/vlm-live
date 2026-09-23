"""엔드투엔드 형상 검증 — 맥/CPU, 가중치 다운로드 없이.

검증 항목:
  1. 기하 인코더 상태가 상수 메모리인가 (라이브 불변식)
  2. 기하 토큰 → 프로젝터 → LLM 비전 토큰 격자 리샘플이 맞는가
  3. temporal_patch=2 로 프레임→패치 풀링이 맞는가 (토큰 수 2배 어긋남 버그 방지)
  4. DeepStack 훅이 실제로 불리고, 주입이 출력을 바꾸는가
  5. zero_init 일 때 주입량이 정확히 0 인가 (베이스 보존)

실행: PYTHONPATH=src python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_model import IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, VOCAB, tiny_model  # noqa: E402

from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig  # noqa: E402
from live3r.model.live3r import Live3RModel  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _fail
    mark = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        _fail += 1


def build(zero_init: bool = True, n_layers_inject: int = 3) -> Live3RModel:
    base = tiny_model()
    cfg = Live3RConfig(
        base_model="tiny",
        dtype="fp32",
        geometry=GeometryConfig(
            name="dummy", tap_layers=tuple(range(n_layers_inject)), hidden_size=96, image_size=224
        ),
        fusion=FusionConfig(
            mode="deepstack",
            inject_layers=tuple(range(n_layers_inject)),
            merge_size=2,
            zero_init=zero_init,
        ),
        lora=LoRAConfig(enabled=False),
    )
    return Live3RModel(cfg, base)


def main() -> int:
    torch.manual_seed(0)
    print("\n=== 1. 기하 인코더 스트리밍 불변식 ===")
    model = build()
    geo = model.geometry
    geo.reset()
    frames = torch.randn(64, 3, 224, 224)
    b0 = None
    for t in range(frames.shape[0]):
        geo.ingest(frames[t : t + 1])
        if t == 3:
            b0 = geo.state_bytes()
    check("상태 바이트가 상수", b0 == geo.state_bytes(), f"{b0} -> {geo.state_bytes()}")
    check("프레임 카운터 증가", geo.frame_index == 64, str(geo.frame_index))

    print("\n=== 2~3. 비전 경로 토큰 수 & temporal 풀링 ===")
    T, H, W = 8, 64, 64                      # 프레임 8장, 64x64
    vis = model.base.config.vision_config
    p, m, tp = vis.patch_size, vis.spatial_merge_size, vis.temporal_patch_size
    t_patches, gh, gw = T // tp, H // p, W // p
    n_vis_tokens = t_patches * (gh // m) * (gw // m)
    pixel_values_videos = torch.randn(t_patches * gh * gw, 3 * tp * p * p)
    video_grid_thw = torch.tensor([[t_patches, gh, gw]])

    with torch.no_grad():
        vfeat = model.base.model.get_video_features(pixel_values_videos, video_grid_thw)
    pooled = vfeat.pooler_output if hasattr(vfeat, "pooler_output") else vfeat
    vfeat_t = pooled[0] if isinstance(pooled, (list, tuple)) else pooled
    check(
        "비전 타워 출력 토큰 수 = t*(h/m)*(w/m)",
        vfeat_t.shape[0] == n_vis_tokens,
        f"{tuple(vfeat_t.shape)} vs 기대 {n_vis_tokens}",
    )

    geo.reset()
    geom_outs = [geo.ingest(torch.randn(1, 3, 224, 224)) for _ in range(T)]
    bundle = model.build_geometry_embeds(geom_outs, llm_grids=(gh // m, gw // m))
    check(
        "기하 임베딩 토큰 수 == 비전 토큰 수",
        bundle.embeds[0].shape[0] == n_vis_tokens,
        f"{tuple(bundle.embeds[0].shape)} vs 기대 ({n_vis_tokens}, {model.d_llm})",
    )
    check("주입 지점 수 == inject_layers", len(bundle.embeds) == 3, str(len(bundle.embeds)))
    check(
        "풀링 없이는 2배로 어긋난다(회귀 방지)",
        model.build_geometry_embeds(geom_outs, (gh // m, gw // m), pool_temporal=False)
        .embeds[0]
        .shape[0]
        == n_vis_tokens * tp,
    )

    print("\n=== 4. DeepStack 주입 ===")
    prompt = torch.randint(0, 900, (1, 6))
    ids = torch.cat(
        [prompt, torch.full((1, n_vis_tokens), VIDEO_TOKEN_ID), torch.randint(0, 900, (1, 4))], 1
    )
    attn = torch.ones_like(ids)
    model.injector.hit_count = 0
    with torch.no_grad():
        out_geo = model(input_ids=ids, attention_mask=attn, geometry=bundle)
    check("훅 호출 횟수 == 주입 레이어 수", model.injector.hit_count == 3, str(model.injector.hit_count))
    with torch.no_grad():
        out_base = model(input_ids=ids, attention_mask=attn, geometry=None)
    same = torch.allclose(out_geo.logits, out_base.logits, atol=1e-6)
    check("zero_init 에서는 출력이 베이스와 동일", same)

    print("\n=== 5. gate 를 열면 출력이 바뀐다 ===")
    with torch.no_grad():
        for proj in model.projectors:
            torch.nn.init.normal_(proj.fc2.weight, std=0.02)
            proj.gate.fill_(1.0)
        bundle2 = model.build_geometry_embeds(geom_outs, (gh // m, gw // m))
        out_geo2 = model(input_ids=ids, attention_mask=attn, geometry=bundle2)
    diff = (out_geo2.logits - out_base.logits).abs().max().item()
    check("주입이 로짓을 바꾼다", diff > 1e-4, f"max|Δlogit| = {diff:.4e}")

    print("\n=== 6. 형상 불일치를 조용히 넘기지 않는다 ===")
    try:
        bad = torch.cat([prompt, torch.full((1, n_vis_tokens - 4), VIDEO_TOKEN_ID)], 1)
        with torch.no_grad():
            model(input_ids=bad, attention_mask=torch.ones_like(bad), geometry=bundle2)
        check("토큰 수 불일치 시 예외", False, "예외가 안 났다")
    except RuntimeError as e:
        check("토큰 수 불일치 시 예외", "기하 임베딩" in str(e), str(e)[:70])

    print("\n=== 7. LiveSession 스트리밍 런타임 ===")
    from live3r.serve.session import LiveSession  # noqa: E402

    sess = LiveSession(model, device="cpu")
    frames = torch.randn(24, 3, H, W)
    geom_frames = torch.randn(24, 3, 224, 224)
    for i in range(frames.shape[0]):
        sess.ingest(frames[i], geom_frames[i])
    s = sess.summary()
    check("블록 수 == 프레임/temporal_patch", s["blocks"] == 24 // tp, str(s["blocks"]))
    from live3r.data.vision import smart_resize

    rh, rw = smart_resize(H, W, model.spec.factor, model.spec.min_pixels, model.spec.max_pixels)
    per_block = (rh // (p * m)) * (rw // (p * m))  # 공식처럼 min_pixels 를 적용한 해상도 기준
    check(
        "누적 비전 토큰 == 블록 × 프레임당 토큰",
        s["visual_tokens"] == s["blocks"] * per_block,
        f'{s["visual_tokens"]} = {s["blocks"]} × {per_block} ({H}x{W} → smart_resize {rh}x{rw})',
    )
    check("기하 상태 상수 유지", s["geometry_state_bytes"] == b0, str(s["geometry_state_bytes"]))
    check("drift < 1.5 (스트리밍 성립)", s["drift"] < 1.5, f'drift={s["drift"]:.3f}')
    q = torch.randint(0, 900, (1, 5))
    r = sess.ask(q, max_new_tokens=8)
    check("질문 응답 길이", r["token_ids"].shape[1] == 8, str(tuple(r["token_ids"].shape)))
    print(f'    ttft={r["ttft_ms"]:.1f}ms tpot={r["tpot_ms"]:.1f}ms '
          f'frame={s["frame_ms_mean"]:.1f}ms drift={s["drift"]:.3f}')

    print("\n=== 파라미터 요약 ===")
    for k, v in model.trainable_parameter_summary().items():
        print(f"  {k:18s} {v:>12,}")

    print()
    if _fail:
        print(f"{RED}{_fail}개 실패{RESET}")
        return 1
    print(f"{GREEN}전부 통과{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
