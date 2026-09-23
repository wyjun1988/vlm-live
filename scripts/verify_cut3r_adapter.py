"""CUT3R 어댑터 구조 검증 — 체크포인트(2GB) 없이 랜덤 소형 모델로 코드 경로 전체를 돈다.

실제 가중치로 품질을 보는 게 아니라, `ingest()` 가 CUT3R 재귀 루프와 **같은 순서로** 돌고
탭 토큰·포즈 토큰·상태 크기가 규약대로 나오는지를 본다. 실가중치 검증은
scripts/verify_geometry_adapter.py 로 GPU 머신에서.

    PYTHONPATH=src:.refs/CUT3R/src python scripts/verify_cut3r_adapter.py
"""

from __future__ import annotations

import sys
from math import inf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_fail = 0


def check(name, cond, detail=""):
    global _fail
    print(f"  [{GREEN}PASS{RESET}]" if cond else f"  [{RED}FAIL{RESET}]", name,
          f" — {detail}" if detail else "")
    if not cond:
        _fail += 1


def tiny_cut3r(enc_dim=64, dec_dim=48, dec_depth=4, img=(64, 64)):
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig

    cfg = ARCroco3DStereoConfig(
        state_size=64,
        pos_embed="RoPE100",
        rgb_head=False,
        pose_head=True,
        img_size=img,
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=enc_dim,
        enc_depth=2,
        enc_num_heads=4,
        dec_embed_dim=dec_dim,
        dec_depth=dec_depth,
        dec_num_heads=4,
        local_mem_size=32,
        # state 디코더는 기본 16헤드다. dec_embed_dim 을 줄이면 head_dim 이 홀수가 되어
        # RoPE2D 가 "multiple of two" 로 터진다 — 소형 구성에서는 반드시 같이 줄여야 한다.
        state_dec_num_heads=4,
    )
    torch.manual_seed(0)
    return ARCroco3DStereo(cfg).eval()


def main() -> int:
    from live3r.geometry.cut3r import CUT3RStream
    from live3r.geometry.rope_patch import patch_rope2d

    patched = patch_rope2d()
    print(f"RoPE2D 음수위치 패치: {'적용' if patched else '불필요(curope 존재) 또는 실패'}")

    print("=== 소형 CUT3R 구성 ===")
    net = tiny_cut3r()
    print(f"  dec_depth={net.dec_depth}  dec_embed_dim={net.dec_embed_dim}  "
          f"pose_head={net.pose_head_flag}  params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    taps = (2, 3, 4)
    geo = CUT3RStream(net=net, tap_layers=taps, image_size=64)
    check("hidden_size == dec_embed_dim", geo.hidden_size == net.dec_embed_dim,
          f"{geo.hidden_size}")

    print("\n=== 프레임 스트리밍 ===")
    H = W = 64
    geo.reset()
    outs = []
    for t in range(6):
        outs.append(geo.ingest(torch.randn(1, 3, H, W)))
    o = outs[-1]
    gh, gw = H // geo.patch_size, W // geo.patch_size
    check("grid_hw", o.grid_hw == (gh, gw), str(o.grid_hw))
    check("탭 수", set(o.tokens) == set(taps), str(sorted(o.tokens)))
    for k, v in o.tokens.items():
        check(f"tap[{k}] 형상", tuple(v.shape) == (1, gh * gw, net.dec_embed_dim), str(tuple(v.shape)))
    check("포즈 토큰 형상", tuple(o.pose_token.shape) == (1, 1, net.dec_embed_dim),
          str(tuple(o.pose_token.shape)))

    print("\n=== 상수 상태 (라이브 불변식) ===")
    b1 = geo.state_bytes()
    for _ in range(40):
        geo.ingest(torch.randn(1, 3, H, W))
    b2 = geo.state_bytes()
    check("상태 바이트 상수", b1 == b2, f"{b1 / 1e3:.1f}KB -> {b2 / 1e3:.1f}KB")

    print("\n=== 재귀성 (상태가 실제로 쓰이는가) ===")
    geo.reset()
    f = torch.randn(1, 3, H, W)
    a = geo.ingest(f).tokens[taps[-1]]
    b = geo.ingest(f).tokens[taps[-1]]        # 같은 프레임, 다른 상태
    check("같은 입력이라도 상태가 다르면 출력이 다르다",
          not torch.allclose(a, b, atol=1e-5),
          f"max|Δ| = {(a - b).abs().max().item():.3e}")
    geo.reset()
    c = geo.ingest(f).tokens[taps[-1]]
    check("reset 후 첫 프레임은 재현된다", torch.allclose(a, c, atol=1e-5))

    print("\n=== 탭 범위 검사 ===")
    try:
        CUT3RStream(net=net, tap_layers=(11, 17, 23))
        check("범위 밖 탭 거부", False, "예외가 안 났다")
    except ValueError as e:
        check("범위 밖 탭 거부", "dec_depth" in str(e), str(e)[:60])

    print("\n=== 전체 파이프라인 (실제 CUT3R + Qwen3.5 구조) ===")
    from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
    from live3r.model.live3r import Live3RModel
    from live3r.serve.session import LiveSession

    sys.path.insert(0, str(Path(__file__).parent))
    from tiny_model import tiny_model  # noqa: E402

    cfg = Live3RConfig(
        base_model="tiny", dtype="fp32",
        geometry=GeometryConfig(name="dummy", tap_layers=taps,
                                hidden_size=net.dec_embed_dim, image_size=64),
        fusion=FusionConfig(inject_layers=(0, 1, 2), merge_size=2, zero_init=False),
        lora=LoRAConfig(enabled=False),
    )
    model = Live3RModel(cfg, tiny_model())
    model.geometry = geo          # 더미 대신 **진짜 CUT3R** 을 꽂는다
    geo.reset()

    sess = LiveSession(model, device="cpu")
    for _ in range(8):
        sess.ingest(torch.randn(3, 64, 64), torch.randn(3, 64, 64))
    summ = sess.summary()
    check("CUT3R 로 스트리밍 성립", summ["blocks"] == 4, f'blocks={summ["blocks"]}')
    check("기하 상태 상수", summ["geometry_state_bytes"] == geo.state_bytes())
    r = sess.ask(torch.randint(0, 900, (1, 4)), max_new_tokens=4)
    check("질의 응답", r["token_ids"].shape == (1, 4),
          f'ttft={r["ttft_ms"]:.1f}ms  frame={summ["frame_ms_mean"]:.1f}ms  drift={summ["drift"]:.3f}')

    print()
    if _fail:
        print(f"{RED}{_fail}개 실패{RESET}")
        return 1
    print(f"{GREEN}전부 통과 — 어댑터가 CUT3R 재귀 규약대로 돈다{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
