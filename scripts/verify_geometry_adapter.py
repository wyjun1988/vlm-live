"""기하 인코더 어댑터 실측 검증 — GPU 머신에서 먼저 돌려라.

설정 YAML 의 `geometry.hidden_size` 를 추측으로 두면 프로젝터 차원이 틀리고
학습이 조용히 망가진다. 이 스크립트가 실제 탭 토큰 차원·격자·지연·상태 크기를 찍어준다.

    PYTHONPATH=src python scripts/verify_geometry_adapter.py \
        --name cut3r --checkpoint checkpoints/cut3r_512_dpt_4_64.pth --frames 200

출력의 `hidden_size` / `grid` 를 configs/*.yaml 에 반영한 뒤 학습을 시작한다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from live3r.config import GeometryConfig  # noqa: E402
from live3r.geometry.registry import available, build_geometry_stream  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dummy", help=f"등록된 것: {available()}")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tap-layers", type=int, nargs="+", default=[11, 17, 23])
    ap.add_argument("--hidden-size", type=int, default=768, help="모르면 그대로 두고 출력값을 봐라")
    ap.add_argument("--image-size", type=int, default=512, help="긴 변 기준 (CUT3R 512 판본)")
    ap.add_argument("--repo-path", default=None, help="CUT3R 클론 경로 (PYTHONPATH 대신)")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    cfg = GeometryConfig(
        name=args.name,
        checkpoint=args.checkpoint,
        tap_layers=tuple(args.tap_layers),
        hidden_size=args.hidden_size,
        image_size=args.image_size,
        options={"repo_path": args.repo_path} if args.repo_path else {},
    )
    print(f"기하 인코더: {cfg.name}  device={device}")
    geo = build_geometry_stream(cfg).to(device).eval()
    print(f"  is_streaming = {geo.is_streaming}"
          + ("" if geo.is_streaming else "   ← 라이브 아님. 비교군 전용"))
    n_params = sum(p.numel() for p in geo.parameters())
    print(f"  파라미터      = {n_params / 1e6:.1f}M")

    # 실제 입력과 같은 모양: 4:3 영상 → 긴 변 512 → 512x384
    s = args.image_size
    frame = torch.randn(1, 3, (s * 3 // 4) // 16 * 16, s, device=device)
    geo.reset()

    print("\n--- 첫 프레임 ---")
    out = geo.ingest(frame)
    print(f"  grid_hw      = {out.grid_hw}   (토큰 {out.num_tokens}개)")
    for k, v in out.tokens.items():
        print(f"  tap[{k:>2}]      = {tuple(v.shape)}  dtype={v.dtype}")
    real_hidden = next(iter(out.tokens.values())).shape[-1]
    print(f"  → configs 의 geometry.hidden_size 는 **{real_hidden}** 이어야 한다"
          + ("  (현재 설정과 일치)" if real_hidden == args.hidden_size else
             f"  ⚠️ 현재 설정 {args.hidden_size} — 고쳐라"))
    if out.pose_token is not None:
        print(f"  pose_token   = {tuple(out.pose_token.shape)}")

    print(f"\n--- 스트리밍 {args.frames} 프레임 ---")
    for _ in range(10):  # warmup
        geo.ingest(frame)
    state_early = geo.state_bytes()
    times = []
    for i in range(args.frames):
        t0 = time.perf_counter()
        geo.ingest(frame)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    state_late = geo.state_bytes()

    n = len(times)
    early = statistics.median(times[: n // 4])
    late = statistics.median(times[-n // 4 :])
    print(f"  프레임당      p50 {statistics.median(times):6.2f} ms   "
          f"p95 {sorted(times)[int(n * 0.95)]:6.2f} ms   (30fps 예산 33)")
    print(f"  drift        {late / max(1e-9, early):6.3f}   (1.0 근처여야 스트리밍)")
    print(f"  상태 크기     {state_early / 1e6:.2f} MB → {state_late / 1e6:.2f} MB"
          + ("  OK 상수" if state_early == state_late else "  ⚠️ 커지고 있다 — 라이브 불가"))
    if device == "cuda":
        print(f"  peak GPU     {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
