"""우리 프레임 단위 CUT3R 재현 == CUT3R 원본 `forward_recurrent` 인가 — 점맵·신뢰도·포즈로 대조.

`CUT3RStream.ingest` 는 원본의 재귀 루프 **몸통 한 스텝**을 따로 재구성한 것이다 (스트리밍이라 프레임을
하나씩 받아야 해서). 지금까지는 형상·상태 크기만 확인했다. 여기서는 같은 전처리 프레임 N 장을

    (a) 원본 ARCroco3DStereo.forward_recurrent(views)   — 시퀀스를 한 번에
    (b) 우리 CUT3RStream.ingest(frame) × N, decode_points=True — 한 장씩

에 넣고 헤드 출력(첫 프레임 좌표계 점맵 · 신뢰도 · 카메라 포즈)을 비교한다. 같으면 재귀 상태·포즈 조회·
메모리 갱신과 헤드 입력 구성이 모두 원본과 같다는 뜻이다 (장면 지도 프롬프트가 이걸 쓴다).

    PYTHONPATH=src python scripts/check_cut3r_stream.py --video data/eval/vsibench/scannet/scene0086_02.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--checkpoint", default="checkpoints/cut3r_512_dpt_4_64.pth")
    ap.add_argument("--repo", default=".refs/CUT3R")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from live3r.data.vision import geometry_frame
    from live3r.eval.consistent import read_video_frames
    from live3r.geometry.cut3r import CUT3RStream

    dev = torch.device(args.device)
    stream = CUT3RStream(checkpoint=args.checkpoint, repo_path=args.repo, device=dev)
    stream.net.to(dev).eval()
    frames, idx = read_video_frames(args.video, args.frames)
    x = [geometry_frame(f, 512, stream.patch_size).unsqueeze(0).to(dev) for f in frames]
    h, w = x[0].shape[-2:]
    print(f"프레임 {len(x)}장 (인덱스 {idx}) · 입력 {w}x{h} · {args.device}")

    # (a) 원본 — CUT3R demo.prepare_input 과 같은 view 사전 (이미지만)
    views = [{
        "img": xi, "ray_map": torch.full((1, 6, h, w), torch.nan, device=dev),
        "true_shape": torch.tensor([[h, w]], device=dev), "idx": i, "instance": str(i),
        "camera_pose": torch.eye(4, device=dev).unsqueeze(0),
        "img_mask": torch.tensor([True], device=dev), "ray_mask": torch.tensor([False], device=dev),
        "update": torch.tensor([True], device=dev), "reset": torch.tensor([False], device=dev),
    } for i, xi in enumerate(x)]
    with torch.no_grad():
        ref, _ = stream.net.forward_recurrent(views, dev)

    # (b) 우리 — 한 장씩
    stream.decode_points = True
    stream.reset()
    ours = [stream.ingest(xi) for xi in x]

    from dust3r.utils.camera import pose_encoding_to_camera  # type: ignore

    worst = 0.0
    for i, (r, o) in enumerate(zip(ref, ours)):
        p_ref = r["pts3d_in_other_view"].permute(0, 3, 1, 2)
        dp = float((p_ref - o.pointmap).abs().max())
        dc = float((r["conf"].unsqueeze(1) - o.conf).abs().max())
        dt = float((pose_encoding_to_camera(r["camera_pose"]) - o.extra["c2w"]).abs().max())
        scale = float(p_ref.abs().mean())
        worst = max(worst, dp / max(scale, 1e-6))
        print(f"  [{i}] 점맵 max|Δ| {dp:.2e} (평균 크기 {scale:.2f}m) · 신뢰도 {dc:.2e} · 포즈 {dt:.2e}")
    ok = worst < 1e-3
    print(f"\n{'원본과 일치' if ok else '불일치'} (점맵 상대 오차 최대 {worst:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
