"""지연 벤치 실행기.

맥(더미 기하 인코더):
    PYTHONPATH=src python scripts/bench_latency.py --tiny --frames 128
GPU 머신(실모델):
    PYTHONPATH=src python scripts/bench_latency.py --config configs/live3r_4b.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from live3r.config import Live3RConfig  # noqa: E402
from live3r.eval.latency import benchmark_stream, save_report  # noqa: E402
from live3r.model.live3r import Live3RModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live3r_dummy.yaml")
    ap.add_argument("--tiny", action="store_true", help="초소형 랜덤 LLM (맥/CI)")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--offline", action="store_true", help="오프라인 재인코딩 대비도 잰다")
    ap.add_argument("--mode", choices=["incremental", "deferred"], default="incremental",
                    help="deferred = 실제 설계 (halving 키프레임을 모았다가 질문 시점에 프리필)")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--geom-stride", type=int, default=3)
    ap.add_argument("--base-model", default=None, help="로컬 경로로 덮어쓰기")
    ap.add_argument("--geometry-checkpoint", default=None)
    ap.add_argument("--cut3r-repo", default=None)
    ap.add_argument("--scene-map", action="store_true",
                    help="(deferred) CUT3R 장면 지도 프롬프트 포함 — 헤드 디코딩·렌더·지도 토큰 비용을 잰다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    cfg = Live3RConfig.from_yaml(args.config)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.geometry_checkpoint:
        cfg.geometry.checkpoint = args.geometry_checkpoint
    if args.cut3r_repo:
        cfg.geometry.options = dict(cfg.geometry.options, repo_path=args.cut3r_repo)

    if args.tiny:
        from tiny_model import tiny_model

        cfg.dtype = "fp32"
        cfg.geometry.name = "dummy"
        cfg.geometry.hidden_size = 256
        cfg.geometry.tap_layers = tuple(range(len(cfg.fusion.inject_layers)))
        model = Live3RModel(cfg, tiny_model())
        label = "tiny+dummy"
    else:
        model = Live3RModel.from_pretrained(cfg)
        label = f"{cfg.base_model}+{cfg.geometry.name}"

    if args.mode == "deferred":
        from live3r.data.prompt import PromptBuilder
        from live3r.eval.latency import benchmark_deferred

        rep = benchmark_deferred(model, PromptBuilder.from_model(model), n_frames=args.frames,
                                 frame_hw=(args.height, args.width), budget=args.budget,
                                 geom_stride=args.geom_stride, device=device, label=label,
                                 scene_map=args.scene_map)
        print()
        print(rep.pretty())
        return 0

    rep = benchmark_stream(
        model,
        n_frames=args.frames,
        frame_hw=(args.height, args.width),
        device=device,
        label=label,
        measure_offline=args.offline,
    )
    print()
    print(rep.pretty())
    print()
    if args.out:
        save_report(rep, args.out)
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
