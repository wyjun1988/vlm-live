"""채택 게이트 판정 — 같은 경로로 잰 베이스 vs 학습 결과의 lmms-eval 결과를 비교한다.

    python scripts/check_gate.py outputs/gate/base outputs/gate/trained \\
        --spatial vsibench --general videomme --reference mmstar

사전 등록 게이트 (docs/BENCHMARKS.md §4):
    1. 공간(spatial) 태스크   Δ ≥ +1.0 점
    2. 일반(general) 태스크   Δ ≥ −1.0 점   ← 특화 역설 방어
    3. 지연 (drift < 1.2, frame_ingest p95 예산 내) — 이 스크립트 밖 (bench_latency.py)

**두 결과는 반드시 같은 평가 경로(eval_path=live3r)에서 가중치만 바꿔 잰 것이어야 한다.**
경로가 다르면 차이에 입력 형식 차이가 섞인다. 베이스 = weights 없이 돌린 것 (zero-init =
베이스 VLM 과 같은 출력).

점수 단위: lmms-eval 태스크마다 0~1 이거나 0~100 이다. 두 값이 모두 1 이하면 ×100 해서
"점" 단위로 맞춘다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKIP = ("stderr", "alias", "submission", "samples")

# 태스크별 대표 지표 (lmms-eval 0.7.3 의 metric_list 에서 확인).
# "첫 번째 수치 지표"에 기대면 안 된다 — MMStar 는 첫 지표가 하위 범주 'coarse perception' 이고
# 전체 평균 'average' 는 맨 마지막이다. 그대로 두면 하위 범주 하나로 게이트를 판정하게 된다.
PRIMARY = {
    "vsibench": "vsibench_overall",
    "vsibench_debiased": "vsibench_overall",
    "videomme": "videomme_perception_score",
    "mmstar": "average",
    "mvbench": "mvbench_accuracy",
}


def latest_results(d: Path) -> dict:
    files = sorted(d.rglob("*results*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"{d} 아래에 lmms-eval 결과(*results*.json)가 없다")
    return json.loads(files[-1].read_text())["results"]


def numeric_metrics(task_res: dict) -> dict[str, float]:
    out = {}
    for k, v in task_res.items():
        if any(s in k for s in SKIP) or not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        out[k] = float(v)
    return out


def to_points(a: float, b: float) -> tuple[float, float]:
    return (a * 100, b * 100) if max(abs(a), abs(b)) <= 1.0 else (a, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("trained_dir")
    ap.add_argument("--spatial", default="vsibench", help="쉼표 구분. Δ ≥ +min_gain 이어야 한다")
    ap.add_argument("--general", default="videomme", help="쉼표 구분. Δ ≥ −max_drop 이어야 한다")
    ap.add_argument("--reference", default="mmstar", help="쉼표 구분. 표시만 하고 판정에는 안 쓴다")
    ap.add_argument("--metric", action="append", default=[],
                    help="태스크별 대표 지표 지정 task=metric (기본: PRIMARY 표. 없으면 첫 지표 + 경고)")
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--max-drop", type=float, default=1.0)
    args = ap.parse_args()

    base, trained = latest_results(Path(args.base_dir)), latest_results(Path(args.trained_dir))
    chosen = dict(m.split("=", 1) for m in args.metric)
    spatial = [t for t in args.spatial.split(",") if t]
    general = [t for t in args.general.split(",") if t]
    reference = {t for t in args.reference.split(",") if t}
    gated = set(spatial) | set(general)

    print(f"{'태스크':<22} {'지표':<34} {'베이스':>8} {'학습':>8} {'Δ(점)':>8}")
    print("-" * 86)
    primary: dict[str, float] = {}
    for task in sorted(set(base) | set(trained)):
        bm, tm = numeric_metrics(base.get(task, {})), numeric_metrics(trained.get(task, {}))
        keys = [k for k in bm if k in tm]
        if not keys:
            print(f"{task:<22} (양쪽에 공통 지표 없음)")
            continue
        want = chosen.get(task) or PRIMARY.get(task)
        main_key = next((k for k in keys if want and (k == want or k.split(",")[0] == want)), None)
        if main_key is None:
            main_key = keys[0]
            if task in gated:
                print(f"  ⚠️ {task}: 대표 지표를 몰라 첫 지표({main_key})를 썼다 — --metric {task}=<지표> 로 명시해라")
        tag = " (참고·판정 안 함)" if task in reference and task not in gated else ""
        if tag:
            print(f"{task}{tag}")
        for k in keys:
            b, t = to_points(bm[k], tm[k])
            mark = " ◀" if k == main_key else ""
            print(f"{task:<22} {k:<34} {b:8.2f} {t:8.2f} {t - b:+8.2f}{mark}")
            if k == main_key:
                primary[task] = t - b

    verdicts = []
    for task in spatial:
        d = primary.get(task)
        ok = d is not None and d >= args.min_gain
        verdicts.append(ok)
        print(f"\n[게이트1 공간] {task}: Δ={d if d is None else f'{d:+.2f}'} "
              f"(기준 ≥ +{args.min_gain}) → {'통과' if ok else '미달'}")
    for task in general:
        d = primary.get(task)
        ok = d is not None and d >= -args.max_drop
        verdicts.append(ok)
        print(f"[게이트2 일반] {task}: Δ={d if d is None else f'{d:+.2f}'} "
              f"(기준 ≥ −{args.max_drop}) → {'통과' if ok else '미달 — 특화 역설'}")
    print("[게이트3 지연] scripts/bench_latency.py 로 따로 확인 (drift < 1.2, ingest p95 예산 내)")

    passed = bool(verdicts) and all(verdicts)
    print(f"\n판정: {'게이트 1·2 통과' if passed else '기각 또는 보류'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
