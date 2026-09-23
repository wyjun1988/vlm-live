"""키프레임 선택기 비교 — 모델 없이 커버리지 성질만 빠르게 본다.

    PYTHONPATH=src python scripts/bench_selectors.py

라이브에서 "미래를 보는 균등 샘플링"을 잃는 비용이 얼마나 되는지 감을 잡는 용도다.
실제 점수는 scripts/run_streaming_eval.sh 로 재야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from live3r.eval.streaming import (  # noqa: E402
    HalvingSelector,
    ReservoirSelector,
    StrideSelector,
    UniformOracleSelector,
)


def run(make, T, budget):
    sel = make(T)
    for i in range(T):
        sel.offer(i, None)
    k = sel.final_selection()
    d = torch.tensor([float(x) for x in k]).diff()
    return {
        "n": len(k),
        "use": len(k) / budget,
        "std": d.std().item() if len(d) > 1 else 0.0,
        "tail": k[-1] / (T - 1) if T > 1 else 1.0,
        "causal": sel.is_causal,
    }


def main() -> int:
    budget = 32
    cands = [
        ("halving(ob2)", lambda T: HalvingSelector(budget, overbuffer=2)),
        ("halving(ob4)", lambda T: HalvingSelector(budget, overbuffer=4)),
        ("reservoir", lambda T: ReservoirSelector(budget, seed=0)),
        ("stride(x8)", lambda T: StrideSelector(budget, stride=8)),
        ("uniform(oracle)", lambda T: UniformOracleSelector(budget, T)),
    ]
    print(f"예산 {budget} 프레임\n")
    print(f"{'스트림':>7} {'선택기':<16} {'인과':>4} {'개수':>6} {'예산사용':>8} {'간격std':>9} {'꼬리도달':>8}")
    print("-" * 66)
    for T in (300, 900, 1800, 3600, 9000):
        for name, mk in cands:
            r = run(mk, T, budget)
            print(f"{T:>7} {name:<16} {'O' if r['causal'] else 'X':>4} "
                  f"{r['n']:>6} {r['use']:>7.0%} {r['std']:>9.1f} {r['tail']:>8.2f}")
        print()
    print("읽는 법: 간격std 는 낮을수록, 꼬리도달은 1.0 에 가까울수록 오프라인 균등추출에 가깝다.")
    print("uniform(oracle) 은 총 길이를 보므로 라이브가 아니다 — 상한선 참고용.")
    print("ob=overbuffer: 버퍼를 예산의 몇 배로 두는지. 크면 균등해지지만 기하토큰 버퍼 메모리가 는다.")
    print("주의: 여기 수치는 프록시다. 실제로 VSI 점수를 얼마나 바꾸는지는 베이크오프에서 재야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
