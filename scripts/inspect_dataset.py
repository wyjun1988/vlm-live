"""받은 데이터셋 형식 확인 — 대용량 원본 영상을 받기 전에 먼저 돌려라.

    PYTHONPATH=src python scripts/inspect_dataset.py data/raw/spatialstack
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live3r.data.datasets import _load_annotations, _split_conversation  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = Path(sys.argv[1])
    recs = _load_annotations(root)
    print(f"{root}: {len(recs):,} 레코드\n")

    print("--- 키 빈도 ---")
    for k, c in Counter(k for r in recs for k in r).most_common():
        print(f"  {k:16s} {c:>8,} ({c / len(recs) * 100:5.1f}%)")

    print("\n--- data_source ---")
    for k, c in Counter(r.get("data_source", "?") for r in recs).most_common(15):
        print(f"  {k:28s} {c:>8,}")

    print("\n--- 참조 영상 ---")
    vids = {r["video"] for r in recs if "video" in r}
    print(f"  고유 영상 {len(vids):,}개, 샘플/영상 {len(recs) / max(1, len(vids)):.1f}")
    missing = [v for v in list(vids)[:200] if not (root / v).exists() and not Path(v).exists()]
    print(f"  (앞 200개 중) 로컬에 없는 영상 {len(missing)}개"
          + ("  → docs/DATA.md §2-2 로 원본을 받아야 한다" if missing else ""))
    for v in list(vids)[:3]:
        print(f"    예: {v}")

    print("\n--- 샘플 3개 ---")
    for r in recs[:3]:
        q, a = _split_conversation(r.get("conversations", []))
        print(f"  Q: {q[:110]}")
        print(f"  A: {a[:110]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
