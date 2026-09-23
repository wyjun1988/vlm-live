"""받은 데이터셋 형식 확인 — 모델 없이, 원본 미디어를 받기 전에도 돈다.

    PYTHONPATH=src python scripts/inspect_dataset.py data/sensenova_si_800k.json \\
        --media-root data/sensenova_media --n 3

어노테이션 스키마·이미지/영상 경로 해석·placeholder 개수·대화 턴을 눈으로 본다.
전체 통계와 JSONL 변환은 scripts/prepare_annotations.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live3r.data.datasets import RecordError, load_records, normalize_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ann")
    ap.add_argument("--media-root", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scan", type=int, default=2000, help="앞에서 N개로 간단 통계")
    args = ap.parse_args()

    recs = load_records(args.ann, max_samples=args.scan)
    print(f"{args.ann}: 앞 {len(recs):,}개 읽음 ({type(recs).__name__})\n")
    first = recs[0]
    print("원본 키:", sorted(first.keys()))
    for k, v in first.items():
        print(f"  {k:14s} {type(v).__name__:6s} {str(v)[:110]!r}")

    media = Path(args.media_root) if args.media_root else None
    kinds, bad, ph_mismatch, missing = {}, 0, 0, 0
    for r in recs:
        try:
            rec = normalize_record(r)
        except RecordError:
            bad += 1
            continue
        kinds[rec.media_type] = kinds.get(rec.media_type, 0) + 1
        ph = sum(len(re.findall(r"<image>|<video>", u)) for u, _ in rec.turns)
        if rec.images and ph not in (0, len(rec.images)):
            ph_mismatch += 1
        if media and rec.images and not (media / rec.images[0]).exists():
            missing += 1
    print(f"\n미디어 유형 {kinds} · 형식 오류 {bad} · placeholder 불일치 {ph_mismatch}"
          + (f" · 첫 이미지 누락 {missing}" if media else ""))

    print(f"\n--- 샘플 {args.n}개 ---")
    for r in recs[: args.n]:
        rec = normalize_record(r)
        print(f"\nid={rec.id} media={rec.media_type} 이미지={len(rec.images or [])} 턴={len(rec.turns)}")
        for p in (rec.images or [])[:3]:
            ok = "" if media is None else ("  ✓" if (media / p).exists() else "  ✗ 없음")
            print(f"  img: {p}{ok}")
        for u, a in rec.turns[:2]:
            print(f"  Q: {u[:160]!r}")
            print(f"  A: {a[:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
