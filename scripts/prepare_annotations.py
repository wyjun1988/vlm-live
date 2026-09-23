"""어노테이션 검증 + JSONL/오프셋 인덱스 변환 — 학습 전에 한 번 돌린다.

    PYTHONPATH=src python scripts/prepare_annotations.py \\
        data/sensenova_si_800k.json data/sensenova.jsonl \\
        --media-root data/sensenova_media --check-files 2000

만드는 것:
    data/sensenova.jsonl          검증 통과한 레코드만, 한 줄에 하나
    data/sensenova.jsonl.idx.npy  줄 시작 바이트 오프셋 → 학습 때 지연 로딩 (프로세스당 수 MB)
    data/sensenova.report.json    통계·버린 사유

왜 필요한가:
  * 83만 레코드 JSON 배열을 DDP 8프로세스 × DataLoader 워커가 각자 통째로 파싱하면
    메모리가 수십 GB 로 불어난다. 인덱스가 있으면 필요한 줄만 읽는다.
  * placeholder 수와 이미지 수가 안 맞는 레코드는 "어느 질문이 어느 이미지를 보는지"가
    깨져 있다. 학습 중에 조용히 건너뛰는 대신 **여기서 개수를 보고** 판단한다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402

from live3r.data.datasets import RecordError, normalize_record  # noqa: E402

_PH = re.compile(r"<image>|<video>")
_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_MC = re.compile(r"^\(?[A-H]\)?[.:)]?(\s|$)")


def answer_kind(a: str) -> str:
    a = a.strip()
    low = a.lower().rstrip(".")
    if _NUM.match(a):
        return "numeric"
    if _MC.match(a):
        return "multiple_choice"
    if low in ("yes", "no"):
        return "yes_no"
    return "free_text"


def iter_raw(path: Path):
    with open(path, "rb") as fh:
        head = fh.read(4096).lstrip()
    if head.startswith(b"["):
        print(f"JSON 배열로 읽는다 (한 번만 전체 로드): {path}", flush=True)
        for rec in json.loads(path.read_text()):
            yield rec
    else:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="원본 어노테이션 (json 배열 또는 jsonl)")
    ap.add_argument("dst", help="출력 jsonl 경로")
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--check-files", type=int, default=2000,
                    help="존재 여부를 실제로 확인할 레코드 수 (무작위). 0=안 함, -1=전부")
    ap.add_argument("--max-images", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (스모크용 부분집합)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    media = Path(args.media_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    stats = Counter()
    dropped = Counter()
    n_images = Counter()
    n_turns = Counter()
    kinds = Counter()
    sources = Counter()
    missing_examples: list[str] = []
    checked = missing = 0
    offsets: list[int] = []

    with open(dst, "wb") as out:
        for i, raw in enumerate(iter_raw(src)):
            if args.limit and i >= args.limit:
                break
            stats["total"] += 1
            try:
                rec = normalize_record(raw)
            except RecordError as exc:
                dropped["형식:" + str(exc)[:40]] += 1
                continue

            imgs = rec.images or []
            ph = sum(len(_PH.findall(u)) for u, _ in rec.turns)
            if imgs and ph not in (0, len(imgs)):
                dropped["placeholder≠이미지수"] += 1
                continue
            if len(imgs) > args.max_images:
                dropped[f"이미지>{args.max_images}"] += 1
                continue

            denom = args.limit or 850_000
            do_check = args.check_files < 0 or (
                args.check_files > 0 and rng.random() < args.check_files / denom
            )
            if do_check and imgs:
                checked += 1
                bad = [p for p in imgs if not (media / p).exists()]
                if bad:
                    missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(str(media / bad[0]))

            n_images[min(len(imgs), 30)] += 1
            n_turns[min(len(rec.turns), 10)] += 1
            for _, a in rec.turns:
                kinds[answer_kind(a)] += 1
            sources[rec.data_source] += 1

            offsets.append(out.tell())
            out.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
            stats["kept"] += 1
            if stats["total"] % 100_000 == 0:
                print(f"  {stats['total']:,} 처리 / {stats['kept']:,} 유지", flush=True)

    np.save(str(dst) + ".idx.npy", np.asarray(offsets, dtype=np.int64))

    report = {
        "src": str(src),
        "total": stats["total"],
        "kept": stats["kept"],
        "dropped": dict(dropped),
        "images_per_record": {str(k): v for k, v in sorted(n_images.items())},
        "turns_per_record": {str(k): v for k, v in sorted(n_turns.items())},
        "answer_kinds": dict(kinds),
        "data_sources_top": dict(sources.most_common(20)),
        "file_check": {"checked_records": checked, "records_with_missing": missing,
                       "examples": missing_examples},
    }
    rpt = dst.with_suffix(".report.json")
    rpt.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print()
    print(f"총 {report['total']:,} → 유지 {report['kept']:,}")
    if dropped:
        print("버린 사유:", dict(dropped))
    print("레코드당 이미지:", report["images_per_record"])
    print("레코드당 턴   :", report["turns_per_record"])
    print("답변 유형     :", dict(kinds))
    print(f"파일 확인     : {checked}건 중 누락 {missing}건", missing_examples[:2])
    print(f"\n출력: {dst}  (+ .idx.npy, {rpt.name})")
    if checked and missing / checked > 0.01:
        print("⚠️ 누락 비율이 1% 를 넘는다 — --media-root 가 맞는지 먼저 확인해라")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
