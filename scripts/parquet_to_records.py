"""HF parquet(이미지 내장) → Live3R 레코드 형식(JSON + 이미지 파일).

SenseNova-SI 는 전체가 397GB 라 로컬에 못 받는다. HF 에 있는 1,000건 미리보기
(`SenseNova-SI-800K_1000samples.parquet`, 941MB)는 이미지가 parquet 안에 들어 있어서
이걸 풀면 로컬에서도 실제 데이터로 작은 S1 파일럿을 돌릴 수 있다.

    PYTHONPATH=src python scripts/parquet_to_records.py \\
        data/sensenova_1k/SenseNova-SI-800K_1000samples.parquet data/sensenova_1k

만드는 것: <out>/records.json (image 필드 = media/ 기준 상대경로), <out>/media/*.jpg|png
그다음 prepare_annotations.py 로 검증·홀드아웃 분리를 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "bin"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    import pyarrow.parquet as pq

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src)
    print(f"{src.name}: {table.num_rows} 행 · 컬럼 {table.column_names}")
    img_col = next((c for c in ("image", "images") if c in table.column_names), None)
    if img_col is None:
        raise SystemExit(f"이미지 컬럼이 없다: {table.column_names}")

    recs, n_img = [], 0
    for row in table.to_pylist():
        rid = str(row.get("id", len(recs)))
        items = row[img_col]
        if not isinstance(items, list):
            items = [items]
        paths = []
        for k, it in enumerate(items):
            if isinstance(it, dict) and it.get("bytes"):
                data = it["bytes"]
                rel = f"{rid}_{k}.{_ext(data)}"
                (media / rel).write_bytes(data)
                paths.append(rel)
                n_img += 1
            elif isinstance(it, dict) and it.get("path"):
                paths.append(it["path"])
            elif isinstance(it, str):
                paths.append(it)
            else:
                raise SystemExit(f"이미지 항목 형식을 모르겠다: {type(it)} {str(it)[:80]}")
        conv = row.get("conversations")
        recs.append({"id": rid, "image": paths, "conversations": conv,
                     **{k: row[k] for k in ("data_source", "source") if k in row}})
    (out / "records.json").write_text(json.dumps(recs, ensure_ascii=False))
    print(f"레코드 {len(recs)} · 이미지 {n_img}장 → {out}/records.json, {media}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
