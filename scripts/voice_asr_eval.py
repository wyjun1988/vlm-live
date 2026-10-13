"""Transcription accuracy of the voice front-end's ASR on real Korean speech (LiveKit eot-bench, `ko`: 377
human-to-agent turns, 16 kHz, with a word-level reference transcript).

    python scripts/voice_asr_eval.py --asr checkpoints/qwen3-asr-1.7b --out outputs/voice/asr_ko_1.7b.json

Each turn is transcribed as the front-end would at its scoring point: the audio up to the end of speech (the start of
the annotated final silence) + 0.2 s. CER = character edit distance after dropping spaces and punctuation (Hangul,
digits and Latin letters, lower-cased), summed over turns. The reference transcript's provenance is not documented
(it looks machine-made: "20명", "010 0528"), so the CER measures agreement with it, number-format differences
included; read the saved pairs before trusting a small difference.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live3r.serve.voice import SR, QwenASR  # noqa: E402

KEEP = re.compile(r"[^0-9a-z가-힣ㄱ-ㆎ]")


def norm(s: str) -> str:
    return KEEP.sub("", s.lower())


def edits(a: str, b: str) -> int:
    """Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return int(prev[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/eot_bench/data/ko/validation-00000-of-00001.parquet")
    ap.add_argument("--asr", default="checkpoints/qwen3-asr-1.7b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--language", default="Korean", help='"auto" detects it')
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/voice/asr_ko.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf

    asr = QwenASR(args.asr, device=args.device)
    asr(np.zeros(SR, np.float32))
    lang = None if args.language == "auto" else args.language
    rows = pq.read_table(args.data).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    out, e_sum, n_sum, t_sum, a_sum = [], 0, 0, 0.0, 0.0
    t0 = time.time()
    for k, r in enumerate(rows):
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == SR
        a = a[: int((r["silence_spans"][-1]["start"] + 0.2) * SR)]
        t = time.perf_counter()
        hyp = asr(a, language=lang)
        dt = time.perf_counter() - t
        ref = " ".join(w["word"].strip() for w in r["words"])
        e, n = edits(norm(hyp), norm(ref)), max(1, len(norm(ref)))
        e_sum, n_sum, t_sum, a_sum = e_sum + e, n_sum + n, t_sum + dt, a_sum + len(a) / SR
        out.append({"id": r["id"], "seconds": len(a) / SR, "asr_s": dt, "cer": e / n, "ref": ref, "hyp": hyp})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)}  CER so far {100 * e_sum / n_sum:.2f} %  ({time.time() - t0:.0f} s)", flush=True)

    cers = np.array([o["cer"] for o in out])
    secs = np.array([o["seconds"] for o in out])
    print(f"\n{len(out)} turns, {a_sum / 60:.1f} min of speech: CER {100 * e_sum / n_sum:.2f} % "
          f"(median per turn {100 * np.median(cers):.1f} %, turns with CER 0: {100 * np.mean(cers == 0):.0f} %)")
    for lo, hi in ((0, 4), (4, 8), (8, 999)):
        sel = [o for o in out if lo <= o["seconds"] < hi]
        if sel:
            ee = sum(o["cer"] * max(1, len(norm(o["ref"]))) for o in sel)
            nn = sum(max(1, len(norm(o["ref"]))) for o in sel)
            print(f"  {lo}-{hi if hi < 999 else '...'} s: {len(sel)} turns, CER {100 * ee / nn:.2f} %")
    print(f"ASR time: {t_sum / len(out):.2f} s per turn, RTF {t_sum / a_sum:.3f} ({args.device}); "
          f"short turns (< 6 s): {np.mean([o['asr_s'] for o in out if o['seconds'] < 6]):.2f} s")
    print("\nlargest disagreements:")
    for o in sorted(out, key=lambda o: -o["cer"])[:5]:
        print(f"  CER {100 * o['cer']:.0f} %  ref: {o['ref'][:70]}\n            hyp: {o['hyp'][:70]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"asr": args.asr, "language": args.language, "cer": e_sum / n_sum, "turns": out},
              open(args.out, "w"), ensure_ascii=False, indent=0)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
