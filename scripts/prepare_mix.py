"""Prepare every source of a training mix once: JSONL + offset index, read lazily by the trainer (--prepared-dir).

    python scripts/prepare_mix.py --mix ref70                 # -> data/vgllm/<name>.jsonl (+ .idx.npy, .holdout.jsonl,
                                                              #    .report.json) and data/vgllm/PREPARED.md

Eight ranks each parsing a 740 MB JSON array (and every DataLoader worker holding it) would not fit; the offset
index keeps a few MB per process. Each source goes through scripts/prepare_annotations.py (records whose placeholders
do not match their images are dropped; a small holdout per source is never trained on; a sample of the media paths is
checked under the source's media root), one source after another - parsing an array takes a few GB while it runs.
A source whose prepared file is newer than its annotation file is skipped. The table at the end: records kept,
dropped, held out, media missing, marked (spar_info - skipped in training until the marks are drawn), per source.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live3r.data.mixture import load_registry, resolve_mix  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mix", required=True, help="a mix or dataset names of the registry (percentages are ignored)")
    ap.add_argument("--registry", default="configs/data/vgllm.yaml")
    ap.add_argument("--root", default=None, help="override the registry's root (another mount point)")
    ap.add_argument("--out", default="data/vgllm")
    ap.add_argument("--holdout", type=int, default=300, help="records per source kept out of training")
    ap.add_argument("--check-files", type=int, default=1000, help="records per source whose media are checked")
    ap.add_argument("--max-images", type=int, default=32)
    ap.add_argument("--drop-marked", action="store_true",
                    help="leave out records with marks (spar_info) while training cannot draw them; the prepared "
                         "files then hold only usable records (an epoch in steps is then an epoch of real samples)")
    ap.add_argument("--force", action="store_true", help="prepare again even when up to date")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = resolve_mix(a.mix, load_registry(a.registry, a.root))
    rows, failed = [], []
    for s in sources:
        dst = out / f"{s.name}.jsonl"
        idx = Path(str(dst) + ".idx.npy")
        if not s.annotation.is_file():
            print(f"[{s.name}] {s.annotation} not found", flush=True)
            failed.append(s.name)
            continue
        rep_path = dst.with_suffix(".report.json")
        same_choice = rep_path.exists() and json.loads(rep_path.read_text()).get("drop_marked", False) == a.drop_marked
        fresh = idx.exists() and idx.stat().st_mtime > s.annotation.stat().st_mtime and same_choice
        if fresh and not a.force:
            print(f"[{s.name}] up to date", flush=True)
        else:
            t0 = time.perf_counter()
            print(f"[{s.name}] {s.annotation} -> {dst}", flush=True)
            cmd = [sys.executable, str(HERE / "prepare_annotations.py"), str(s.annotation), str(dst),
                   "--media-root", str(s.media_root), "--holdout", str(a.holdout), "--check-files", str(a.check_files),
                   "--max-images", str(a.max_images), "--records", str(s.records or 0)]
            if a.drop_marked:
                cmd.append("--drop-marked")
            rc = subprocess.run(cmd).returncode
            print(f"[{s.name}] exit {rc} in {time.perf_counter() - t0:.0f} s", flush=True)
            if rc not in (0, 2):                          # 2 = done, but more than 1 % of the checked media missing
                failed.append(s.name)
                continue
        rep = json.loads(dst.with_suffix(".report.json").read_text())
        fc = rep.get("file_check", {})
        rows.append((s.name, rep.get("total", 0), rep.get("kept", 0), sum(rep.get("dropped", {}).values()),
                     rep.get("holdout", 0), fc.get("records_with_missing", 0), fc.get("checked_records", 0),
                     rep.get("marked", 0), rep.get("media", {})))
    lines = [f"# Prepared sources of `{a.mix}`", "", f"Registry `{a.registry}` · written by scripts/prepare_mix.py", "",
             "| source | records | kept | dropped | holdout | media missing (checked) | marked | media |",
             "|---|---|---|---|---|---|---|---|"]
    for name, tot, kept, drop, ho, miss, chk, marked, media in rows:
        lines.append(f"| {name} | {tot:,} | {kept:,} | {drop:,} | {ho:,} | {miss} ({chk}) | {marked:,} "
                     f"({100 * marked / max(1, tot):.0f} %) | {media} |")
    tk = sum(r[2] for r in rows)
    # marked among the kept (holdout drawn at random); none are kept with --drop-marked
    tm = 0 if a.drop_marked else sum(round(r[7] * r[2] / max(1, r[1])) for r in rows)
    lines += ["", f"**{tk:,} records kept**, about {tm:,} of them marked (skipped while data.marks is 'skip') -> "
              f"about {tk - tm:,} usable now." + (f" FAILED: {failed}" if failed else "")]
    (out / "PREPARED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
