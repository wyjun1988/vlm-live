"""Read the training data of the 70.83 reference (the VG-LLM data on the shared volume) - the data only, never code.

    python scripts/inspect_reference_data.py                       # every dataset of configs/data/vgllm.yaml
    python scripts/inspect_reference_data.py --names spar_234k,sqa3d
    -> outputs/reference_data/DATA.md, summary.json, records/<name>.json  and  reference_data_bundle.tar.gz (to send)

What our own loader needs to know, per dataset:
  * records: the count, the fields and their types, extra fields (e.g. question types)
  * media: images per record or a video (file or frame directory), placeholders (<image>/<video>) against media,
    the directory layout of the paths, whether they resolve under the media root, image sizes, video length and fps
  * conversations: turns, answer kinds (number / option / yes-no / text), answer lengths, chain-of-thought markers
  * marks: `spar_info` - how often, its form and keys, examples (the questions that refer to points or boxes drawn
    on the frames)
  * VSI-Bench overlap: scene ids found in the media paths against VSI-Bench's 288 scenes (configs/data/
    vsibench_scenes.txt). An overlap would put evaluation scenes in the training data.
Record samples: the first 3 and 5 random records of every dataset, whole.

Only the annotation files and media under the registry's paths are opened (read-only); CPU only, safe next to a
training run. JSON arrays are parsed one record at a time (the text is held once, the records are not).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live3r.data.datasets import iter_json_records as iter_records  # noqa: E402
from live3r.data.mixture import Source, load_registry  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
PLACEHOLDER = re.compile(r"<image>|<video>")
SCENE_PATTERNS = {                      # scene ids as they appear in paths of the three VSI-Bench collections
    "scannet": re.compile(r"(scene\d{4}_\d{2})"),
    "arkitscenes": re.compile(r"(?<![0-9a-f])(4\d{7})(?![0-9a-f])"),
    "scannetpp": re.compile(r"(?<![0-9a-z])([0-9a-f]{10})(?![0-9a-z])"),
}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
OPTION = re.compile(r"^\(?[A-H]\)?(?:[.:)]|$)")     # "B", "B.", "(B)", "B. chair" - not "A man ..."
COT = re.compile(r"<think>|</think>|step by step|let me|first,|therefore", re.I)


# --------------------------------------------------------------------------------------------- reading
def conversations(rec: dict) -> list[dict]:
    conv = rec.get("conversations")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            return []
    return conv if isinstance(conv, list) else []


def media_of(rec: dict) -> tuple[str, list[str]]:
    """('images' | 'video' | 'none', paths)."""
    img = rec.get("image", rec.get("images"))
    if isinstance(img, str):
        img = [img]
    if img:
        return "images", [str(x) for x in img]
    vid = rec.get("video", rec.get("videos"))
    if isinstance(vid, str):
        return "video", [vid]
    if isinstance(vid, list) and vid:
        return "video", [str(x) for x in vid]
    return "none", []


def answer_kind(a: str) -> str:
    a = a.strip()
    if NUM.match(a):
        return "number"
    if OPTION.match(a):
        return "option"
    if a.lower().rstrip(".") in ("yes", "no"):
        return "yes/no"
    return "text"


def shape_of(v, depth: int = 0) -> str:
    """A short type signature of a value (for spar_info and unknown fields)."""
    if isinstance(v, dict):
        if depth > 1:
            return "dict"
        return "{" + ", ".join(f"{k}: {shape_of(x, depth + 1)}" for k, x in list(v.items())[:8]) + "}"
    if isinstance(v, list):
        return f"list[{len(v)}]" + (f" of {shape_of(v[0], depth + 1)}" if v and depth <= 1 else "")
    if isinstance(v, str):
        return "str"
    return type(v).__name__


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f} %" if b else "-"


def quantiles(xs: list[float]) -> str:
    if not xs:
        return "-"
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
    return f"min {s[0]:g} · median {q(0.5):g} · p95 {q(0.95):g} · max {s[-1]:g}"


# --------------------------------------------------------------------------------------------- per dataset
class Stats:
    def __init__(self, src: Source, sample_media: int, seed: int) -> None:
        self.src = src
        self.n = 0
        self.keys: Counter = Counter()
        self.kind: Counter = Counter()
        self.n_media: list[int] = []
        self.turns: Counter = Counter()
        self.mismatch = 0
        self.answer_kind: Counter = Counter()
        self.answer_len: list[int] = []
        self.cot = 0
        self.spar = 0
        self.spar_form: Counter = Counter()
        self.spar_keys: Counter = Counter()
        self.spar_examples: list = []
        self.layout: Counter = Counter()
        self.ext: Counter = Counter()
        self.fields: dict[str, Counter] = {}
        self.scenes: dict[str, set] = {k: set() for k in SCENE_PATTERNS}
        self.first: list[dict] = []
        self.reservoir: list[dict] = []
        self.media_pool: list[tuple[str, list[str]]] = []
        self.sample_media = sample_media
        self.rng = random.Random(seed)

    def add(self, rec: dict) -> None:
        self.n += 1
        if not isinstance(rec, dict):
            self.keys["<not a dict>"] += 1
            return
        self.keys.update(rec.keys())
        kind, paths = media_of(rec)
        self.kind[kind] += 1
        self.n_media.append(len(paths))
        for p in paths[:2]:
            parts = Path(p).parts
            self.layout["/".join(parts[:2]) if len(parts) > 2 else (parts[0] if parts else "")] += 1
            self.ext[Path(p).suffix.lower() or "(none: directory?)"] += 1
        for p in paths:
            for coll, rx in SCENE_PATTERNS.items():
                m = rx.search(p)
                if m:
                    self.scenes[coll].add(m.group(1))
        conv = conversations(rec)
        humans = [c for c in conv if (c.get("from") or c.get("role")) in ("human", "user")]
        gpts = [c for c in conv if (c.get("from") or c.get("role")) in ("gpt", "assistant")]
        self.turns[len(gpts)] += 1
        n_ph = sum(len(PLACEHOLDER.findall(str(c.get("value", c.get("content", ""))))) for c in humans)
        if n_ph and kind == "images" and n_ph != len(paths):
            self.mismatch += 1
        if n_ph and kind == "video" and n_ph != 1:
            self.mismatch += 1
        if gpts:
            a = str(gpts[0].get("value", gpts[0].get("content", "")))
            self.answer_kind[answer_kind(a)] += 1
            self.answer_len.append(len(a))
            self.cot += bool(COT.search(a)) or len(a) > 400
        for k, v in rec.items():
            if k in ("conversations", "image", "images", "video", "videos", "id", "spar_info"):
                continue
            if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 60:
                self.fields.setdefault(k, Counter())[str(v)] += 1
            else:
                self.fields.setdefault(k, Counter())[shape_of(v)] += 1
        si = rec.get("spar_info")
        if si not in (None, "", {}, []):
            self.spar += 1
            parsed = si
            if isinstance(si, str):
                try:
                    parsed = json.loads(si)
                    self.spar_form["str (JSON)"] += 1
                except json.JSONDecodeError:
                    self.spar_form["str (not JSON)"] += 1
            else:
                self.spar_form[type(si).__name__] += 1
            if isinstance(parsed, dict):
                self.spar_keys.update(parsed.keys())
            if len(self.spar_examples) < 8 and self.rng.random() < 0.5:
                q = str(humans[0].get("value", ""))[:300] if humans else ""
                self.spar_examples.append({"spar_info": si, "question": q, "n_media": len(paths)})
        if len(self.first) < 3:
            self.first.append(rec)
        # reservoirs: 5 whole records, `sample_media` media lists
        if len(self.reservoir) < 5:
            self.reservoir.append(rec)
        elif (j := self.rng.randrange(self.n)) < 5:
            self.reservoir[j] = rec
        if paths:
            if len(self.media_pool) < self.sample_media:
                self.media_pool.append((kind, paths))
            elif (j := self.rng.randrange(self.n)) < self.sample_media:
                self.media_pool[j] = (kind, paths)


def check_media(st: Stats, probe: int) -> dict:
    """Resolve the sampled media under the media root; sizes of some images, length and fps of some videos."""
    root = st.src.media_root
    found = missing = 0
    missing_ex: list[str] = []
    sizes: list[tuple[int, int]] = []
    videos: list[dict] = []
    for kind, paths in st.media_pool:
        for p in paths:
            q = Path(p) if Path(p).is_absolute() else root / p
            if q.exists():
                found += 1
            else:
                missing += 1
                if len(missing_ex) < 5:
                    missing_ex.append(str(q))
                continue
            if kind == "images" and len(sizes) < probe:
                try:
                    from PIL import Image

                    with Image.open(q) as im:
                        sizes.append(im.size)
                except Exception:  # noqa: BLE001
                    pass
            if kind == "video" and len(videos) < probe:
                videos.append(probe_video(q))
    return {"found": found, "missing": missing, "missing_examples": missing_ex, "image_sizes": Counter(sizes),
            "videos": videos}


def probe_video(q: Path) -> dict:
    if q.is_dir():
        frames = sorted(f.name for f in q.iterdir() if f.suffix.lower() in IMAGE_EXT)
        return {"path": q.name, "kind": "frame directory", "frames": len(frames),
                "names": frames[:3] + (["..."] if len(frames) > 3 else [])}
    out = {"path": q.name, "kind": "file", "bytes": q.stat().st_size}
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(str(q))
        out.update(frames=len(vr), fps=round(float(vr.get_avg_fps()), 2))
        return out
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(q))
        out.update(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=round(cap.get(cv2.CAP_PROP_FPS), 2))
        cap.release()
    except Exception:  # noqa: BLE001
        out["probe"] = "no decord/cv2"
    return out


def vsi_scenes(path: Path) -> dict[str, set]:
    out: dict[str, set] = {"scannet": set(), "arkitscenes": set(), "scannetpp": set()}
    if not path.is_file():
        return out
    for ln in path.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            coll, sid = ln.split()
            out.setdefault(coll, set()).add(sid)
    return out


# --------------------------------------------------------------------------------------------- report
def report(st: Stats, media: dict, vsi: dict[str, set], seconds: float) -> tuple[list[str], dict]:
    s, n = st.src, st.n
    L = [f"## {s.name}", "",
         f"`{s.annotation}` · media root `{s.media_root}` · **{n:,} records** "
         f"(registry: {s.records if s.records is not None else '-'}{'' if s.records in (None, n) else ' - DIFFERS'}) "
         f"· read in {seconds:.0f} s", ""]
    L += ["| | |", "|---|---|",
          f"| fields | {', '.join(f'`{k}` {pct(v, n)}' for k, v in st.keys.most_common(15))} |",
          f"| media | {', '.join(f'{k} {pct(v, n)}' for k, v in st.kind.most_common())} · per record "
          f"{quantiles(st.n_media)} |",
          f"| path layout (first two parts) | {', '.join(f'`{k}` {v}' for k, v in st.layout.most_common(6))} |",
          f"| extensions | {', '.join(f'`{k}` {v}' for k, v in st.ext.most_common(6))} |",
          f"| placeholders != media | {st.mismatch} ({pct(st.mismatch, n)}) |",
          f"| answer turns per record | {', '.join(f'{k}: {v}' for k, v in sorted(st.turns.items())[:8])} |",
          f"| first answer | {', '.join(f'{k} {pct(v, n)}' for k, v in st.answer_kind.most_common())} · "
          f"chars {quantiles(st.answer_len)} |",
          f"| reasoning-like answers | {st.cot} ({pct(st.cot, n)}) |",
          f"| spar_info | {st.spar} ({pct(st.spar, n)}) · form {dict(st.spar_form)} · keys "
          f"{', '.join(f'`{k}` {v}' for k, v in st.spar_keys.most_common(12)) or '-'} |"]
    for k, c in sorted(st.fields.items()):
        L.append(f"| field `{k}` | {', '.join(f'`{v}` {m}' for v, m in c.most_common(8))}"
                 f"{' ...' if len(c) > 8 else ''} ({len(c)} distinct) |")
    tot = media["found"] + media["missing"]
    L.append(f"| media resolved (sample) | {media['found']}/{tot} ({pct(media['found'], tot)})"
             f"{' - missing e.g. ' + ', '.join(f'`{m}`' for m in media['missing_examples'][:2]) if media['missing'] else ''} |")
    if media["image_sizes"]:
        L.append(f"| image sizes (sample) | {', '.join(f'{w}x{h} ×{c}' for (w, h), c in media['image_sizes'].most_common(5))} |")
    if media["videos"]:
        v = media["videos"]
        fps = [x["fps"] for x in v if x.get("fps")]
        fr = [x["frames"] for x in v if x.get("frames")]
        L.append(f"| videos (sample of {len(v)}) | {Counter(x['kind'] for x in v)} · frames {quantiles(fr)} · "
                 f"fps {quantiles(fps)} · e.g. {json.dumps(v[0], ensure_ascii=False)[:160]} |")
    overlap = {}
    for coll, ids in st.scenes.items():
        if ids:
            ov = ids & vsi.get(coll, set())
            overlap[coll] = {"scenes": len(ids), "in_vsibench": len(ov), "examples": sorted(ov)[:20]}
    scenes = ", ".join(f"{c}: {o['scenes']} scenes, **{o['in_vsibench']} in VSI-Bench**" for c, o in overlap.items())
    L.append(f"| scene ids in paths | {scenes or 'none recognised'} |")
    if st.spar_examples:
        L += ["", "spar_info examples:", ""]
        for ex in st.spar_examples[:4]:
            L.append(f"- `{json.dumps(ex['spar_info'], ensure_ascii=False)[:400]}` · media {ex['n_media']} · "
                     f"Q: {ex['question'][:200]!r}")
    L.append("")
    summary = {"name": s.name, "records": n, "fields": dict(st.keys), "media": dict(st.kind),
               "media_per_record": quantiles(st.n_media), "mismatch": st.mismatch, "turns": dict(st.turns),
               "answer_kind": dict(st.answer_kind), "reasoning_like": st.cot, "spar_info": st.spar,
               "spar_form": dict(st.spar_form), "spar_keys": dict(st.spar_keys), "layout": dict(st.layout.most_common(20)),
               "media_found": media["found"], "media_missing": media["missing"], "videos": media["videos"],
               "vsibench_overlap": overlap}
    return L, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", default="configs/data/vgllm.yaml")
    ap.add_argument("--root", default=None, help="override the registry's root (another mount point)")
    ap.add_argument("--names", default=None, help="comma list of datasets (default: all in the registry)")
    ap.add_argument("--vsi-scenes", default="configs/data/vsibench_scenes.txt")
    ap.add_argument("--sample-media", type=int, default=200, help="records per dataset whose media are resolved")
    ap.add_argument("--probe", type=int, default=20, help="images sized / videos probed per dataset")
    ap.add_argument("--out", default="outputs/reference_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    reg = load_registry(a.registry, a.root)
    names = [x.strip() for x in a.names.split(",")] if a.names else list(reg.datasets)
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    vsi = vsi_scenes(Path(a.vsi_scenes))
    lines = ["# The VG-LLM training data, read (data only)", "",
             f"Registry `{a.registry}`, {len(names)} datasets, VSI-Bench scenes from `{a.vsi_scenes}` "
             f"({sum(len(v) for v in vsi.values())}). Written by scripts/inspect_reference_data.py.", ""]
    summaries = []
    for name in names:
        src = reg.datasets[name]
        print(f"[{name}] {src.annotation}", flush=True)
        if not src.annotation.is_file():
            lines += [f"## {name}", "", f"`{src.annotation}` not found.", ""]
            continue
        t0 = time.perf_counter()
        st = Stats(src, a.sample_media, a.seed)
        try:
            for rec in iter_records(src.annotation):
                st.add(rec)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the others
            lines += [f"## {name}", "", f"read failed after {st.n:,} records: {exc!r}", ""]
            continue
        media = check_media(st, a.probe)
        L, summ = report(st, media, vsi, time.perf_counter() - t0)
        lines += L
        summaries.append(summ)
        (out / "records" / f"{name}.json").write_text(json.dumps({"first": st.first, "random": st.reservoir},
                                                                 indent=1, ensure_ascii=False, default=str))
        (out / "DATA.md").write_text("\n".join(lines) + "\n")          # after every dataset: partial results survive
        print(f"  {st.n:,} records, {time.perf_counter() - t0:.0f} s", flush=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=1, ensure_ascii=False, default=str))
    tar = out / "reference_data_bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for f in ["DATA.md", "summary.json", "records"]:
            t.add(out / f, arcname=f"reference_data/{f}")
    print(f"{out / 'DATA.md'}\n{tar} ({tar.stat().st_size / 1e6:.1f} MB) - send this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
