"""One Markdown report of everything the weekend run produced (scripts/server_weekend.sh).

    python scripts/weekend_report.py outputs/weekend > outputs/weekend/REPORT.md

Safe to run at any time: each section reports what exists and names what is missing, so the report is useful
while the run is going and after a partial failure. No GPU, no model, no lmms-eval import. VSI comparisons are a
video-level paired bootstrap on the per-question scores that eval_vsi_local.py saves (both runs must have
answered the same questions).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# lmms-eval's VSI aggregation (_compute_all_subscores): the mean of each question type, the three relative-
# direction difficulties averaged into one type first, then the mean over the 8 types
REL_DIR = "object_rel_direction"
TYPE_SHORT = {
    "obj_appearance_order": "appear", "object_abs_distance": "abs dist", "object_counting": "count",
    "object_rel_distance": "rel dist", "object_size_estimation": "obj size", "room_size_estimation": "room size",
    "route_planning": "route", REL_DIR: "rel dir",
}
PHASES = [("p0", "preflight"), ("p1", "smoke (training)"), ("p1e", "smoke (evaluation paths)"),
          ("p2_vsibench", "VSI-Bench data"), ("p2_mmstar", "MMStar data"), ("p2_videomme", "VideoMME data"),
          ("p3", "zero-shot VSI baseline"), ("s1_real", "S1 real"), ("s1_control", "S1 control"),
          ("p4", "S1 pair"), ("p5", "S1 ablation"), ("s2_real", "S2 real"), ("s2_control", "S2 control"),
          ("p6", "S2 pair"), ("p7", "evaluation")]
ARMS = ["s1_real", "s1_control", "s2_real", "s2_control"]
# (label, run A, run B): A - B. A run is "<vsi json name>:<mode>"; "base_best" is the base's better format.
COMPARISONS = [
    ("Base: image mode - video mode (the format confound)", "base_plain:oracle-image", "base_plain:oracle-video"),
    ("Base: + format hint - video mode (gate baseline)", "base_hint:oracle-image", "base_plain:oracle-video"),
    ("Base: routed facts - format hint (value of measurements)", "base_routed:oracle-image", "base_hint:oracle-image"),
    ("Base: routed facts - video mode (best zero-shot)", "base_routed:oracle-image", "base_plain:oracle-video"),
    ("S1: real - control (geometry content, S1)", "s1_real:oracle-image", "s1_control:oracle-image"),
    ("S2: real - control (geometry content, S2)  <- key", "s2_real:oracle-image", "s2_control:oracle-image"),
    ("S2 real - base best format (gate 1, local eval)", "s2_real:oracle-image", "base_best"),
    ("S2 real - base + format hint (training vs one line)", "s2_real:oracle-image", "base_hint:oracle-image"),
    ("S2 real: + format hint - plain", "s2_real_hint:oracle-image", "s2_real:oracle-image"),
    ("S2 real: routed - plain (measurements after training)", "s2_real_routed:oracle-image", "s2_real:oracle-image"),
    ("Routed: S2 real - base (training on top of prompting)", "s2_real_routed:oracle-image", "base_routed:oracle-image"),
    ("Routed: S2 real - S2 control", "s2_real_routed:oracle-image", "s2_control_routed:oracle-image"),
]


# ------------------------------------------------------------------------------------------------ VSI
def vsi_type(t: str) -> str:
    return REL_DIR if t.startswith(REL_DIR) else t


def overall(preds: list[dict]) -> tuple[float, dict[str, float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for p in preds:
        by[p["type"]].append(p["score"])
    means = {t: float(np.mean(v)) for t, v in by.items()}
    rel = [means.pop(t) for t in list(means) if t.startswith(REL_DIR)]
    if rel:
        means[REL_DIR] = float(np.mean(rel))
    return (float(np.mean(list(means.values()))) if means else float("nan")), means


def paired_bootstrap(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0):
    """Δ = overall(a) - overall(b) with a 95% interval, resampling videos (questions of a video move together)."""
    ka = {p["id"]: p for p in a}
    kb = {p["id"]: p for p in b}
    ids = sorted(set(ka) & set(kb))
    if not ids:
        return None
    types = sorted({p["type"] for p in ka.values()})
    vids = sorted({ka[i]["video"] for i in ids})
    ti, vi = {t: k for k, t in enumerate(types)}, {v: k for k, v in enumerate(vids)}
    cnt = np.zeros((len(vids), len(types)))
    sa, sb = np.zeros_like(cnt), np.zeros_like(cnt)
    for i in ids:
        r, c = vi[ka[i]["video"]], ti[ka[i]["type"]]
        cnt[r, c] += 1
        sa[r, c] += ka[i]["score"]
        sb[r, c] += kb[i]["score"]
    rel = [k for k, t in enumerate(types) if t.startswith(REL_DIR)]
    other = [k for k, t in enumerate(types) if not t.startswith(REL_DIR)]

    def score(s, n):  # rows already summed over the resampled videos: [..., types]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / n
        parts = [m[..., other]] + ([np.nanmean(m[..., rel], axis=-1, keepdims=True)] if rel else [])
        return np.nanmean(np.concatenate(parts, axis=-1), axis=-1)

    point = score(sa.sum(0), cnt.sum(0)) - score(sb.sum(0), cnt.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    d = score(sa[idx].sum(1), cnt[idx].sum(1)) - score(sb[idx].sum(1), cnt[idx].sum(1))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return 100 * float(point), 100 * float(lo), 100 * float(hi), len(vids), len(ids)


def load_vsi(out: Path) -> dict[str, dict]:
    runs = {}
    for f in sorted((out / "vsi").glob("*.json")):
        try:
            runs[f.stem] = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def vsi_preds(runs: dict, spec: str) -> tuple[str, list[dict] | None]:
    if spec == "base_best":  # the base's better format, as in gate 1
        best = None
        for mode in ("oracle-image", "oracle-video"):
            r = runs.get("base_plain", {})
            if mode in r.get("scores", {}) and (best is None or r["scores"][mode]["overall"] > best[1]):
                best = (mode, r["scores"][mode]["overall"])
        return (f"base_plain:{best[0]}", runs["base_plain"]["predictions"][best[0]]) if best else (spec, None)
    name, _, mode = spec.partition(":")
    preds = runs.get(name, {}).get("predictions", {}).get(mode)
    if preds and not all("score" in p and "video" in p for p in preds):
        preds = None                     # an output from before scores were saved - cannot be paired
    return spec, preds


def section_vsi(out: Path) -> list[str]:
    runs = load_vsi(out)
    lines = ["## VSI-Bench (local harness, live path: 32 uniform keyframes)", ""]
    if not runs:
        return lines + ["No VSI results yet (outputs in `vsi/`).", ""]
    cols = list(TYPE_SHORT)
    lines += ["| run | mode | overall | " + " | ".join(TYPE_SHORT[c] for c in cols) + " | format fail | n |",
              "|---|---|---|" + "---|" * len(cols) + "---|---|"]
    for name, r in runs.items():
        for mode, sc in r.get("scores", {}).items():
            per = {c: None for c in cols}
            for k, v in sc.items():
                t = k.split("_MRA")[0].removesuffix("_accuracy")
                if t in per:
                    per[t] = v
            ff = r.get("format_fail", {}).get(mode)
            cells = " | ".join("-" if per[c] is None else f"{100 * per[c]:.1f}" for c in cols)
            lines.append(f"| {name} | {mode} | **{100 * sc['overall']:.1f}** | {cells} | "
                         f"{'-' if ff is None else f'{100 * ff:.1f}%'} | {r.get('n_questions', '-')} |")
    lines += ["", "M2 reference (60 videos, 1,151 questions): base video mode 48.6 · + format hint 53.7 · "
              "routed facts 56.0 · S2 pilot (905 samples) 44.1 vs control 42.3.", "",
              "### Paired comparisons (Δ in points, 95% interval from resampling videos)", "",
              "| comparison | A | B | Δ | 95% interval | videos / questions |", "|---|---|---|---|---|---|"]
    for label, a, b in COMPARISONS:
        sa, pa = vsi_preds(runs, a)
        sb, pb = vsi_preds(runs, b)
        if not pa or not pb:
            continue
        res = paired_bootstrap(pa, pb)
        if res is None:
            lines.append(f"| {label} | {sa} | {sb} | no common questions | | |")
            continue
        d, lo, hi, nv, nq = res
        sig = " *" if lo > 0 or hi < 0 else ""
        lines.append(f"| {label} | {sa} | {sb} | **{d:+.1f}**{sig} | [{lo:+.1f}, {hi:+.1f}] | {nv} / {nq} |")
    lines += ["", "`*` = the interval excludes 0.", ""]
    return lines


# ------------------------------------------------------------------------------------------------ training
LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?step (?P<step>\d+)/(?P<total>\d+) loss (?P<loss>[\d.]+) "
                  r"lr (?P<lr>\S+) \| inj (?P<inj>.*?) \| gate (?P<gate>.*?) \| (?P<sps>[\d.]+) samp/s .*?"
                  r"\| mem (?P<mem>\S+) \| skip (?P<skip>\d+) \| sync (?P<sync>\S+)")


def parse_train_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    rows = [m.groupdict() for m in map(LINE.match, text.splitlines()) if m]
    info: dict = {"rows": rows, "error": None, "early": None, "donor_same": None}
    for line in text.splitlines():
        if "--max-hours" in line and "reached" in line:
            info["early"] = line.split("WARNING", 1)[-1].strip()
        m = re.search(r"같은 격자 기증자 (\d+)%", line)
        if m:
            info["donor_same"] = int(m.group(1))
    tb = text.rfind("Traceback")
    if tb >= 0:
        tail = [ln for ln in text[tb:].splitlines() if ln.strip()]
        info["error"] = tail[-1][:200] if tail else "Traceback"
    return info


def fmt_hours(rows: list[dict]) -> str:
    if len(rows) < 2:
        return "-"
    t0, t1 = (datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S") for r in (rows[0], rows[-1]))
    return f"{(t1 - t0).total_seconds() / 3600:.1f} h"


def section_training(out: Path) -> list[str]:
    lines = ["## Training", ""]
    env = {}
    for f in ("budget.env", "budget_s2.env"):
        p = out / f
        if p.exists():
            env.update(dict(ln.split("=", 1) for ln in p.read_text().split() if "=" in ln))
    if env:
        lines += ["Budget: " + " · ".join(f"{k} {v}" for k, v in env.items()) + " (steps x 128 = samples per arm)", ""]
    lines += ["| run | steps | first -> last loss | last inj (L0..) | samples/s | peak mem | skipped | sync | "
              "elapsed | notes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ["smoke_a", "smoke_b", "smoke_c", "smoke_d"] + ARMS:
        log = out / (f"p1_{name}.log" if name.startswith("smoke") else f"{name}.log")
        info = parse_train_log(log)
        if info is None:
            continue
        rows = info["rows"]
        notes = []
        if info["early"]:
            notes.append("stopped by --max-hours")
        if info["donor_same"] is not None:
            notes.append(f"same-grid donors {info['donor_same']}%")
        if info["error"]:
            notes.append(f"error: {info['error']}")
        if (out / f"{name}.done").exists() or (out / name / "final.pt").exists():
            notes.append("final.pt")
        if not rows:
            lines.append(f"| {name} | - | - | - | - | - | - | - | - | {'; '.join(notes) or 'no log lines yet'} |")
            continue
        first, last = rows[0], rows[-1]
        sps = np.mean([float(r["sps"]) for r in rows[1:]] or [float(first["sps"])])
        mem = max(rows, key=lambda r: float(re.sub(r"[^\d.]", "", r["mem"].split("/")[0]) or 0))["mem"]
        lines.append(f"| {name} | {last['step']}/{last['total']} | {float(first['loss']):.3f} -> "
                     f"{float(last['loss']):.3f} | {last['inj']} | {sps:.1f} | {mem} | {last['skip']} | "
                     f"{last['sync']} | {fmt_hours(rows)} | {'; '.join(notes)} |")
    lines += ["", "`inj` = per-layer ||injection|| / ||vision hidden||; above 3 the geometry drowns the image "
              "(M2 at lr 1e-3: 15-30).", ""]
    return lines


# ------------------------------------------------------------------------------------------------ ablation
def verdict(mean: float, ci: float, pos: str, neg: str, none: str) -> str:
    return pos if mean - ci > 0 else neg if mean + ci < 0 else none


def section_ablation(out: Path) -> list[str]:
    lines = ["## Geometry ablation (holdout loss, lower is better)", ""]
    found = False
    for label, f in [("S1 (projector only)", "p5_ablation_s1.json"), ("S2 (LoRA)", "p7_ablation_s2.json")]:
        p = out / f
        if not p.exists():
            lines.append(f"- {label}: not run yet (`{f}`)")
            continue
        found = True
        r = json.loads(p.read_text())
        loss = r.get("loss", {})
        lines.append(f"- **{label}**, n = {r.get('n')}: " + " · ".join(
            f"{k} {v[0]:.4f} ± {v[1]:.4f}" for k, v in loss.items()))
        m, ci = r["shuffled_minus_real"]
        lines.append(f"  - shuffled - real = {m:+.4f} ± {ci:.4f} -> " + verdict(
            m, ci, "reads the geometry's content", "shuffled is better (?)",
            "content not distinguishable from a signal") + f" · same grid shape {100 * r.get('same_shape_rate', 0):.0f}%")
        m, ci = r["none_minus_real"]
        lines.append(f"  - none - real = {m:+.4f} ± {ci:.4f}")
        c = r.get("control")
        if c:
            m, ci = c["value_of_content"]
            lines.append(f"  - **value of content = control(shuffled) - real = {m:+.4f} ± {ci:.4f}** -> " + verdict(
                m, ci, "real geometry beats the control: the content is worth something",
                "the control is better: real geometry gets in the way",
                "not distinguishable from the control at this scale")
                + f" · real wins on {100 * c.get('win_rate_vs_control', 0):.0f}% of samples")
    lines.append("")
    if not found:
        lines.insert(2, "M2 pilot (0.8B, 905 samples): value of content -0.04 ± 0.05 - the projector learned the "
                        "answer format, not the geometry.")
    return lines


# ------------------------------------------------------------------------------------------------ gate
def section_gate(out: Path) -> list[str]:
    lines = ["## Pre-registered gate (lmms-eval, eval_path=live3r)", ""]
    cg_path = Path(__file__).with_name("check_gate.py")
    spec = importlib.util.spec_from_file_location("check_gate", cg_path)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    table = {}
    for run in ("base", "base_video", "trained"):
        d = out / "gate" / run
        try:
            res = cg.latest_results(d)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        for task, want in cg.PRIMARY.items():
            m = cg.numeric_metrics(res.get(task, {}))
            k = next((k for k in m if k.split(",")[0] == want), None)
            if k is not None:
                v = m[k]
                table.setdefault(task, {})[run] = 100 * v if abs(v) <= 1.0 else v
    if table:
        lines += ["| task | base (image) | base (video) | S2 real | Δ vs best base |", "|---|---|---|---|---|"]
        for task, v in table.items():
            bases = [v[r] for r in ("base", "base_video") if r in v]
            delta = f"{v['trained'] - max(bases):+.2f}" if "trained" in v and bases else "-"
            lines.append(f"| {task} | " + " | ".join(f"{v[r]:.2f}" if r in v else "-"
                                                    for r in ("base", "base_video", "trained")) + f" | {delta} |")
        lines += ["", "Gate 1: vsibench Δ >= +1.0 over the base's best format · Gate 2: videomme Δ >= -1.0 · "
                  "MMStar reported only.", ""]
    else:
        lines += ["No gate results yet (`gate/base`, `gate/base_video`, `gate/trained`).", ""]
    chk = out / "gate_check.txt"
    if chk.exists():
        lines += ["check_gate.py output:", "", "```", chk.read_text().strip(), "```", ""]
    return lines


def section_latency(out: Path) -> list[str]:
    lines = ["## Latency (gate 3)", ""]
    p = out / "latency_stream.json"
    if p.exists():
        r = json.loads(p.read_text())
        keep = {k: v for k, v in r.items() if isinstance(v, (int, float, bool)) and not isinstance(v, dict)}
        lines += ["Streaming ingest: " + " · ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}"
                                                   for k, v in keep.items()), ""]
    p = out / "latency_live.log"
    if p.exists():
        body = [ln for ln in p.read_text(errors="replace").splitlines() if "ms" in ln or "TTFT" in ln][-12:]
        if body:
            lines += ["Live design (cached keyframe prefix, question-only TTFT):", "", "```", *body, "```", ""]
    if len(lines) == 2:
        lines += ["Not run yet.", ""]
    return lines


def section_status(out: Path) -> list[str]:
    lines = ["## Status", ""]
    status = (out / "STATUS").read_text(errors="replace").splitlines() if (out / "STATUS").exists() else []
    failed = [ln for ln in status if "FAILED" in ln]
    lines += ["| step | state |", "|---|---|"]
    for key, label in PHASES:
        hit = re.compile(rf"FAILED {re.escape(key)}\b")   # "p1" must not match a "p1e" failure
        state = "done" if (out / f"{key}.done").exists() else (
            "FAILED" if any(hit.search(ln) for ln in failed) else "-")
        lines.append(f"| {key} {label} | {state} |")
    lines.append("")
    if failed:
        lines += ["Failures (from STATUS):", "", "```", *failed[-15:], "```", ""]
    if status:
        lines += [f"Last line of STATUS: `{status[-1]}`", ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="outputs/weekend")
    args = ap.parse_args()
    out = Path(args.out)
    parts = ["# Weekend run report - Sensenova-only baseline", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{out}`. Plan and reading guide: "
             "docs/SERVER_WEEKEND.md.", ""]
    for section in (section_status, section_training, section_ablation, section_vsi, section_gate, section_latency):
        try:
            parts += section(out)
        except Exception as exc:  # noqa: BLE001 - one broken section must not hide the others
            parts += [f"## {section.__name__.removeprefix('section_')}", "", f"(could not build: {exc!r})", ""]
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
