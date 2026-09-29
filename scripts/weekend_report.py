"""One Markdown report of everything a server run produced (scripts/server_weekend.sh, scripts/server_run2.sh).

    python scripts/weekend_report.py outputs/weekend > outputs/weekend/REPORT.md
    python scripts/weekend_report.py outputs/run2 > outputs/run2/REPORT.md      # run 2 (detected from the RUN file)

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

RUN = "weekend"   # set in main() from --run or <out>/RUN

# lmms-eval's VSI aggregation (_compute_all_subscores): the mean of each question type, the three relative-
# direction difficulties averaged into one type first, then the mean over the 8 types
REL_DIR = "object_rel_direction"
TYPE_SHORT = {
    "obj_appearance_order": "appear", "object_abs_distance": "abs dist", "object_counting": "count",
    "object_rel_distance": "rel dist", "object_size_estimation": "obj size", "room_size_estimation": "room size",
    "route_planning": "route", REL_DIR: "rel dir",
}
# markers a run leaves (outputs/weekend/<name>.done); the roles of scripts/server_weekend.sh produce them
PHASES = [("p0", "shared preflight"), ("p2_vsibench", "VSI-Bench data"), ("p2_mmstar", "MMStar data"),
          ("p2_videomme", "VideoMME data"), ("p3", "zero-shot VSI baseline"),
          ("s1_real", "S1 real"), ("s1_control", "S1 control"), ("s2_real", "S2 real"), ("s2_control", "S2 control"),
          ("s2_sft", "plain SFT (no geometry)"), ("s2_joint", "S2 joint (no S1)"),
          ("s1_real_seed1", "S1 real, seed 1"), ("s1_control_seed1", "S1 control, seed 1"),
          ("s2_real_seed1", "S2 real, seed 1"), ("s2_control_seed1", "S2 control, seed 1"),
          ("s1_real_2b", "2B S1 real"), ("s1_control_2b", "2B S1 control"), ("s2_real_2b", "2B S2 real"),
          ("s2_control_2b", "2B S2 control"), ("gate_base", "gate: base"), ("gate_base_video", "gate: base video"),
          ("gate_trained", "gate: trained"), ("gate_check", "gate verdict"),
          ("role.real", "role real"), ("role.control", "role control"), ("role.sft", "role sft"), ("role.small", "role small")]
# run 2 (scripts/server_run2.sh): one epoch per arm, the new stack (linear patch embed, fla), all weekend checkpoints
# re-evaluated on the same stack under the prefix wk_
PHASES_RUN2 = [("p0", "shared preflight"), ("d0", "throughput diagnostic (first node)"), ("calib", "speed calibration"),
               ("s1_real", "S1 real (short)"), ("s1_control", "S1 control (short)"), ("s2_real", "S2 real"),
               ("s2_control", "S2 control"), ("s2_sft", "plain SFT (no geometry)"), ("recover_vsi", "weekend re-evaluation"),
               ("recover_abl", "weekend ablations"), ("s2_joint", "S2 joint (no S1)"), ("gate_base", "gate: base"),
               ("gate_base_video", "gate: base video"), ("gate_trained", "gate: trained"), ("gate_check", "gate verdict"),
               ("role.real", "role real"), ("role.control", "role control"), ("role.sft", "role sft"),
               ("role.recover", "role recover")]
COMPARISONS_RUN2 = [
    ("S2: real - control (geometry content)  <- key", "s2_real:oracle-image", "s2_control:oracle-image"),
    ("S2 real - plain SFT (value of the whole geometry path)", "s2_real:oracle-image", "s2_sft:oracle-image"),
    ("S2 control - plain SFT (a content-free injection)", "s2_control:oracle-image", "s2_sft:oracle-image"),
    ("S2 joint (no S1) - S2 real (is S1 needed?)", "s2_joint:oracle-image", "s2_real:oracle-image"),
    ("S1: real - control", "s1_real:oracle-image", "s1_control:oracle-image"),
    ("Hint, same prompt: S2 real - control", "s2_real_hint:oracle-image", "s2_control_hint:oracle-image"),
    ("Hint, same prompt: S2 real - plain SFT", "s2_real_hint:oracle-image", "s2_sft_hint:oracle-image"),
    ("Hint, same prompt: plain SFT - base (what SFT buys)", "s2_sft_hint:oracle-image", "base_hint:oracle-image"),
    ("Hint, same prompt: S2 real - base", "s2_real_hint:oracle-image", "base_hint:oracle-image"),
    ("S2 real - base best format (gate 1, local eval)", "s2_real:oracle-image", "base_best"),
    ("Room facts, same prompt: S2 real - plain SFT", "s2_real_routed_room:oracle-image", "s2_sft_routed_room:oracle-image"),
    ("S2 real: room facts - hint (measurements after training)", "s2_real_routed_room:oracle-image", "s2_real_hint:oracle-image"),
    ("Base: room facts - hint", "base_routed_room:oracle-image", "base_hint:oracle-image"),
    ("Base: + hint - video mode", "base_hint:oracle-image", "base_plain:oracle-video"),
    ("1 epoch vs 2 %: S2 real - weekend S2 real", "s2_real:oracle-image", "wk_s2_real:oracle-image"),
    ("1 epoch vs 2 %: plain SFT - weekend plain SFT", "s2_sft:oracle-image", "wk_s2_sft:oracle-image"),
    ("Weekend: S2 real - control (rule 1 at 2 %)", "wk_s2_real:oracle-image", "wk_s2_control:oracle-image"),
    ("Weekend seed 1: S2 real - control", "wk_s2_real_seed1:oracle-image", "wk_s2_control_seed1:oracle-image"),
    ("Weekend: S2 real seed 1 - seed 0 (training-seed noise)", "wk_s2_real_seed1:oracle-image", "wk_s2_real:oracle-image"),
    ("Weekend: S2 real - plain SFT", "wk_s2_real:oracle-image", "wk_s2_sft:oracle-image"),
    ("Weekend: S2 joint - S2 real", "wk_s2_joint:oracle-image", "wk_s2_real:oracle-image"),
    ("Weekend: S1 real - control", "wk_s1_real:oracle-image", "wk_s1_control:oracle-image"),
    ("Weekend 2B: S2 real - control", "wk_s2_real_2b:oracle-image", "wk_s2_control_2b:oracle-image"),
    ("Weekend 2B: S2 real - 2B base + hint", "wk_s2_real_2b_hint:oracle-image", "base_hint_2b:oracle-image"),
    ("Stack check: weekend S2 real, new stack - old stack", "wk_s2_real:oracle-image", "old_s2_real:oracle-image"),
]
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
    ("Plain SFT - base best format (what SFT alone buys)", "s2_sft:oracle-image", "base_best"),
    ("Plain SFT + hint - plain SFT", "s2_sft_hint:oracle-image", "s2_sft:oracle-image"),
    ("S2 real - plain SFT (value of the whole geometry path)", "s2_real:oracle-image", "s2_sft:oracle-image"),
    ("S2 control - plain SFT (a content-free injection)", "s2_control:oracle-image", "s2_sft:oracle-image"),
    ("S2 joint (no S1) - S2 real (is S1 needed?)", "s2_joint:oracle-image", "s2_real:oracle-image"),
    ("Seed 1: S2 real - control", "s2_real_seed1:oracle-image", "s2_control_seed1:oracle-image"),
    ("S2 real: seed 1 - seed 0 (training-seed noise)", "s2_real_seed1:oracle-image", "s2_real:oracle-image"),
    ("S2 control: seed 1 - seed 0 (training-seed noise)", "s2_control_seed1:oracle-image", "s2_control:oracle-image"),
    ("2B: base + hint - base video mode", "base_hint_2b:oracle-image", "base_plain_2b:oracle-video"),
    ("2B: S2 real - control (geometry content, 2B)", "s2_real_2b:oracle-image", "s2_control_2b:oracle-image"),
    ("2B: S2 real - 2B base + hint", "s2_real_2b:oracle-image", "base_hint_2b:oracle-image"),
    ("2B routed: S2 real - control", "s2_real_2b_routed:oracle-image", "s2_control_2b_routed:oracle-image"),
    ("2B S2 real - 4B S2 real (the size gap after training)", "s2_real_2b:oracle-image", "s2_real:oracle-image"),
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


def per_type_scores(sc: dict) -> dict[str, float | None]:
    """The 8 VSI types from a scores dict (metric suffixes stripped; the direction difficulties already averaged)."""
    per: dict[str, float | None] = {c: None for c in TYPE_SHORT}
    for k, v in sc.items():
        t = k.split("_MRA")[0].removesuffix("_accuracy")
        if t in per:
            per[t] = v
    return per


def debiased_ids(vsi_root: str = "data/eval/vsibench") -> set | None:
    """VSI-Debiased = the questions NOT in pruned_ids.txt (the ones the Debiased pilot found answerable without the
    video). None when the file is not there."""
    f = Path(vsi_root) / "pruned_ids.txt"
    if not f.exists():
        return None
    return {int(x) for x in f.read_text().split()}


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
            per = per_type_scores(sc)
            ff = r.get("format_fail", {}).get(mode)
            cells = " | ".join("-" if per[c] is None else f"{100 * per[c]:.1f}" for c in cols)
            lines.append(f"| {name} | {mode} | **{100 * sc['overall']:.1f}** | {cells} | "
                         f"{'-' if ff is None else f'{100 * ff:.1f}%'} | {r.get('n_questions', '-')} |")
    lines += ["", "M2 reference (60 videos, 1,151 questions): base video mode 48.6 · + format hint 53.7 · "
              "routed facts 56.0 · S2 pilot (905 samples) 44.1 vs control 42.3.", "",
              "### Paired comparisons (Δ in points, 95% interval from resampling videos)", "",
              "| comparison | A | B | Δ | 95% interval | videos / questions | Δ Debiased [95%] |",
              "|---|---|---|---|---|---|---|"]
    pruned = debiased_ids()
    for label, a, b in (COMPARISONS_RUN2 if RUN == "run2" else COMPARISONS):
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
        deb = "-"
        if pruned is not None:
            rd = paired_bootstrap([p for p in pa if p["id"] not in pruned], [p for p in pb if p["id"] not in pruned])
            if rd is not None:
                deb = f"{rd[0]:+.1f} [{rd[1]:+.1f}, {rd[2]:+.1f}]" + (" *" if rd[1] > 0 or rd[2] < 0 else "")
        lines.append(f"| {label} | {sa} | {sb} | **{d:+.1f}**{sig} | [{lo:+.1f}, {hi:+.1f}] | {nv} / {nq} | {deb} |")
    lines += ["", "`*` = the interval excludes 0. Debiased = the questions not in `pruned_ids.txt` (2,362 of 5,130).", ""]
    return lines


def section_curve(out: Path) -> list[str]:
    """VSI at the saved S2 checkpoints: does more Sensenova keep helping, or does the numeric prior take over?"""
    runs = load_vsi(out)
    pts = sorted((int(m.group(1)), name) for name in runs for m in [re.match(r"s2_real_step(\d+)$", name)] if m)
    if not pts:
        return []
    cols = list(TYPE_SHORT)
    lines = ["## S2 learning curve (VSI, image mode, real arm)", "",
             "| checkpoint | overall | Δ vs base + format hint | " + " | ".join(TYPE_SHORT[c] for c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    _, hint = vsi_preds(runs, "base_hint:oracle-image")
    rows = [(0, "base_hint")] if "base_hint" in runs else []
    rows += pts + ([(None, "s2_real")] if "s2_real" in runs else [])
    for step, name in rows:
        sc = runs[name].get("scores", {}).get("oracle-image")
        if not sc:
            continue
        per = per_type_scores(sc)
        cells = " | ".join("-" if per[c] is None else f"{100 * per[c]:.1f}" for c in cols)
        if name == "base_hint":
            lines.append(f"| base + format hint | {100 * sc['overall']:.1f} | — | {cells} |")
            continue
        _, preds = vsi_preds(runs, f"{name}:oracle-image")
        d = paired_bootstrap(preds, hint) if preds and hint else None
        delta = f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]" if d else "-"
        lines.append(f"| {'step ' + str(step) if step else 'final'} | {100 * sc['overall']:.1f} | {delta} | {cells} |")
    return lines + ["", "Per type along the curve: where more Sensenova helps and where it hurts - the numeric types "
                    "(abs dist, obj size, room size) are where the M2 pilot's answers collapsed.", ""]


# ------------------------------------------------------------------------------------------------ training
LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?step (?P<step>\d+)/(?P<total>\d+) loss (?P<loss>[\d.]+) "
                  r"lr (?P<lr>\S+) \| inj (?P<inj>.*?) \| gate (?P<gate>.*?) \| (?P<sps>[\d.]+) samp/s .*?"
                  r"\| mem (?P<mem>\S+) \| skip (?P<skip>\d+) \| sync (?P<sync>\S+)")


def parse_train_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    rows = [m.groupdict() for m in map(LINE.match, text.splitlines()) if m]
    info: dict = {"rows": rows, "error": None, "early": None, "donor_same": None, "resumed": 0, "nonfinite": None}
    for line in text.splitlines():
        if "--max-hours" in line and "reached" in line:
            info["early"] = line.split("WARNING", 1)[-1].strip()
        if "Resumed from" in line:
            info["resumed"] += 1
        m = re.search(r"같은 격자 기증자 (\d+)%", line)
        if m:
            info["donor_same"] = int(m.group(1))
        m = re.search(r"non-finite: (\d+) samples skipped, (\d+) steps without update", line)
        if m:
            info["nonfinite"] = (int(m.group(1)), int(m.group(2)))
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
        n = int(env.get("N_TRAIN", 0) or 0)
        if n:
            lines += [" · ".join(f"{st} {int(env[k]) * 128 / n:.2f} epochs" for st, k in (("S1", "S1_STEPS"), ("S2", "S2_STEPS"))
                                 if k in env), ""]
    lines += ["| run | steps | first -> last loss | last inj (L0..) | samples/s | peak mem | skipped | sync | "
              "elapsed | notes |", "|---|---|---|---|---|---|---|---|---|---|"]
    logs = sorted(out.glob("p1_*_smoke_*.log")) + sorted(out.glob("s[12]_*.log"))
    for log in logs:
        name = log.stem.removeprefix("p1_")
        info = parse_train_log(log)
        if info is None:
            continue
        rows = info["rows"]
        notes = []
        if info["early"]:
            notes.append("stopped by --max-hours")
        if info["donor_same"] is not None:
            notes.append(f"same-grid donors {info['donor_same']}%")
        if info["resumed"]:
            notes.append(f"resumed {info['resumed']}x")
        if info["nonfinite"] and any(info["nonfinite"]):
            notes.append(f"non-finite: {info['nonfinite'][0]} samples / {info['nonfinite'][1]} steps skipped")
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


# -------------------------------------------------------------------------------------------------- probe
def load_probe(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def section_probe(out: Path) -> list[str]:
    """The holdout probe train.py runs every --probe-every steps (HoldoutProbe): when, if ever, real geometry starts
    beating shuffled geometry, and what the answers look like along the way - the path, not just the endpoint."""
    files = sorted(out.glob("s[12]_*/probe.jsonl"))
    if not files:
        return []
    lines = ["## Holdout probe during training (loss, lower is better)", "",
             "`shuffled - real` > 0 and growing = the model reads the geometry's content; `none - real` = the value "
             "of the whole injection path; `real wins` = share of samples where real beats shuffled. Answers are "
             "greedy, on the first probe samples.", ""]

    def f4(v):
        return "-" if v is None else f"{v:.4f}"

    for f in files:
        rows = load_probe(f)
        if not rows:
            continue
        lines += [f"### {f.parent.name}", "",
                  "| step | real | shuffled | none | shuffled - real | real wins | none - real | s |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            loss = r.get("loss", {})
            wins = r.get("real_wins")
            lines.append(f"| {r.get('step')} | {f4(loss.get('real'))} | {f4(loss.get('shuffled'))} | "
                         f"{f4(loss.get('none'))} | {f4(r.get('shuffled_minus_real'))} | "
                         f"{'-' if wins is None else f'{100 * wins:.0f}%'} | {f4(r.get('none_minus_real'))} | "
                         f"{r.get('seconds', 0):.0f} |")
        gens = rows[-1].get("generations") or []
        if gens:
            lines += ["", f"Answers at step {rows[-1].get('step')}: " + " · ".join(
                f"`{g.get('answer', '')[:30]}` (truth `{g.get('truth', '')[:30]}`)" for g in gens[:6])]
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------ ablation
def verdict(mean: float, ci: float, pos: str, neg: str, none: str) -> str:
    return pos if mean - ci > 0 else neg if mean + ci < 0 else none


def section_ablation(out: Path) -> list[str]:
    lines = ["## Geometry ablation (holdout loss, lower is better)", ""]
    found = False
    files = [("S1 (projector only)", "p5_ablation_s1.json"), ("S2 (LoRA)", "p7_ablation_s2.json"),
             ("2B S2 (LoRA)", "p7_ablation_2b.json")]
    if RUN == "run2":
        files += [("weekend S1, new stack", "wk_p5_ablation_s1.json"), ("weekend S2, new stack", "wk_p7_ablation_s2.json"),
                  ("weekend S2 seed 1, new stack", "wk_p7_ablation_s2_seed1.json"),
                  ("weekend 2B S2, new stack", "wk_p7_ablation_2b.json")]
    for label, f in files:
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
    nodes = sorted(p.name.removeprefix("PID.") for p in out.glob("PID.*"))
    roles = sorted({m.group(1) for ln in status for m in [re.search(r"\[([a-z+]+)\] started on", ln)] if m})
    if nodes:
        lines += [f"Nodes seen: {', '.join(nodes)} · roles: {', '.join(roles) or '-'}", ""]
    lines += ["| step | state |", "|---|---|"]
    done = {p.stem for p in out.glob("*.done")}
    phases = PHASES_RUN2 if RUN == "run2" else PHASES
    for key, label in phases:
        hit = re.compile(rf"FAILED {re.escape(key)}\b")   # "p1" must not match a "p1e" failure
        state = "done" if key in done else ("FAILED" if any(hit.search(ln) for ln in failed) else "-")
        lines.append(f"| {key} {label} | {state} |")
    other = sorted(k for k in done if k not in {k for k, _ in phases} and not k.startswith(("p0.", "p1.")))
    if other:
        lines.append(f"| other markers | {', '.join(other)} |")
    lines.append("")
    if failed:
        lines += [f"Failures (from STATUS, all {len(failed)}):", "", "```", *failed, "```", ""]
    if status:
        lines += [f"Last line of STATUS: `{status[-1]}`", ""]
    return lines


def section_diagnostic(out: Path) -> list[str]:
    """Run 2: the throughput diagnostic (scripts/diagnose_throughput.py) and the calibrated budget."""
    if RUN != "run2":
        return []
    lines = ["## Throughput: diagnostic and budget", ""]
    d = out / "d0_throughput.txt"
    if d.exists():
        keep = [ln.rstrip() for ln in d.read_text(errors="replace").splitlines()
                if ln.strip().startswith(("python", "cuda", "transformers", "packages", "qwen3_5.", "Gated", "conv3d",
                                          "linear", "stock", "loss fused", "FLA", "F1", "F2", "Conv3d", "vision tower",
                                          "fla check", "[llm]"))]
        lines += ["```", *keep, "```", ""]
    else:
        lines += ["Diagnostic not run yet (`d0_throughput.txt`).", ""]
    b = out / "budget.env"
    if b.exists():
        env = dict(ln.split("=", 1) for ln in b.read_text().split() if "=" in ln)
        n = int(env.get("N_TRAIN", 0) or 0)
        lines += ["Budget: " + " · ".join(f"{k} {v}" for k, v in env.items()), ""]
        if n:
            for st in ("S1", "S2"):
                steps = int(env.get(f"{st}_STEPS", 0) or 0)
                sps = float(env.get(f"CALIB_SPS_{st}", 0) or 0)
                if steps:
                    lines.append(f"- {st}: {steps} steps x 128 = {steps * 128:,} samples = {steps * 128 / n:.2f} epochs"
                                 + (f" at {sps:.2f} samples/s = {steps * 128 / sps / 3600:.1f} h" if sps else ""))
            lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="outputs/weekend")
    ap.add_argument("--run", choices=("weekend", "run2"), default=None,
                    help="which run's phases and comparisons (default: the RUN file in the output directory)")
    args = ap.parse_args()
    out = Path(args.out)
    global RUN
    RUN = args.run or ((out / "RUN").read_text().strip() if (out / "RUN").exists() else "weekend")
    if RUN == "run2":
        parts = ["# Run 2 report - Sensenova, one epoch per arm, new stack", "",
                 f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{out}`. Plan and reading guide: docs/SERVER_RUN2.md; "
                 "why this run: docs/SERVER_RESULTS_20260929.md.", ""]
    else:
        parts = ["# Weekend run report - Sensenova-only baseline", "",
                 f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{out}`. Plan and reading guide: "
                 "docs/SERVER_WEEKEND.md.", ""]
    for section in (section_status, section_diagnostic, section_training, section_probe, section_ablation, section_vsi,
                    section_curve, section_gate, section_latency):
        try:
            parts += section(out)
        except Exception as exc:  # noqa: BLE001 - one broken section must not hide the others
            parts += [f"## {section.__name__.removeprefix('section_')}", "", f"(could not build: {exc!r})", ""]
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
