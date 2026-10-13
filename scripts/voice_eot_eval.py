"""How well the voice front-end finds the end of a spoken turn, on real Korean speech (LiveKit eot-bench, `ko`).

    python scripts/voice_eot_eval.py --out outputs/voice/eot_ko.json

Two views of the same 377 turns (real human-to-agent turns, every pause annotated):
  1. eot-bench's own protocol - the annotated pauses, one Smart Turn score 0.2 s into each, the threshold x action
     delay x timeout sweep, cut-offs counted per pause. It should reproduce the benchmark's published Korean numbers
     for Smart Turn v3.2 (30.1 % cut-offs at a 0.3 s mean wait, 9.7 % at 0.6 s, 734 ms at 5 %, 586 ms at 10 %), which
     checks our preprocessing;
  2. our front-end end to end - Silero VAD finds the pauses with live3r.serve.voice's hysteresis and frame timing,
     and a cut-off is counted per turn: the share of questions that would be truncated, against the wait from the
     end of speech (the start of the annotated final silence).
The scores are saved, so trying another policy is a replay (--replay), not a rerun.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live3r.serve.voice import FRAME, SR, SileroVAD, SmartTurn, TurnPolicy, VoiceFrontend  # noqa: E402

THRESHOLDS = np.round(np.arange(0.0, 1.001, 0.01), 2)
DELAYS = np.round(np.arange(0.2, 1.001, 0.1), 2)       # eot-bench's grids
TIMEOUTS = np.round(np.arange(1.0, 3.501, 0.5), 2)
SCORE_POINT = 0.2
EPS = 1e-9
PUBLISHED_KO = {"cut@0.3": 0.301, "cut@0.6": 0.097, "wait@5%": 0.734, "wait@10%": 0.586}  # eot-bench, SmartTurn v3.2


def load_rows(path: str):
    import pyarrow.parquet as pq
    import soundfile as sf

    for r in pq.read_table(path).to_pylist():
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        assert sr == SR, sr
        yield r["id"], a, r["silence_spans"], r["words"]


def score_rows(rows, vad: SileroVAD, turn: SmartTurn, limit: int | None) -> list[dict]:
    """Per turn: the annotated pauses with their Smart Turn score (view 1) and the pauses our VAD finds, with their
    score and where speech resumed (view 2). The front-end runs with a policy that never commits, so it scores every
    pause of the turn on everything heard so far - exactly what it would score live up to its first commit."""
    never = TurnPolicy(threshold=2.0, timeout=1e9, max_turn=1e9)
    out = []
    for k, (rid, a, spans, words) in enumerate(rows):
        if limit and k >= limit:
            break
        oracle = []
        for i, s in enumerate(spans):
            dur = float(s["end"]) - float(s["start"])
            if dur < 0.1 - EPS:                        # eot-bench drops pauses under 0.1 s before scoring
                continue
            p = turn(a[: int(math.floor((float(s["start"]) + SCORE_POINT) * SR + 1e-6))]) if dur >= SCORE_POINT - EPS else None
            oracle.append({"label": "eot" if i == len(spans) - 1 else "hold", "dur": dur, "p": p})
        fe = VoiceFrontend(vad, turn, None, never)
        events = fe.push(np.concatenate([a, np.zeros(int(4.0 * SR), np.float32)]))
        pauses, open_ = [], {}
        for e in events:
            if e.kind == "pause":
                open_ = {"since": e.since, "p": e.p, "end": math.inf}
                pauses.append(open_)
            elif e.kind == "resume":
                open_["end"] = e.t
        out.append({"id": rid, "end_of_speech": float(spans[-1]["start"]), "duration": len(a) / SR,
                    "words": " ".join(w["word"].strip() for w in words), "oracle": oracle, "pauses": pauses})
    return out


def sweep_pauses(turns: list[dict]) -> list[dict]:
    """View 1, eot-bench's metric: hold pauses of 0.2-5 s, cut off when the policy fires inside them; the mean wait
    over the final pauses."""
    hold = [s for t in turns for s in t["oracle"] if s["label"] == "hold" and 0.2 - EPS <= s["dur"] <= 5.0 + EPS]
    eot = [s for t in turns for s in t["oracle"] if s["label"] == "eot"]
    hd = np.array([s["dur"] for s in hold])
    hp = np.array([np.nan if s["p"] is None else s["p"] for s in hold])
    ep = np.array([np.nan if s["p"] is None else s["p"] for s in eot])
    rows = []
    for thr in THRESHOLDS:
        hf = np.where(hp > thr, SCORE_POINT, np.inf)
        ef = np.where(ep > thr, SCORE_POINT, np.inf)
        for d in DELAYS:
            model_cut = np.isfinite(hf) & (hd > np.maximum(d, hf) + EPS)
            em = np.maximum(d, ef)
            for to in TIMEOUTS:
                if to + EPS < d:
                    continue
                cut = model_cut | (hd > to + EPS)
                det = np.isfinite(ef) & (em <= to + EPS)
                rows.append({"threshold": float(thr), "action_delay": float(d), "timeout": float(to),
                             "cutoff": float(cut.mean()), "wait": float(np.where(det, em, to).mean())})
    vad = [{"delay": float(d), "cutoff": float((hd > d + EPS).mean()), "wait": float(d)}
           for d in sorted(set(DELAYS) | set(TIMEOUTS))]
    return rows, vad, len(hold), len(eot)


def frames_up(seconds: float) -> float:
    """The front-end acts at frame ends: a delay of `seconds` from a pause start lands on the next frame boundary."""
    return math.ceil(round(seconds * SR) / FRAME - 1e-9) * FRAME / SR


def replay_turn(t: dict, thr: float, d: float, to: float) -> tuple[bool, float]:
    """View 2 for one turn and one policy: (truncated, wait from the end of speech). The first pause whose fire time
    comes before speech resumes commits the turn; committing before the annotated end of speech truncates it."""
    for pz in t["pauses"]:
        fire = pz["since"] + frames_up(max(d, SCORE_POINT) if pz["p"] > thr else to)
        if fire <= pz["end"] + EPS:
            return fire < t["end_of_speech"] - EPS, fire - t["end_of_speech"]
    return False, math.nan  # never committed (cannot happen with 4 s of trailing silence and timeouts <= 3.5 s)


def sweep_turns(turns: list[dict]) -> list[dict]:
    rows = []
    for thr in THRESHOLDS:
        for d in DELAYS:
            for to in TIMEOUTS:
                if to + EPS < d:
                    continue
                res = [replay_turn(t, thr, d, to) for t in turns]
                trunc = np.array([r[0] for r in res])
                wait = np.array([r[1] for r in res if not r[0] and np.isfinite(r[1])])
                rows.append({"threshold": float(thr), "action_delay": float(d), "timeout": float(to),
                             "cutoff": float(trunc.mean()), "wait": float(wait.mean()) if len(wait) else math.nan,
                             "wait_p50": float(np.median(wait)) if len(wait) else math.nan,
                             "wait_p95": float(np.percentile(wait, 95)) if len(wait) else math.nan})
    return rows


def best_wait(rows, cut_budget):
    ok = [r for r in rows if r["cutoff"] <= cut_budget + EPS and np.isfinite(r["wait"])]
    return min(ok, key=lambda r: (r["wait"], r["cutoff"], r["timeout"], r["action_delay"], r["threshold"])) if ok else None


def best_cut(rows, wait_budget):
    ok = [r for r in rows if np.isfinite(r["wait"]) and r["wait"] <= wait_budget + EPS]
    return min(ok, key=lambda r: (r["cutoff"], r["wait"], r["timeout"], r["action_delay"], r["threshold"])) if ok else None


def fmt(r) -> str:
    if r is None:
        return "-"
    extra = f", p95 {r['wait_p95']:.2f} s" if "wait_p95" in r else ""
    return (f"cut-off {100 * r['cutoff']:.1f} %, mean wait {1000 * r['wait']:.0f} ms{extra}  "
            f"(threshold {r['threshold']:.2f}, action {r['action_delay']:.1f} s, timeout {r['timeout']:.1f} s)")


def report(turns: list[dict]) -> dict:
    rows, vad, n_hold, n_eot = sweep_pauses(turns)
    print(f"\n== 1. eot-bench protocol (annotated pauses, per pause): {n_hold} hold pauses, {n_eot} turns ==")
    ours = {"cut@0.3": best_cut(rows, 0.3), "cut@0.6": best_cut(rows, 0.6),
            "wait@5%": best_wait(rows, 0.05), "wait@10%": best_wait(rows, 0.10)}
    for k, r in ours.items():
        pub = PUBLISHED_KO[k]
        pub_s = f"{100 * pub:.1f} %" if k.startswith("cut") else f"{1000 * pub:.0f} ms"
        print(f"  {k:9s} ours: {fmt(r)}   | published: {pub_s}")
    print(f"  2 % budget: {fmt(best_wait(rows, 0.02))}")
    vad_cut = {b: min((v for v in vad if v['wait'] <= b + EPS), key=lambda v: v['cutoff'], default=None)
               for b in (0.3, 0.6)}
    vad_wait = {b: min((v for v in vad if v['cutoff'] <= b + EPS), key=lambda v: v['wait'], default=None)
                for b in (0.05, 0.10)}
    print("  VAD only: " + ", ".join(f"cut@{b}s {100 * v['cutoff']:.1f} %" for b, v in vad_cut.items() if v)
          + "; " + ", ".join(f"wait@{int(100 * b)}% {1000 * v['wait']:.0f} ms" for b, v in vad_wait.items() if v)
          + "   | published: 58.6 %, 15.2 %; 900 ms, 700 ms")

    trows = sweep_turns(turns)
    n_p = sum(len(t["pauses"]) for t in turns)
    print(f"\n== 2. our front-end (Silero VAD pauses, per turn = truncated questions): {len(turns)} turns, "
          f"{n_p} scored pauses ==")
    for b in (0.02, 0.05, 0.10):
        print(f"  truncation <= {int(100 * b):2d} %: {fmt(best_wait(trows, b))}")
    for w in (0.5, 0.6, 0.7, 0.8, 1.0):
        print(f"  mean wait <= {w:.1f} s: {fmt(best_cut(trows, w))}")
    vad_only = [r for r in trows if r["threshold"] == 1.0 and r["action_delay"] == 0.2]  # never "finished"
    for to in (1.0, 1.5):
        r = next((r for r in vad_only if r["timeout"] == to), None)
        print(f"  silence timeout {to:.1f} s alone: {fmt(r)}")
    return {"pauses": {"best": ours, "vad": vad}, "turns": trows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/eot_bench/data/ko/validation-00000-of-00001.parquet")
    ap.add_argument("--vad", default="checkpoints/silero-vad/silero_vad.onnx")
    ap.add_argument("--smart-turn", default="checkpoints/smart-turn-v3/smart-turn-v3.2-cpu.onnx")
    ap.add_argument("--limit", type=int, default=0, help="first N turns only (0 = all)")
    ap.add_argument("--out", default="outputs/voice/eot_ko.json")
    ap.add_argument("--replay", help="a previous --out file: re-run the sweeps on its saved scores")
    args = ap.parse_args()

    if args.replay:
        turns = json.load(open(args.replay))["turns"]
    else:
        t0 = time.time()
        turns = score_rows(load_rows(args.data), SileroVAD(args.vad), SmartTurn(args.smart_turn), args.limit)
        print(f"scored {len(turns)} turns in {time.time() - t0:.0f} s")
    summary = report(turns)
    if not args.replay:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"data": args.data, "smart_turn": args.smart_turn, "turns": turns,
                   "summary_pauses": summary["pauses"]["best"]}, open(args.out, "w"), ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
