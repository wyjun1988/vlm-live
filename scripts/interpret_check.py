"""How well does the model read questions (live3r.serve.interpret, the llm backend) - before any LoRA for it.

    PYTHONPATH=src python scripts/interpret_check.py --model checkpoints/qwen3.5-4b --device mps \\
        --vsi-n 160 --out outputs/interpret/q4b.json

Two question sets: (1) a stratified sample of VSI-Bench questions, where the rules are exact, so the LLM reading is
scored by agreement with them (type, roles, and whether the canonical rendering routes identically); (2) the
paraphrase / Korean fixture (tests/fixtures/question_paraphrases.jsonl), where the rules read nothing and only the
LLM can - scored against the hand-written expectation. Prints a per-type table and the disagreements; the JSON keeps
every reply, which is also the seed of a (question -> JSON) training set for the reading LoRA.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live3r.data.prompt import ASSISTANT_PREFIX  # noqa: E402
from live3r.serve.interpret import Interpretation, interpret, interpret_rules  # noqa: E402

SYNONYMS = {"television": "tv", "tv set": "tv", "fridge": "refrigerator", "couch": "sofa", "trash can": "trash bin",
            "garbage can": "trash bin", "bin": "trash bin", "cooker": "stove", "night stand": "nightstand"}


def norm(v):
    if isinstance(v, list):
        return [norm(x) for x in v]
    s = str(v).strip().lower()
    s = s[4:] if s.startswith("the ") else s
    s = SYNONYMS.get(s, s)
    return s[:-1] if s.endswith("s") and s[:-1] in SYNONYMS.values() else s


def roles_match(got: dict, want: dict) -> bool:
    return all(norm(got.get(k)) == norm(v) for k, v in want.items())


def make_ask(model_path: str, device: str, max_new_tokens: int):
    """(ask, last): ask(prompt) -> reply; last["tokens"] / last["reply"] describe the most recent reply (the output
    tokens are the live cost - the prompt's fixed part would be cached)."""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    last: dict = {}

    @torch.no_grad()
    def ask(prompt: str) -> str:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n{ASSISTANT_PREFIX}"
        ids = torch.tensor(tok(text, add_special_tokens=False).input_ids).unsqueeze(0).to(device)
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False)
        last["tokens"] = int(out.shape[1] - ids.shape[1])
        last["prompt_tokens"] = int(ids.shape[1])
        last["reply"] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return last["reply"]

    return ask, last


def vsi_sample(root: Path, n: int, seed: int) -> list[dict]:
    docs = [json.loads(ln) for ln in open(root / "test.jsonl") if ln.strip()]
    by = collections.defaultdict(list)
    for d in docs:
        by[d["question_type"]].append(d)
    rng = random.Random(seed)
    per = max(1, n // len(by))
    out = []
    for t in sorted(by):
        out += rng.sample(by[t], min(per, len(by[t])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/qwen3.5-4b")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--vsi-n", type=int, default=160, help="VSI questions, spread evenly over the types (0 = skip)")
    ap.add_argument("--fixture", default="tests/fixtures/question_paraphrases.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--compact", action="store_true", help="the short reply format (PROMPT_COMPACT)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ask, last = make_ask(args.model, args.device, args.max_new_tokens)
    rows = []
    t0 = time.time()

    def run(question: str, want_type: str, want_roles: dict, source: str) -> dict:
        t = time.time()
        last.clear()
        it = interpret(question, ask, compact=args.compact)
        ok_type = it.type == want_type
        ok_roles = ok_type and roles_match(it.roles, want_roles)
        return {"set": source, "question": question, "want_type": want_type, "want_roles": want_roles,
                "type": it.type, "roles": it.roles, "objects": it.objects, "question_en": it.question_en,
                "source": it.source, "ok_type": ok_type, "ok_roles": ok_roles, "seconds": time.time() - t,
                "reply": last.get("reply"), "reply_tokens": last.get("tokens"), "prompt_tokens": last.get("prompt_tokens")}

    if args.vsi_n:
        for d in vsi_sample(Path(args.vsi_root), args.vsi_n, args.seed):
            rules = interpret_rules(d["question"])
            r = run(d["question"], rules.type, rules.roles, "vsi")
            got = Interpretation(r["type"], r["objects"], r["roles"], r["question_en"], d["question"], r["source"])
            r["ok_route"] = interpret_rules(got.canonical()).roles == rules.roles and got.type == rules.type
            r["vsi_type"] = d["question_type"]
            rows.append(r)
            print(f"  vsi {len(rows):>4} {d['question_type']:<28} {'ok' if r['ok_roles'] else 'MISS':<4} "
                  f"{r['seconds']:.1f}s", flush=True)
    fx = Path(args.fixture)
    if fx.exists():
        for ln in fx.read_text().splitlines():
            if ln.strip():
                f = json.loads(ln)
                rows.append(run(f["question"], f["type"], f["roles"], "fixture"))
                r = rows[-1]
                print(f"  fixture {r['want_type']:<14} {'ok' if r['ok_roles'] else 'MISS':<4} {f['question'][:60]}",
                      flush=True)

    def table(rs: list[dict], key: str) -> list[str]:
        by = collections.defaultdict(list)
        for r in rs:
            by[r[key]].append(r)
        lines = [f"{'type':<30}{'n':>4}{'type ok':>9}{'roles ok':>10}{'llm src':>9}"]
        for t, g in sorted(by.items()):
            lines.append(f"{t:<30}{len(g):>4}{100 * sum(r['ok_type'] for r in g) / len(g):>8.0f}%"
                         f"{100 * sum(r['ok_roles'] for r in g) / len(g):>9.0f}%"
                         f"{100 * sum(r['source'] == 'llm' for r in g) / len(g):>8.0f}%")
        lines.append(f"{'all':<30}{len(rs):>4}{100 * sum(r['ok_type'] for r in rs) / len(rs):>8.0f}%"
                     f"{100 * sum(r['ok_roles'] for r in rs) / len(rs):>9.0f}%"
                     f"{100 * sum(r['source'] == 'llm' for r in rs) / len(rs):>8.0f}%")
        return lines

    report = []
    vsi = [r for r in rows if r["set"] == "vsi"]
    if vsi:
        report += [f"VSI questions (agreement with the rules; routes identically {100 * sum(r['ok_route'] for r in vsi) / len(vsi):.0f}%)",
                   *table(vsi, "vsi_type"), ""]
    fxr = [r for r in rows if r["set"] == "fixture"]
    if fxr:
        report += ["Paraphrases / Korean (against the hand-written reading)", *table(fxr, "want_type"), ""]
    misses = [r for r in rows if not r["ok_roles"]]
    if misses:
        report.append(f"{len(misses)} misses:")
        for r in misses[:40]:
            report.append(f"  [{r['set']}] {r['question'][:90]}\n      want {r['want_type']} {r['want_roles']}\n"
                          f"      got  {r['type']} {r['roles']} ({r['source']})")
    sec = sum(r["seconds"] for r in rows) / max(1, len(rows))
    toks = [r["reply_tokens"] for r in rows if r.get("reply_tokens") is not None]
    ptoks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    report.append(f"\n{len(rows)} questions · {sec:.2f} s per reading on {args.device} · {args.model} · "
                  f"{'compact' if args.compact else 'full'} format · reply {statistics.mean(toks) if toks else 0:.0f} tokens "
                  f"(prompt {statistics.mean(ptoks) if ptoks else 0:.0f})")
    print("\n".join(report))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "device": args.device, "compact": args.compact, "rows": rows,
                                          "report": "\n".join(report), "elapsed": time.time() - t0},
                                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
