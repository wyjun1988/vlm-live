"""Read the reference system that reached VSI 70.83 directly from the shared volume, instead of asking about it.

    python scripts/inspect_reference_system.py          # the paths the reference team gave on 2026-09-30
    -> outputs/reference_70/INVENTORY.md  and  outputs/reference_70/reference_70_bundle.tar.gz (small files, to send)

The reference: VG-LLM codebase, Qwen/Qwen3.5-4B + LoRA `ckpt/cot/qwen3.5-4b-allcot-sqa-gt-cut3r-lora2`, CUT3R
`cut3r_512_dpt_4_64.pth`, lmms-eval vsibench with thinking off, overall 70.83 (results 20260728_140541). Their reply
was an answer-generation runbook, not answers to docs/QUESTIONS_FOR_EXISTING_SYSTEM.md; everything the questionnaire
asks is in their files, so this script reads them:
  [paths]     what exists; whether their CUT3R checkpoint is byte-identical to ours (sha256)
  [lora]      the checkpoint directory: files, adapter config (r, alpha, target and saved modules), trainer state and
              training arguments (epochs, steps, lr, batch), tensor groups (what besides LoRA was trained)
  [results]   the lmms-eval results JSON: every metric (per-type scores), the task config (prompts, generation
              kwargs), the model arguments (frames, pixels, thinking)
  [samples]   the per-question samples: count, the full prompt of one question, answer lengths and chain-of-thought
              markers (a CoT answer is a latency question for the live design), per-type accuracy recomputed
  [repo]      VG-LLM: commit, the evaluation script and what it calls, the training script(s) of this run, the dataset
              registry entries and their annotation files (record counts), where the geometry is fused in the model
Read-only on every reference path; CPU only (safe next to a training run); secrets in copied text are masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

DEFAULTS = {
    "vg": "/group-volume/jcho/VG-LLM",
    "lora": "/group-volume/jcho/VG-LLM/ckpt/cot/qwen3.5-4b-allcot-sqa-gt-cut3r-lora2",
    "cut3r": "/group-volume/jcho/CUT3R/src/cut3r_512_dpt_4_64.pth",
    "results": "/group-volume/jcho/VG-LLM/logs/vsibench_qwen35_nothink/cot__qwen3.5-4b-allcot-sqa-gt-cut3r-lora2/"
               "20260728_140541_results.json",
    "our_cut3r": "checkpoints/cut3r_512_dpt_4_64.pth",
    "out": "outputs/reference_70",
    # second pass (2026-09-30): what the first bundle could not show - the dataset annotations of the training mix,
    # the gated-token fusion code, their own 1 fps latency profiles, the train.log
}
RUN_NAME = "qwen3.5-4b-allcot-sqa-gt-cut3r-lora2"
SECRET = re.compile(r"(?i)\b((?:[A-Z0-9_]*API[_-]?KEY|[A-Z0-9_]*(?:ACCESS|AUTH|HF|HUB|WANDB|GITHUB|GH|OPENAI)[_-]?TOKEN|"
                    r"SECRET[_-]?KEY|PASSW(?:OR)?D)\s*[=:]\s*[\"']?)([A-Za-z0-9_\-\.]{8,})")
URL_CRED = re.compile(r"(https?://)[^/\s:@]+(?::[^/\s@]+)?@")
TEXT_EXT = {".json", ".yaml", ".yml", ".txt", ".md", ".sh", ".py", ".cfg", ".toml", ".jsonl"}
MAX_COPY = 2 << 20                     # copy text files up to 2 MB into the bundle


def redact(text: str) -> str:
    return URL_CRED.sub(r"\1***@", SECRET.sub(r"\1***", text))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sh(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"({exc!r})"


class Inventory:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.bundle = out / "bundle"
        self.lines: list[str] = []
        self.bundle.mkdir(parents=True, exist_ok=True)

    def h(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def add(self, *lines: str) -> None:
        self.lines += list(lines)

    def code(self, text: str, lang: str = "") -> None:
        self.lines += [f"```{lang}", text.rstrip(), "```"]

    def copy(self, src: Path, rel: str) -> str | None:
        """A small text file into the bundle, secrets masked; returns the bundle path."""
        try:
            if not src.is_file() or src.stat().st_size > MAX_COPY or src.suffix not in TEXT_EXT:
                return None
            dst = self.bundle / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(redact(src.read_text(errors="replace")))
            return rel
        except OSError:
            return None


# ------------------------------------------------------------------------------------------------ sections
def section_paths(inv: Inventory, a) -> None:
    inv.h("Paths")
    inv.add("| path | exists | size |", "|---|---|---|")
    for k in ("vg", "lora", "cut3r", "results", "our_cut3r"):
        p = Path(getattr(a, k))
        size = human(p.stat().st_size) if p.is_file() else ("dir" if p.is_dir() else "-")
        inv.add(f"| `{p}` | {'yes' if p.exists() else '**no**'} | {size} |")
    ours, theirs = Path(a.our_cut3r), Path(a.cut3r)
    if ours.is_file() and theirs.is_file():
        if ours.stat().st_size != theirs.stat().st_size:
            inv.add("", f"CUT3R checkpoints differ in size: ours {ours.stat().st_size}, theirs {theirs.stat().st_size}.")
        else:
            same = sha256(ours) == sha256(theirs)
            inv.add("", f"CUT3R checkpoint: **{'byte-identical to ours' if same else 'SAME SIZE BUT DIFFERENT BYTES'}** (sha256).")


def tensor_groups(path: Path) -> list[str]:
    """Tensor names grouped by module prefix, with parameter counts - from safetensors metadata or a CPU torch load."""
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    try:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as f:
                for k in f.keys():
                    shape = f.get_slice(k).get_shape()
                    n = 1
                    for s in shape:
                        n *= s
                    g = re.sub(r"\.\d+\.", ".N.", k)
                    g = re.sub(r"(lora_[AB])\..*", r"\1", g)
                    groups[".".join(g.split(".")[:6])][0] += 1
                    groups[".".join(g.split(".")[:6])][1] += n
        elif path.stat().st_size < (3 << 30):
            import torch

            sd = torch.load(path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else {}
            for k, v in sd.items():
                if hasattr(v, "numel"):
                    g = ".".join(re.sub(r"\.\d+\.", ".N.", k).split(".")[:6])
                    groups[g][0] += 1
                    groups[g][1] += int(v.numel())
        else:
            return [f"(not loaded: {human(path.stat().st_size)})"]
    except Exception as exc:  # noqa: BLE001
        return [f"(could not read: {exc!r})"]
    rows = sorted(groups.items(), key=lambda kv: -kv[1][1])
    return [f"{g}  x{c}  {n / 1e6:.2f}M params" for g, (c, n) in rows[:60]] + (
        [f"... {len(rows) - 60} more groups"] if len(rows) > 60 else [])


def section_lora(inv: Inventory, a) -> None:
    inv.h("Checkpoint directory (LoRA and anything else saved)")
    d = Path(a.lora)
    if not d.is_dir():
        inv.add(f"`{d}` not found.")
        return
    files = sorted(p for p in d.rglob("*") if p.is_file())
    inv.add(f"{len(files)} files, {human(sum(p.stat().st_size for p in files))}:", "")
    inv.code("\n".join(f"{human(p.stat().st_size):>10}  {p.relative_to(d)}" for p in files[:200]))
    for p in files:
        inv.copy(p, f"lora/{p.relative_to(d)}")
    cfg = d / "adapter_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        keys = ("base_model_name_or_path", "peft_type", "r", "lora_alpha", "lora_dropout", "target_modules",
                "modules_to_save", "bias", "use_rslora", "use_dora")
        inv.add("", "adapter_config.json:", "")
        inv.code(json.dumps({k: c.get(k) for k in keys if k in c}, indent=1, ensure_ascii=False), "json")
    for name in ("trainer_state.json",):
        for p in d.rglob(name):
            try:
                st = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            hist = st.get("log_history", [])
            losses = [h for h in hist if "loss" in h]
            inv.add("", f"`{p.relative_to(d)}`: global_step {st.get('global_step')}, max_steps {st.get('max_steps')}, "
                        f"epochs {st.get('num_train_epochs')} (reached {st.get('epoch')}), batch {st.get('train_batch_size')}; "
                        f"{len(losses)} logged losses"
                        + (f", first {losses[0].get('loss')} -> last {losses[-1].get('loss')}" if losses else ""))
    for p in list(d.rglob("training_args.bin"))[:2]:
        try:
            import torch

            ta = torch.load(p, map_location="cpu", weights_only=False)
            td = ta.to_dict() if hasattr(ta, "to_dict") else vars(ta)
            keep = ("learning_rate", "num_train_epochs", "max_steps", "per_device_train_batch_size",
                    "gradient_accumulation_steps", "warmup_ratio", "warmup_steps", "lr_scheduler_type", "weight_decay",
                    "bf16", "gradient_checkpointing", "deepspeed", "model_max_length", "max_pixels", "min_pixels",
                    "video_max_frames", "video_min_frames", "base_interval", "mm_projector_lr", "vision_tower_lr",
                    "tune_mm_llm", "tune_mm_vision", "tune_mm_mlp", "dataset_use", "data_flatten", "output_dir",
                    "geometry_encoder_path", "geometry_encoder_type", "use_geometry_encoder", "feature_fusion_method",
                    "geometry_merger_type", "report_to", "seed")
            extra = {k: v for k, v in td.items() if any(t in k.lower() for t in ("geom", "cut3r", "vggt", "fusion",
                                                                                  "frame", "pixel", "dataset", "data_"))}
            inv.add("", f"`{p.relative_to(d)}` (TrainingArguments; the fields that define the recipe):", "")
            inv.code(redact(json.dumps({**{k: td.get(k) for k in keep if k in td}, **extra}, indent=1, default=str,
                                       ensure_ascii=False)), "json")
        except Exception as exc:  # noqa: BLE001
            inv.add("", f"`{p.relative_to(d)}`: could not be read ({exc!r})")
    for p in files:
        if p.suffix in (".safetensors", ".bin", ".pt", ".pth") and p.name != "training_args.bin":
            inv.add("", f"Tensor groups in `{p.relative_to(d)}` ({human(p.stat().st_size)}):", "")
            inv.code("\n".join(tensor_groups(p)))


def section_results(inv: Inventory, a) -> dict:
    inv.h("lmms-eval results (the 70.83)")
    p = Path(a.results)
    if not p.exists():
        inv.add(f"`{p}` not found.")
        return {}
    r = json.loads(p.read_text())
    inv.copy(p, "results/" + p.name)
    inv.add("`results` (every metric of every task; the per-type VSI scores are here):", "")
    inv.code(json.dumps(r.get("results", {}), indent=1, ensure_ascii=False)[:12000], "json")
    cfg = r.get("config", {})
    inv.add("", "`config` (model, model_args, batch, limit, generation kwargs):", "")
    inv.code(redact(json.dumps(cfg, indent=1, default=str, ensure_ascii=False))[:6000], "json")
    for task, tc in (r.get("configs") or {}).items():
        keep = {k: tc.get(k) for k in ("dataset_path", "dataset_kwargs", "test_split", "generation_kwargs",
                                        "lmms_eval_specific_kwargs", "metric_list", "process_results") if k in tc}
        inv.add("", f"task config `{task}`:", "")
        inv.code(json.dumps(keep, indent=1, default=str, ensure_ascii=False)[:6000], "json")
    inv.add("", f"n-samples: `{json.dumps(r.get('n-samples', {}))}` · date `{r.get('date')}` · "
                f"lmms_eval `{r.get('lmms_eval_version', r.get('git_hash', '?'))}`")
    return r


def section_samples(inv: Inventory, a) -> None:
    inv.h("Per-question samples (prompt, answer length, chain of thought)")
    res = Path(a.results)
    stamp = res.name.split("_results")[0]
    cands = sorted(res.parent.glob(f"{stamp}*samples*.jsonl")) or sorted(res.parent.glob("*samples*.jsonl"))
    if not cands:
        inv.add(f"No `*samples*.jsonl` next to `{res}`.")
        return
    for sp in cands[:3]:
        rows = []
        with open(sp, errors="replace") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        inv.add(f"`{sp.name}`: {len(rows)} records. Keys: `{sorted(rows[0].keys()) if rows else []}`", "")
        if not rows:
            continue

        def resp(r):
            x = r.get("resps", r.get("filtered_resps", ""))
            while isinstance(x, list) and x:
                x = x[0]
            return str(x)

        def filt(r):
            x = r.get("filtered_resps", "")
            while isinstance(x, list) and x:
                x = x[0]
            return str(x)

        lens = [len(resp(r)) for r in rows]
        words = [len(resp(r).split()) for r in rows]
        cot = sum(bool(re.search(r"<think>|step by step|therefore|because|first,|let's", resp(r), re.I)) for r in rows)
        multi = sum("\n" in resp(r).strip() for r in rows)
        inv.add(f"- raw answer length: median {statistics.median(lens):.0f} chars / {statistics.median(words):.0f} words, "
                f"p90 {sorted(lens)[int(0.9 * (len(lens) - 1))]} chars, max {max(lens)} chars",
                f"- answers with chain-of-thought markers: {cot} ({100 * cot / len(rows):.1f} %); multi-line: {multi} "
                f"({100 * multi / len(rows):.1f} %) - long answers cost decode time in the live design",
                f"- raw answer differs from the filtered (scored) answer on "
                f"{sum(resp(r).strip() != filt(r).strip() for r in rows)} records")
        prompt = rows[0].get("input") or rows[0].get("arguments") or rows[0].get("prompt")
        if prompt is not None:
            inv.add("", "The full prompt of the first record (the template: pre/post prompt, how frames are referenced):", "")
            inv.code(redact(json.dumps(prompt, ensure_ascii=False, default=str) if not isinstance(prompt, str)
                            else prompt)[:4000])
        # per-type accuracy from whatever per-doc score the task stored
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            doc = r.get("doc", {}) or {}
            t = doc.get("question_type") or r.get("question_type")
            score = None
            for k, v in r.items():
                if isinstance(v, dict) and ("accuracy" in v or "MRA:.5:.95:.05" in v):
                    score = v.get("accuracy", v.get("MRA:.5:.95:.05"))
            if t is not None and isinstance(score, (int, float)):
                by_type[t].append(float(score))
        if by_type:
            inv.add("", "| question type | n | mean score |", "|---|---|---|")
            for t, v in sorted(by_type.items()):
                inv.add(f"| {t} | {len(v)} | {100 * statistics.mean(v):.1f} |")
        ex_by_type: dict[str, list] = defaultdict(list)
        for r in rows:
            t = (r.get("doc", {}) or {}).get("question_type", "?")
            if len(ex_by_type[t]) < 2:
                ex_by_type[t].append(r)
        inv.add("", "Two answers per type (raw -> filtered; target):", "")
        ex = []
        for t, rs in sorted(ex_by_type.items()):
            for r in rs:
                ex.append(f"[{t}] raw={resp(r)[:300]!r} -> filtered={filt(r)[:60]!r} ; target={str(r.get('target', ''))[:40]!r}")
        inv.code(redact("\n".join(ex)))
        # the samples file itself is large; keep a 50-record excerpt for the bundle
        ex_path = inv.bundle / "results" / f"{sp.stem}.first50.jsonl"
        ex_path.parent.mkdir(parents=True, exist_ok=True)
        ex_path.write_text("".join(redact(json.dumps(r, ensure_ascii=False, default=str)) + "\n" for r in rows[:50]))


def grep(root: Path, pattern: str, globs: tuple[str, ...], limit: int = 80) -> list[str]:
    out = []
    rx = re.compile(pattern, re.I)
    for g in globs:
        for p in sorted(root.rglob(g)):
            if any(part in (".git", "__pycache__", "node_modules", "ckpt", "logs", "wandb", "lmms_eval") for part in p.parts):
                continue
            try:
                if p.stat().st_size > (1 << 20):
                    continue
                for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(ln):
                        out.append(f"{p.relative_to(root)}:{i}: {redact(ln.strip())[:600]}")
                        if len(out) >= limit:
                            return out
            except OSError:
                continue
    return out


def section_repo(inv: Inventory, a) -> None:
    inv.h("VG-LLM repository")
    vg = Path(a.vg)
    if not vg.is_dir():
        inv.add(f"`{vg}` not found.")
        return
    inv.add(f"commit: `{sh(['git', 'log', '-1', '--format=%H %ci %s'], vg)}`",
            f"remote: `{redact(sh(['git', 'remote', '-v'], vg).splitlines()[0] if sh(['git', 'remote', '-v'], vg) else '-')}`",
            f"uncommitted changes: {len(sh(['git', 'status', '--short'], vg).splitlines())} files")
    ev = vg / "scripts" / "evaluation" / "eval_lora_multi.sh"
    if ev.exists():
        inv.copy(ev, "repo/scripts/evaluation/eval_lora_multi.sh")
        text = ev.read_text(errors="replace")
        inv.add("", "`scripts/evaluation/eval_lora_multi.sh` (frames, pixels, model_args, prompts are set here):", "")
        inv.code(redact(text)[:8000], "bash")
        for ref in sorted(set(re.findall(r"[\w./-]+\.(?:sh|py|yaml)", text)))[:20]:
            rp = (vg / ref) if not ref.startswith("/") else Path(ref)
            if rp.is_file():
                inv.copy(rp, "repo/" + (str(rp.relative_to(vg)) if str(rp).startswith(str(vg)) else rp.name))
    hits = grep(vg, re.escape(RUN_NAME) + r"|allcot|lora2", ("*.sh", "*.py", "*.yaml", "*.json"), 60)
    inv.add("", f"Where this run is defined (`{RUN_NAME}`, allcot, lora2):", "")
    inv.code("\n".join(hits) or "(no match)")
    for f in sorted({h.split(":")[0] for h in hits})[:15]:
        inv.copy(vg / f, "repo/" + f)
    reg = grep(vg / "src" if (vg / "src").is_dir() else vg,
               r"(annotation_path|data_path|json_path|DATASET|dataset_use|\.jsonl?[\"'])", ("*.py",), 80)
    inv.add("", "Dataset registry lines (annotation files and dataset names):", "")
    inv.code("\n".join(reg) or "(no match)")
    paths = sorted({m for h in reg for m in re.findall(r"[\"'](/[^\"']+\.jsonl?)[\"']", h)})
    if paths:
        inv.add("", "| annotation file | exists | size | records |", "|---|---|---|---|")
        for ap in paths[:40]:
            p = Path(ap)
            n = "-"
            if p.is_file():
                try:
                    if p.suffix == ".jsonl":
                        n = str(sum(1 for _ in open(p, "rb")))
                    elif p.stat().st_size < (600 << 20):
                        d = json.loads(p.read_text())
                        n = str(len(d)) if isinstance(d, (list, dict)) else "?"
                except Exception:  # noqa: BLE001
                    n = "?"
            inv.add(f"| `{ap}` | {'yes' if p.is_file() else 'no'} | {human(p.stat().st_size) if p.is_file() else '-'} | {n} |")
    frames = grep(vg, r"max_num_frames|max_frames|nframes|num_frames|video_fps|\bfps\b|max_pixels|min_pixels|"
                      r"total_pixels|frame_sample|sample_frames|uniform", ("*.py", "*.sh", "*.yaml"), 80)
    inv.add("", "Evaluation / training frame protocol (frames per video, pixels, sampling) - wherever it is set:", "")
    inv.code("\n".join(frames) or "(no match)")
    fus = grep(vg / "src" if (vg / "src").is_dir() else vg,
               r"geometry|cut3r|vggt|fusion|merger|spatial_merge", ("*.py",), 120)
    inv.add("", "Where geometry enters the model (fusion code):", "")
    inv.code("\n".join(fus) or "(no match)")
    for f in [f for f, n in Counter(h.split(":")[0] for h in fus).most_common(6)]:
        base = vg / "src" if (vg / "src").is_dir() else vg
        inv.copy(base / f, "repo/src/" + f)


def training_package(vg: Path) -> Path:
    """The package the training scripts run (src/<pkg>/train/train_*.py), e.g. src/qwen35."""
    for sc in sorted((vg / "scripts" / "train").glob("*.sh")):
        m = re.search(r"src/(\w+)/train/\w+\.py", sc.read_text(errors="replace"))
        if m and (vg / "src" / m.group(1)).is_dir():
            return vg / "src" / m.group(1)
    return vg / "src"


def section_data(inv: Inventory, a) -> None:
    """Every dataset name the training scripts use, resolved in the training package's registry."""
    inv.h("Training data (the datasets named in the training scripts, resolved)")
    vg = Path(a.vg)
    pkg = training_package(vg)
    names: dict[str, set] = defaultdict(set)
    for sc in sorted((vg / "scripts" / "train").glob("*.sh")):
        for m in re.finditer(r'^DATASETS="([^"]+)"', sc.read_text(errors="replace"), re.M):
            for n in m.group(1).split(","):
                names[n.strip().split("%")[0]].add(sc.name)
    if not names:
        inv.add("No DATASETS= line in scripts/train/*.sh.")
        return
    inv.add(f"Training package: `{pkg.relative_to(vg)}`. {len(names)} dataset names across the training scripts.", "")
    py = [p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts]
    text = {p: p.read_text(errors="replace") for p in py if p.stat().st_size < (2 << 20)}
    inv.add("| dataset | used by (scripts) | defined in | annotation | size | records | media root |", "|---|---|---|---|---|---|---|")
    for n in sorted(names):
        hit, ann, root = "-", "-", "-"
        for p, t in text.items():
            # NAME = {...} dict entries (VG-LLM / Qwen2.5-VL finetune style), keyed by the upper-case name
            m = re.search(rf"^{re.escape(n.upper())}\s*=\s*\{{(.*?)\}}", t, re.M | re.S) or \
                re.search(rf"[\"']{re.escape(n)}[\"']\s*:\s*([A-Z0-9_]+|\{{.*?\}})", t, re.S)
            if m:
                hit = str(p.relative_to(vg))
                body = m.group(1)
                if re.fullmatch(r"[A-Z0-9_]+", body.strip()):
                    m2 = re.search(rf"^{body.strip()}\s*=\s*\{{(.*?)\}}", t, re.M | re.S)
                    body = m2.group(1) if m2 else body
                am = re.search(r"annotation_path[\"']?\s*[:=]\s*[\"']([^\"']+)", body)
                rm = re.search(r"data_path[\"']?\s*[:=]\s*[\"']([^\"']+)", body)
                ann = am.group(1) if am else "-"
                root = rm.group(1) if rm else "-"
                break
        size, recs = "-", "-"
        ap_ = Path(ann) if ann != "-" else None
        if ap_ is not None and not ap_.is_absolute():
            ap_ = vg / ap_
        if ap_ is not None and ap_.is_file():
            size = human(ap_.stat().st_size)
            try:
                if ap_.suffix == ".jsonl":
                    recs = str(sum(1 for _ in open(ap_, "rb")))
                elif ap_.stat().st_size < (1500 << 20):
                    d = json.loads(ap_.read_text())
                    recs = str(len(d)) if isinstance(d, (list, dict)) else "?"
                else:
                    recs = "(too large to load)"
            except Exception as exc:  # noqa: BLE001
                recs = f"({type(exc).__name__})"
        inv.add(f"| {n} | {', '.join(sorted(names[n]))[:80]} | `{hit}` | `{ann}` | {size} | {recs} | `{root}` |")
    for p in py:
        if re.search(r"annotation_path", text.get(p, "")):
            inv.copy(p, "repo/" + str(p.relative_to(vg)))


def section_fusion(inv: Inventory, a) -> None:
    """The model code of the training package: the geometry merger, the feature fusion, the encoder call."""
    inv.h("Geometry fusion in the training package")
    vg = Path(a.vg)
    pkg = training_package(vg)
    hits = grep(pkg, r"gated_token|feature_fusion|geometry_merger|class \w*(Fusion|Merger)\w*|geometry_encoder\(|"
                     r"reference_frame|geometry_layer_index", ("*.py",), 150)
    inv.code("\n".join(hits) or "(no match)")
    for f, _ in Counter(h.split(":")[0] for h in hits).most_common(8):
        inv.copy(pkg / f, "repo/" + str((pkg / f).relative_to(vg)))


def section_profiling(inv: Inventory, a) -> None:
    """Their own latency profiles of the 70.83 model at 1 fps on long videos (profiling/results/*/profile.jsonl)."""
    inv.h("Their latency profiles (profiling/results)")
    root = Path(a.vg) / "profiling" / "results"
    files = sorted(root.glob("*/profile.jsonl")) if root.is_dir() else []
    if not files:
        inv.add(f"No `{root}/*/profile.jsonl`.")
        return
    for f in files:
        rows = []
        for ln in f.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        keys = sorted({k for r in rows for k in r})
        num = {k: [r[k] for r in rows if isinstance(r.get(k), (int, float))] for k in keys}
        summ = ", ".join(f"{k} median {statistics.median(v):.3g}" for k, v in num.items() if v)[:1500]
        inv.add(f"- `{f.parent.name}`: {len(rows)} rows · keys {keys[:30]}", f"  {summ}")
        inv.copy(f, f"profiling/{f.parent.name}/profile.jsonl")
    for f in sorted(root.glob("*/run.log"))[:8]:
        tail = f.read_text(errors="replace").splitlines()[-40:]
        (inv.bundle / "profiling" / f.parent.name).mkdir(parents=True, exist_ok=True)
        (inv.bundle / "profiling" / f.parent.name / "run.log.tail").write_text(redact("\n".join(tail)))


def section_trainlog(inv: Inventory, a) -> None:
    """The head and tail of the checkpoint's train.log (the whole file is over the copy limit)."""
    p = Path(a.lora) / "train.log"
    if not p.is_file():
        return
    lines = p.read_text(errors="replace").splitlines()
    (inv.bundle / "lora").mkdir(parents=True, exist_ok=True)
    (inv.bundle / "lora" / "train.log.head_tail").write_text(redact("\n".join(lines[:300] + ["...", ""] + lines[-300:])))
    inv.h("train.log (head and tail copied to the bundle)")
    inv.add(f"{len(lines)} lines; first lines:", "")
    inv.code(redact("\n".join(lines[:40]))[:6000])


PORT_GLOBS = (   # what porting their model into our live harness needs (relative to the repo)
    "src/qwen35/model/geometry_encoders/*.py", "src/qwen35/model/cut3r/*.py", "src/qwen35/data/draw_marker.py",
    "src/qwen35/train/train_qwen.py", "src/qwen35/train/argument.py", "src/qwen35/train/*.py",
    "src/lmms_eval/models/simple/vgllm*.py", "src/lmms_eval/models/chat/vgllm*.py", "src/lmms_eval/models/*vgllm*.py",
    "src/lmms_eval/tasks/vsibench/*.py", "src/lmms_eval/tasks/vsibench/*.yaml",
)
MIX_70 = ("spar_234k", "llava_hound_64k", "vlm3r_scannet", "vlm3r_scannetpp", "vlm3r_arkit", "sensenova_800k",
          "spar_234k_cot", "llava_hound_64k_cot", "vlm3r_scannet_cot", "vlm3r_scannetpp_cot", "vlm3r_arkit_cot",
          "sensenova_800k_cot", "sqa3d", "scanqa")


def section_port(inv: Inventory, a) -> None:
    """Files for porting (their CUT3R wrapper, eval model, SPAR marks) and 3 records of every dataset of the mix."""
    inv.h("For porting: encoder wrapper, eval model, marker drawing; record samples of the 70.83 mix")
    vg = Path(a.vg)
    got = []
    for g in PORT_GLOBS:
        for f in sorted(vg.glob(g)):
            if "__pycache__" in f.parts or ".ipynb_checkpoints" in f.parts:
                continue
            if inv.copy(f, "repo/" + str(f.relative_to(vg))):
                got.append(str(f.relative_to(vg)))
    inv.add(f"Copied {len(got)} files:", "")
    inv.code("\n".join(got) or "(none found)")
    pkg = training_package(vg)
    reg = (pkg / "data" / "__init__.py")
    text = reg.read_text(errors="replace") if reg.exists() else ""
    out = inv.bundle / "records"
    out.mkdir(parents=True, exist_ok=True)
    inv.add("", "| dataset | annotation | first record keys | image / video field |", "|---|---|---|---|")
    for n in MIX_70:
        m = re.search(rf"^{re.escape(n.upper())}\s*=\s*\{{(.*?)\}}", text, re.M | re.S)
        ap_ = re.search(r"annotation_path[\"']?\s*[:=]\s*[\"']([^\"']+)", m.group(1)).group(1) if m else None
        p = Path(ap_) if ap_ else None
        if p is None or not p.is_file():
            inv.add(f"| {n} | `{ap_}` | (not found) | |")
            continue
        try:
            if p.suffix == ".jsonl":
                recs = []
                with open(p, errors="replace") as f:
                    for ln in f:
                        recs.append(json.loads(ln))
                        if len(recs) == 3:
                            break
            elif p.stat().st_size < (900 << 20):
                d = json.loads(p.read_text(errors="replace"))
                recs = (d[:3] if isinstance(d, list) else list(d.values())[:3])
            else:
                inv.add(f"| {n} | `{ap_}` | (too large for a sample: {human(p.stat().st_size)}) | |")
                continue
        except Exception as exc:  # noqa: BLE001
            inv.add(f"| {n} | `{ap_}` | ({exc!r})[:80] | |")
            continue
        (out / f"{n}.first3.json").write_text(redact(json.dumps(recs, indent=1, ensure_ascii=False, default=str))[:200000])
        r0 = recs[0] if recs else {}
        vis = {k: (v if isinstance(v, str) else (f"list[{len(v)}]: {v[0] if v else ''}" if isinstance(v, list) else type(v).__name__))
               for k, v in r0.items() if k in ("image", "images", "video", "spar_info")}
        inv.add(f"| {n} | `{ap_}` | `{sorted(r0.keys())}` | `{json.dumps(vis, ensure_ascii=False)[:160]}` |")


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    a = ap.parse_args()
    out = Path(a.out)
    inv = Inventory(out)
    inv.add("# Reference system inventory (VSI 70.83)",
            "", "Written by scripts/inspect_reference_system.py - read-only on the reference paths. Maps onto "
                "docs/QUESTIONS_FOR_EXISTING_SYSTEM.md: data (repo: registry + training script), evaluation protocol "
                "(results config + eval script + samples prompt), per-type scores (results), encoder operating point "
                "and fusion (repo, checkpoint tensor groups), recipe (adapter config + training args).")
    for section in (section_paths, section_lora, section_results, section_samples, section_repo, section_data,
                    section_fusion, section_profiling, section_trainlog, section_port):
        try:
            section(inv, a)
        except Exception as exc:  # noqa: BLE001 - one unreadable part must not hide the others
            inv.add("", f"({section.__name__} failed: {exc!r})")
    (out / "INVENTORY.md").write_text("\n".join(inv.lines) + "\n")
    shutil.copy(out / "INVENTORY.md", inv.bundle / "INVENTORY.md")
    tar = out / "reference_70_bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(inv.bundle, arcname="reference_70")
    print(f"{out / 'INVENTORY.md'}\n{tar} ({human(tar.stat().st_size)}) - send this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
