"""Training mixtures: several annotation files, each with its own media root, sampled as one dataset.

The registry (configs/data/vgllm.yaml) names every annotation file with the directory its media paths are relative
to; a mix is a list of names, each optionally with a sampling percentage:

    ref70                              a named mix from the registry
    spar_234k,sensenova_800k%25        explicit names; sensenova_800k contributes a fixed 25 % subset
    all6,sqa3d%200                     a named mix plus SQA3D seen twice

`%N` below 100 keeps a fixed random subset (the same one on every rank and every resume - it depends on the seed
only); above 100 it repeats the file (whole repeats, then a fixed subset for the remainder).

`MixtureDataset` concatenates one `SpatialVQADataset` per source, so every source keeps its own media root and
records, a broken record is replaced by another record of the same source (the mix proportions hold), and each
sample carries its source's name as `data_source`.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Source:
    name: str
    annotation: Path
    media_root: Path
    records: int | None = None
    fraction: float = 1.0

    def annotation_for_training(self, prepared_dir: str | Path | None) -> Path:
        """Our JSONL + offset index of this file (scripts/prepare_mix.py) when it exists, else the file itself."""
        if prepared_dir:
            p = Path(prepared_dir) / f"{self.name}.jsonl"
            if p.exists() and Path(str(p) + ".idx.npy").exists():
                return p
        return self.annotation


@dataclass
class Registry:
    datasets: dict[str, Source]
    mixes: dict[str, list[str]]
    path: Path | None = None


def load_registry(path: str | Path, root: str | Path | None = None) -> Registry:
    """The registry YAML -> Sources with absolute paths. `root` overrides the file's `root` (another mount point)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    base = Path(root if root is not None else raw.get("root", "."))
    datasets: dict[str, Source] = {}
    for name, d in (raw.get("datasets") or {}).items():
        if not isinstance(d, dict) or "annotation" not in d or "media" not in d:
            raise ValueError(f"{path}: dataset {name!r} needs 'annotation' and 'media'")
        ann, med = Path(d["annotation"]), Path(d["media"])
        datasets[name] = Source(name=name, annotation=ann if ann.is_absolute() else base / ann,
                                media_root=med if med.is_absolute() else base / med, records=d.get("records"))
    mixes = {k: list(v) for k, v in (raw.get("mixes") or {}).items()}
    for mix, names in mixes.items():
        unknown = [n for n in names if n not in datasets]
        if unknown:
            raise ValueError(f"{path}: mix {mix!r} names unknown datasets {unknown}")
    return Registry(datasets=datasets, mixes=mixes, path=Path(path))


def resolve_mix(spec: str, registry: Registry) -> list[Source]:
    """'ref70' / 'spar_234k,sensenova_800k%25' / 'all6,sqa3d%200' -> Sources with their fractions."""
    out: list[Source] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        name, _, pct = part.partition("%")
        frac = float(pct) / 100.0 if pct else 1.0
        if frac <= 0:
            raise ValueError(f"{part!r}: the percentage must be positive")
        if name in registry.mixes:
            out += [replace(registry.datasets[n], fraction=frac) for n in registry.mixes[name]]
        elif name in registry.datasets:
            out.append(replace(registry.datasets[name], fraction=frac))
        else:
            known = sorted(registry.mixes) + sorted(registry.datasets)
            raise KeyError(f"{name!r} is neither a mix nor a dataset of {registry.path} (known: {known})")
    names = [s.name for s in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{spec!r} names {dup} more than once - give one entry with a percentage instead")
    if not out:
        raise ValueError(f"{spec!r} names no dataset")
    return out


def sample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    """The record indices a source contributes: whole repeats, then a fixed random subset (seed-only)."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    whole = int(math.floor(fraction))
    rest = int(round((fraction - whole) * n))
    parts = [np.arange(n, dtype=np.int64)] * whole
    if rest:
        rng = np.random.default_rng(seed)
        parts.append(np.sort(rng.permutation(n)[:rest]).astype(np.int64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


class MixtureDataset(Dataset):
    """One dataset over several sources. `build(source) -> Dataset` makes each part (a SpatialVQADataset)."""

    def __init__(self, sources: list[Source], build: Callable[[Source], Dataset], seed: int = 0) -> None:
        self.sources = list(sources)
        self.parts = [build(s) for s in self.sources]
        self.index: list[np.ndarray] = []
        for k, (s, part) in enumerate(zip(self.sources, self.parts)):
            # a per-source seed: adding or removing another source must not change this one's subset
            self.index.append(sample_indices(len(part), s.fraction, seed * 1_000_003 + _stable_hash(s.name)))
        self.offsets = np.cumsum([0] + [len(ix) for ix in self.index]).tolist()

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, i: int) -> tuple[int, int]:
        """Global index -> (part, record index within that part)."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        k = bisect.bisect_right(self.offsets, i) - 1
        return k, int(self.index[k][i - self.offsets[k]])

    def __getitem__(self, i: int) -> dict:
        k, j = self.locate(i)
        item = self.parts[k][j]
        item["data_source"] = self.sources[k].name
        return item

    def build(self, i: int) -> dict:
        """One sample without the retry loop (the trainer's donor priming uses it)."""
        k, j = self.locate(i)
        item = self.parts[k].build(j)
        item["data_source"] = self.sources[k].name
        return item

    def summary(self) -> list[tuple[str, int, int, float]]:
        """(name, records in the file, samples in one pass, share of the pass)."""
        total = max(1, len(self))
        return [(s.name, len(p), len(ix), len(ix) / total) for s, p, ix in zip(self.sources, self.parts, self.index)]

    # the trainer reads these from its dataset for the failure summary
    @property
    def n_fail(self) -> int:
        return sum(getattr(p, "n_fail", 0) for p in self.parts)

    @property
    def fail_reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.parts:
            for k, v in getattr(p, "fail_reasons", {}).items():
                out[k] = out.get(k, 0) + v
        return out


def _stable_hash(name: str) -> int:
    """A process-independent hash (Python's hash() of a str changes per process)."""
    h = 0
    for ch in name.encode():
        h = (h * 131 + ch) % 1_000_000_007
    return h
