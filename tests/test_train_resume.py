"""Resume and divergence handling in the training loop - what a multi-day unattended run relies on.

Runs the real trainer (tiny random Qwen3.5 + dummy geometry, CPU) on a few generated records.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

from live3r.train.train import DivergenceGuard, SkipFirst

TOK = os.environ.get("LIVE3R_TOKENIZER")
needs_tok = pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정")


def test_skip_first_drops_indices_once_then_gives_full_passes():
    s = SkipFirst(range(5), skip=2)
    assert list(s) == [2, 3, 4] and list(s) == [0, 1, 2, 3, 4]


def test_divergence_guard_tolerates_isolated_bad_steps_but_not_a_run_of_them():
    g = DivergenceGuard(limit=3)
    for ok in (False, True, False, False):
        g.step(ok, at_step=1)
    assert g.bad_steps == 3 and g.consecutive == 2
    with pytest.raises(FloatingPointError, match="diverging"):
        g.step(False, at_step=5)
    assert "3 steps without update" not in g.summary() and "4 steps" in g.summary()


def _dataset(tmp_path, n=8):
    from PIL import Image

    media = tmp_path / "media"
    media.mkdir()
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        names = [f"{i}_{k}.jpg" for k in range(2)]
        for nm in names:
            Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(media / nm)
        recs.append({"id": i, "image": names,
                     "conversations": [{"from": "human", "value": "<image><image>How far?"},
                                       {"from": "gpt", "value": f"{1 + i % 3}.5 m"}]})
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(recs))
    return ann, media


def _run(monkeypatch, tmp_path, out, *extra):
    from live3r.train import train as T

    ann, media = _dataset(tmp_path) if not (tmp_path / "ann.json").exists() else (tmp_path / "ann.json",
                                                                                  tmp_path / "media")
    argv = ["train", "--config", "configs/live3r_dummy.yaml", "--base-model", "tiny", "--tokenizer", TOK,
            "--geometry", "dummy", "--stage", "align", "--ann", str(ann), "--media-root", str(media),
            "--output", str(out), "--device", "cpu", "--workers", "0", "--grad-accum", "2", "--lr", "1e-2",
            "--log-every", "1", "--warmup-ratio", "0", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert T.main() == 0


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@needs_tok
def test_resume_reproduces_the_uninterrupted_run_exactly(monkeypatch, tmp_path):
    """Stop after step 2 of 4, resume, and end up with the tensors of a run that was never stopped: same data
    order, same optimizer moments, same schedule. Fewer than this and the two training arms would no longer
    have seen the same thing when one of them crashed and came back."""
    a, b = tmp_path / "a", tmp_path / "b"
    _run(monkeypatch, tmp_path, a, "--max-steps", "4", "--save-every", "2", "--stop-at-step", "2")
    assert (a / "resume.pt").exists() and torch.load(a / "resume.pt")["step"] == 2
    interrupted = torch.load(a / "final.pt")
    _run(monkeypatch, tmp_path, a, "--max-steps", "4", "--save-every", "2", "--resume")
    _run(monkeypatch, tmp_path, b, "--max-steps", "4", "--save-every", "2")
    resumed, straight = torch.load(a / "final.pt"), torch.load(b / "final.pt")
    assert not _same(interrupted, resumed), "the resumed run made no progress"
    assert _same(resumed, straight)
    assert _same(torch.load(a / "step2.pt"), torch.load(b / "step2.pt"))


@needs_tok
def test_resume_without_a_checkpoint_starts_fresh_and_refuses_a_finished_one(monkeypatch, tmp_path):
    out = tmp_path / "c"
    _run(monkeypatch, tmp_path, out, "--max-steps", "2", "--save-every", "1", "--resume")
    assert torch.load(out / "resume.pt")["step"] == 2
    with pytest.raises(RuntimeError, match="nothing to resume"):
        _run(monkeypatch, tmp_path, out, "--max-steps", "2", "--save-every", "1", "--resume")


@needs_tok
def test_non_finite_gradient_skips_the_update_and_a_run_of_them_stops(monkeypatch, tmp_path):
    """One bad step must cost one update, not the run; three in a row is divergence and must stop it."""
    real = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def nan_first(params, clip, **kw):
        calls["n"] += 1
        norm = real(params, clip, **kw)
        return torch.tensor(float("nan")) if calls["n"] == 1 else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", nan_first)
    _run(monkeypatch, tmp_path, tmp_path / "d", "--max-steps", "2", "--save-every", "1")
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda p, c, **kw: torch.tensor(float("nan")))
    _run(monkeypatch, tmp_path, tmp_path / "e", "--max-steps", "2", "--save-every", "1")   # 2 bad: tolerated
    untouched = torch.load(tmp_path / "e" / "final.pt")
    assert _same(torch.load(tmp_path / "d" / "step1.pt"), untouched), "a skipped step changed the weights"
    assert not _same(torch.load(tmp_path / "d" / "step2.pt"), untouched), "the next good step made no update"
    with pytest.raises(FloatingPointError, match="diverging"):
        _run(monkeypatch, tmp_path, tmp_path / "f", "--max-steps", "3", "--save-every", "0")
