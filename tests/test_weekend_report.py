"""scripts/weekend_report.py - the numbers it prints must mean what lmms-eval and train.py mean."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("weekend_report", Path(__file__).parent.parent / "scripts" /
                                              "weekend_report.py")
wr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wr)


def _p(i, t, s, video="scannet/a"):
    return {"id": i, "type": t, "score": s, "video": video}


PREDS = [_p(0, "object_counting", 1.0), _p(1, "object_counting", 0.0), _p(2, "room_size_estimation", 0.2),
         _p(3, "object_rel_direction_easy", 1.0), _p(4, "object_rel_direction_medium", 0.0),
         _p(5, "object_rel_direction_medium", 0.0), _p(6, "object_rel_direction_hard", 0.6)]


def test_overall_averages_the_direction_difficulties_first_like_lmms_eval():
    """lmms-eval: the mean of easy / medium / hard is one type (0.533 here), not the mean of all direction
    questions (0.4) - the second would silently weight medium double."""
    ov, per = wr.overall(PREDS)
    assert per["object_rel_direction"] == pytest.approx((1.0 + 0.0 + 0.6) / 3)
    assert ov == pytest.approx((0.5 + 0.2 + (1.6 / 3)) / 3)


def test_paired_bootstrap_point_is_the_difference_and_same_runs_give_zero():
    other = [dict(p, score=min(1.0, p["score"] + 0.3), video=f"v{p['id'] % 3}") for p in PREDS]
    base = [dict(p, video=f"v{p['id'] % 3}") for p in PREDS]
    d, lo, hi, nv, nq = wr.paired_bootstrap(other, base)
    assert d == pytest.approx(100 * (wr.overall(other)[0] - wr.overall(base)[0]))
    assert (nv, nq) == (3, len(PREDS)) and lo <= d <= hi
    assert wr.paired_bootstrap(base, base)[:3] == (0.0, 0.0, 0.0)


def test_training_log_line_is_parsed(tmp_path):
    # the format of train.py's progress line - keep the two in sync
    log = tmp_path / "s2_real.log"
    log.write_text(
        "2026-09-26 01:00:00,001 [r0] INFO ep 0 step 20/1562 loss 1.2345 lr 3.0e-05/1.0e-04 | inj L0:0.123 "
        "L1:0.100 | gate +0.01 +0.02 | 12.3 samp/s | data 5 geom 80 fb 300 ms/샘플 | mem 45.2GB | skip 3 | sync OK\n"
        "2026-09-26 03:30:00,001 [r0] INFO ep 0 step 40/1562 loss 0.9876 lr 3.0e-05/1.0e-04 | inj L0:0.200 "
        "L1:0.150 | gate +0.02 +0.03 | 13.1 samp/s | data 5 geom 80 fb 300 ms/샘플 | mem 51.0GB | skip 4 | sync OK\n"
        "2026-09-26 03:31:00,001 [r0] WARNING --max-hours 25.0 reached at step 40/1562 — stopping and saving.\n"
        "2026-09-26 03:31:01,001 [r0] INFO 대조(shuffled) 기하: 5120샘플 · 같은 격자 기증자 71% · 기증자 없어 주입 없이 0\n")
    info = wr.parse_train_log(log)
    first, last = info["rows"][0], info["rows"][-1]
    assert (last["step"], last["total"], last["sync"], last["skip"]) == ("40", "1562", "OK", "4")
    assert float(first["loss"]) == 1.2345 and last["inj"] == "L0:0.200 L1:0.150" and last["mem"] == "51.0GB"
    assert info["early"] and info["donor_same"] == 71 and info["error"] is None
    assert wr.fmt_hours(info["rows"]) == "2.5 h"


def test_status_does_not_blame_p1_for_a_p1e_failure(tmp_path):
    (tmp_path / "p0.done").touch()
    (tmp_path / "STATUS").write_text("2026-09-26 01:00:00 FAILED p1e ablation - x.log\n"
                                     "2026-09-26 01:00:01 FAILED s1_control - s1_control.log\n")
    rows = {ln.split(" | ")[0].strip("| "): ln for ln in wr.section_status(tmp_path) if ln.startswith("| ")}
    assert rows["p1 smoke (training)"].endswith("| - |")
    assert rows["p1e smoke (evaluation paths)"].endswith("| FAILED |")
    assert rows["s1_control S1 control"].endswith("| FAILED |")
    assert rows["p0 preflight"].endswith("| done |")


def test_report_on_an_empty_run_directory_says_what_is_missing(tmp_path, capsys):
    import sys

    argv, sys.argv = sys.argv, ["weekend_report.py", str(tmp_path)]
    try:
        assert wr.main() == 0
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "No VSI results yet" in text and "not run yet" in text and "could not build" not in text


def test_vsi_outputs_without_per_question_scores_are_not_paired(tmp_path):
    """Outputs written before eval_vsi_local.py saved scores cannot be bootstrapped - skip, do not crash."""
    (tmp_path / "vsi").mkdir()
    old = {"scores": {"oracle-image": {"overall": 0.5}}, "predictions": {"oracle-image": [{"id": 0, "type": "x"}]}}
    for name in ("s2_real", "s2_control"):
        (tmp_path / "vsi" / f"{name}.json").write_text(json.dumps(old))
    text = "\n".join(wr.section_vsi(tmp_path))
    assert "| s2_real | oracle-image | **50.0**" in text and "S2: real - control" not in text
