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


def test_status_does_not_blame_an_arm_for_its_namesakes_failure(tmp_path):
    """A 'FAILED s1_control' line must not mark s1_control_seed1 or s1_control_2b as failed, and vice versa."""
    (tmp_path / "p0.done").touch()
    (tmp_path / "s1_control_2b.done").touch()
    (tmp_path / "PID.node_control").touch()
    (tmp_path / "STATUS").write_text("2026-09-26 01:00:00 [control] started on node_control at abc\n"
                                     "2026-09-26 01:00:01 [control] FAILED s1_control - s1_control.log\n"
                                     "2026-09-26 01:00:02 [small] FAILED s1_real_2b - x.log\n")
    text = wr.section_status(tmp_path)
    rows = {ln.split(" | ")[0].strip("| "): ln for ln in text if ln.startswith("| ")}
    assert rows["s1_control S1 control"].endswith("| FAILED |")
    assert rows["s1_control_seed1 S1 control, seed 1"].endswith("| - |")
    assert rows["s1_control_2b 2B S1 control"].endswith("| done |")
    assert rows["s1_real_2b 2B S1 real"].endswith("| FAILED |")
    assert rows["s1_real S1 real"].endswith("| - |")
    assert rows["p0 shared preflight"].endswith("| done |")
    assert any("Nodes seen: node_control" in ln and "roles: control" in ln for ln in text)


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


def test_probe_section_lists_every_probe_row_and_the_last_answers(tmp_path):
    (tmp_path / "s2_real").mkdir()
    rows = [{"step": 0, "n": 32, "loss": {"real": 1.5, "shuffled": 1.5, "none": 1.52}, "shuffled_minus_real": 0.0,
             "none_minus_real": 0.02, "real_wins": 0.5, "generations": [], "seconds": 40.0},
            {"step": 500, "n": 32, "loss": {"real": 0.9, "shuffled": 0.95, "none": 1.0}, "shuffled_minus_real": 0.05,
             "none_minus_real": 0.1, "real_wins": 0.72, "seconds": 41.0,
             "generations": [{"id": "7", "truth": "1.8 m", "answer": "1.1"}]}]
    (tmp_path / "s2_real" / "probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    text = "\n".join(wr.section_probe(tmp_path))
    assert "### s2_real" in text and "| 0 | 1.5000 | 1.5000 | 1.5200 | 0.0000 | 50% | 0.0200 | 40 |" in text
    assert "| 500 | 0.9000 | 0.9500 | 1.0000 | 0.0500 | 72% | 0.1000 | 41 |" in text
    assert "Answers at step 500: `1.1` (truth `1.8 m`)" in text
    assert wr.section_probe(tmp_path / "nothing") == []


def test_learning_curve_lists_checkpoints_with_per_type_scores(tmp_path):
    (tmp_path / "vsi").mkdir()

    def run(overall, count, room, preds_score):
        preds = [{"id": i, "type": "object_counting" if i % 2 else "room_size_estimation", "video": f"v{i // 2}",
                  "score": preds_score} for i in range(6)]
        return {"scores": {"oracle-image": {"overall": overall, "object_counting_MRA:.5:.95:.05": count,
                                            "room_size_estimation_MRA:.5:.95:.05": room,
                                            "object_rel_direction_accuracy": 0.7}},
                "predictions": {"oracle-image": preds}}

    (tmp_path / "vsi" / "base_hint.json").write_text(json.dumps(run(0.50, 0.60, 0.40, 0.5)))
    (tmp_path / "vsi" / "s2_real_step500.json").write_text(json.dumps(run(0.52, 0.61, 0.45, 0.5)))
    (tmp_path / "vsi" / "s2_real.json").write_text(json.dumps(run(0.55, 0.62, 0.50, 0.6)))
    text = "\n".join(wr.section_curve(tmp_path))
    assert "| base + format hint | 50.0 | — |" in text
    assert "| step 500 | 52.0 |" in text and "| final | 55.0 | +10.0 [" in text
    row = next(ln for ln in text.splitlines() if ln.startswith("| final"))
    assert row.count("|") == 3 + len(wr.TYPE_SHORT) + 1 and "| 62.0 |" in row and "| 50.0 |" in row and "| 70.0 |" in row
    assert wr.section_curve(tmp_path / "nothing") == []


def test_run2_mode_uses_its_phases_comparisons_and_budget(tmp_path):
    (tmp_path / "RUN").write_text("run2\n")
    (tmp_path / "vsi").mkdir()
    for name, sc in (("s2_real", 0.6), ("s2_control", 0.5)):
        preds = [{"id": i, "type": "object_counting", "video": f"v{i // 2}", "score": sc} for i in range(8)]
        (tmp_path / "vsi" / f"{name}.json").write_text(json.dumps(
            {"scores": {"oracle-image": {"overall": sc}}, "predictions": {"oracle-image": preds}, "n_questions": 8}))
    (tmp_path / "budget.env").write_text("N_TRAIN=830135\nCALIB_SPS_S1=16.0\nCALIB_SPS_S2=15.0\nS1_STEPS=324\nS2_STEPS=6485\n")
    (tmp_path / "d0_throughput.txt").write_text("[verdict]\n  F1 NEEDED: the Conv3d patch embed is 40x slower\n")
    wr.RUN = "run2"
    try:
        vsi = "\n".join(wr.section_vsi(tmp_path))
        diag = "\n".join(wr.section_diagnostic(tmp_path))
        status = "\n".join(wr.section_status(tmp_path))
    finally:
        wr.RUN = "weekend"
    assert "S2: real - control (geometry content)  <- key" in vsi and "**+10.0**" in vsi and "Debiased" in vsi
    assert "F1 NEEDED" in diag and "S2: 6485 steps x 128 = 830,080 samples = 1.00 epochs at 15.00 samples/s" in diag
    assert "recover_vsi" in status and "s2_joint" in status
