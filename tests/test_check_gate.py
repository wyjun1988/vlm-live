"""게이트 판정 스크립트 — 사전 등록 기준을 사람이 옮겨 적다 틀리지 않게.

결과 형식은 lmms-eval 0.7.3 실제 metric_list 순서를 따른다:
  vsibench: vsibench_overall 이 먼저, 하위 문항 지표가 뒤 (0~1 스케일)
  videomme: videomme_perception_score (0~100)
  mmstar:   **하위 범주가 먼저, average 가 맨 마지막** ← 첫 지표를 쓰면 틀린다
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_gate.py"


def vsi(overall, order=0.5):
    return {"alias": "vsibench", "vsibench_overall,none": overall,
            "vsibench_overall_stderr,none": 0.01, "obj_appearance_order_accuracy,none": order}


def vmme(score):
    return {"alias": "videomme", "videomme_perception_score,none": score}


def mmstar(avg, coarse=0.7):
    return {"alias": "mmstar", "coarse perception,none": coarse, "fine-grained perception,none": 0.5,
            "math,none": 0.4, "average,none": avg}


def _write(d: Path, results: dict):
    (d / "model").mkdir(parents=True)
    (d / "model" / "20260923_results.json").write_text(json.dumps({"results": results}))


def _run(tmp_path, base, trained, *extra):
    _write(tmp_path / "b", base)
    _write(tmp_path / "t", trained)
    p = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "b"), str(tmp_path / "t"), *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout


def test_default_gate_is_vsibench_and_videomme(tmp_path):
    code, out = _run(
        tmp_path,
        {"vsibench": vsi(0.600), "videomme": vmme(60.0), "mmstar": mmstar(0.60)},
        {"vsibench": vsi(0.625), "videomme": vmme(59.5), "mmstar": mmstar(0.50)},
    )
    assert code == 0 and "게이트 1·2 통과" in out
    assert "vsibench_overall" in out and "+2.50" in out   # 0~1 → 점 단위
    assert "참고·판정 안 함" in out                          # mmstar 는 보고만


def test_general_regression_fails(tmp_path):
    code, out = _run(
        tmp_path,
        {"vsibench": vsi(0.60), "videomme": vmme(60.0)},
        {"vsibench": vsi(0.65), "videomme": vmme(58.0)},
    )
    assert code == 1 and "특화 역설" in out


def test_spatial_gain_too_small_fails(tmp_path):
    code, out = _run(tmp_path, {"vsibench": vsi(0.600), "videomme": vmme(60.0)},
                     {"vsibench": vsi(0.605), "videomme": vmme(60.0)})
    assert code == 1 and "미달" in out


def test_mmstar_uses_average_not_first_subcategory(tmp_path):
    """회귀 방지 — 첫 지표(coarse perception)가 아니라 average 로 판정해야 한다."""
    code, out = _run(
        tmp_path,
        {"vsibench": vsi(0.60), "mmstar": mmstar(avg=0.60, coarse=0.70)},
        # average 는 2점 하락, coarse perception 은 오히려 상승
        {"vsibench": vsi(0.62), "mmstar": mmstar(avg=0.58, coarse=0.80)},
        "--general", "mmstar", "--reference", "",
    )
    assert code == 1, "mmstar 를 하위 범주로 판정했다"
    assert "average,none" in out and "◀" in out


def test_vsibench_uses_overall_not_subtask(tmp_path):
    code, out = _run(
        tmp_path,
        {"vsibench": vsi(0.60, order=0.9), "videomme": vmme(60.0)},
        {"vsibench": vsi(0.60, order=0.1), "videomme": vmme(60.0)},  # overall 동일 → 공간 게이트 미달
    )
    assert code == 1


def test_unknown_task_warns_and_explicit_metric_works(tmp_path):
    code, out = _run(
        tmp_path,
        {"foo": {"a,none": 10.0, "b,none": 60.0}, "videomme": vmme(60.0)},
        {"foo": {"a,none": 10.0, "b,none": 62.0}, "videomme": vmme(60.0)},
        "--spatial", "foo",
    )
    assert code == 1 and "대표 지표를 몰라" in out
    code, out = _run(
        tmp_path / "x",
        {"foo": {"a,none": 10.0, "b,none": 60.0}, "videomme": vmme(60.0)},
        {"foo": {"a,none": 10.0, "b,none": 62.0}, "videomme": vmme(60.0)},
        "--spatial", "foo", "--metric", "foo=b",
    )
    assert code == 0


def _run_alt(tmp_path, base, trained, alt, *extra):
    _write(tmp_path / "v", alt)
    return _run(tmp_path, base, trained, "--base-alt", f"video={tmp_path / 'v'}", *extra)


def test_best_format_baseline_blocks_format_only_gain(tmp_path):
    """베이스가 이미지 모드에서 형식 실패로 낮으면, 형식만 배운 학습이 같은 경로 비교로는 통과한다.
    기준선을 베이스의 최선 형식(비디오 모드)으로 잡으면 막힌다 (2026-09-24 결정, M2 실측 교란)."""
    base = {"vsibench": vsi(0.24), "videomme": vmme(60.0)}          # 이미지 모드: 형식 실패로 낮다
    trained = {"vsibench": vsi(0.45), "videomme": vmme(60.0)}       # 형식을 배워 크게 오름
    code, out = _run(tmp_path / "same", base, trained)
    assert code == 0, "같은 경로 비교로는 통과해야 한다 (교란 재현)"
    code, out = _run_alt(tmp_path / "best", base, trained, {"vsibench": vsi(0.49)})
    assert code == 1 and "기준선 video 49.00" in out and "-4.00" in out


def test_best_format_baseline_passes_real_gain(tmp_path):
    code, out = _run_alt(tmp_path, {"vsibench": vsi(0.24), "videomme": vmme(60.0)},
                         {"vsibench": vsi(0.51), "videomme": vmme(59.5)}, {"vsibench": vsi(0.49)})
    assert code == 0 and "+2.00" in out and "베이스 최선 형식" in out


def test_best_format_uses_image_when_it_is_higher(tmp_path):
    """후보가 더 낮으면 기본(이미지 모드) 기준선 그대로."""
    code, out = _run_alt(tmp_path, {"vsibench": vsi(0.50), "videomme": vmme(60.0)},
                         {"vsibench": vsi(0.52), "videomme": vmme(60.0)}, {"vsibench": vsi(0.40)})
    assert code == 0 and "기준선 기본 50.00" in out


def test_best_format_applies_only_to_spatial(tmp_path):
    """일반 게이트(VideoMME)는 결정대로 같은 경로 비교 — 후보가 있어도 기준선을 바꾸지 않는다."""
    code, out = _run_alt(tmp_path, {"vsibench": vsi(0.50), "videomme": vmme(55.0)},
                         {"vsibench": vsi(0.52), "videomme": vmme(55.0)},
                         {"vsibench": vsi(0.40), "videomme": vmme(65.0)})
    assert code == 0
