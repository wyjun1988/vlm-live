"""lmms-eval 어댑터 — 설치돼 있을 때만 돈다."""

import pytest

lmms = pytest.importorskip("lmms_eval", reason="lmms-eval 미설치")


def test_live3r_model_registered():
    import live3r.eval.lmms_live3r  # noqa: F401
    from lmms_eval.api.registry import MODEL_REGISTRY

    assert "live3r" in MODEL_REGISTRY


def test_adapter_requires_config():
    from live3r.eval.lmms_live3r import Live3R

    with pytest.raises(ValueError, match="config"):
        Live3R()


@pytest.mark.parametrize(
    "task", ["vsibench", "mmsi_bench", "cv_bench", "videomme", "sparbench", "mmsi_video"]
)
def test_target_tasks_exist(task):
    """평가 계획에 적은 태스크가 업스트림에 실제로 있는지 — 오타·이름변경 방지."""
    from lmms_eval.tasks import TaskManager

    assert task in set(TaskManager().all_tasks)
