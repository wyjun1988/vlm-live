import pytest

from live3r.config import Live3RConfig


@pytest.mark.parametrize("name", ["live3r_4b", "live3r_2b", "live3r_08b", "live3r_dummy"])
def test_shipped_configs_load(name):
    cfg = Live3RConfig.from_yaml(f"configs/{name}.yaml")
    assert cfg.base_model.startswith("Qwen/")
    assert len(cfg.geometry.tap_layers) == len(cfg.fusion.inject_layers)


def test_mismatched_tap_and_inject_rejected():
    with pytest.raises(ValueError, match="주입 지점 하나당 탭 하나"):
        Live3RConfig.from_dict(
            {"geometry": {"tap_layers": [1, 2, 3]}, "fusion": {"inject_layers": [0, 1]}}
        )


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="없는 키"):
        Live3RConfig.from_dict({"fusion": {"nonexistent": 1}})
