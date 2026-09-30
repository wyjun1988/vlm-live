"""Training mixtures (live3r.data.mixture) and the data-only inspection of the reference data."""

import json
from pathlib import Path

import numpy as np
import pytest

from live3r.data.mixture import MixtureDataset, Source, load_registry, resolve_mix, sample_indices

REPO = Path(__file__).resolve().parents[1]


def write_registry(tmp_path: Path) -> Path:
    reg = tmp_path / "reg.yaml"
    reg.write_text(
        f"root: {tmp_path}\n"
        "datasets:\n"
        "  a: {annotation: train/a.json, media: media, records: 3}\n"
        "  b: {annotation: /abs/b.json, media: media/b}\n"
        "  c: {annotation: train/c.json, media: media}\n"
        "mixes:\n"
        "  ab: [a, b]\n")
    return reg


def test_registry_resolves_paths_and_mixes(tmp_path):
    reg = load_registry(write_registry(tmp_path))
    assert reg.datasets["a"].annotation == tmp_path / "train/a.json"
    assert reg.datasets["a"].media_root == tmp_path / "media" and reg.datasets["a"].records == 3
    assert reg.datasets["b"].annotation == Path("/abs/b.json") and reg.datasets["b"].media_root == tmp_path / "media/b"
    assert [s.name for s in resolve_mix("ab", reg)] == ["a", "b"]
    mix = resolve_mix("ab%50, c%200", reg)
    assert [(s.name, s.fraction) for s in mix] == [("a", 0.5), ("b", 0.5), ("c", 2.0)]
    # another mount point
    assert load_registry(write_registry(tmp_path), root="/mnt/x").datasets["a"].media_root == Path("/mnt/x/media")


@pytest.mark.parametrize("spec,err", [("zzz", KeyError), ("a,a", ValueError), ("ab,a", ValueError), ("a%0", ValueError),
                                      ("", ValueError)])
def test_bad_mix_specs(tmp_path, spec, err):
    with pytest.raises(err):
        resolve_mix(spec, load_registry(write_registry(tmp_path)))


def test_registry_rejects_unknown_names_in_a_mix(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("datasets:\n  a: {annotation: a.json, media: m}\nmixes:\n  m: [a, nope]\n")
    with pytest.raises(ValueError, match="nope"):
        load_registry(p)


def test_the_project_registry_reconciles_the_reference_mix():
    """configs/data/vgllm.yaml: the ref70 mix is the 2,029,000 records the 70.83 run's 31,704 steps x 64 consumed."""
    reg = load_registry(REPO / "configs/data/vgllm.yaml")
    ref = resolve_mix("ref70", reg)
    assert len(ref) == 14 and sum(s.records for s in ref) == 2_029_000
    assert sum(s.records for s in resolve_mix("all6", reg)) == 1_337_817
    live = resolve_mix("live", reg)
    assert sum(s.records for s in live) == 1_443_777 and not any(s.name.endswith("_cot") for s in live)
    assert reg.datasets["sensenova_800k"].media_root == Path("/group-volume/VG-LLM_data/media/sensenova")


def test_sample_indices_subset_repeat_and_determinism():
    assert list(sample_indices(4, 1.0, 0)) == [0, 1, 2, 3]
    half = sample_indices(10, 0.5, 7)
    assert len(half) == 5 and len(set(half)) == 5 and list(half) == sorted(half)
    assert list(half) == list(sample_indices(10, 0.5, 7))          # seed-only: the same on every rank and resume
    assert list(half) != list(sample_indices(10, 0.5, 8))
    twice = sample_indices(3, 2.0, 0)
    assert sorted(twice.tolist()) == [0, 0, 1, 1, 2, 2]
    assert len(sample_indices(10, 1.3, 0)) == 13
    assert len(sample_indices(0, 1.0, 0)) == 0


class Part:
    """A stand-in for SpatialVQADataset: records are their own index."""

    def __init__(self, n: int, tag: str) -> None:
        self.n, self.tag = n, tag
        self.n_fail, self.fail_reasons = 1, {f"E:{tag}": 1}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: int) -> dict:
        return {"id": f"{self.tag}{j}", "data_source": "unknown"}

    build = __getitem__


def test_mixture_locates_samples_and_labels_the_source():
    srcs = [Source("a", Path("a"), Path("m"), fraction=1.0), Source("b", Path("b"), Path("m"), fraction=0.5),
            Source("c", Path("c"), Path("m"), fraction=2.0)]
    mix = MixtureDataset(srcs, lambda s: Part({"a": 3, "b": 4, "c": 2}[s.name], s.name), seed=0)
    assert len(mix) == 3 + 2 + 4
    ids = [mix[i]["id"] for i in range(len(mix))]
    assert ids[:3] == ["a0", "a1", "a2"] and all(x.startswith("b") for x in ids[3:5]) and ids[5:] == ["c0", "c1", "c0", "c1"]
    assert {mix[i]["data_source"] for i in range(len(mix))} == {"a", "b", "c"}
    assert mix.build(0)["data_source"] == "a"
    assert [(n, r, k) for n, r, k, _ in mix.summary()] == [("a", 3, 3), ("b", 4, 2), ("c", 2, 4)]
    assert mix.n_fail == 3 and mix.fail_reasons == {"E:a": 1, "E:b": 1, "E:c": 1}
    with pytest.raises(IndexError):
        mix.locate(len(mix))


def test_a_source_subset_does_not_depend_on_the_other_sources():
    b = Source("b", Path("b"), Path("m"), fraction=0.5)
    one = MixtureDataset([b], lambda s: Part(10, "b"), seed=3)
    two = MixtureDataset([Source("a", Path("a"), Path("m")), b], lambda s: Part(10, s.name), seed=3)
    assert list(one.index[0]) == list(two.index[1])


# ------------------------------------------------------------------------------------ the data-only inspector
def test_inspector_reads_records_media_marks_and_scene_overlap(tmp_path):
    import inspect_reference_data as ird
    from PIL import Image

    media = tmp_path / "media"
    (media / "scannet/scene0025_01").mkdir(parents=True)
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(media / "scannet/scene0025_01/0.jpg")
    (media / "hound/v1").mkdir(parents=True)
    for k in range(3):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(media / f"hound/v1/{k:03d}.jpg")
    recs = [
        {"id": 0, "image": ["scannet/scene0025_01/0.jpg", "scannet/scene0999_00/1.jpg"],
         "conversations": [{"from": "human", "value": "<image>\n<image>\nHow far is the red point from the blue?"},
                           {"from": "gpt", "value": "1.5"}],
         "spar_info": json.dumps({"red_point": [[10, 20]], "blue_point": [[30, 40]]}), "type": "distance"},
        {"id": 1, "video": "hound/v1",
         "conversations": [{"from": "human", "value": "<video>\nWhat happens?"}, {"from": "gpt", "value": "A man walks."}]},
        {"id": 2, "image": "scannet/scene0025_01/0.jpg",
         "conversations": [{"from": "human", "value": "<image><image>Q"}, {"from": "gpt", "value": "B"}]},
    ]
    (tmp_path / "train").mkdir()
    (tmp_path / "train/a.json").write_text(json.dumps(recs, indent=1))
    reg = load_registry(write_registry(tmp_path))
    st = ird.Stats(reg.datasets["a"], sample_media=10, seed=0)
    for r in ird.iter_records(reg.datasets["a"].annotation):
        st.add(r)
    assert st.n == 3 and st.kind == {"images": 2, "video": 1} and st.mismatch == 1
    assert st.spar == 1 and st.spar_form == {"str (JSON)": 1} and set(st.spar_keys) == {"red_point", "blue_point"}
    assert st.answer_kind == {"number": 1, "text": 1, "option": 1}
    assert st.scenes["scannet"] == {"scene0025_01", "scene0999_00"}
    media_res = ird.check_media(st, probe=5)
    assert media_res["found"] == 3 and media_res["missing"] == 1
    assert media_res["videos"][0]["kind"] == "frame directory" and media_res["videos"][0]["frames"] == 3
    vsi = ird.vsi_scenes(REPO / "configs/data/vsibench_scenes.txt")
    assert sum(len(v) for v in vsi.values()) == 288 and "scene0025_01" in vsi["scannet"]
    lines, summ = ird.report(st, media_res, vsi, 0.1)
    assert summ["vsibench_overlap"]["scannet"] == {"scenes": 2, "in_vsibench": 1, "examples": ["scene0025_01"]}
    assert any("1 in VSI-Bench" in ln for ln in lines) and any("`type`" in ln for ln in lines)


def test_inspector_streams_json_arrays_and_jsonl(tmp_path):
    import inspect_reference_data as ird

    recs = [{"id": i, "s": "x, ] [ {"} for i in range(4)]
    (tmp_path / "a.json").write_text("  [\n" + ",\n".join(json.dumps(r) for r in recs) + "\n]\n")
    assert [r["id"] for r in ird.iter_records(tmp_path / "a.json")] == [0, 1, 2, 3]
    (tmp_path / "b.json").write_text("\n".join(json.dumps(r) for r in recs))
    assert len(list(ird.iter_records(tmp_path / "b.json"))) == 4
    (tmp_path / "c.json").write_text("[]")
    assert list(ird.iter_records(tmp_path / "c.json")) == []
