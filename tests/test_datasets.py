"""데이터셋 — SenseNova-SI 형식(이미지 시퀀스·문자열 conversations·멀티턴)과 영상 레코드."""

import json
import os

import numpy as np
import pytest

from live3r.data.datasets import LazyJsonl, RecordError, load_records, normalize_record


def test_conversations_as_json_string_are_parsed():
    conv = [{"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "A"}]
    rec = normalize_record({"id": 1, "image": ["a.jpg"], "conversations": json.dumps(conv)})
    assert rec.turns == [("<image>Q", "A")] and rec.images == ["a.jpg"] and rec.media_type == "images"


def test_multi_turn_and_system_message():
    conv = [{"from": "system", "value": "sys"},
            {"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]
    rec = normalize_record({"image": "a.jpg", "conversations": conv})
    assert rec.turns == [("Q1", "A1"), ("Q2", "A2")] and rec.images == ["a.jpg"]


@pytest.mark.parametrize("conv", [[], [{"from": "gpt", "value": "A"}], [{"from": "bot", "value": "x"}]])
def test_malformed_records_raise(conv):
    with pytest.raises(RecordError):
        normalize_record({"conversations": conv})


def test_load_json_array_jsonl_and_indexed(tmp_path):
    recs = [{"id": i, "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
            for i in range(5)]
    (tmp_path / "a.json").write_text(json.dumps(recs))
    assert [r["id"] for r in load_records(tmp_path / "a.json")] == list(range(5))

    lines = tmp_path / "b.json"  # 확장자는 .json 인데 내용은 JSONL
    lines.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert len(load_records(lines)) == 5

    jl = tmp_path / "c.jsonl"
    offsets = []
    with open(jl, "wb") as f:
        for r in recs:
            offsets.append(f.tell())
            f.write(json.dumps(r).encode() + b"\n")
    np.save(str(jl) + ".idx.npy", np.asarray(offsets))
    lazy = load_records(jl)
    assert isinstance(lazy, LazyJsonl) and len(lazy) == 5 and lazy[3]["id"] == 3


TOK = os.environ.get("LIVE3R_TOKENIZER")


@pytest.mark.skipif(not TOK, reason="LIVE3R_TOKENIZER 미설정")
def test_dataset_builds_image_sequence_sample(tmp_path):
    from PIL import Image
    from transformers import AutoTokenizer

    from live3r.config import DataConfig
    from live3r.data.datasets import SpatialVQADataset
    from live3r.data.prompt import PromptBuilder
    from live3r.data.vision import VisionSpec, tokens_per_step

    media = tmp_path / "media"
    media.mkdir()
    rng = np.random.default_rng(0)
    for i, (h, w) in enumerate([(480, 640), (968, 1296), (512, 512)]):
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(media / f"{i}.jpg")
    recs = [
        {"id": 0, "image": ["0.jpg", "1.jpg", "2.jpg"],
         "conversations": [{"from": "human", "value": "<image>\n<image>\n<image>\nQ?"},
                           {"from": "gpt", "value": "1.43"}]},
        {"id": 1, "image": ["missing.jpg"],
         "conversations": [{"from": "human", "value": "<image>Q"}, {"from": "gpt", "value": "x"}]},
    ]
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(recs))
    tok = AutoTokenizer.from_pretrained(TOK)
    pb = PromptBuilder(tok, 248056, 248057, 248053, 248054)
    spec = VisionSpec()
    ds = SpatialVQADataset(ann, media, pb, spec, geom_long_side=512, geom_unit=16,
                           data_cfg=DataConfig(), max_fail_rate=1.0)

    item = ds[0]
    grids = item["image_grid_thw"]
    assert item["media_type"] == "images" and grids.shape == (3, 3)
    assert int((item["mm_token_type_ids"] == 1).sum()) == sum(tokens_per_step(g, spec) for g in grids)
    assert len(item["geom_frames"]) == 3 and len(item["llm_grids"]) == 3
    assert max(item["geom_frames"][1].shape[-2:]) == 512, "기하 입력은 긴 변 512 여야 한다"
    assert tok.decode(item["labels"][0][item["labels"][0] != -100]) == "1.43<|im_end|>"

    other = ds[1]  # 이미지 누락 → 다른 샘플로 대체되고 재시도 횟수가 실린다
    assert other["retries"] >= 1 and other["id"] == "0"


def test_lazy_jsonl_is_picklable_after_use(tmp_path):
    """macOS 등 spawn 방식 DataLoader 워커는 데이터셋을 피클링한다 — 파일을 연 뒤에도 돼야 한다."""
    import pickle

    jl = tmp_path / "c.jsonl"
    offsets = []
    with open(jl, "wb") as f:
        for i in range(3):
            offsets.append(f.tell())
            f.write(json.dumps({"id": i}).encode() + b"\n")
    np.save(str(jl) + ".idx.npy", np.asarray(offsets))
    recs = load_records(jl)
    assert recs[1]["id"] == 1          # 파일 핸들이 열린다
    again = pickle.loads(pickle.dumps(recs))
    assert again[2]["id"] == 2 and len(again) == 3
