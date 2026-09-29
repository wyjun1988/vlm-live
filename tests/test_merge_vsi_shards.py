"""scripts/merge_vsi_shards.py: shards of one sharded VSI run combine into what one process would have written."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("merge_vsi_shards", Path(__file__).parent.parent / "scripts" /
                                              "merge_vsi_shards.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)
pytest.importorskip("lmms_eval")


def shard(k, n, videos, **over):
    recs, preds = [], []
    for vi, v in enumerate(videos):
        ds, scene = v.split("/")
        for q in range(2):
            qid = 100 * (vi * n + k) + q
            recs.append({"id": qid, "dataset": ds, "scene_name": scene, "question_type": "object_counting",
                         "prediction": "3", "ground_truth": "3", "MRA:.5:.95:.05": 1.0 if q == 0 else 0.5})
            preds.append({"id": qid, "type": "object_counting", "pred": "3", "gt": "3", "attached": "", "video": v,
                          "score": 1.0 if q == 0 else 0.5})
    d = {k2: None for k2 in mv.SETTINGS}
    d.update(model="m", budget=32, videos=videos, shard=[k, n], records={"oracle-image": recs},
             predictions={"oracle-image": preds}, maps={v: {"area_m2": 1.0} for v in videos}, measurements=[])
    d.update(over)
    return d


def test_merge_restores_the_unsharded_order_and_rescores():
    # the unsharded selection v0..v4 went round-robin: shard 0 = v0, v2, v4 ; shard 1 = v1, v3
    vids = [f"scannet/v{i}" for i in range(5)]
    s0, s1 = shard(0, 2, vids[0::2]), shard(1, 2, vids[1::2])
    m = mv.merge([s1, s0])                                  # any file order
    assert m["videos"] == vids and m["n_questions"] == 10 and m["shards"] == 2
    assert [p["video"] for p in m["predictions"]["oracle-image"]][::2] == vids
    assert m["scores"]["oracle-image"]["overall"] == pytest.approx(0.75)
    assert set(m["maps"]) == set(vids) and m["records"] is None


def test_merge_refuses_mismatched_or_incomplete_shards():
    vids = [f"scannet/v{i}" for i in range(4)]
    with pytest.raises(ValueError, match="exactly once"):
        mv.merge([shard(0, 3, vids[0::3]), shard(1, 3, vids[1::3])])
    with pytest.raises(ValueError, match="disagree on 'weights'"):
        mv.merge([shard(0, 2, vids[0::2], weights="a.pt"), shard(1, 2, vids[1::2], weights="b.pt")])
    with pytest.raises(ValueError, match="scored records"):
        mv.merge([shard(0, 2, vids[0::2], records=None), shard(1, 2, vids[1::2])])
