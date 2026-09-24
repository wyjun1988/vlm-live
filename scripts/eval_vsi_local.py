"""VSI-Bench 로컬 평가 — 스트리밍(halving) vs 오프라인 오라클 vs 비디오 모드.

프롬프트·채점은 lmms-eval 의 vsibench 함수를 **그대로** 쓴다 (doc_to_text / process_results /
_compute_all_subscores) — 공식 점수와 같은 잣대다.

모드 (`--modes`, 쉼표 구분):
    halving-image   라이브 — 인과적 halving 선택기, 키프레임을 이미지로 (우리 기본)
    oracle-image    오프라인 상한 — 총 길이를 알고 균등 선택, 이미지로
    oracle-video    오프라인 + Qwen 비디오 모드 (2프레임=1블록, 타임스탬프) — 베이스 모델의 네이티브 형식
    halving-video   라이브 + 비디오 모드

비교가 뜻하는 것:
    halving-image − oracle-image  = **키프레임 선택 비용** (손실② — 라이브가 미래를 못 봐서 잃는 것)
    oracle-image  − oracle-video  = 입력 형식 효과 (우리가 고른 이미지 모드가 베이스에 유리/불리한지)

학습 전(weights 없음)에는 프로젝터가 zero-init 이라 기하 주입량이 **정확히 0** 이다
(실가중치 확인: scripts/check_real_weights.py). 그래서 기본 기하 인코더는 dummy — 느린 CUT3R
(M2 307ms/프레임) 없이 같은 숫자가 나온다. 주입 경로 자체는 그대로 탄다.

학습 결과를 평가할 때는 `--config <학습 설정 YAML> --weights <final.pt>` — 학습과 같은 방식으로 모델을 만든다
(실제 CUT3R, LoRA 가 켜져 있으면 LoRA 까지). 기하는 키프레임에서만 돈다 (오라클이면 균등 32장을 순서대로 —
Sensenova 이미지 시퀀스 학습·오프라인 lmms 경로와 같은 입력). 실제 기하는 세션들이 인코더 상태를 공유하므로
모드는 하나씩만 돌린다.

    PYTHONPATH=src python scripts/eval_vsi_local.py --config configs/m2_s2_4b.yaml \
        --weights outputs/m2_s2_4b_real/final.pt --modes oracle-image --videos 60 --seed 1

영상은 **한 번만 디코딩**해서 모든 모드의 세션에 push 한다. 한 영상의 문항들(평균 18개)은
같은 스트림에 대한 질문이라 세션을 재사용한다 — 라이브 시나리오와 같은 구조다.

    PYTHONPATH=src python scripts/eval_vsi_local.py --model checkpoints/qwen3.5-0.8b --videos 24
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ALL_TYPES = [
    "object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
    "object_rel_distance", "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "route_planning", "obj_appearance_order",
]
LMMS_KWARGS = {  # vsibench/_default_template_yaml 의 default
    "pre_prompt": "",
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a single word or phrase.",
}


def pick_videos(docs, k: int, seed: int) -> list[tuple[str, str]]:
    """영상 k 개 — 10개 문항 유형이 모두 들어가게 고른다.

    방향 문항은 easy·medium·hard 가 전부 있어야 lmms-eval 집계가 돈다 (하나라도 빠지면 KeyError).
    """
    by_vid = collections.defaultdict(set)
    for d in docs:
        by_vid[(d["dataset"], d["scene_name"])].add(d["question_type"])
    vids = sorted(by_vid)
    random.Random(seed).shuffle(vids)
    chosen, covered = [], set()
    for v in vids:  # 먼저 유형 커버
        if by_vid[v] - covered:
            chosen.append(v)
            covered |= by_vid[v]
        if covered >= set(ALL_TYPES):
            break
    for v in vids:  # 나머지를 k 까지
        if len(chosen) >= k:
            break
        if v not in chosen:
            chosen.append(v)
    return chosen


_VOCAB_PATTERNS = [
    r"How many (.+?)\(s\) are in this room",
    r"dimension \(length, width, or height\) of the (.+?), measured",
    r"distance between the (.+?) and the (.+?) \(in meters\)",
    r"which of these objects \((.+?)\) is the closest to the (.+?)\?",
    r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to|in) ",
    r"categories in the video: (.+?)\?",
    r"beginning at the (.+?) facing the (.+?)\. You want to navigate to the (.+?)\.",
]


def vsi_vocab(questions: list[str]) -> list[str]:
    """Object names mentioned in a video's VSI questions (ORACLE vocabulary — a live system would have to
    detect open-vocabulary objects without knowing the questions; results using this are an optimistic probe)."""
    import re

    names: list[str] = []
    for q in questions:
        for pat in _VOCAB_PATTERNS:
            for m in re.finditer(pat, q):
                for g in m.groups():
                    names += [n.strip() for n in g.split(",")]
    # some VSI questions are malformed ("beginning at the standing by the window and facing ...") -> drop phrases
    bad = {"standing", "facing", "by", "and", "the"}
    return sorted({n for n in names if n and len(n) < 40 and not (set(n.split()) & bad)})


def detect_objects(live, pb, frame, vocab, device, max_new_tokens: int = 256):
    """The VLM's own grounding on one frame (no geometry injection) -> parsed detections."""
    from dataclasses import replace

    from live3r.data.vision import prepare_image, tokens_per_step
    from live3r.serve.object_map import detection_prompt, parse_detections

    spec = replace(live.spec, max_pixels=640 * 480)          # native VSI resolution — small objects matter
    pv, grid = prepare_image(frame, spec)
    ids = pb.build_query("<image>" + detection_prompt(vocab), [pb.image_segment(tokens_per_step(grid[0], spec))])
    ids = ids.to(device)
    with torch.no_grad():
        out = live.base.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 mm_token_type_ids=(ids == live.image_token_id).long(),
                                 pixel_values=pv.to(device), image_grid_thw=grid.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False)
    return parse_detections(live.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


COGMAP_PROMPT = (
    "You are watching a video of an indoor scene. Write a brief cognitive map of the scene: list the main objects "
    "and give each one's approximate position on a 10 by 10 top-down grid of the whole scene, as JSON like "
    '{"sofa": [[2, 3]], "tv": [[7, 3]]} (one [x, y] per instance). After the JSON, estimate the room\'s width '
    "and length in meters in one short sentence."
)


def wants_room_facts(question: str) -> bool:
    """I-27 routing rule: attach the measured room facts only to questions about the room's size."""
    q = question.lower()
    return "room" in q and any(k in q for k in ("size", "square meter", "area"))


def object_facts_for(question: str, om) -> str:
    """I-28: at question time, look up only the objects the question names and attach measured closest-point
    distances — or nothing if an object is missing / seen in fewer than 2 keyframes. No counts (the model counts
    better from the images). NOTE: the names are read with VSI's question templates (benchmark-specific); a
    general system would have the LLM extract them."""
    import re

    m = re.search(r"distance between the (.+?) and the (.+?) \(in meters\)", question)
    if m:
        d = om.pair_distance(m.group(1), m.group(2))
        if d is None:
            return ""
        return (f"Measured from a 3D reconstruction of the video: the closest points of the {m.group(1)} and the "
                f"{m.group(2)} are about {d:.1f} m apart.\n")
    m = re.search(r"which of these objects \((.+?)\) is the closest to the (.+?)\?", question)
    if m:
        target = m.group(2)
        opts = [o.strip() for o in m.group(1).split(",") if o.strip() and o.strip() != target]
        meas = [(o, om.pair_distance(o, target)) for o in opts]
        meas = [(o, d) for o, d in meas if d is not None]
        if len(meas) >= 2:
            return ("Measured from a 3D reconstruction of the video, closest-point distances to the " + target
                    + ": " + ", ".join(f"{o} {d:.1f} m" for o, d in meas) + ".\n")
    return ""


def count_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        n = c.streams.video[0].frames or 0
    if n <= 0:
        with av.open(str(path)) as c:
            n = sum(1 for _ in c.decode(video=0))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/qwen3.5-0.8b")
    ap.add_argument("--vsi-root", default="data/eval/vsibench")
    ap.add_argument("--videos", type=int, default=24)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--modes", default="halving-image,oracle-image,oracle-video")
    ap.add_argument("--geometry", default="dummy", help="학습 전이면 dummy 로 충분 (zero-init)")
    ap.add_argument("--weights", default=None, help="학습 결과 (.pt) — --config 와 같이 준다")
    ap.add_argument("--config", default=None,
                    help="학습 설정 YAML — 주면 학습과 같은 방식으로 모델을 만든다 (base_model·실제 기하·LoRA)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-map", action="store_true",
                    help="CUT3R 장면 지도 프롬프트 (지도 이미지 + 측정값 텍스트, 키프레임 뒤) — 실제 기하 필요 (--config)")
    ap.add_argument("--map-image", action=argparse.BooleanOptionalAction, default=True,
                    help="with --scene-map: include the map image (--no-map-image = measured facts as text only)")
    ap.add_argument("--object-map", action="store_true",
                    help="with --scene-map: object-level cognitive map as text (VLM grounding on keyframes, "
                         "lifted to 3D with CUT3R). Uses the ORACLE vocabulary from the video's questions")
    ap.add_argument("--detect-every", type=int, default=2, help="--object-map: ground every N-th keyframe")
    ap.add_argument("--detector", choices=["vlm", "owlv2"], default="vlm",
                    help="--object-map: the VLM's own grounding, or OWLv2 (open vocabulary, fixed indoor list)")
    ap.add_argument("--oracle-vocab", action=argparse.BooleanOptionalAction, default=None,
                    help="object names from the video's questions (upper-bound probe only). "
                         "Default: on for --detector vlm, off for owlv2 (fixed indoor vocabulary)")
    ap.add_argument("--first-videos", type=int, default=0,
                    help="evaluate only the first N videos of the --videos selection (quick probes that stay "
                         "comparable with full runs)")
    ap.add_argument("--route-facts", action="store_true",
                    help="I-27: keep the prefix = keyframes + format instruction; attach the measured room facts "
                         "(CUT3R) only to room-size questions when they arrive. Needs --scene-map --no-map-image")
    ap.add_argument("--route-objects", action="store_true",
                    help="I-28: build the object map in the background (needs --object-map) but attach only "
                         "measured distances for the objects a distance question names, at question time")
    ap.add_argument("--detector-device", default="cpu",
                    help="device for OWLv2 (cpu keeps the GPU for the VLM — MPS contention made T3 slow)")
    ap.add_argument("--self-map", action="store_true",
                    help="background thinking (I-20): the VLM first writes a cognitive map of the scene from the "
                         "keyframes (question-agnostic); the text goes into the prompt for every question")
    ap.add_argument("--format-hint", action="store_true",
                    help="지도 없이 형식 지시 한 줄만 (지도 프롬프트의 대조군 — 지시 효과와 지도 효과를 가른다)")
    ap.add_argument("--geom-stride", type=int, default=10**9,
                    help="키프레임 말고도 N 프레임마다 기하를 돌린다 (지도가 촘촘해진다). 기본은 키프레임만")
    ap.add_argument("--save-maps", type=int, default=6, help="--scene-map 일 때 앞쪽 N 개 영상의 지도 이미지를 저장")
    ap.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="같은 영상의 질문들이 시각 프리픽스 캐시를 공유한다 (답은 동일, 수 배 빠름)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import av
    from lmms_eval.tasks.vsibench.utils import (
        MCA_QUESTION_TYPES,
        _compute_all_subscores,
        extract_number,
        fuzzy_matching,
        vsibench_doc_to_text,
        vsibench_process_results,
    )
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from live3r.config import FusionConfig, GeometryConfig, Live3RConfig, LoRAConfig
    from live3r.data.prompt import PromptBuilder
    from live3r.eval.prefix_cache import PrefixCache
    from live3r.eval.streaming import (
        HalvingSelector,
        StreamingAudit,
        StreamingSession,
        UniformOracleSelector,
    )
    from live3r.model.live3r import Live3RModel
    from live3r.serve.scene_map import SceneMap

    root = Path(args.vsi_root)
    docs = [json.loads(line) for line in open(root / "test.jsonl")]
    vids = pick_videos(docs, args.videos, args.seed)
    if args.first_videos:
        vids = vids[: args.first_videos]
    todo = [d for d in docs if (d["dataset"], d["scene_name"]) in set(vids)]
    modes = [m for m in args.modes.split(",") if m]
    print(f"영상 {len(vids)} · 문항 {len(todo)} · 모드 {modes} · 키프레임 {args.budget}")

    dev = torch.device(args.device)
    t0 = time.time()
    if args.config:  # 학습 결과 — 학습과 같은 방식으로 만든다 (실제 기하, LoRA)
        from live3r.train.train import build_model

        cfg = Live3RConfig.from_yaml(args.config)
        if args.scene_map and cfg.geometry.name == "dummy":
            raise SystemExit("--scene-map 은 점·포즈를 내는 실제 기하(CUT3R)가 필요하다 — 설정의 geometry 를 봐라")
        if cfg.geometry.name != "dummy" and len(modes) > 1:
            raise SystemExit("실제 기하는 세션들이 인코더 상태를 공유한다 — --modes 는 하나씩 돌려라")
        stage = "sft" if cfg.lora.enabled else "align"
        live = build_model(argparse.Namespace(stage=stage, init_from=args.weights, tokenizer=None), cfg, True)
        if stage == "align" and args.weights:
            res = live.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
            print(f"가중치 로드 (unexpected {len(res.unexpected_keys)})")
        live = live.to(dev).eval()
        args.model = cfg.base_model
        print(f"학습 설정 {args.config} · 기하 {cfg.geometry.name} · LoRA {cfg.lora.enabled} · 가중치 {args.weights}")
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(dev).eval()
        cfg = Live3RConfig(
            base_model=args.model, dtype=args.dtype,
            geometry=GeometryConfig(name=args.geometry, tap_layers=(6, 9, 12), hidden_size=768, image_size=512),
            fusion=FusionConfig(inject_layers=(0, 1, 2), zero_init=True),
            lora=LoRAConfig(enabled=False),
        )
        live = Live3RModel(cfg, base, tok).to(dev).eval()
        if args.weights:
            res = live.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
            print(f"가중치 로드 (unexpected {len(res.unexpected_keys)})")
    pb = PromptBuilder.from_model(live)
    print(f"모델 로드 {time.time() - t0:.1f}s ({args.model}, {args.device} {args.dtype})")

    scored = {m: [] for m in modes}
    maps: dict[str, dict] = {}
    oracle_vocab = args.oracle_vocab if args.oracle_vocab is not None else (args.detector == "vlm")
    owl = None
    if args.object_map and args.detector == "owlv2":
        from live3r.serve.detector import OWLv2Detector

        owl = OWLv2Detector(device=args.detector_device)
    objects: dict[str, dict] = {}
    self_maps: dict[str, str] = {}
    timing = collections.defaultdict(float)
    for vi, (ds, scene) in enumerate(vids):
        path = root / ds / f"{scene}.mp4"
        qs = [d for d in todo if d["dataset"] == ds and d["scene_name"] == scene]
        total = count_frames(path)
        sessions = {}
        for m in modes:
            sel_name, vmode = m.split("-")
            sel = HalvingSelector(args.budget) if sel_name == "halving" else UniformOracleSelector(args.budget, total)
            # 기하는 키프레임에서만 돌린다 (dummy·zero-init 이라 출력과 무관 — 속도만 아낀다)
            sessions[m] = StreamingSession(live, sel, geom_stride=args.geom_stride, visual_mode=vmode,
                                           device=dev, audit=StreamingAudit(),
                                           scene_map=SceneMap() if args.scene_map else None,
                                           map_image=args.map_image)

        t1 = time.time()
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            fps = float(st.average_rate) if st.average_rate else 30.0
            for s in sessions.values():
                s.begin(fps=fps)
            for i, frame in enumerate(c.decode(video=0)):
                raw = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                for s in sessions.values():
                    s.push(i, raw)
        timing["decode"] += time.time() - t1

        for m, s in sessions.items():
            t2 = time.time()
            pre = s.prefill()
            if s.scene_map is not None and s.scene_map.n_frames:
                maps[f"{ds}/{scene}"] = s.scene_map.facts()
                if pre.get("map_tokens") and len(maps) <= args.save_maps and args.out:
                    mdir = Path(args.out).with_suffix("").parent / (Path(args.out).stem + "_maps")
                    mdir.mkdir(parents=True, exist_ok=True)
                    s.last_map_image.save(mdir / f"{ds}_{scene}.png")
            segs = StreamingSession.segments(pb, pre)
            om = None
            if args.object_map and s.scene_map is not None:
                from live3r.serve.object_map import ObjectMap

                t_det = time.time()
                vocab = vsi_vocab([d["question"] for d in qs]) if oracle_vocab else None
                om = ObjectMap()
                n_det = 0
                for rank, fi in enumerate(pre["frame_indices"]):
                    if rank % args.detect_every:
                        continue
                    frame = s._kept[fi].raw.numpy()
                    if args.detector == "owlv2":
                        dets = owl(frame, vocab)   # vocab None -> fixed indoor list (question-agnostic)
                    else:
                        dets = detect_objects(live, pb, frame, vocab or [], dev)
                    n_det += om.add(rank + 1, dets, s._kept[fi].geom)
                obj_text = om.text(to_map=s.scene_map.to_map, n_frames=len(pre["frame_indices"]))
                timing["detect"] += time.time() - t_det
                objects[f"{ds}/{scene}"] = {"vocab": vocab, "lifted": n_det, "text": obj_text}
                if obj_text and not args.route_objects:
                    segs = (segs[:-1] + [obj_text, segs[-1]]) if pre.get("map_text") else segs + [obj_text]
            if args.self_map:  # I-20: "look at the geometry first", done before any question arrives
                t_sm = time.time()
                pc0 = PrefixCache(live, pb, segs, pre["pixel_kwargs"], pre["geometry"], dev)
                cm, _ = pc0.answer(COGMAP_PROMPT, max_new_tokens=400, do_sample=False)
                del pc0
                cm = cm.strip()
                segs = segs + ["Notes written after watching the video (a top-down cognitive map of the scene):\n"
                               + cm + "\nAnswer directly in the requested format without explanation.\n"]
                self_maps[f"{ds}/{scene}"] = cm
                timing["selfmap"] += time.time() - t_sm
            room_facts = ""
            if (args.route_facts or args.route_objects) and s.scene_map is not None:
                room_facts = s.scene_map.room_facts() if args.route_facts else ""
                if pre.get("map_text") and segs and segs[-1] == pre["map_text"]:
                    segs = segs[:-1]          # no always-on facts in the cached prefix
                segs = segs + ["Answer directly in the requested format without explanation.\n"]
            if args.format_hint:  # 지도 텍스트 끝의 지시문과 같은 자리·같은 문장 (지도·측정값만 없다)
                segs = segs + ["Answer directly in the requested format without explanation.\n"]
            s.report(strict=not m.startswith("oracle"))
            pc = PrefixCache(live, pb, segs, pre["pixel_kwargs"], pre["geometry"], dev) if args.prefix_cache else None
            for d in qs:
                text = vsibench_doc_to_text(dict(d), LMMS_KWARGS)
                attached = ""
                if room_facts and wants_room_facts(d["question"]):
                    attached += room_facts        # question-time routing: a few tokens after the cached prefix
                if args.route_objects and om is not None:
                    attached = object_facts_for(d["question"], om) + attached
                text = attached + text
                if pc is not None:
                    ans, _ = pc.answer(text, max_new_tokens=16, do_sample=False)
                else:
                    ids = pb.build_query(text, segs).to(dev)
                    mm = torch.zeros_like(ids)
                    mm[ids == live.image_token_id] = 1
                    mm[ids == live.video_token_id] = 2
                    with torch.no_grad(), live.injector.primed(pre["geometry"].embeds, live.visual_pos_mask(ids)):
                        out = live.base.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            mm_token_type_ids=mm, max_new_tokens=16, do_sample=False,
                                            **{k: v.to(dev) for k, v in pre["pixel_kwargs"].items()})
                    ans = live.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                ans = ans.strip()
                r = vsibench_process_results(dict(d), [ans])["vsibench_overall"]
                r["attached"] = attached        # what was routed to this question (for diagnosis)
                scored[m].append(r)
            timing[m] += time.time() - t2
        done = sum(len(v) for v in scored.values()) // max(1, len(modes))
        print(f"  [{vi + 1}/{len(vids)}] {ds}/{scene} ({total}프레임, {len(qs)}문항) · 누적 {done}문항 · "
              + " ".join(f"{m} {timing[m]:.0f}s" for m in modes), flush=True)

    # ------------------------------------------------------------------ 집계
    table = {m: _compute_all_subscores(scored[m]) for m in modes}
    keys = [k for k in table[modes[0]] if k != "overall"] + ["overall"]
    print("\n" + f"{'유형':<34}" + "".join(f"{m:>16}" for m in modes))
    print("-" * (34 + 16 * len(modes)))
    for k in keys:
        print(f"{k:<34}" + "".join(f"{100 * table[m][k]:>16.1f}" for m in modes))
    if "halving-image" in modes and "oracle-image" in modes:
        d = 100 * (table["halving-image"]["overall"] - table["oracle-image"]["overall"])
        print(f"\n키프레임 선택 비용 (halving − oracle, 이미지 모드): {d:+.1f} 점")
    if "oracle-image" in modes and "oracle-video" in modes:
        d = 100 * (table["oracle-image"]["overall"] - table["oracle-video"]["overall"])
        print(f"입력 형식 효과 (이미지 − 비디오 모드, 오라클 선택): {d:+.1f} 점")
    # 답 형식 실패율 — lmms-eval 파싱 함수 그대로. 보는 능력이 아니라 "지시대로 답했나"의 문제.
    def format_fail(r):
        if r["question_type"] in MCA_QUESTION_TYPES:
            return fuzzy_matching(r["prediction"]) not in ("A", "B", "C", "D")
        try:
            return extract_number(r["prediction"]) is None
        except Exception:
            return True

    print("\n답 형식 실패율 (선택형: 첫 단어가 A~D 아님 / 수치형: 숫자 추출 실패):")
    for m in modes:
        f = sum(format_fail(r) for r in scored[m]) / max(1, len(scored[m]))
        print(f"  {m:16s} {100 * f:5.1f}%")
    n = len(scored[modes[0]])
    print(f"\n문항 {n} · 디코딩 {timing['decode']:.0f}s · " + " · ".join(f"{m} {timing[m] / max(1, n):.2f}s/문항" for m in modes))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "model": args.model, "videos": [f"{a}/{b}" for a, b in vids], "budget": args.budget,
            "n_questions": n, "scores": {m: {k: float(v) for k, v in table[m].items()} for m in modes},
            "format_fail": {m: sum(format_fail(r) for r in scored[m]) / max(1, len(scored[m])) for m in modes},
            "predictions": {m: [{"id": r["id"], "type": r["question_type"], "pred": r["prediction"],
                                 "gt": r["ground_truth"], "attached": r.get("attached", "")}
                                for r in scored[m]] for m in modes},
            "config": args.config, "weights": args.weights, "scene_map": args.scene_map, "maps": maps,
            "format_hint": args.format_hint, "map_image": args.map_image,
            "object_map": args.object_map, "detector": args.detector if args.object_map else None,
            "oracle_vocab": bool(args.object_map and oracle_vocab), "objects": objects,
            "self_map": args.self_map, "self_maps": self_maps, "route_facts": args.route_facts,
            "route_objects": args.route_objects,
        }, indent=2, ensure_ascii=False))
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
