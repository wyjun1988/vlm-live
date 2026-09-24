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
                                           scene_map=SceneMap() if args.scene_map else None)

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
            if pre.get("map_tokens"):
                maps[f"{ds}/{scene}"] = s.scene_map.facts()
                if len(maps) <= args.save_maps and args.out:
                    mdir = Path(args.out).with_suffix("").parent / (Path(args.out).stem + "_maps")
                    mdir.mkdir(parents=True, exist_ok=True)
                    s.last_map_image.save(mdir / f"{ds}_{scene}.png")
            segs = StreamingSession.segments(pb, pre)
            if args.format_hint:  # 지도 텍스트 끝의 지시문과 같은 자리·같은 문장 (지도·측정값만 없다)
                segs = segs + ["Answer directly in the requested format without explanation.\n"]
            s.report(strict=not m.startswith("oracle"))
            pc = PrefixCache(live, pb, segs, pre["pixel_kwargs"], pre["geometry"], dev) if args.prefix_cache else None
            for d in qs:
                text = vsibench_doc_to_text(dict(d), LMMS_KWARGS)
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
                                 "gt": r["ground_truth"]} for r in scored[m]] for m in modes},
            "config": args.config, "weights": args.weights, "scene_map": args.scene_map, "maps": maps,
            "format_hint": args.format_hint,
        }, indent=2, ensure_ascii=False))
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
