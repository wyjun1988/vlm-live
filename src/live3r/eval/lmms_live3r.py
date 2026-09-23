"""lmms-eval 모델 어댑터.

lmms-eval 의 `qwen3_5` 모델(= Qwen3_VL 얇은 래퍼)을 상속해서 **로딩만** 바꾼다.
생성 경로(`generate_until`)는 손대지 않는다 — `Live3RModel.enable_auto_geometry()` 가
`base.forward` 를 감싸 기하 인코딩·주입을 자동으로 하기 때문에, 상위 코드는 자기가
평범한 Qwen3.5 를 돌리고 있다고 믿으면 된다.

**이렇게 한 이유**: lmms-eval 의 generate_until 을 복사해 고치면 업스트림이 바뀔 때마다
깨진다. 상속 + forward 래핑이면 태스크 50개가 수정 없이 그대로 돈다.

사용:
    pip install lmms-eval
    python -m lmms_eval --model live3r \
        --model_args pretrained=Qwen/Qwen3.5-4B,config=configs/live3r_4b.yaml,weights=outputs/4b_s2/final.pt \
        --tasks vsibench,mmsi_bench,videomme --batch_size 1

지원 태스크(업스트림 0.7.3 확인): vsibench, mmsi_bench, mmsi_video, cv_bench,
sparbench, blink, videomme, vsisuper, revsi
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

try:
    from lmms_eval.api.registry import register_model
    from lmms_eval.models.simple.qwen3_5 import Qwen3_5 as _Qwen3_5Base

    _HAVE_LMMS = True
except ImportError:  # pragma: no cover - lmms-eval 미설치 환경
    _HAVE_LMMS = False

    def register_model(*a, **k):  # type: ignore
        def deco(cls):
            return cls

        return deco

    class _Qwen3_5Base:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("lmms-eval 이 필요하다: pip install lmms-eval")


@register_model("live3r")
class Live3R(_Qwen3_5Base):
    """Live3R = Qwen3.5 + 스트리밍 기하 인코더 + deepstack 주입.

    Args (model_args 로 전달):
        pretrained: 베이스 VLM (기본은 config 의 base_model)
        config: Live3R YAML 경로 (필수)
        weights: 학습된 프로젝터/LoRA 체크포인트 (.pt). 없으면 무작위 초기화 상태 =
            zero_init 이므로 **베이스 VLM 과 동일하게 동작한다** (베이스라인 측정용).
        auto_geometry: 기본 True. False 면 기하 주입 없이 순수 베이스로 돈다 (음성 대조군).
    """

    def __init__(
        self,
        pretrained: str | None = None,
        config: str | None = None,
        weights: str | None = None,
        auto_geometry: bool = True,
        streaming: bool = False,
        selector: str = "halving",
        keyframe_budget: int = 32,
        geom_stride: int = 3,
        stream_mode: str = "deferred",
        vlm_short_side: int = 224,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        from ..config import Live3RConfig
        from ..model.live3r import Live3RModel
        from ..model.lora import apply_lora

        if config is None:
            raise ValueError("model_args 에 config=<live3r yaml> 이 필요하다")
        cfg = Live3RConfig.from_yaml(config)
        # thinking 은 전부 끈다 (사내 기준선 73.3 도 off. docs/DIRECTION_20260923.md)
        super().__init__(
            pretrained=pretrained or cfg.base_model, enable_thinking=enable_thinking, **kwargs
        )
        self.streaming = bool(streaming)
        self.selector_name = selector
        self.keyframe_budget = int(keyframe_budget)
        self.geom_stride = int(geom_stride)
        self.stream_mode = stream_mode
        self.vlm_short_side = int(vlm_short_side)
        self.stream_reports: list[dict] = []

        # 상위 클래스가 이미 올려둔 베이스 모델을 그대로 감싼다 (두 번 로드하지 않는다)
        live = Live3RModel(cfg, self._model)
        live.freeze_base()

        if cfg.lora.enabled:
            live = apply_lora(live, cfg.lora)
        if auto_geometry and not self.streaming:
            live.enable_auto_geometry()   # LoRA 적용 뒤에 걸어야 한다 (base 가 래핑되므로)
        elif self.streaming:
            # 스트리밍에서는 자동 기하를 쓰면 안 된다 — 그건 키프레임 픽셀만 보고 기하를
            # 다시 계산하는 것이라, 전 구간을 먹은 스트리밍 상태를 버리는 셈이 된다.
            logger.info("스트리밍 모드 — auto_geometry 비활성 (기하는 StreamingSession 이 관리)")
        else:
            logger.warning("auto_geometry=False — 기하 주입 없이 베이스 VLM 으로 평가한다")

        if weights:
            _load_trained(live, weights)
        else:
            logger.warning(
                "weights 가 없다. 프로젝터가 zero_init 이라 **베이스 VLM 과 동일한 출력**이 나온다. "
                "베이스라인 측정이면 정상이고, 학습 결과를 보려는 거면 weights 를 줘라."
            )

        live = live.to(self._model.device if hasattr(self._model, "device") else "cpu")
        live.eval()
        self.live3r = live
        self._model = live.base      # generate() 는 이쪽으로 간다 (forward 가 래핑돼 있다)

    @property
    def model(self):
        return self._model

    def generate_until(self, requests):
        if not self.streaming:
            return super().generate_until(requests)
        return _streaming_generate(self, requests)


def _streaming_generate(self, requests):
    """스트리밍 제약 하에서 답한다. 프레임은 시간순 1패스, 총 길이는 보지 않는다.

    부모의 generate_until 을 쓰지 않는 유일한 경로다 — 부모는 _probe_video_metadata 로
    **총 프레임 수를 먼저 조회**해서 균등 샘플링한다. 그게 정확히 우리가 금지한 것이다.
    """
    import torch
    from tqdm import tqdm

    from .streaming import SELECTORS, CausalVideoFeed, StreamingAudit, StreamingSession

    live, tok = self.live3r, self.tokenizer
    results = []
    for req in tqdm(requests, disable=(self.rank != 0), desc="Streaming"):
        context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
        visuals = doc_to_visual(self.task_dict[task][split][doc_id])
        video = next((v for v in (visuals or []) if isinstance(v, str)), None)
        if video is None:
            raise ValueError(f"스트리밍 평가는 영상 태스크 전용이다 (doc {doc_id})")

        sel_cls = SELECTORS[self.selector_name]
        sel = sel_cls(self.keyframe_budget)
        audit = StreamingAudit()
        sess = StreamingSession(
            live, sel, geom_stride=self.geom_stride, vlm_short_side=self.vlm_short_side,
            mode=self.stream_mode, device=live.base.device, audit=audit,
        )
        sess.consume(CausalVideoFeed(video, audit))
        pre = sess.prefill()

        ids, mm = _build_stream_prompt(self, pre, context.replace("<image>", "").strip())
        mask = live.visual_pos_mask(ids)
        gk = self._build_generate_kwargs(gen_kwargs)
        with torch.no_grad(), live.injector.primed(pre["geometry"].embeds, mask):
            out = live.base.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                pixel_values_videos=pre["pixel_values_videos"],
                video_grid_thw=pre["video_grid_thw"], **gk,
            )
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        for term in gen_kwargs.get("until", []) or []:
            if term:
                ans = ans.split(term)[0]
        results.append(self._strip_thinking(ans))
        self.stream_reports.append(sess.report(strict=True) | {"doc_id": doc_id})
    return results


def _build_stream_prompt(self, pre, question: str):
    """Qwen3.5 비디오 규약(프레임마다 별도 vision 세그먼트)으로 프롬프트를 만든다."""
    import torch

    from ..data.collate import Live3RCollator

    tok = self.tokenizer
    grid = pre["video_grid_thw"]
    n_patches = int(grid[0, 0])
    per_patch = pre["n_visual_tokens"] // max(1, n_patches)
    col = Live3RCollator(self.processor, self.live3r.vision_patch,
                         self.live3r.temporal_patch, self.live3r.spatial_merge)
    body = col.build_video_prompt(n_patches, per_patch)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": body + "\n" + question}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        self.live3r.base.device
    )
    vid = tok.convert_tokens_to_ids(col.video_token)
    return ids, (ids == vid).long() * 2


def _load_trained(live, path: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"학습 체크포인트가 없다: {p}")
    state = torch.load(p, map_location="cpu")
    result = live.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    if loaded == 0:
        raise RuntimeError(
            f"{p} 에서 로드된 텐서가 0개다. 키 이름이 안 맞는다 — "
            "LoRA 를 켠 채 저장했는데 끄고 로드하는(또는 반대) 경우가 흔하다."
        )
    logger.info("학습 가중치 로드: %d/%d 텐서 (unexpected %d)",
                loaded, len(state), len(result.unexpected_keys))


def manifest():
    """lmms-eval 레지스트리 entry point 용.

    pyproject 의 [project.entry-points."lmms_eval.models"] 에 등록하면
    `pip install -e .` 만으로 `--model live3r` 가 잡힌다.
    """
    from lmms_eval.models.registry_v2 import ModelManifest

    return ModelManifest(
        model_id="live3r",
        simple_class_path="live3r.eval.lmms_live3r.Live3R",
    )
