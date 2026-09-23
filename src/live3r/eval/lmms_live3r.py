"""lmms-eval 모델 어댑터 (`--model live3r`).

lmms-eval 의 `qwen3_5` 모델을 상속해 **모델 로딩**과 태스크 연동(task_dict·요청 형식·메트릭)은
그대로 쓰고, **입력 구성과 생성은 학습과 같은 경로**로 한다.

평가 경로 (`eval_path`, 기본 "live3r"):
    live3r  학습과 같은 형식 — 시스템 프롬프트 없음, 같은 VisionSpec, 영상은 균등 키프레임을
            이미지 모드로 (`eval/consistent.py`). 게이트 비교는 반드시 이 경로에서 한다.
    lmms    부모(qwen3_5) 경로 + `auto_geometry` — 참고용. 학습 분포와 세 군데가 다르다
            (시스템 프롬프트 · 비디오 모드 · 총 픽셀 예산). 이걸로 게이트를 재면
            "일반 능력 회귀"가 아니라 "처음 보는 입력에 대한 반응"을 잰다.

영상 태스크에서 `streaming=True` 면 라이브 제약 하의 스트리밍 경로(`eval/streaming.py`)로 간다.

생성은 기본 **greedy** 다. qwen3_5 의 기본값(temperature 0.7)을 그대로 두면 온도를 안 정한
태스크(예: cv_bench)가 샘플링으로 돌아 점수가 실행마다 달라진다 — 1점짜리 게이트를 못 잰다.

사용:
    python -m lmms_eval --model live3r \\
        --model_args pretrained=Qwen/Qwen3.5-4B,config=configs/live3r_4b.yaml,weights=outputs/4b_s1/final.pt \\
        --tasks videomme,vsibench --batch_size 1

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
        DEFAULT_GEN_KWARGS: dict = {}

        def __init__(self, *a, **k):
            raise ImportError("lmms-eval 이 필요하다: pip install lmms-eval")


@register_model("live3r")
class Live3R(_Qwen3_5Base):
    """Live3R = Qwen3.5 + 스트리밍 기하 인코더 + deepstack 주입.

    Args (model_args 로 전달):
        pretrained: 베이스 VLM (기본은 config 의 base_model)
        config: Live3R YAML 경로 (필수)
        weights: 학습된 프로젝터/LoRA 체크포인트 (.pt). 없으면 zero-init 이라
            **베이스 VLM 과 같은 출력**이 나온다 (같은 경로의 베이스라인).
        eval_path: "live3r"(기본, 학습과 같은 입력) | "lmms"(부모 경로, 참고용)
        use_geometry: False 면 기하 주입 없이 (음성 대조군)
        keyframe_budget: 영상에서 LLM 이 볼 키프레임 수 (오프라인·스트리밍 공통)
        streaming, selector, geom_stride, stream_mode, visual_mode: 스트리밍 평가 설정
    """

    #: greedy 기본값 — 태스크가 온도를 명시하면 그걸 따른다 (모듈 docstring 참고)
    DEFAULT_GEN_KWARGS = {"max_new_tokens": 1024, "temperature": 0}

    def __init__(
        self,
        pretrained: str | None = None,
        config: str | None = None,
        weights: str | None = None,
        eval_path: str = "live3r",
        use_geometry: bool = True,
        keyframe_budget: int = 32,
        streaming: bool = False,
        selector: str = "halving",
        geom_stride: int = 3,
        stream_mode: str = "deferred",
        visual_mode: str = "image",
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        from ..config import Live3RConfig
        from ..model.live3r import Live3RModel
        from ..model.lora import apply_lora

        if config is None:
            raise ValueError("model_args 에 config=<live3r yaml> 이 필요하다")
        if eval_path not in ("live3r", "lmms"):
            raise ValueError(f"eval_path 는 live3r | lmms — 받은 값 {eval_path!r}")
        cfg = Live3RConfig.from_yaml(config)
        # thinking 은 전부 끈다 (사내 기준선 73.3 도 off. docs/DIRECTION_20260923.md)
        super().__init__(
            pretrained=pretrained or cfg.base_model, enable_thinking=enable_thinking, **kwargs
        )
        self.eval_path = eval_path
        self.use_geometry = _as_bool(use_geometry)
        self.keyframe_budget = int(keyframe_budget)
        self.streaming = _as_bool(streaming)
        self.selector_name = selector
        self.geom_stride = int(geom_stride)
        self.stream_mode = stream_mode
        self.visual_mode = visual_mode
        self.stream_reports: list[dict] = []

        # 상위 클래스가 이미 올려둔 베이스 모델을 그대로 감싼다 (두 번 로드하지 않는다)
        live = Live3RModel(cfg, self._model)
        live.freeze_base()
        if cfg.lora.enabled:
            live = apply_lora(live, cfg.lora)
        if weights:
            _load_trained(live, weights)
        else:
            logger.warning(
                "weights 가 없다 — 프로젝터가 zero-init 이라 **베이스 VLM 과 같은 출력**이 나온다. "
                "같은 경로의 베이스라인이면 정상이다."
            )

        # auto_geometry(부모 forward 래핑)는 부모 경로에서만 켠다. live3r 경로에서 켜두면
        # 픽셀에서 기하를 한 번 더 계산해 우리가 넣은 기하를 덮어쓴다.
        if self.eval_path == "lmms" and self.use_geometry and not self.streaming:
            live.enable_auto_geometry()   # LoRA 적용 뒤에 걸어야 한다 (base 가 래핑되므로)
            logger.warning(
                "eval_path=lmms — 부모 경로는 학습 분포와 다르다 (시스템 프롬프트·비디오 모드·해상도). "
                "게이트 비교에는 eval_path=live3r 를 써라."
            )

        live = live.to(self._model.device if hasattr(self._model, "device") else "cpu")
        live.eval()
        self.live3r = live
        self._model = live.base

    @property
    def model(self):
        return self._model

    def generate_until(self, requests):
        if self.eval_path == "lmms" and not self.streaming:
            return super().generate_until(requests)
        return _live3r_generate(self, requests)


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


# ------------------------------------------------------------------- 생성 경로
def _live3r_generate(self, requests) -> list[str]:
    """학습과 같은 입력 경로로 요청들을 처리한다 (영상 + streaming 이면 스트리밍 경로).

    `self` 는 Live3R 인스턴스 — 필요한 속성: live3r, tokenizer, task_dict, rank,
    _build_generate_kwargs, _strip_thinking, 설정값들. (테스트는 가짜 객체로 부른다)
    """
    from tqdm import tqdm

    from ..data.prompt import PromptBuilder
    from .consistent import generate, is_video, prepare_inputs, read_video_frames

    live = self.live3r
    prompt = PromptBuilder.from_model(live, tokenizer=self.tokenizer)
    results = []
    for req in tqdm(requests, disable=(self.rank != 0), desc="Live3R"):
        context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
        visuals = doc_to_visual(self.task_dict[task][split][doc_id]) or []
        video = next((v for v in visuals if is_video(v)), None)
        gk = self._build_generate_kwargs(gen_kwargs)

        if video is not None and self.streaming:
            ans = _stream_one(self, live, prompt, video, context, gk, doc_id)
        else:
            if video is not None:
                frames, idx = read_video_frames(video, self.keyframe_budget)
                items, fidx = list(frames), idx
            else:
                items, fidx = [v for v in visuals if not is_video(v)], None
            prepared = prepare_inputs(live, prompt, context, items, fidx,
                                      use_geometry=self.use_geometry)
            ans = generate(live, prepared, self.tokenizer, **gk)

        for term in gen_kwargs.get("until", []) or []:
            if term:
                ans = ans.split(term)[0]
        results.append(self._strip_thinking(ans))
    return results


def _stream_one(self, live, prompt, video, context, gk, doc_id) -> str:
    """라이브 제약 하에서 한 문항. 프레임은 시간순 1패스, 총 길이는 보지 않는다.

    부모(공식 비디오 프로세서)는 `linspace(0, total_frames-1, n)` 으로 **총 프레임 수를 알고**
    균등 샘플링한다. 그게 정확히 우리가 금지한 것이라 여기서는 쓰지 않는다.
    """
    from .streaming import SELECTORS, CausalVideoFeed, StreamingAudit, StreamingSession

    cls = SELECTORS[self.selector_name]
    if cls.is_causal:
        sel = cls(self.keyframe_budget)
    else:
        # 오프라인 상한선(오라클)은 총 길이를 **알아야** 한다 — 비교군이라 허용된다.
        # 감사는 이 실행을 '인과적이지 않음'으로 기록하고, 결과는 라이브 점수로 쓰면 안 된다.
        sel = cls(self.keyframe_budget, _count_frames(video))
    audit = StreamingAudit()
    sess = StreamingSession(
        live, sel, geom_stride=self.geom_stride, mode=self.stream_mode,
        visual_mode=self.visual_mode, device=next(live.base.parameters()).device, audit=audit,
    )
    sess.consume(CausalVideoFeed(video, audit))
    pre = sess.prefill()

    question = context.replace("<image>", "").replace("<video>", "").strip()
    ids = prompt.build_query(question, StreamingSession.segments(prompt, pre))
    ids = ids.to(next(live.base.parameters()).device)
    mm = torch.zeros_like(ids)
    mm[ids == live.image_token_id] = 1
    mm[ids == live.video_token_id] = 2
    with torch.no_grad(), live.injector.primed(pre["geometry"].embeds, live.visual_pos_mask(ids)):
        out = live.base.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
            **pre["pixel_kwargs"], **gk,
        )
    # 라이브 선택기면 위반 시 예외(strict). 오라클은 위반이 '정상'이므로 기록만 한다.
    self.stream_reports.append(sess.report(strict=sel.is_causal) | {"doc_id": doc_id})
    return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _count_frames(video) -> int:
    """총 프레임 수 — 오라클 전용. 라이브 경로에서는 절대 부르지 않는다."""
    try:
        import decord  # type: ignore

        return len(decord.VideoReader(str(video)))
    except ImportError:
        import av  # type: ignore

        with av.open(str(video)) as c:
            n = c.streams.video[0].frames or 0
        if n <= 0:
            with av.open(str(video)) as c:
                n = sum(1 for _ in c.decode(video=0))
        return n


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
