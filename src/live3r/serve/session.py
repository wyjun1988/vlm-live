"""LiveSession — 라이브 스트리밍 런타임.

오프라인 VLM 과의 결정적 차이:
    오프라인: 질문이 올 때마다 전체 영상을 다시 인코딩한다 → 질문 지연 = O(전체 길이)
    Live3R  : 프레임은 들어오는 대로 KV 캐시에 누적하고, 질문은 캐시 뒤에 붙여 디코딩만 한다
              → 질문 지연 = O(질문 길이)

Qwen3.5 구조가 여기에 유리하다:
  * 32레이어 중 24개가 linear_attention(Gated DeltaNet) = **상태 크기 고정**
  * full_attention 은 8개뿐 → KV 캐시 선형 증가 압박이 1/4로 준다

temporal_patch=2 때문에 비전 타워는 프레임을 2장씩 먹는다. 그래서 프레임 1장 분량의
버퍼링 지연(30fps 에서 33ms)이 구조적으로 생긴다. 이건 숨기지 말고 계측에 포함한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from ..model.live3r import Live3RModel

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    frame_index: int
    geometry_ms: float
    vision_ms: float
    prefill_ms: float
    total_ms: float
    visual_tokens: int
    geometry_state_bytes: int

    @property
    def emitted(self) -> bool:
        """이번 호출이 실제로 비전 블록을 KV 캐시에 넣었는가 (temporal 버퍼가 찼는가)."""
        return self.visual_tokens > 0


@dataclass
class SessionState:
    cache = None
    seq_len: int = 0
    visual_token_count: int = 0
    frames_seen: int = 0
    #: M-RoPE 커서. Qwen3.5 는 비전 블록마다 3D 위치를 새로 계산해야 한다.
    #: 스트리밍에서는 모델의 rope_deltas 캐시가 첫 프리필 값에 고정되므로(past_len>0 분기)
    #: **우리가 직접 position_ids 를 만들어 넘겨야 한다.** 안 그러면 두 번째 블록부터
    #: 비전 토큰이 텍스트처럼 1차원 위치를 받아 공간 정보가 뭉개진다.
    mrope_pos: int = 0
    stats: list[IngestStats] = field(default_factory=list)


class LiveSession:
    """프레임을 계속 먹으면서 아무 때나 질문받는 세션.

    사용:
        sess = LiveSession(model)
        for frame in camera:        # [3, H, W] 정규화된 텐서
            sess.ingest(frame)
        print(sess.ask("소파에서 냉장고까지 몇 미터?"))
    """

    def __init__(
        self,
        model: Live3RModel,
        device: str | torch.device = "cpu",
        max_visual_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.cfg = model.cfg
        self.max_visual_tokens = max_visual_tokens or self.cfg.stream.max_visual_tokens
        if not model.geometry.is_streaming:
            logger.warning(
                "비스트리밍 기하 인코더로 LiveSession 을 돌리고 있다 — 지연 수치를 라이브로 보고하지 마라"
            )
        self.state = SessionState()
        self._frame_buf: list[torch.Tensor] = []
        self._geom_buf: list = []
        self.reset()

    # ------------------------------------------------------------------ 라이프사이클
    def reset(self) -> None:
        self.model.geometry.reset()
        self.state = SessionState()
        self._frame_buf.clear()
        self._geom_buf.clear()

    # ------------------------------------------------------------------- 인제스트
    @torch.no_grad()
    def ingest(self, frame: torch.Tensor, geom_frame: torch.Tensor | None = None) -> IngestStats:
        """프레임 1장을 먹인다.

        Args:
            frame: [3, H, W] — Qwen 비전 타워용 전처리(patch 16 배수, mean/std 0.5)
            geom_frame: [3, Hg, Wg] — 기하 인코더용 전처리(해상도/정규화가 다르다).
                None 이면 frame 을 재사용 (해상도가 맞을 때만).
        """
        t0 = time.perf_counter()
        gf = (geom_frame if geom_frame is not None else frame).unsqueeze(0).to(self.device)

        do_geom = (self.state.frames_seen % max(1, self.cfg.geometry.stride)) == 0
        if do_geom:
            g0 = time.perf_counter()
            gout = self.model.geometry.ingest(gf)
            geometry_ms = (time.perf_counter() - g0) * 1e3
            self._last_geom = gout
        else:
            geometry_ms = 0.0
            gout = self._last_geom  # 보간: 직전 상태를 재사용 (0.8B 비용 절감 축)

        self._frame_buf.append(frame.to(self.device))
        self._geom_buf.append(gout)
        self.state.frames_seen += 1

        tp = self.model.temporal_patch
        if len(self._frame_buf) < tp:
            total = (time.perf_counter() - t0) * 1e3
            st = IngestStats(
                frame_index=self.state.frames_seen - 1,
                geometry_ms=geometry_ms,
                vision_ms=0.0,
                prefill_ms=0.0,
                total_ms=total,
                visual_tokens=0,
                geometry_state_bytes=self.model.geometry.state_bytes(),
            )
            self.state.stats.append(st)
            return st

        # temporal patch 가 찼다 → 비전 인코딩 + KV 프리필
        v0 = time.perf_counter()
        n_vis, geom_bundle, input_ids, pixel_kwargs = self._prepare_block()
        vision_ms = (time.perf_counter() - v0) * 1e3

        p0 = time.perf_counter()
        self._prefill(input_ids, geom_bundle, pixel_kwargs)
        prefill_ms = (time.perf_counter() - p0) * 1e3

        self._frame_buf.clear()
        self._geom_buf.clear()
        total = (time.perf_counter() - t0) * 1e3
        st = IngestStats(
            frame_index=self.state.frames_seen - 1,
            geometry_ms=geometry_ms,
            vision_ms=vision_ms,
            prefill_ms=prefill_ms,
            total_ms=total,
            visual_tokens=n_vis,
            geometry_state_bytes=self.model.geometry.state_bytes(),
        )
        self.state.stats.append(st)
        return st

    def _prepare_block(self):
        """버퍼에 찬 temporal_patch 장의 프레임을 비전 토큰 + 기하 임베딩으로 만든다."""
        m = self.model
        frames = torch.stack(self._frame_buf, 0)  # [tp, 3, H, W]
        _, _, H, W = frames.shape
        p, sm = m.vision_patch, m.spatial_merge
        gh, gw = H // p, W // p
        llm_grid = (gh // sm, gw // sm)
        n_vis = llm_grid[0] * llm_grid[1]

        # Qwen 비디오 전처리 형식으로 평탄화: [t*gh*gw, 3*tp*p*p]
        tp = m.temporal_patch
        x = frames.reshape(1, tp, 3, gh, p, gw, p)
        x = x.permute(0, 3, 5, 2, 1, 4, 6).reshape(gh * gw, 3 * tp * p * p)
        grid_thw = torch.tensor([[1, gh, gw]], device=self.device)

        bundle = m.build_geometry_embeds(self._geom_buf, llm_grid, pool_temporal=True)

        # Qwen3.5 비디오 규약과 동일하게 vision_start/end 로 감싼다.
        # 학습(collator)과 추론(여기)의 프롬프트 형식이 다르면 조용히 성능만 깎인다.
        vs = m.base.config.vision_start_token_id
        ve = m.base.config.vision_end_token_id
        input_ids = torch.cat(
            [
                torch.tensor([[vs]], dtype=torch.long, device=self.device),
                torch.full((1, n_vis), m.video_token_id, dtype=torch.long, device=self.device),
                torch.tensor([[ve]], dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        # M-RoPE: 텍스트(vision_start) → 비전 블록 → 텍스트(vision_end) 순으로 커서를 굴린다.
        cur = self.state.mrope_pos
        pos_start = self._text_positions(cur, 1)                       # [3,1,1]
        cur += 1
        pos_vis = m.base.model.get_vision_position_ids(
            cur, grid_thw[0], 1, sm, device=self.device
        ).unsqueeze(1)                                                  # [3,1,n_vis]
        cur += max(gh, gw) // sm
        pos_end = self._text_positions(cur, 1)
        cur += 1
        pos = torch.cat([pos_start, pos_vis, pos_end], dim=2)           # [3,1,n_vis+2]
        self.state.mrope_pos = cur
        pixel_kwargs = {
            "pixel_values_videos": x,
            "video_grid_thw": grid_thw,
            "position_ids": pos,  # [3, 1, n_vis+2]
        }
        return n_vis, bundle, input_ids, pixel_kwargs

    @torch.no_grad()
    def _prefill(self, input_ids, bundle, pixel_kwargs) -> None:
        m = self.model
        mask = m.visual_pos_mask(input_ids)
        ctx = (
            m.injector.primed(bundle.embeds, mask)
            if m.injector is not None
            else _nullcontext()
        )
        with ctx:
            out = m.base(
                input_ids=input_ids,
                past_key_values=self.state.cache,
                use_cache=True,
                **pixel_kwargs,
            )
        self.state.cache = out.past_key_values
        self.state.seq_len += input_ids.shape[1]
        self.state.visual_token_count += int(mask.sum().item())

    # ---------------------------------------------------------------------- 질의
    @torch.no_grad()
    def ask(self, question_ids: torch.Tensor, max_new_tokens: int = 64) -> dict:
        """누적된 캐시 뒤에 질문을 붙여 답한다. 프레임 재인코딩 없음.

        Args:
            question_ids: [1, L] — 토크나이즈된 질문 (채팅 템플릿 적용 후)
        Returns:
            {"token_ids", "ttft_ms", "tpot_ms", "total_ms"}
        """
        m = self.model
        t0 = time.perf_counter()
        ids = question_ids.to(self.device)
        pos = self.state.mrope_pos
        out = m.base(
            input_ids=ids,
            past_key_values=self.state.cache,
            use_cache=True,
            position_ids=self._text_positions(pos, ids.shape[1]),
        )
        pos += ids.shape[1]
        cache = out.past_key_values
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        ttft = (time.perf_counter() - t0) * 1e3

        gen = [next_tok]
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            out = m.base(
                input_ids=next_tok,
                past_key_values=cache,
                use_cache=True,
                position_ids=self._text_positions(pos, 1),
            )
            pos += 1
            cache = out.past_key_values
            next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(next_tok)
        decode_ms = (time.perf_counter() - t1) * 1e3
        # 질문 캐시는 세션에 남기지 않는다 — 다음 질문이 이전 답변에 오염되지 않도록
        return {
            "token_ids": torch.cat(gen, dim=1),
            "ttft_ms": ttft,
            "tpot_ms": decode_ms / max(1, len(gen) - 1),
            "total_ms": (time.perf_counter() - t0) * 1e3,
        }

    def _text_positions(self, start: int, length: int) -> torch.Tensor:
        """텍스트 토큰의 M-RoPE 위치 [3, 1, L] — 세 축이 모두 같은 값."""
        p = torch.arange(start, start + length, device=self.device)
        return p.view(1, 1, -1).expand(3, 1, -1).contiguous()

    # ---------------------------------------------------------------------- 계측
    def summary(self) -> dict:
        emitted = [s for s in self.state.stats if s.emitted]
        if not emitted:
            return {"frames": self.state.frames_seen, "blocks": 0}
        tot = sorted(s.total_ms for s in emitted)
        per_frame = [s.total_ms / self.model.temporal_patch for s in emitted]
        n = len(tot)
        early = per_frame[: max(1, n // 10)]
        late = per_frame[-max(1, n // 10) :]
        return {
            "frames": self.state.frames_seen,
            "blocks": n,
            "visual_tokens": self.state.visual_token_count,
            "geometry_state_bytes": emitted[-1].geometry_state_bytes,
            "block_ms_p50": tot[n // 2],
            "block_ms_p95": tot[min(n - 1, int(n * 0.95))],
            "frame_ms_mean": sum(per_frame) / n,
            # drift: 스트림 후반 프레임 비용 / 초반 비용. 1.0 에 가까워야 진짜 스트리밍.
            "drift": (sum(late) / len(late)) / max(1e-9, sum(early) / len(early)),
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
