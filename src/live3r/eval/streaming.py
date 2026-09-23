"""스트리밍 평가 하니스 — 제약을 사람이 지키는 게 아니라 **기계가 강제**한다.

오프라인 벤치를 "라이브로 돌렸다"고 주장하기는 쉽고, 실수로 규칙을 깨기는 더 쉽다.
가장 흔한 사고는 `len(video)` 를 아는 것이다 — 총 길이를 아는 순간 균등 샘플링이
가능해지고, 그건 미래를 본 것이다. 그래서 여기서는 영상 길이를 **구조적으로 숨긴다.**

강제하는 제약 (docs/DIRECTION_20260923.md §5):
  1. 프레임은 시간순 1패스. 되돌아가 재인코딩 금지
  2. 기하 상태 크기 상수
  3. 키프레임 선택은 인과적 — 미래 프레임도, 총 길이도 못 본다
  4. LLM 비전 토큰 예산 상한 고정
  5. 질문 도착 후 프레임 재인코딩 금지

## 왜 모드가 두 개인가

Qwen3.5 는 32층 중 24층이 Gated DeltaNet(선형 어텐션)이다. 이 층들의 "캐시"는 KV 목록이
아니라 **재귀 상태**라서, 한 번 넣은 프레임을 나중에 빼는 게 불가능하다.
→ 키프레임을 LLM 에 바로 흘려넣으면(**incremental**) 선택이 돌이킬 수 없다.
→ 프레임을 밖에 모아뒀다가 질문 시점에 넣으면(**deferred**) reservoir 처럼
  "나중에 더 좋은 프레임이 오면 교체"가 가능하다. 대신 TTFT 에 프리필이 붙는다.

둘 다 라이브 제약을 만족한다. 어느 쪽이 유리한지는 재봐야 아는 것이라 둘 다 구현했다.
"""

from __future__ import annotations

import abc
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from ..data.vision import (
    frame_timestamps,
    geometry_frame,
    llm_grid,
    prepare_image,
    prepare_video,
    tokens_per_step,
)

logger = logging.getLogger(__name__)


class StreamingViolation(RuntimeError):
    """라이브 제약을 깼다. 조용히 넘기면 안 되는 종류의 사고다."""


# --------------------------------------------------------------------------- 감사
@dataclass
class StreamingAudit:
    frames_seen: int = 0
    geometry_calls: int = 0
    vision_encoder_calls: int = 0
    llm_visual_tokens: int = 0
    keyframes_kept: int = 0
    keyframes_evicted: int = 0
    geometry_state_bytes: list[int] = field(default_factory=list)
    buffer_peak_frames: int = 0
    reencoded_frames: int = 0
    length_peeks: int = 0
    violations: list[str] = field(default_factory=list)

    def violate(self, msg: str) -> None:
        self.violations.append(msg)

    def assert_clean(self, token_budget: int | None = None) -> None:
        bad = list(self.violations)
        if self.reencoded_frames:
            bad.append(f"프레임 재인코딩 {self.reencoded_frames}회 — 1패스 위반")
        if self.length_peeks:
            bad.append(f"영상 총 길이 조회 {self.length_peeks}회 — 미래 정보")
        states = set(self.geometry_state_bytes)
        if len(states) > 1:
            bad.append(f"기하 상태 크기가 변했다: {sorted(states)} — 상수 메모리 위반")
        if token_budget is not None and self.llm_visual_tokens > token_budget:
            bad.append(f"비전 토큰 {self.llm_visual_tokens} > 예산 {token_budget}")
        if bad:
            raise StreamingViolation("라이브 제약 위반:\n  - " + "\n  - ".join(bad))

    def to_dict(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "geometry_calls": self.geometry_calls,
            "vision_encoder_calls": self.vision_encoder_calls,
            "llm_visual_tokens": self.llm_visual_tokens,
            "keyframes_kept": self.keyframes_kept,
            "keyframes_evicted": self.keyframes_evicted,
            "geometry_state_bytes": (
                self.geometry_state_bytes[-1] if self.geometry_state_bytes else 0
            ),
            "buffer_peak_frames": self.buffer_peak_frames,
            "violations": self.violations,
        }


# ----------------------------------------------------------------------- 프레임 피드
class CausalVideoFeed:
    """프레임을 시간순으로 한 번만 내준다. **총 길이를 알려주지 않는다.**

    `len()` 이나 `.total_frames` 를 부르면 감사에 기록하고 예외를 던진다.
    이게 있어야 "실수로 오프라인" 을 구조적으로 막는다.
    """

    def __init__(self, path: str | Path, audit: StreamingAudit | None = None) -> None:
        self.path = Path(path)
        self.audit = audit or StreamingAudit()
        self._consumed = False
        #: 카메라 fps 는 스트림 시작 시점에 아는 메타데이터라 미래 정보가 아니다 (총 길이와 다르다)
        self.fps: float | None = None

    def __len__(self):  # noqa: D105
        self.audit.length_peeks += 1
        raise StreamingViolation(
            "영상 총 길이는 미래 정보다. 라이브 시스템은 스트림이 언제 끝나는지 모른다. "
            "균등 샘플링이 필요하면 ReservoirSelector 를 써라 — 길이를 몰라도 균등해진다."
        )

    @property
    def total_frames(self):  # noqa: D102
        return len(self)

    def frames(self) -> Iterator[tuple[int, "torch.Tensor"]]:
        """(index, frame[H,W,3] uint8) 을 순서대로 딱 한 번 내준다."""
        if self._consumed:
            self.audit.reencoded_frames += 1
            raise StreamingViolation("피드를 두 번 소비했다 — 1패스 위반")
        self._consumed = True
        for i, arr in enumerate(self._iter_raw()):
            self.audit.frames_seen += 1
            yield i, arr

    def _iter_raw(self):
        try:
            import av  # type: ignore
        except ImportError:
            av = None
        if av is not None:
            with av.open(str(self.path)) as c:
                st = c.streams.video[0]
                self.fps = float(st.average_rate) if st.average_rate else None
                for frame in c.decode(video=0):
                    yield torch.from_numpy(frame.to_ndarray(format="rgb24"))
            return
        # decord 폴백 — 순차 접근만 쓴다 (len 은 일부러 안 부른다)
        import decord  # type: ignore

        vr = decord.VideoReader(str(self.path))
        self.fps = float(vr.get_avg_fps())
        i = 0
        while True:
            try:
                yield torch.from_numpy(vr[i].asnumpy())
            except (IndexError, StopIteration):
                return
            i += 1


class TensorFeed(CausalVideoFeed):
    """테스트·합성용 — 미리 만든 텐서 [T,H,W,3] 를 같은 규약으로 흘린다."""

    def __init__(
        self, frames: torch.Tensor, audit: StreamingAudit | None = None, fps: float = 30.0
    ) -> None:
        self.path = Path("<tensor>")
        self.audit = audit or StreamingAudit()
        self._consumed = False
        self._frames = frames
        self.fps = fps

    def _iter_raw(self):
        for t in range(self._frames.shape[0]):
            yield self._frames[t]


# ------------------------------------------------------------------ 키프레임 선택기
class KeyframeSelector(abc.ABC):
    """인과적 키프레임 선택. **총 길이를 모른 채** 예산 N장을 고른다."""

    #: 라이브에서 쓸 수 있는가. False 면 오프라인 상한선(oracle) 전용.
    is_causal: bool = True
    #: 이미 넣은 프레임을 되물릴 수 있어야 하는가 (deferred 모드 필요)
    needs_deferred: bool = False

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.reset()

    def reset(self) -> None:
        self.kept: list[int] = []

    def final_selection(self) -> list[int]:
        """질문 시점에 LLM 에 넣을 최종 프레임.

        **이미 보유한 것 중에서 고르는 건 미래 정보가 아니다** — 버린 프레임은 돌아오지 않는다.
        기본은 보유분 전체. 오버샘플링하는 선택기가 여기서 예산에 맞춘다.
        """
        return sorted(self.kept)

    @abc.abstractmethod
    def offer(self, index: int, frame: torch.Tensor) -> tuple[bool, int | None]:
        """프레임 하나를 제안한다.

        Returns:
            (keep, evicted) — keep 이면 채택.
            evicted 는 버릴 프레임 인덱스(int) 또는 인덱스 리스트 또는 None.
        """


class StrideSelector(KeyframeSelector):
    """k 프레임마다 하나. 가장 단순하고 incremental 모드에서 쓸 수 있다.

    한계: 스트림이 길어지면 예산을 앞부분에서 다 써버린다. 짧은 영상 전용.
    """

    needs_deferred = False

    def __init__(self, budget: int, stride: int = 8) -> None:
        self.stride = stride
        super().__init__(budget)

    def offer(self, index: int, frame: torch.Tensor):
        if index % self.stride:
            return False, None
        if len(self.kept) >= self.budget:
            return False, None
        self.kept.append(index)
        return True, None


class ReservoirSelector(KeyframeSelector):
    """Reservoir sampling — **총 길이를 몰라도 결과가 균등**하다.

    오프라인 균등 샘플링이 가진 "미래 지식" 이점을 길이 없이 근사하는 유일한 정공법이다.
    다만 나중 프레임이 앞 프레임을 교체하므로 **deferred 모드가 필요하다**
    (Qwen3.5 의 선형 어텐션 층은 한 번 넣은 프레임을 되물릴 수 없다).
    """

    needs_deferred = True

    def __init__(self, budget: int, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._n = 0
        super().__init__(budget)

    def reset(self) -> None:
        super().reset()
        self._n = 0

    def offer(self, index: int, frame: torch.Tensor):
        self._n += 1
        if len(self.kept) < self.budget:
            self.kept.append(index)
            return True, None
        j = self._rng.randrange(self._n)
        if j < self.budget:
            evicted = self.kept[j]
            self.kept[j] = index
            return True, evicted
        return False, None


class HalvingSelector(KeyframeSelector):
    """**길이를 몰라도 항상 본 구간 전체에 균등하게 깔린다.** 기본값.

    간격 s 로 받다가 예산이 차면 보유분에서 하나 걸러 버리고 s 를 2배로 올린다.
    어느 시점에 잘라도 보유 프레임이 [0, 현재] 에 등간격으로 남는다 —
    오프라인 균등 샘플링을 길이 없이 **결정적으로** 흉내내는 방법이다.

    reservoir 와 비교: reservoir 는 기대값으로만 균등해서 한 번 뽑으면 뭉치고
    꼬리를 자주 놓친다. 공간 커버리지가 중요한 VSI 에서는 이쪽이 낫다.
    (스트림 총 길이가 미지일 때의 표준 해법 — 이벤트 로그 다운샘플링에서 쓰던 것과 같다)
    """

    needs_deferred = True

    def __init__(self, budget: int, overbuffer: int = 2) -> None:
        """
        Args:
            budget: 질문 시점에 LLM 에 넣을 프레임 수 (최종)
            overbuffer: 내부 버퍼를 budget 의 몇 배로 둘지.
                반감 직후 보유분이 절반으로 떨어지므로 1배만 두면 예산의 50~100% 만 쓰게 된다.
                2배를 들고 있다가 질문 때 정확히 budget 개를 균등 추출하면 **항상 100%** 를 쓴다.
                비용은 프레임·기하토큰 버퍼 메모리뿐이다.
        """
        self.overbuffer = max(1, overbuffer)
        self.stride = 1
        super().__init__(budget)

    @property
    def capacity(self) -> int:
        return self.budget * self.overbuffer

    def reset(self) -> None:
        super().reset()
        self.stride = 1

    def offer(self, index: int, frame: torch.Tensor):
        if index % self.stride:
            return False, None
        self.kept.append(index)
        if len(self.kept) <= self.capacity:
            return True, None
        dropped = self.kept[1::2]
        self.kept = self.kept[0::2]
        self.stride *= 2
        if index in self.kept:
            return True, dropped
        return False, dropped

    def final_selection(self) -> list[int]:
        """보유분에서 budget 개를 **인덱스 값 기준** 등간격으로 고른다.

        버퍼 '위치' 기준(k[i*len/n])으로 뽑으면 반감 타이밍에 따라 간격이 들쭉날쭉해진다.
        실제 프레임 인덱스 공간에서 목표 지점을 잡고 가장 가까운 보유 프레임을 고르는 쪽이
        오프라인 균등추출에 훨씬 가깝다 (간격 std 가 절반 이하로 준다).
        """
        k = sorted(self.kept)
        n = self.budget
        if len(k) <= n:
            return k
        if n == 1:
            return k[:1]
        lo, hi = k[0], k[-1]
        out: list[int] = []
        j = 0
        for i in range(n):
            target = lo + (hi - lo) * i / (n - 1)
            while j + 1 < len(k) and abs(k[j + 1] - target) <= abs(k[j] - target):
                j += 1
            if not out or k[j] != out[-1]:
                out.append(k[j])
        # 중복 제거로 모자라면 남은 것으로 채운다 (순서 유지)
        if len(out) < n:
            for x in k:
                if x not in out:
                    out.append(x)
                    if len(out) == n:
                        break
            out.sort()
        return out


class UniformOracleSelector(KeyframeSelector):
    """**라이브 아님.** 총 길이를 알고 균등 추출하는 오프라인 상한선.

    비교군 전용이다. 이걸로 잰 점수를 라이브 점수로 보고하면 안 된다 —
    감사기가 `is_causal=False` 를 보고 결과에 표시한다.
    """

    is_causal = False
    needs_deferred = True

    def __init__(self, budget: int, total_frames: int) -> None:
        self.total = total_frames
        super().__init__(budget)
        k = min(budget, total_frames)
        step = total_frames / k
        self._want = {int(i * step + step / 2) for i in range(k)}

    def offer(self, index: int, frame: torch.Tensor):
        if index in self._want:
            self.kept.append(index)
            return True, None
        return False, None


SELECTORS = {
    "halving": HalvingSelector,          # 기본값 — 길이 미지에서 결정적 균등
    "reservoir": ReservoirSelector,      # 확률적 균등 (비교군)
    "stride": StrideSelector,            # incremental 모드용 (되물림 불필요)
    "uniform_oracle": UniformOracleSelector,  # ⚠️ 라이브 아님. 오프라인 상한선
}
DEFAULT_SELECTOR = "halving"


# ------------------------------------------------------------------- 스트리밍 세션
@dataclass
class _Kept:
    index: int
    raw: torch.Tensor        # [H,W,3] uint8 — VLM 전처리는 질문 시점에 (모드가 정해진 뒤) 한다
    geom: object             # GeomOutput — 이 프레임을 먹은 직후의 기하 출력


class StreamingSession:
    """라이브 제약 하에서 스트림을 먹고, 질문이 오면 답한다.

    핵심 구조 (docs/DIRECTION_20260923.md §3 이중 레이트):
        * 기하 인코더 — `geom_stride` 간격으로 **계속** 돌아 전 구간을 상수 상태에 압축
        * LLM 비전 토큰 — 예산 내 키프레임만. 선택은 인과적

    모드 (`mode`):
        deferred(기본)   키프레임을 밖에 모아뒀다가 질문 시점에 프리필.
                         되물림이 가능해 halving/reservoir 를 쓸 수 있다. TTFT 에 프리필 포함.
        incremental      받는 대로 LLM 에 흘려넣는다. TTFT 최소. 되물림 불가라
                         stride 처럼 되물림이 필요 없는 선택기만 쓸 수 있다.

    키프레임을 LLM 에 넣는 형식 (`visual_mode`):
        image(기본)  키프레임마다 별도 이미지 블록. 기하 토큰이 키프레임과 1:1 로 붙는다.
                     SenseNova-SI(이미지 시퀀스)로 학습한 프로젝터와 **입력 분포가 같다.**
        video        Qwen 비디오 형식(2프레임=1블록, 타임스탬프). 토큰은 절반이지만,
                     띄엄띄엄 뽑힌 키프레임 두 장이 한 블록으로 섞이고 기하도 2장 평균이 된다.
                     이미지로만 학습한 프로젝터에게는 본 적 없는 분포다.
    """

    def __init__(
        self,
        model,
        selector: KeyframeSelector,
        geom_stride: int = 3,
        mode: str = "deferred",
        visual_mode: str = "image",
        device: str | torch.device = "cpu",
        audit: StreamingAudit | None = None,
    ) -> None:
        if mode not in ("deferred", "incremental"):
            raise ValueError(f"모르는 모드 {mode}")
        if visual_mode not in ("image", "video"):
            raise ValueError(f"모르는 visual_mode {visual_mode}")
        if mode == "incremental" and selector.needs_deferred:
            raise ValueError(
                f"{type(selector).__name__} 는 되물림이 필요하다(needs_deferred). "
                "Qwen3.5 는 24/32 층이 선형 어텐션이라 넣은 프레임을 뺄 수 없다 → mode='deferred' 를 써라."
            )
        self.model = model
        self.selector = selector
        self.geom_stride = max(1, geom_stride)
        self.mode = mode
        self.visual_mode = visual_mode
        self.device = torch.device(device)
        self.audit = audit or StreamingAudit()
        if not selector.is_causal:
            self.audit.violate(
                f"{type(selector).__name__} 는 인과적이지 않다 — 오프라인 상한선이지 라이브 점수가 아니다"
            )
        self._kept: dict[int, _Kept] = {}
        self._last_geom = None
        self.fps: float = 30.0

    # ------------------------------------------------------------------ 인제스트
    @torch.no_grad()
    def consume(self, feed: CausalVideoFeed) -> "StreamingSession":
        """피드를 끝까지 먹는다. 프레임은 시간순 1패스."""
        feed.audit = self.audit
        self.fps = float(getattr(feed, "fps", None) or 30.0)
        m = self.model
        m.geometry.reset()
        self.selector.reset()
        self._kept.clear()

        for i, raw in feed.frames():
            keep, evicted = self.selector.offer(i, raw)
            run_geom = (i % self.geom_stride == 0) or keep
            if run_geom:
                self._last_geom = m.geometry.ingest(self._prep_geom(raw))
                self.audit.geometry_calls += 1
                self.audit.geometry_state_bytes.append(m.geometry.state_bytes())

            if evicted is not None:
                for e in (evicted if isinstance(evicted, (list, tuple, set)) else [evicted]):
                    if self._kept.pop(e, None) is not None:
                        self.audit.keyframes_evicted += 1
            if keep:
                self._kept[i] = _Kept(i, raw, self._last_geom)
                self.audit.keyframes_kept += 1
                self.audit.buffer_peak_frames = max(
                    self.audit.buffer_peak_frames, len(self._kept)
                )
        return self

    def _prep_geom(self, raw: torch.Tensor) -> torch.Tensor:
        unit = getattr(self.model.geometry, "patch_size", 16)
        size = self.model.cfg.geometry.image_size
        return geometry_frame(raw, size, unit).unsqueeze(0).to(self.device)

    # ---------------------------------------------------------------------- 질의
    @torch.no_grad()
    def prefill(self) -> dict:
        """버퍼의 키프레임을 LLM 입력 재료로 만든다 (deferred 모드).

        Returns dict:
            pixel_kwargs   모델에 그대로 넘길 픽셀 인자 (pixel_values/image_grid_thw 또는 비디오판)
            geometry       GeometryBundle (주입용)
            step_tokens    스텝(이미지 또는 temporal patch)별 LLM 토큰 수
            timestamps     비디오 모드의 temporal patch 별 시각(초). 이미지 모드는 None
            frame_indices  실제로 넣은 키프레임 인덱스
        """
        chosen = [i for i in self.selector.final_selection() if i in self._kept]
        if not chosen:
            raise StreamingViolation("키프레임이 하나도 안 남았다 — 예산·선택기 설정을 봐라")
        items = [self._kept[i] for i in chosen]
        m, spec = self.model, self.model.spec
        self.audit.vision_encoder_calls += 1

        if self.visual_mode == "image":
            vlm = [prepare_image(it.raw, spec) for it in items]
            grids = torch.cat([g for _, g in vlm], 0)
            pixel_kwargs = {
                "pixel_values": torch.cat([pv for pv, _ in vlm], 0).to(self.device),
                "image_grid_thw": grids.to(self.device),
            }
            llm_grids = [llm_grid(g, spec) for g in grids]
            step_tokens = [tokens_per_step(g, spec) for g in grids]
            bundle = m.build_geometry_embeds([it.geom for it in items], llm_grids, pool_temporal=False)
            timestamps = None
        else:
            frames = torch.stack([it.raw for it in items], 0)
            pv, grid = prepare_video(frames, spec)
            pixel_kwargs = {
                "pixel_values_videos": pv.to(self.device),
                "video_grid_thw": grid.to(self.device),
            }
            geoms = [it.geom for it in items]
            while len(geoms) % spec.temporal_patch:
                geoms.append(geoms[-1])  # patchify 가 마지막 프레임을 복제하는 것과 맞춘다
            bundle = m.build_geometry_embeds(geoms, llm_grid(grid[0], spec), pool_temporal=True)
            step_tokens = [tokens_per_step(grid[0], spec)] * int(grid[0, 0])
            timestamps = frame_timestamps(chosen, self.fps, spec.temporal_patch)

        n_vis = sum(step_tokens)
        self.audit.llm_visual_tokens = n_vis
        return {
            "pixel_kwargs": pixel_kwargs,
            "geometry": bundle,
            "step_tokens": step_tokens,
            "timestamps": timestamps,
            "n_visual_tokens": n_vis,
            "n_keyframes": len(items),
            "frame_indices": chosen,
            "visual_mode": self.visual_mode,
        }

    @staticmethod
    def segments(prompt, pre: dict) -> list[str]:
        """prefill 결과 → 프롬프트 세그먼트 문자열 (PromptBuilder 필요)."""
        if pre["visual_mode"] == "image":
            return [prompt.image_segment(n) for n in pre["step_tokens"]]
        return [prompt.video_segment(pre["step_tokens"][0], pre["timestamps"])]

    def report(self, token_budget: int | None = None, strict: bool = True) -> dict:
        if strict:
            self.audit.assert_clean(token_budget)
        d = self.audit.to_dict()
        d.update(
            selector=type(self.selector).__name__,
            causal=self.selector.is_causal,
            mode=self.mode,
            visual_mode=self.visual_mode,
            geom_stride=self.geom_stride,
        )
        return d
