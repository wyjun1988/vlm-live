"""지연 계측 하니스.

공개 스트리밍 벤치(OVO-S-Bench / StreamingBench / RTV-Bench)는 정확도 위주고
**지연 자체를 표준화해 재는 하니스는 빈약하다.** 그래서 직접 만든다.

재는 것:
    frame_ingest_ms   프레임 1장을 캐시에 넣는 비용 (p50/p95)
    ttft_ms           쿼리 도착 → 첫 토큰
    tpot_ms           토큰당 디코딩 비용
    peak_mem_mb       피크 메모리
    drift             스트림 후반 프레임 비용 / 초반 비용. **1.0 근처여야 진짜 스트리밍**
    offline_speedup   같은 질문을 오프라인 방식(전 프레임 재인코딩)으로 했을 때 대비 배수

목표선 (Gemini Live 체감 기준):
    ttft < 500ms, frame_ingest < 33ms (30fps), drift < 1.2
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..serve.session import LiveSession


@dataclass
class LatencyReport:
    label: str
    device: str
    frames: int
    visual_tokens: int
    frame_ingest_ms_p50: float
    frame_ingest_ms_p95: float
    ttft_ms: float
    tpot_ms: float
    drift: float
    peak_mem_mb: float
    geometry_state_mb: float
    offline_ttft_ms: float | None = None

    @property
    def meets_live_budget(self) -> bool:
        return (
            self.ttft_ms < 500.0
            and self.frame_ingest_ms_p95 < 33.0
            and self.drift < 1.2
        )

    def pretty(self) -> str:
        ok = "OK " if self.meets_live_budget else "MISS"
        lines = [
            f"[{ok}] {self.label}  ({self.device}, {self.frames} frames)",
            f"  frame ingest  p50 {self.frame_ingest_ms_p50:7.2f} ms   p95 {self.frame_ingest_ms_p95:7.2f} ms   (예산 33)",
            f"  TTFT          {self.ttft_ms:7.2f} ms   (예산 500)",
            f"  TPOT          {self.tpot_ms:7.2f} ms",
            f"  drift         {self.drift:7.3f}       (예산 1.2)",
            f"  peak mem      {self.peak_mem_mb:7.1f} MB   기하상태 {self.geometry_state_mb:.2f} MB",
            f"  visual tokens {self.visual_tokens}",
        ]
        if self.offline_ttft_ms is not None:
            lines.append(
                f"  오프라인 대비  {self.offline_ttft_ms / max(1e-9, self.ttft_ms):6.1f}× 빠름"
                f"  (오프라인 TTFT {self.offline_ttft_ms:.0f} ms)"
            )
        return "\n".join(lines)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


@torch.no_grad()
def benchmark_stream(
    model,
    n_frames: int = 256,
    frame_hw: tuple[int, int] = (256, 448),
    geom_hw: tuple[int, int] | None = None,
    question_len: int = 24,
    max_new_tokens: int = 32,
    device: str | torch.device = "cpu",
    label: str = "live3r",
    warmup: int = 4,
    measure_offline: bool = False,
) -> LatencyReport:
    """합성 프레임으로 라이브 경로 지연을 잰다.

    합성 프레임인 이유: 지연은 내용이 아니라 형상·경로에 좌우된다. 실데이터 대기 없이
    구조 변경(주입 깊이, merge_size, stride)의 지연 영향을 바로 볼 수 있어야 한다.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    H, W = frame_hw
    sess = LiveSession(model, device=device)

    def frame():
        return torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)

    for i in range(warmup):
        sess.ingest(frame())
    sess.reset()

    for _ in range(n_frames):
        sess.ingest(frame())
    s = sess.summary()

    per_frame = sorted(
        st.total_ms / model.temporal_patch for st in sess.state.stats if st.emitted
    )
    n = len(per_frame)
    q = torch.randint(0, 1000, (1, question_len))
    r = sess.ask(q, max_new_tokens=max_new_tokens)

    offline_ttft = None
    if measure_offline:
        offline_ttft = _measure_offline_ttft(model, n_frames, (H, W), q, device)

    return LatencyReport(
        label=label,
        device=str(device),
        frames=n_frames,
        visual_tokens=s["visual_tokens"],
        frame_ingest_ms_p50=per_frame[n // 2],
        frame_ingest_ms_p95=per_frame[min(n - 1, int(n * 0.95))],
        ttft_ms=r["ttft_ms"],
        tpot_ms=r["tpot_ms"],
        drift=s["drift"],
        peak_mem_mb=_peak_mem_mb(device),
        geometry_state_mb=s["geometry_state_bytes"] / 1e6,
        offline_ttft_ms=offline_ttft,
    )


@torch.no_grad()
def _measure_offline_ttft(model, n_frames, frame_hw, question, device) -> float:
    """오프라인 방식: 질문이 올 때마다 전 프레임을 처음부터 다시 인코딩한다.

    SpatialStack(VGGT) 같은 구성이 실제로 하는 일이고, 우리가 이기려는 기준선이다.
    """
    H, W = frame_hw
    sess = LiveSession(model, device=device)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        sess.ingest(torch.randint(0, 255, (H, W, 3), dtype=torch.uint8))
    sess.ask(question, max_new_tokens=1)
    return (time.perf_counter() - t0) * 1e3


def save_report(report: LatencyReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(report) | {"meets_live_budget": report.meets_live_budget}, indent=2))


# --------------------------------------------------------------- 지연 모드 (실제 설계)
def _sync(device: torch.device) -> None:
    """GPU 는 비동기라 동기화 없이 재면 시간이 거짓말을 한다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class DeferredReport:
    label: str
    device: str
    frames: int
    budget: int
    geom_stride: int
    geom_frame_ms_p50: float     # 기하 인코더가 도는 프레임의 비용
    other_frame_ms_p50: float    # 선택기만 도는 프레임의 비용
    geom_fps_max: float          # 기하 인코더가 낼 수 있는 최대 fps (= 1000 / geom_frame_ms)
    live_fps_max: float          # 스트림을 실시간으로 따라갈 수 있는 입력 fps (geom_stride 반영)
    prefill_ms: float            # 질문 도착 → 비전 인코딩 + 기하 임베딩 + LLM 프리필
    ttft_ms: float               # 질문 도착 → 첫 토큰 (지연 모드: 시각 프리필 전부 포함)
    tpot_ms: float
    visual_tokens: int
    select_embed_ms: float = 0.0   # 키프레임 확정 + 픽셀 준비 + 기하 임베딩 (prefill() 호출)
    vision_llm_prefill_ms: float = 0.0  # 비전 타워 + LLM 이 시각 프리픽스를 먹는 시간
    question_only_ttft_ms: float = 0.0  # 시각 프리픽스가 미리 캐시돼 있을 때 질문 → 첫 토큰

    def pretty(self) -> str:
        ok_ttft = "OK " if self.ttft_ms < 1000 else "MISS"
        return "\n".join([
            f"[{self.label}]  {self.device} · {self.frames}프레임 · 키프레임 {self.budget} · 기하 stride {self.geom_stride}",
            f"  기하 프레임     {self.geom_frame_ms_p50:8.1f} ms  → 기하 최대 {self.geom_fps_max:5.1f} fps",
            f"  나머지 프레임   {self.other_frame_ms_p50:8.2f} ms",
            f"  실시간 추종     입력 {self.live_fps_max:5.1f} fps 까지 — 프레임당 실측 평균 기준 (30fps 웹캠이면 "
            f"{'가능' if self.live_fps_max >= 30 else '불가 — 기하 stride 를 늘려야 한다'})",
            f"  [{ok_ttft}] TTFT {self.ttft_ms:8.0f} ms  (질문 도착 후 시각 프리필 전부 포함, 목표 < 1000)",
            f"      = 키프레임 확정·기하 임베딩 {self.select_embed_ms:.0f} ms"
            f" + 비전 타워·LLM 시각 프리필 {self.vision_llm_prefill_ms:.0f} ms + 질문·첫 토큰",
            f"  [{'OK ' if self.question_only_ttft_ms < 1000 else 'MISS'}] 질문만 TTFT "
            f"{self.question_only_ttft_ms:6.0f} ms  (시각 프리픽스를 스트림 중에 미리 만들어 뒀을 때)",
            f"  TPOT            {self.tpot_ms:8.1f} ms · 비전 토큰 {self.visual_tokens}",
        ])


@torch.no_grad()
def benchmark_deferred(
    model,
    prompt,
    n_frames: int = 300,
    frame_hw: tuple[int, int] = (480, 640),
    budget: int = 32,
    geom_stride: int = 3,
    device: str | torch.device = "cpu",
    visual_mode: str = "image",
    question: str = "How many chairs are in this room?",
    max_new_tokens: int = 16,
    label: str = "live3r-deferred",
) -> DeferredReport:
    """우리 실제 설계(halving 지연 모드)의 지연을 잰다.

    프레임 단계: 기하 인코더는 geom_stride 프레임마다만 돈다 → 그 프레임과 나머지 프레임의 비용을 따로 본다.
    질문 단계: 모은 키프레임을 비전 인코딩 + 기하 임베딩 + LLM 프리필 → 첫 토큰.
    """
    from .streaming import HalvingSelector, StreamingSession

    device = torch.device(device)
    model = model.to(device).eval()
    sess = StreamingSession(model, HalvingSelector(budget), geom_stride=geom_stride,
                            visual_mode=visual_mode, device=device)
    sess.begin(fps=30.0)
    h, w = frame_hw
    g_ms, o_ms, all_ms = [], [], []
    for i in range(n_frames):
        raw = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8)
        calls = sess.audit.geometry_calls
        _sync(device)
        t0 = time.perf_counter()
        sess.push(i, raw)
        _sync(device)
        dt = (time.perf_counter() - t0) * 1e3
        # 프레임 번호가 아니라 **실제로 기하가 돌았는지**로 나눈다 — halving 은 초반에 거의 모든
        # 프레임을 키프레임으로 받고, 키프레임에서는 stride 와 무관하게 기하가 돈다.
        (g_ms if sess.audit.geometry_calls > calls else o_ms).append(dt)
        if i >= 2:
            all_ms.append(dt)
    g_ms, o_ms = sorted(g_ms[2:] or g_ms), sorted(o_ms or [0.0])  # 앞 2개는 워밍업

    from .prefix_cache import PrefixCache

    # (1) 질문 도착 → 키프레임 확정 + 기하 임베딩
    _sync(device)
    t0 = time.perf_counter()
    pre = sess.prefill()
    _sync(device)
    t_sel = (time.perf_counter() - t0) * 1e3
    segs = StreamingSession.segments(prompt, pre)
    # (2) 비전 타워 + LLM 이 시각 프리픽스를 먹는다
    t1 = time.perf_counter()
    pc = PrefixCache(model, prompt, segs, pre["pixel_kwargs"], pre["geometry"], device)
    _sync(device)
    t_vis = (time.perf_counter() - t1) * 1e3
    # (3) 질문 토큰 → 첫 토큰 (시각 프리픽스는 캐시)
    t2 = time.perf_counter()
    pc.answer(question, max_new_tokens=1, do_sample=False)
    _sync(device)
    t_q = (time.perf_counter() - t2) * 1e3
    prefill, ttft = t_sel + t_vis, t_sel + t_vis + t_q
    # (4) TPOT — 캐시에서 이어서 토큰을 하나씩 (프리필을 다시 하지 않는다)
    t3 = time.perf_counter()
    _, _ = pc.answer(question, max_new_tokens=max_new_tokens, do_sample=False,
                     min_new_tokens=max_new_tokens)
    _sync(device)
    tpot = ((time.perf_counter() - t3) * 1e3 - t_q) / max(1, max_new_tokens - 1)

    gms = g_ms[len(g_ms) // 2]
    # 실시간 추종: 가정 공식이 아니라 실측 평균 (키프레임 때문에 도는 기하까지 포함)
    per_frame_avg = sum(all_ms) / max(1, len(all_ms))
    return DeferredReport(
        label=label, device=str(device), frames=n_frames, budget=budget, geom_stride=geom_stride,
        geom_frame_ms_p50=gms, other_frame_ms_p50=o_ms[len(o_ms) // 2],
        geom_fps_max=1000.0 / max(gms, 1e-6), live_fps_max=1000.0 / max(per_frame_avg, 1e-6),
        prefill_ms=prefill, ttft_ms=ttft, tpot_ms=max(0.0, tpot), visual_tokens=pre["n_visual_tokens"],
        select_embed_ms=t_sel, vision_llm_prefill_ms=t_vis, question_only_ttft_ms=t_q,
    )
