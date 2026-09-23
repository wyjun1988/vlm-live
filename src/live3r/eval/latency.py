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
