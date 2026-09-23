"""공간 VQA 데이터셋 — 이미지 시퀀스(SenseNova-SI)와 영상(VSI-590K 등) 공용.

레코드 형식 (둘 다 LLaVA/InternVL 계열 `conversations`):

    SenseNova-SI (이미지 1~28장):
      {"id": 123, "image": ["a.jpg", "b.jpg", ...],
       "conversations": [{"from": "human", "value": "<image>\\n<image>\\n질문"},
                         {"from": "gpt", "value": "1.43"}, ...]}
      * `conversations` 가 JSON **문자열**로 들어 있는 경우도 있다 (HF 뷰어 기준) → 파싱한다
      * 멀티턴 가능. `<image>` 는 첫 human 턴에 인라인으로 들어간다
    영상:
      {"id": ..., "video": "scannet/videos/scene0191_00.mp4", "conversations": [...]}

이미지 시퀀스는 **이미지 모드**로 넣는다 (이미지마다 별도 비전 블록).
비디오 모드로 넣으면 Qwen 의 temporal_patch=2 때문에 서로 다른 시점 두 장이 한 블록으로
섞이고, "3번째 이미지에서 보면…" 같은 질문이 가리킬 대상이 사라진다.

로딩 비용: 83만 레코드를 JSON 배열로 두면 DDP 8프로세스가 각자 통째로 파싱해 메모리에 올린다.
`scripts/prepare_annotations.py` 로 JSONL + 오프셋 인덱스로 한 번 바꿔두면 레코드를
필요할 때 한 줄씩 읽는다 (프로세스당 메모리 = 인덱스 6.6MB).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .prompt import PromptBuilder, PromptError
from .vision import (
    VisionSpec,
    frame_timestamps,
    geometry_frame,
    llm_grid,
    prepare_image,
    prepare_video,
    tokens_per_step,
)

logger = logging.getLogger(__name__)


class RecordError(ValueError):
    """레코드 자체가 망가졌다 (필드 누락·형식 오류). 해당 샘플은 건너뛴다."""


# ------------------------------------------------------------------ 레코드 로딩
class LazyJsonl:
    """JSONL + 오프셋 인덱스(`<path>.idx.npy`) — 레코드를 필요할 때 한 줄씩 읽는다.

    DataLoader 워커마다 파일 핸들을 따로 연다 (fork 뒤 핸들 공유는 seek 가 꼬인다).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets = np.load(str(self.path) + ".idx.npy", mmap_mode="r")
        self._fh = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        # 파일 핸들은 피클링이 안 된다 — macOS 등 spawn 방식 DataLoader 워커가 데이터셋을
        # 피클링할 때 터진다 (리눅스 fork 에서는 안 드러난다). 핸들은 빼고, 워커에서 다시 연다.
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_pid"] = None
        state["offsets"] = np.asarray(self.offsets)  # memmap 도 평범한 배열로 (수 MB)
        return state

    def __getitem__(self, i: int) -> dict:
        import os

        if self._fh is None or self._pid != os.getpid():
            self._fh = open(self.path, "rb")
            self._pid = os.getpid()
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())


def load_records(path: str | Path, max_samples: int | None = None):
    """JSON 배열 / JSONL / 인덱스 달린 JSONL / 디렉터리를 모두 받는다."""
    path = Path(path)
    if path.is_dir():
        files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
        if not files:
            raise FileNotFoundError(f"{path} 안에 json/jsonl 이 없다")
        out: list = []
        for f in files:
            out.extend(list(load_records(f)))
        return out[:max_samples] if max_samples else out
    if Path(str(path) + ".idx.npy").exists():
        recs = LazyJsonl(path)
        if max_samples:
            return [recs[i] for i in range(min(max_samples, len(recs)))]
        return recs

    with open(path, "rb") as fh:
        head = fh.read(4096).lstrip()
    if head.startswith(b"["):
        data = json.loads(path.read_text())
    else:
        # JSONL (확장자가 .json 이어도 줄단위일 수 있다 — 첫 문자로 판별)
        data = []
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    data.append(json.loads(line))
                    if max_samples and len(data) >= max_samples:
                        break
    if isinstance(data, dict):
        data = data.get("data") or data.get("annotations") or list(data.values())
    return data[:max_samples] if max_samples else data


# ------------------------------------------------------------------ 레코드 정규화
@dataclass
class Record:
    id: str
    turns: list[tuple[str, str]]
    images: list[str] | None = None
    video: str | None = None
    frames: list[int] | None = None
    data_source: str = "unknown"

    @property
    def media_type(self) -> str:
        if self.images:
            return "images"
        if self.video:
            return "video"
        return "text"


def normalize_record(raw: dict) -> Record:
    conv = raw.get("conversations")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError as exc:
            raise RecordError(f"conversations 문자열 파싱 실패: {exc}") from exc
    if not isinstance(conv, list) or not conv:
        raise RecordError("conversations 가 비었다")

    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in conv:
        role = msg.get("from") or msg.get("role")
        val = msg.get("value") if "value" in msg else msg.get("content", "")
        if role in ("system",):
            continue
        if role in ("human", "user"):
            # 연속 user 메시지는 이어붙인다
            pending_user = val if pending_user is None else pending_user + "\n" + val
        elif role in ("gpt", "assistant"):
            if pending_user is None:
                raise RecordError("user 없이 assistant 가 먼저 나왔다")
            turns.append((pending_user, val))
            pending_user = None
        else:
            raise RecordError(f"모르는 role: {role}")
    if not turns:
        raise RecordError("user→assistant 쌍이 하나도 없다")

    images = raw.get("image", raw.get("images"))
    if isinstance(images, str):
        images = [images]
    return Record(
        id=str(raw.get("id", "")),
        turns=turns,
        images=list(images) if images else None,
        video=raw.get("video"),
        frames=raw.get("frames"),
        data_source=str(raw.get("data_source", raw.get("source", "unknown"))),
    )


# ------------------------------------------------------------------- 영상 유틸
@dataclass
class FrameSpec:
    num_frames: int = 16
    #: uniform: 균등 / prefix: 앞쪽만 (스트리밍 학습용)
    mode: str = "uniform"
    jitter: bool = True


def sample_indices(n_total: int, spec: FrameSpec, rng: np.random.Generator | None = None) -> list[int]:
    """학습용 프레임 샘플링. **학습**에서는 총 길이를 알아도 된다 — 라이브 제약은 평가 경로의 것이다."""
    if n_total <= 0:
        return []
    k = min(spec.num_frames, n_total)
    if spec.mode == "prefix":
        return list(range(k))
    edges = np.linspace(0, n_total, k + 1)
    if spec.jitter and rng is not None:
        pos = [int(rng.integers(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1])))) for i in range(k)]
    else:
        pos = [int((edges[i] + edges[i + 1]) / 2) for i in range(k)]
    return [min(n_total - 1, p) for p in pos]


def _read_video(path: Path, indices: list[int]) -> tuple[np.ndarray, float | None]:
    """[T,H,W,3] uint8 와 fps. decord 우선, 없으면 PyAV."""
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(str(path))
        return vr.get_batch(indices).asnumpy(), float(vr.get_avg_fps())
    except ImportError:
        pass
    import av  # type: ignore

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack([frames[min(i, len(frames) - 1)] for i in indices]), fps


def _probe_len(path: Path) -> int:
    try:
        import decord  # type: ignore

        return len(decord.VideoReader(str(path)))
    except ImportError:
        import av  # type: ignore

        with av.open(str(path)) as c:
            return sum(1 for _ in c.decode(video=0))


# -------------------------------------------------------------------- 데이터셋
class SpatialVQADataset(Dataset):
    """정규화된 공간 VQA 데이터셋. 샘플 하나 = 모델 입력 완성본 (배치 1).

    망가진 샘플(이미지 누락·placeholder 불일치·과도한 길이)은 **건너뛰고 다른 샘플을 준다.**
    단, 실패율이 `max_fail_rate` 를 넘으면 멈춘다 — 조용히 절반을 버리며 학습하는 사고를 막는다.
    """

    def __init__(
        self,
        ann_path: str | Path,
        media_root: str | Path,
        prompt: PromptBuilder,
        spec: VisionSpec,
        geom_long_side: int,
        geom_unit: int,
        data_cfg: DataConfig | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        max_retries: int = 20,
        max_fail_rate: float = 0.05,
    ) -> None:
        self.records = load_records(ann_path, max_samples)
        self.media_root = Path(media_root)
        self.prompt = prompt
        self.spec = spec
        self.geom_long_side = geom_long_side
        self.geom_unit = geom_unit
        self.cfg = data_cfg or DataConfig()
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self.max_retries = max_retries
        self.max_fail_rate = max_fail_rate
        self.n_ok = 0
        self.n_fail = 0
        self.fail_reasons: dict[str, int] = {}
        logger.info("%s: %d 레코드 (%s)", ann_path, len(self.records), type(self.records).__name__)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        idx = i
        for attempt in range(self.max_retries):
            try:
                item = self.build(idx)
                self.n_ok += 1
                # 워커 프로세스의 카운터는 메인 프로세스에서 안 보인다 → 샘플에 실어 보낸다
                item["retries"] = attempt
                return item
            except (RecordError, PromptError, FileNotFoundError, OSError, ValueError) as exc:
                self._record_fail(idx, exc)
                idx = self._py_rng.randrange(len(self.records))
        raise RuntimeError(
            f"{self.max_retries}번 연속 샘플 구성 실패. 사유 분포: {self.fail_reasons}"
        )

    def _record_fail(self, idx: int, exc: Exception) -> None:
        self.n_fail += 1
        key = type(exc).__name__ + ":" + str(exc).split("\n")[0][:60]
        self.fail_reasons[key] = self.fail_reasons.get(key, 0) + 1
        if self.fail_reasons[key] <= 3:
            logger.warning("샘플 %d 건너뜀 — %s", idx, str(exc).split("\n")[0])
        total = self.n_ok + self.n_fail
        if total >= 200 and self.n_fail / total > self.max_fail_rate:
            raise RuntimeError(
                f"샘플 실패율 {self.n_fail / total:.1%} > {self.max_fail_rate:.0%}. "
                f"데이터 경로나 형식이 잘못됐다. 사유 분포: {self.fail_reasons}"
            )

    def _resolve(self, p: str) -> Path:
        cand = self.media_root / p
        if cand.exists():
            return cand
        if Path(p).is_absolute() and Path(p).exists():
            return Path(p)
        raise FileNotFoundError(f"미디어 파일이 없다: {cand}")

    # ------------------------------------------------------------------ 구성
    def build(self, i: int) -> dict:
        rec = normalize_record(self.records[i])
        mt = rec.media_type
        if mt == "images":
            return self._build_images(rec)
        if mt == "video":
            return self._build_video(rec)
        return self._build_text(rec)

    def _pack(self, rec: Record, built, **extra) -> dict:
        return {
            "id": rec.id,
            "data_source": rec.data_source,
            "input_ids": built.input_ids,
            "labels": built.labels,
            "attention_mask": torch.ones_like(built.input_ids),
            "mm_token_type_ids": built.mm_token_type_ids,
            "turns_used": built.turns_used,
            **extra,
        }

    def _build_images(self, rec: Record) -> dict:
        from PIL import Image

        if len(rec.images) > self.cfg.max_images:
            raise RecordError(f"이미지 {len(rec.images)}장 > max_images {self.cfg.max_images}")
        pil = []
        for p in rec.images:
            with Image.open(self._resolve(p)) as im:
                pil.append(im.convert("RGB"))
        vlm = [prepare_image(im, self.spec) for im in pil]
        segments = [self.prompt.image_segment(tokens_per_step(g[0], self.spec)) for _, g in vlm]
        built = self.prompt.build(rec.turns, segments, "images", self.cfg.max_length)
        return self._pack(
            rec,
            built,
            media_type="images",
            pixel_values=torch.cat([pv for pv, _ in vlm], 0),
            image_grid_thw=torch.cat([g for _, g in vlm], 0),
            geom_frames=[geometry_frame(im, self.geom_long_side, self.geom_unit) for im in pil],
            llm_grids=[llm_grid(g[0], self.spec) for _, g in vlm],
            pool_temporal=False,
            cache_key="img|" + "|".join(rec.images),
        )

    def _build_video(self, rec: Record) -> dict:
        path = self._resolve(rec.video)
        idx = rec.frames or sample_indices(
            _probe_len(path), FrameSpec(num_frames=self.cfg.num_frames), self.rng
        )
        raw, fps = _read_video(path, idx)
        pv, grid = prepare_video(raw, self.spec)
        ts = frame_timestamps(idx, fps or self.cfg.default_fps, self.spec.temporal_patch)
        segment = self.prompt.video_segment(tokens_per_step(grid[0], self.spec), ts)
        built = self.prompt.build(rec.turns, [segment], "video", self.cfg.max_length)
        return self._pack(
            rec,
            built,
            media_type="video",
            pixel_values_videos=pv,
            video_grid_thw=grid,
            geom_frames=[geometry_frame(f, self.geom_long_side, self.geom_unit) for f in raw],
            llm_grids=llm_grid(grid[0], self.spec),
            pool_temporal=True,
            cache_key=f"vid|{rec.video}|{','.join(map(str, idx))}",
        )

    def _build_text(self, rec: Record) -> dict:
        built = self.prompt.build(rec.turns, [], "text", self.cfg.max_length)
        return self._pack(rec, built, media_type="text", geom_frames=[], cache_key="")
