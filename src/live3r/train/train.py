"""Live3R 학습 — 2단계.

    S1 정렬 : 프로젝터만 학습 (LLM 동결, LoRA 없음)
    S2 SFT  : 프로젝터 + LoRA

S1 을 따로 두는 이유: 프로젝터가 zero-init 이라 시작 시 기하 기여가 정확히 0이다.
LoRA 를 동시에 풀면 LLM 이 **기하 없이 푸는 지름길**을 먼저 배워버리고, 그 뒤에는
기하 토큰이 들어와도 무시한다. 정렬을 먼저 끝내고 LoRA 를 연다.

기하 인코더는 항상 동결이고 no_grad 로 돈다 — 그래서 옵티마이저 메모리가 안 든다.
대신 매 스텝 인코더 forward 비용이 든다. 같은 비디오가 여러 QA 에 재사용되면
`--geom-cache` 로 프로세스 내 LRU 캐시를 켜라 (디스크 캐시는 토큰이 커서 권장 안 함).

실행 예:
    PYTHONPATH=src python -m live3r.train.train \
        --config configs/live3r_4b.yaml --stage align \
        --ann data/raw/vsi590k --video-root data/raw \
        --output outputs/4b_s1 --epochs 1 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Live3RConfig
from ..data.collate import Live3RCollator
from ..data.datasets import FrameSpec, SpatialVQADataset
from ..model.live3r import Live3RModel
from ..model.lora import apply_lora

logger = logging.getLogger(__name__)


class GeomCache:
    """(영상, 프레임인덱스) → GeomOutput 리스트.

    ⚠️ 키를 영상만으로 잡으면 안 된다 — 프레임 샘플링에 jitter 가 있어 같은 영상이라도
    에폭마다 다른 프레임을 본다. 잘못된 키는 **다른 샘플의 기하 토큰을 조용히 먹인다**.
    """

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)


def run_geometry(model: Live3RModel, geom_frames: torch.Tensor, device) -> list:
    """동결 인코더를 프레임 순서대로 돌린다 (no_grad)."""
    model.geometry.reset()
    outs = []
    with torch.no_grad():
        for t in range(geom_frames.shape[0]):
            outs.append(model.geometry.ingest(geom_frames[t : t + 1].to(device)))
    return outs


def build_optimizer(model: Live3RModel, lr: float, weight_decay: float = 0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    logger.info("학습 파라미터 %.2fM", n / 1e6)
    if n == 0:
        raise RuntimeError("학습할 파라미터가 하나도 없다 — 동결 설정을 확인해라")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))


def train(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Live3RConfig.from_yaml(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = Live3RModel.from_pretrained(cfg)
    model.freeze_base()  # 항상 먼저 전부 동결하고 필요한 것만 연다
    for p in model.projectors.parameters():
        p.requires_grad_(True)
    if model.pose_projector is not None:
        for p in model.pose_projector.parameters():
            p.requires_grad_(True)

    if args.stage == "sft":
        if args.init_from:
            state = torch.load(args.init_from, map_location="cpu")
            missing = model.load_state_dict(state, strict=False)
            logger.info("S1 프로젝터 로드: missing=%d", len(missing.missing_keys))
        else:
            logger.warning(
                "S2(sft) 를 S1 정렬 없이 시작한다 — LLM 이 기하를 무시하는 지름길을 배울 수 있다. "
                "--init-from 으로 S1 결과를 넣는 걸 권장."
            )
        model = apply_lora(model, cfg.lora)

    model = model.to(device)
    if cfg.geometry.freeze:
        model.geometry.eval()
    if args.grad_checkpointing:
        # 4B + 16프레임이면 활성화 메모리가 병목이다. 인코더는 no_grad 라 영향 없음.
        base = model.base
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.use_cache = False
            logger.info("gradient checkpointing 켬")
        else:
            logger.warning("gradient_checkpointing_enable 이 없다 — 건너뜀")

    ds = SpatialVQADataset(
        ann_path=args.ann,
        video_root=args.video_root,
        processor=model.processor,
        frame_spec=FrameSpec(num_frames=args.num_frames, mode=args.frame_mode),
        vlm_short_side=args.vlm_short_side,
        geom_short_side=cfg.geometry.image_size,
    )
    collator = Live3RCollator(
        model.processor, model.vision_patch, model.temporal_patch, model.spatial_merge
    )
    dl = DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=args.workers, collate_fn=collator
    )

    opt = build_optimizer(model, args.lr, args.weight_decay)
    steps = args.max_steps or (len(dl) * args.epochs // args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, steps), pct_start=0.03
    )
    cache = GeomCache(args.geom_cache) if args.geom_cache else None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    running = 0.0

    for epoch in range(args.epochs):
        for i, batch in enumerate(dl):
            key = batch["cache_key"]
            geom_outs = cache.get(key) if cache else None
            if geom_outs is None:
                geom_outs = run_geometry(model, batch["geom_frames"], device)
                if cache:
                    cache.put(key, geom_outs)

            bundle = model.build_geometry_embeds(geom_outs, batch["llm_grid_hw"])
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                mm_token_type_ids=batch["mm_token_type_ids"].to(device),
                pixel_values_videos=batch["pixel_values_videos"].to(device),
                video_grid_thw=batch["video_grid_thw"].to(device),
                geometry=bundle,
            )
            (out.loss / args.grad_accum).backward()
            running += out.loss.item()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    gates = [float(p.gate.detach()) for p in model.projectors]
                    logger.info(
                        "epoch %d step %d loss %.4f lr %.2e gate %s",
                        epoch, step, running / (args.log_every * args.grad_accum),
                        sched.get_last_lr()[0], [f"{g:+.3f}" for g in gates],
                    )
                    running = 0.0
                if args.save_every and step % args.save_every == 0:
                    _save(model, out_dir / f"step{step}.pt")
                if args.max_steps and step >= args.max_steps:
                    _save(model, out_dir / "final.pt")
                    return 0
    _save(model, out_dir / "final.pt")
    return 0


def _save(model: Live3RModel, path: Path) -> None:
    """학습된 것만 저장한다 — 베이스 가중치를 복사하지 않는다."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if ("projector" in k or "lora" in k.lower())}
    torch.save(state, path)
    logger.info("저장 %s (%d 텐서, %.1f MB)", path, len(state),
                sum(v.numel() * v.element_size() for v in state.values()) / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["align", "sft"], default="align")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--init-from", default=None, help="S2 에서 S1 프로젝터를 이어받는다")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--frame-mode", default="uniform", choices=["uniform", "fps", "prefix"])
    ap.add_argument("--vlm-short-side", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--geom-cache", type=int, default=0,
                    help="LRU 용량 (0=끔). 키는 영상+프레임인덱스라 같은 코호트 재사용에만 걸린다")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default=None)
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
