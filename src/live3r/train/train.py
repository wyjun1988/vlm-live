"""Live3R 학습 — 2단계, torchrun DDP.

    S1 정렬 (align) : 프로젝터만 학습 (LLM 동결, LoRA 없음)
    S2 SFT   (sft)  : 프로젝터 + LoRA

S1 을 따로 두는 이유: 프로젝터가 zero-init 이라 시작 시 기하 기여가 정확히 0이다.
LoRA 를 동시에 풀면 LLM 이 **기하 없이 푸는 지름길**을 먼저 배워버린다. 정렬을 먼저 한다.

실행:
    # 1 GPU
    PYTHONPATH=src python -m live3r.train.train --config configs/live3r_4b.yaml ...
    # 8 GPU
    PYTHONPATH=src torchrun --nproc_per_node 8 -m live3r.train.train --config configs/live3r_4b.yaml ...

첫 실행은 반드시 `--dump-samples 3 --max-steps 20` 으로 — 프롬프트·라벨이 눈으로 맞는지,
손실이 내려가는지, 주입 비율(inj)이 0 에서 움직이는지 본다.

학습 루프의 순서가 중요하다:
    forward(keep_primed=True) → backward → injector.clear()
gradient checkpointing 은 역전파 때 레이어 forward 를 재실행하고, 그때 주입 훅도 다시 불린다.
주입 상태가 이미 비워져 있으면 재실행 그래프가 원래와 달라진다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from ..config import Live3RConfig
from ..data.collate import Live3RCollator
from ..data.datasets import SpatialVQADataset
from ..data.prompt import BuiltPrompt, PromptBuilder
from ..model.live3r import Live3RModel
from ..model.lora import apply_lora

logger = logging.getLogger("live3r.train")


# ------------------------------------------------------------------------ 분산
def setup_distributed(device_arg: str | None):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        use_cuda = torch.cuda.is_available() and device_arg != "cpu"
        dist.init_process_group("nccl" if use_cuda else "gloo")
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return dist.get_rank(), dist.get_world_size(), local_rank, device
    dev = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, torch.device(dev)


def all_mean(x: float, world: int, device) -> float:
    if world == 1:
        return x
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return float(t.item() / world)


class GeomCache:
    """(영상/이미지 세트 + 프레임 인덱스) → GeomOutput 리스트.

    ⚠️ 키에 프레임 인덱스가 반드시 들어가야 한다 — 같은 영상이라도 에폭마다 다른 프레임을 본다.
    예전엔 키가 data_source + input_ids.shape 라서 **다른 샘플의 기하 토큰을 먹일 수 있었다.**
    """

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key and key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        if not key:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)


@torch.no_grad()
def check_rank_sync(params, world: int, device) -> float:
    """모든 랭크의 학습 파라미터가 같은지 — DDP 가 기울기를 제대로 동기화하고 있다는 직접 증거.

    파라미터 체크섬을 all_gather 해서 랭크 간 최대 차이를 돌려준다. 0 이 아니면 랭크마다
    다른 모델을 학습하고 있는 것이다 (no_sync 누적이 틀렸거나, 미사용 파라미터 처리 문제).
    """
    if world == 1:
        return 0.0
    local = torch.stack([p.detach().double().sum() + p.detach().double().abs().sum() for p in params]).to(device)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return float(torch.stack(gathered).std(0).max())


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# ---------------------------------------------------------------------- 모델
def build_model(args, cfg: Live3RConfig, is_main: bool) -> Live3RModel:
    model = Live3RModel.from_pretrained(cfg, tokenizer_path=args.tokenizer)
    model.freeze_base()  # 항상 먼저 전부 동결하고 필요한 것만 연다
    for p in model.projectors.parameters():
        p.requires_grad_(True)
    if model.pose_projector is not None:
        for p in model.pose_projector.parameters():
            p.requires_grad_(True)

    if args.stage == "sft":
        if cfg.lora.enabled:
            model = apply_lora(model, cfg.lora)
        if args.init_from:
            state = torch.load(args.init_from, map_location="cpu")
            res = model.load_state_dict(state, strict=False)
            loaded = len(state) - len(res.unexpected_keys)
            if is_main:
                logger.info("S1 결과 로드: %d/%d 텐서 (%s)", loaded, len(state), args.init_from)
            if loaded == 0:
                raise RuntimeError(f"{args.init_from} 에서 로드된 텐서가 0개다 — 키가 안 맞는다")
        elif is_main:
            logger.warning(
                "S2(sft) 를 S1 정렬 없이 시작한다 — LLM 이 기하를 무시하는 지름길을 배울 수 있다. "
                "--init-from 으로 S1 결과를 넣는 걸 권장."
            )
    return model


def dump_samples(ds: SpatialVQADataset, prompt: PromptBuilder, n: int) -> None:
    """학습 전에 눈으로 확인 — 프롬프트 형식·라벨 구간·토큰 수."""
    print("\n" + "=" * 78)
    print(f"샘플 {n}개 (전체 {len(ds):,})")
    print("=" * 78)
    for i in range(min(n, len(ds))):
        item = ds[i]
        built = BuiltPrompt(item["input_ids"], item["labels"], item["mm_token_type_ids"],
                            prompt.tok.decode(item["input_ids"][0]), item["turns_used"])
        n_img = int((item["mm_token_type_ids"] == 1).sum())
        n_vid = int((item["mm_token_type_ids"] == 2).sum())
        grids = item.get("image_grid_thw", item.get("video_grid_thw"))
        print(f"\n[{i}] id={item['id']} media={item['media_type']} "
              f"토큰={item['input_ids'].shape[1]} (이미지 {n_img} / 비디오 {n_vid}) "
              f"턴={item['turns_used']} 기하프레임={len(item['geom_frames'])}")
        if grids is not None:
            print(f"    grid_thw={grids.tolist()}")
        if item["geom_frames"]:
            print(f"    기하 입력 크기={[tuple(f.shape[-2:]) for f in item['geom_frames'][:4]]}")
        print("    " + prompt.summarize(built, 500).replace("\n", "\n    "))
    print("=" * 78 + "\n")


# ------------------------------------------------------------------------ 학습
def train(args) -> int:
    rank, world, local_rank, device = setup_distributed(args.device)
    is_main = rank == 0
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(message)s",
    )
    torch.manual_seed(args.seed + rank)

    cfg = Live3RConfig.from_yaml(args.config)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.geometry:
        cfg.geometry.name = args.geometry
    if args.geometry_checkpoint:
        cfg.geometry.checkpoint = args.geometry_checkpoint
    if args.cut3r_repo:
        cfg.geometry.options = dict(cfg.geometry.options, repo_path=args.cut3r_repo)
    if args.max_pixels:
        cfg.data.max_pixels = args.max_pixels

    model = build_model(args, cfg, is_main)
    model.to(device)
    if args.grad_checkpointing:
        base = model.base
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(base, "config"):
            base.config.use_cache = False
    model.train()  # 동결된 기하 인코더는 eval 로 남는다 (Live3RModel.train 참고)
    live = model

    if is_main:
        s = live.trainable_parameter_summary()
        logger.info("파라미터: 전체 %.2fB / 학습 %.1fM (프로젝터 %.1fM, LoRA %.1fM)",
                    s["total"] / 1e9, s["trainable"] / 1e6, s["projector"] / 1e6, s["lora"] / 1e6)

    prompt = PromptBuilder.from_model(live)
    ds = SpatialVQADataset(
        ann_path=args.ann,
        media_root=args.media_root,
        prompt=prompt,
        spec=live.spec,
        geom_long_side=cfg.geometry.image_size,
        geom_unit=getattr(live.geometry, "patch_size", 16),
        data_cfg=cfg.data,
        max_samples=args.max_samples,
        seed=args.seed + rank,
    )
    if is_main and args.dump_samples:
        dump_samples(ds, prompt, args.dump_samples)

    sampler = (
        DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        if world > 1
        else RandomSampler(ds)
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=Live3RCollator(),
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    ddp = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            # 텍스트 전용 샘플이 섞이면 그 스텝엔 프로젝터가 안 쓰인다 → True 가 안전하다
            find_unused_parameters=args.find_unused,
            # 버퍼는 전부 상수(rotary 등)거나 랭크별 상태(기하 인코더)다 — 동기화하면 안 된다
            broadcast_buffers=False,
        )

    params = [p for p in live.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = args.max_steps or max(1, math.ceil(len(dl) * args.epochs / args.grad_accum))
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, total_steps, warmup)
    )
    cache = GeomCache(args.geom_cache) if args.geom_cache else None

    out_dir = Path(args.output)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        logger.info("스텝 %d (워밍업 %d) · GPU %d × 누적 %d = 유효 배치 %d · 샘플 %s",
                    total_steps, warmup, world, args.grad_accum, world * args.grad_accum, f"{len(ds):,}")

    step, micro = 0, 0
    skipped = 0
    run_loss, run_n = 0.0, 0
    t_data = t_geom = t_fb = 0.0
    t_last = time.perf_counter()
    samples_since = 0

    for epoch in range(max(1, args.epochs)):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        for batch in dl:
            t1 = time.perf_counter()
            t_data += t1 - t0

            # ---- 기하 (동결, no_grad) ----
            geom_outs = None
            if batch["geom_frames"]:
                geom_outs = cache.get(batch["cache_key"]) if cache else None
                if geom_outs is None:
                    geom_outs = live.run_geometry(batch["geom_frames"])
                    if cache:
                        cache.put(batch["cache_key"], geom_outs)
            t2 = time.perf_counter()
            t_geom += t2 - t1

            pix = {k: batch[k].to(device, non_blocking=True)
                   for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
                   if k in batch}
            measure = is_main and ((step + 1) % args.log_every == 0) and micro == args.grad_accum - 1
            live.injector.measure = measure

            is_sync = (micro == args.grad_accum - 1)
            ctx = ddp.no_sync() if (world > 1 and not is_sync) else contextlib.nullcontext()
            with ctx:
                out = ddp(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    mm_token_type_ids=batch["mm_token_type_ids"].to(device),
                    geom_outs=geom_outs,
                    llm_grids=batch.get("llm_grids"),
                    pool_temporal=batch.get("pool_temporal", False),
                    keep_primed=True,
                    **pix,
                )
                loss = out.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"손실이 {loss.item()} — 샘플 id={batch['id']}")
                (loss / args.grad_accum).backward()
            live.injector.clear()  # 역전파 재계산까지 끝난 뒤에만 비운다
            t0 = time.perf_counter()
            t_fb += t0 - t2
            run_loss += float(loss.detach())
            run_n += 1
            skipped += int(batch.get("retries", 0))
            samples_since += 1
            micro += 1

            if micro < args.grad_accum:
                continue
            micro = 0
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                drift = check_rank_sync(params, world, device)
                if drift > 1e-6 and is_main:
                    logger.error("랭크 간 파라미터가 다르다 (체크섬 std %.3e) — DDP 동기화 문제", drift)
                loss_avg = all_mean(run_loss / max(1, run_n), world, device)
                dt = time.perf_counter() - t_last
                sps = all_mean(samples_since / max(dt, 1e-9), world, device) * world
                if is_main:
                    ratios = " ".join(f"L{k}:{v:.3f}" for k, v in sorted(live.injector.last_ratio.items()))
                    gates = " ".join(f"{p.gate.item():+.2f}" for p in live.projectors)
                    mem = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0
                    n = max(1, run_n)
                    logger.info(
                        "ep %d step %d/%d loss %.4f lr %.2e | inj %s | gate %s | %.1f samp/s | "
                        "data %.0f geom %.0f fb %.0f ms/샘플 | mem %.1fGB | skip %d | sync %s",
                        epoch, step, total_steps, loss_avg, sched.get_last_lr()[0], ratios or "-",
                        gates, sps, 1e3 * t_data / n, 1e3 * t_geom / n, 1e3 * t_fb / n, mem, skipped,
                        "-" if world == 1 else ("OK" if drift <= 1e-6 else f"DIFF {drift:.1e}"),
                    )
                run_loss, run_n = 0.0, 0
                t_data = t_geom = t_fb = 0.0
                t_last = time.perf_counter()
                samples_since = 0

            if is_main and args.save_every and step % args.save_every == 0:
                _save(live, out_dir / f"step{step}.pt")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if is_main:
        _save(live, out_dir / "final.pt")
        logger.info("완료. 건너뛴 샘플 %d (랭크0 기준, 사유는 워커 로그의 '건너뜀' 경고 참고)", skipped)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _save(live: Live3RModel, path: Path) -> None:
    """학습된 것만 저장한다 — 베이스 가중치를 복사하지 않는다."""
    state = live.trainable_state_dict()
    torch.save(state, path)
    logger.info("저장 %s (%d 텐서, %.1f MB)", path, len(state),
                sum(v.numel() * v.element_size() for v in state.values()) / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live3R 학습 (torchrun 지원)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["align", "sft"], default="align")
    ap.add_argument("--ann", required=True, help="어노테이션 (json/jsonl, 인덱스 있으면 지연 로딩)")
    ap.add_argument("--media-root", "--video-root", dest="media_root", required=True,
                    help="이미지/영상 경로의 기준 디렉터리")
    ap.add_argument("--output", required=True)
    ap.add_argument("--init-from", default=None, help="S2 에서 S1 결과(프로젝터)를 이어받는다")
    # 설정 덮어쓰기 (스모크용)
    ap.add_argument("--base-model", default=None, help="로컬 경로 등으로 덮어쓰기 ('tiny' = 랜덤 초소형)")
    ap.add_argument("--tokenizer", default=None, help="base-model=tiny 일 때 쓸 실제 토크나이저 경로")
    ap.add_argument("--geometry", default=None, help="dummy 로 두면 CUT3R 없이 데이터·LLM 경로만 본다")
    ap.add_argument("--geometry-checkpoint", default=None)
    ap.add_argument("--cut3r-repo", default=None)
    ap.add_argument("--max-pixels", type=int, default=None, help="이미지당 픽셀 상한 (토큰 수 조절)")
    # 최적화
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--find-unused", action=argparse.BooleanOptionalAction, default=True)
    # 입출력
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--geom-cache", type=int, default=0, help="LRU 용량 (0=끔)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--dump-samples", type=int, default=0, help="학습 전 샘플 N개를 사람이 읽게 출력")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
