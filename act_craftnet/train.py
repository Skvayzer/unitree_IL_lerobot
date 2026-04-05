"""
Multi-GPU training for ACT-CraftNet v2 on block_stacking dataset.
Launch with torchrun (PyTorch DDP):
  torchrun --nproc_per_node=8 train.py [options]

Architecture changes vs v1:
  - FrozenDINOv2 (ViT-B/14) replaces ResNet-18 visual backbone
  - iDP3DepthEncoder for all 3 depth views (head + 2 wrists)
  - System0Policy MoE finger correction with bidirectional S1↔S0 connections
  - 8 encoder tokens: [z | state | tactile_fb | env | depth | cam0 | cam1 | cam2]
  - DINOv2 params fully frozen (requires_grad=False, excluded from optimizer)

Training fixes vs v1:
  - DDP forward: call model(batch) not model.module(batch)
  - Explicit epoch counter for DistributedSampler
  - kl_weight=1.0 (conservative for 152-episode dataset)
  - checkpoint_latest.pt only (overwrites, no accumulation)
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from dataset import BlockStackingDataset, make_splits
from model import ACTCraftNet, ACTConfig, compute_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/vast/users/chenyuan.chen/constantine/block_stacking")
    p.add_argument("--output_dir", default="/vast/users/chenyuan.chen/constantine/act_craftnet/runs")
    p.add_argument("--run_name",   default="act_craftnet_v3")

    # Model
    p.add_argument("--chunk_size",   type=int,   default=50)
    p.add_argument("--dim_model",    type=int,   default=512)
    p.add_argument("--n_enc_layers", type=int,   default=4)
    p.add_argument("--n_dec_layers", type=int,   default=7)
    p.add_argument("--latent_dim",   type=int,   default=32)
    p.add_argument("--kl_weight",    type=float, default=1.0)

    # Training
    p.add_argument("--batch_size",   type=int,   default=8,   help="per-GPU batch size")
    p.add_argument("--steps",        type=int,   default=100_000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int,   default=1_000)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--skip_frames",  type=int,   default=1)
    p.add_argument("--val_frac",     type=float, default=0.1)

    # Checkpointing
    p.add_argument("--save_every",   type=int,   default=5_000)
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--resume",       default=None)
    p.add_argument("--wandb_project", default="act-craftnet")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_off",    action="store_true")
    return p.parse_args()


def get_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def setup_dist():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def unwrap(model):
    """Unwrap DDP for state_dict access only."""
    return model.module if isinstance(model, DDP) else model


def main():
    args = parse_args()
    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    out_dir = Path(args.output_dir) / args.run_name
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main() and not args.wandb_off:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            resume="allow",
            id=args.run_name,
        )

    log(f"=== ACT-CraftNet training v3: {args.run_name} ===")
    log(f"  data_dir={args.data_dir}  GPUs={dist.get_world_size() if dist.is_initialized() else 1}")
    log(f"  kl_weight={args.kl_weight}  batch/gpu={args.batch_size}  steps={args.steps}")
    log(f"  Visual: FrozenDINOv2 ViT-B/14  |  Depth: iDP3 3-view  |  System0: MoE top-2/4")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = make_splits(
        args.data_dir, val_frac=args.val_frac,
        chunk_size=args.chunk_size, skip_frames=args.skip_frames, augment=True,
    )
    log(f"  Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = ACTConfig(
        chunk_size=args.chunk_size, dim_model=args.dim_model,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        latent_dim=args.latent_dim, kl_weight=args.kl_weight,
    )
    model = ACTCraftNet(cfg).to(device)

    if dist.is_initialized():
        # find_unused_parameters=False: all trainable params used in every forward
        # DINOv2 params are frozen (requires_grad=False) so DDP ignores them
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main():
        total_params   = sum(p.numel() for p in unwrap(model).parameters()) / 1e6
        trainable_params = sum(p.numel() for p in unwrap(model).parameters()
                               if p.requires_grad) / 1e6
        frozen_params  = total_params - trainable_params
        log(f"Model params: {total_params:.1f}M total  |  "
            f"{trainable_params:.1f}M trainable  |  {frozen_params:.1f}M frozen (DINOv2)")

    # ── Optimizer: only trainable (non-frozen) parameters ─────────────────────
    # DINOv2 params have requires_grad=False — excluded from optimizer automatically
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        log(f"Resumed from step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    epoch = 0
    loader_iter = iter(train_loader)
    log_accum = {"total_loss": 0.0, "l1_loss": 0.0, "kl_loss": 0.0}
    t0 = time.time()

    for step in range(start_step, args.steps):
        # LR schedule (cosine with warmup)
        lr_now = get_lr(step, args.warmup_steps, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Fetch batch — explicit epoch counter for correct DDP reshuffling
        try:
            batch = next(loader_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)   # must use epoch, not step
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        batch = to_device(batch, device)

        # Forward (through DDP wrapper to preserve gradient sync hooks)
        optimizer.zero_grad()
        actions_hat, mu, log_sigma = model(batch)
        loss, info = compute_loss(actions_hat, mu, log_sigma, batch, args.kl_weight)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for k in log_accum:
            log_accum[k] += info.get(k, 0.0)

        # ── Logging ────────────────────────────────────────────────────────────
        if (step + 1) % args.log_every == 0 and is_main():
            n = args.log_every
            log(f"[{step+1:6d}/{args.steps}] "
                f"loss={log_accum['total_loss']/n:.4f}  "
                f"l1={log_accum['l1_loss']/n:.4f}  "
                f"kl={log_accum['kl_loss']/n:.5f}  "
                f"lr={lr_now:.2e}  epoch={epoch}  "
                f"{time.time()-t0:.1f}s")
            if not args.wandb_off:
                wandb.log({
                    "train/loss":  log_accum["total_loss"] / n,
                    "train/l1":    log_accum["l1_loss"] / n,
                    "train/kl":    log_accum["kl_loss"] / n,
                    "train/lr":    lr_now,
                    "train/epoch": epoch,
                }, step=step + 1)
            log_accum = {k: 0.0 for k in log_accum}
            t0 = time.time()

        # ── Checkpoint + validation ────────────────────────────────────────────
        if (step + 1) % args.save_every == 0 and is_main():
            ckpt_path = out_dir / "checkpoint_latest.pt"
            torch.save({
                "step":      step + 1,
                "epoch":     epoch,
                "model":     unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "args":      vars(args),
            }, ckpt_path)
            log(f"Saved checkpoint_latest.pt at step {step+1}")

            # Validation
            model.eval()
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    vb = to_device(vb, device)
                    ah, vmu, vls = model(vb)
                    vl, _ = compute_loss(ah, vmu, vls, vb, args.kl_weight)
                    val_sum += vl.item()
                    val_n   += 1
            val_loss = val_sum / max(1, val_n)
            log(f"  Val loss: {val_loss:.4f}")
            if not args.wandb_off:
                wandb.log({"val/loss": val_loss}, step=step + 1)
            model.train()

    # ── Final checkpoint ───────────────────────────────────────────────────────
    if is_main():
        torch.save(
            {"step": args.steps, "model": unwrap(model).state_dict()},
            out_dir / "checkpoint_final.pt",
        )
        log("Training complete.")

    if is_main() and not args.wandb_off:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
