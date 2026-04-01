"""
Fine-tune NAFNet-width64 on SCUNet pseudo ground truth for video denoising.

Distillation training: NAFNet learns to replicate SCUNet's denoising quality
at much higher speed (pure CNN, torch.compile friendly).

Training recipe:
    - Start from SIDD-pretrained NAFNet-width64 checkpoint
    - Random 256x256 crops from full-frame 1080p pairs
    - Charbonnier loss (smooth L1)
    - AdamW optimizer, cosine LR with warmup
    - Random flip/rotation augmentation
    - Validation PSNR every N iterations

Usage:
    # Quick local test (5 frames, 100 iters)
    python training/train_nafnet.py --data-dir data/train_pairs --max-iters 100 --val-freq 50 --batch-size 4

    # Full local training
    python training/train_nafnet.py --data-dir data/train_pairs --max-iters 50000 --batch-size 4 --lr 2e-4

    # Arguments can also be set via environment variables (for Modal):
    #   DATA_DIR, CHECKPOINT_DIR, PRETRAINED_PATH, MAX_ITERS, etc.
"""
import os
import sys
import glob
import time
import math
import random
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.nafnet_arch import NAFNet
from lib.paths import PROJECT_ROOT, REFERENCE_CODE, CHECKPOINTS_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1): sqrt((pred - target)^2 + eps^2)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class PSNRLoss(nn.Module):
    """PSNR-based loss (as used in NAFNet original training)."""
    def __init__(self):
        super().__init__()
        self.scale = 10 / math.log(10)

    def forward(self, pred, target):
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return self.scale * torch.log(mse + 1e-8).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PairedFrameDataset(Dataset):
    """
    Loads paired input/target frames, returns random crops with augmentation.

    Directory structure:
        data_dir/input/frame_XXXXX.png   (compressed input)
        data_dir/target/frame_XXXXX.png  (SCUNet denoised)
    """
    def __init__(self, data_dir, crop_size=256, augment=True, max_frames=-1):
        self.crop_size = crop_size
        self.augment = augment

        input_dir = os.path.join(data_dir, "input")
        target_dir = os.path.join(data_dir, "target")

        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No PNG files in {input_dir}")

        # Match input/target pairs by filename
        self.pairs = []
        for inp_path in input_files:
            fname = os.path.basename(inp_path)
            tgt_path = os.path.join(target_dir, fname)
            if os.path.exists(tgt_path):
                self.pairs.append((inp_path, tgt_path))

        if max_frames > 0:
            self.pairs = self.pairs[:max_frames]

        if not self.pairs:
            raise FileNotFoundError(f"No matching pairs found in {data_dir}")
        print(f"Dataset: {len(self.pairs)} pairs, crop={crop_size}, augment={augment}")

    def __len__(self):
        # Return a large number so DataLoader can iterate indefinitely
        # (we control training by iteration count, not epochs)
        return len(self.pairs) * 100

    def __getitem__(self, idx):
        # Pick a random pair (better than cycling through in order)
        pair_idx = idx % len(self.pairs)
        inp_path, tgt_path = self.pairs[pair_idx]

        # Read images (BGR -> RGB -> float32 [0,1])
        inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w, _ = inp.shape
        cs = self.crop_size

        # Random crop (same location for both)
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        inp = inp[top:top + cs, left:left + cs]
        tgt = tgt[top:top + cs, left:left + cs]

        # Augmentation: random flip and rotation
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                inp = inp[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()
            # Vertical flip
            if random.random() < 0.5:
                inp = inp[::-1, :, :].copy()
                tgt = tgt[::-1, :, :].copy()
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                inp = np.rot90(inp, k).copy()
                tgt = np.rot90(tgt, k).copy()

        # HWC -> CHW tensor
        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        tgt = torch.from_numpy(tgt.transpose(2, 0, 1))
        return inp, tgt


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_dir, device, num_frames=5, crop_size=512):
    """
    Validate on a few full-resolution crops (larger than training crops).
    Returns average PSNR in dB.
    """
    model.eval()
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))[:num_frames]
    psnrs = []

    for inp_path in input_files:
        fname = os.path.basename(inp_path)
        tgt_path = os.path.join(target_dir, fname)
        if not os.path.exists(tgt_path):
            continue

        inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w, _ = inp.shape
        # Center crop for validation (deterministic)
        cs = min(crop_size, h, w)
        top = (h - cs) // 2
        left = (w - cs) // 2
        inp = inp[top:top + cs, left:left + cs]
        tgt = tgt[top:top + cs, left:left + cs]

        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(tgt.transpose(2, 0, 1)).unsqueeze(0).to(device)

        out_t = model(inp_t)
        out_t = out_t.clamp(0, 1)

        mse = ((out_t - tgt_t) ** 2).mean().item()
        if mse > 0:
            psnr = 10 * math.log10(1.0 / mse)
        else:
            psnr = 100.0
        psnrs.append(psnr)

    model.train()
    return np.mean(psnrs) if psnrs else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Model ----
    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    # Load pretrained weights (skip if resuming from a training checkpoint)
    if args.resume and os.path.exists(args.resume):
        pass  # will load model + optimizer + scheduler below
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded SIDD pretrained weights")
    else:
        print("WARNING: Training from scratch (no pretrained weights)")

    model = model.to(device)
    model.train()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"NAFNet-width64: {params_m:.2f}M parameters")
    if device.type == "cuda":
        print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # ---- Dataset ----
    dataset = PairedFrameDataset(
        args.data_dir,
        crop_size=args.crop_size,
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- Loss ----
    if args.loss == "charbonnier":
        criterion = CharbonnierLoss(eps=1e-6)
    elif args.loss == "psnr":
        criterion = PSNRLoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    print(f"Loss: {args.loss}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.9),  # NAFNet uses (0.9, 0.9)
    )

    # ---- LR Scheduler: cosine with linear warmup ----
    warmup_iters = args.warmup_iters
    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(warmup_iters, 1)
        progress = (step - warmup_iters) / max(args.max_iters - warmup_iters, 1)
        return max(args.eta_min / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Checkpoint dir ----
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- AMP (mixed precision) ----
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP (mixed precision): {'ON' if use_amp else 'OFF'}")

    # ---- Resume ----
    start_iter = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["iteration"]
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"  Resumed at iter {start_iter}, best_psnr={best_psnr:.2f}")

    # ---- Training ----
    print(f"\nTraining config:")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  crop_size:   {args.crop_size}")
    print(f"  lr:          {args.lr}")
    print(f"  max_iters:   {args.max_iters}")
    print(f"  warmup:      {warmup_iters}")
    print(f"  loss:        {args.loss}")
    print(f"  amp:         {use_amp}")
    print(f"  checkpoints: {ckpt_dir}")
    print()

    torch.backends.cudnn.benchmark = True

    data_iter = iter(dataloader)
    loss_sum = 0.0
    loss_count = 0
    start_time = time.time()

    for iteration in range(start_iter, args.max_iters):
        # Get batch (restart iterator if exhausted)
        try:
            inp_batch, tgt_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inp_batch, tgt_batch = next(data_iter)

        inp_batch = inp_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # Forward with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(inp_batch)
            loss = criterion(pred, tgt_batch)

        # Backward with scaler
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping (prevents explosion during fine-tuning)
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item()
        loss_count += 1

        # ---- Logging ----
        if (iteration + 1) % args.print_freq == 0:
            avg_loss = loss_sum / loss_count
            elapsed = time.time() - start_time
            iters_per_sec = (iteration + 1 - start_iter) / elapsed
            eta = (args.max_iters - iteration - 1) / max(iters_per_sec, 0.01)
            lr_now = optimizer.param_groups[0]["lr"]
            vram = torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
            print(f"  iter {iteration + 1:6d}/{args.max_iters} | "
                  f"loss={avg_loss:.6f} | lr={lr_now:.2e} | "
                  f"{iters_per_sec:.1f} it/s | ETA: {eta / 60:.1f}min | "
                  f"VRAM: {vram:.0f}MB")
            loss_sum = 0.0
            loss_count = 0

        # ---- Validation ----
        if (iteration + 1) % args.val_freq == 0:
            psnr = validate(model, args.data_dir, device,
                          num_frames=min(10, len(dataset.pairs)),
                          crop_size=512)
            is_best = psnr > best_psnr
            if is_best:
                best_psnr = psnr
            print(f"  VALIDATION iter {iteration + 1}: PSNR={psnr:.2f} dB "
                  f"{'(BEST)' if is_best else ''} [best={best_psnr:.2f}]")
            model.train()

            # Save best model
            if is_best:
                best_path = os.path.join(ckpt_dir, "nafnet_best.pth")
                torch.save({"params": model.state_dict()}, best_path)
                print(f"  Saved best model: {best_path}")

        # ---- Checkpoint ----
        if (iteration + 1) % args.save_freq == 0:
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "iteration": iteration + 1,
                "best_psnr": best_psnr,
                "args": vars(args),
            }
            ckpt_path = os.path.join(ckpt_dir, f"nafnet_iter{iteration + 1:06d}.pth")
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Also save latest (for easy resume)
            latest_path = os.path.join(ckpt_dir, "nafnet_latest.pth")
            torch.save(ckpt_data, latest_path)

    # ---- Final save ----
    final_path = os.path.join(ckpt_dir, "nafnet_final.pth")
    torch.save({"params": model.state_dict()}, final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best PSNR: {best_psnr:.2f} dB")

    # Also save the best model in a standalone format (just params, like NAFNet expects)
    best_ckpt = os.path.join(ckpt_dir, "nafnet_best.pth")
    if os.path.exists(best_ckpt):
        print(f"Best model: {best_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train NAFNet on SCUNet pseudo-GT pairs")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("DATA_DIR", "data/train_pairs"),
                        help="Training pairs directory (input/ + target/)")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Random crop size for training (default: 256)")
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("BATCH_SIZE", "4")),
                        help="Batch size (4 fits in 6GB with 256 crops)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")

    # Model
    parser.add_argument("--pretrained", type=str,
                        default=os.environ.get("PRETRAINED_PATH",
                            str(REFERENCE_CODE / "NAFNet" / "experiments" / "pretrained_models" / "NAFNet-SIDD-width64.pth")),
                        help="Pretrained checkpoint path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")

    # Training
    parser.add_argument("--max-iters", type=int,
                        default=int(os.environ.get("MAX_ITERS", "50000")),
                        help="Total training iterations")
    parser.add_argument("--lr", type=float,
                        default=float(os.environ.get("LR", "2e-4")),
                        help="Learning rate (lower than from-scratch since fine-tuning)")
    parser.add_argument("--eta-min", type=float, default=1e-7,
                        help="Minimum LR for cosine schedule")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay")
    parser.add_argument("--warmup-iters", type=int, default=500,
                        help="Linear warmup iterations")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--loss", type=str, default="charbonnier",
                        choices=["charbonnier", "psnr", "l1"],
                        help="Loss function")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use AMP mixed precision (default: on, essential for 6GB VRAM)")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable AMP mixed precision")

    # Logging / checkpoints
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.environ.get("CHECKPOINT_DIR",
                            str(CHECKPOINTS_DIR / "nafnet_distill")),
                        help="Checkpoint save directory")
    parser.add_argument("--print-freq", type=int, default=50,
                        help="Print every N iterations")
    parser.add_argument("--val-freq", type=int, default=1000,
                        help="Validate every N iterations")
    parser.add_argument("--save-freq", type=int, default=5000,
                        help="Save checkpoint every N iterations")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
