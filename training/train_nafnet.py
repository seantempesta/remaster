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
import gc
import glob
import time
import math
import random
import argparse
import json
import signal
import warnings
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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


class VGGFeatureExtractor(nn.Module):
    """Extract multi-layer features from VGG19 for perceptual loss.

    Adapted from KAIR (github.com/cszn/KAIR). Extracts features at
    conv layers [2, 7, 16, 25, 34] (one per VGG block), applies
    ImageNet normalization, and freezes all weights.
    """
    def __init__(self, feature_layers=[2, 7, 16, 25, 34]):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        # Split VGG into sub-networks ending at each feature layer
        self.features = nn.Sequential()
        prev = -1
        for i, layer in enumerate(feature_layers):
            self.features.add_module(
                f'block{i}',
                nn.Sequential(*list(vgg.features.children())[(prev + 1):(layer + 1)])
            )
            prev = layer

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for block in self.features.children():
            x = block(x)
            feats.append(x.clone())
        return feats


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for image restoration.

    Computes weighted L1 distance between VGG19 feature maps of
    prediction and target. Runs in eval mode with frozen weights.
    Must be called in fp32 (not inside AMP autocast).
    """
    def __init__(self, weights=[0.1, 0.1, 1.0, 1.0, 1.0]):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.weights = weights
        self.criterion = nn.L1Loss()
        self.eval()

    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        with torch.no_grad():
            target_feats = self.vgg(target)
        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss += w * self.criterion(pf, tf)
        return loss

    def train(self, mode=True):
        # Always stay in eval mode (BN stats, dropout)
        return super().train(False)


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
    def __init__(self, data_dir, crop_size=256, augment=True,
                 max_frames=-1, cache_in_ram=False):
        self.crop_size = crop_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cached_images = None

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
            raise FileNotFoundError(
                f"No matching pairs found in {data_dir}")

        if cache_in_ram:
            self._load_all_into_ram()

        print(f"Dataset: {len(self.pairs)} pairs, crop={crop_size}, "
              f"augment={augment}, cached={cache_in_ram}")

    def _load_all_into_ram(self):
        """Pre-load all images as float32 numpy arrays."""
        import psutil
        print(f"  Caching {len(self.pairs)} pairs into RAM...")
        t0 = time.time()
        self.cached_images = []
        for i, (inp_path, tgt_path) in enumerate(self.pairs):
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            # Store as uint8 to save RAM (convert to float in __getitem__)
            self.cached_images.append((inp, tgt))
            if (i + 1) % 200 == 0:
                mb = psutil.Process().memory_info().rss / 1024**2
                print(f"    {i+1}/{len(self.pairs)} loaded, "
                      f"RAM: {mb:.0f}MB")
        elapsed = time.time() - t0
        mb = psutil.Process().memory_info().rss / 1024**2
        print(f"  Cached {len(self.pairs)} pairs in {elapsed:.1f}s, "
              f"RAM: {mb:.0f}MB")

    def __len__(self):
        return len(self.pairs) * 100

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)

        if self.cached_images is not None:
            inp, tgt = self.cached_images[pair_idx]
            inp = inp.astype(np.float32) / 255.0
            tgt = tgt.astype(np.float32) / 255.0
        else:
            inp_path, tgt_path = self.pairs[pair_idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0

        h, w, _ = inp.shape
        cs = self.crop_size

        # Random crop (same location for both)
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        inp = inp[top:top + cs, left:left + cs]
        tgt = tgt[top:top + cs, left:left + cs]

        # Augmentation: random flip and rotation
        if self.augment:
            if random.random() < 0.5:
                inp = inp[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()
            if random.random() < 0.5:
                inp = inp[::-1, :, :].copy()
                tgt = tgt[::-1, :, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                inp = np.rot90(inp, k).copy()
                tgt = np.rot90(tgt, k).copy()

        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        tgt = torch.from_numpy(tgt.transpose(2, 0, 1))
        return inp, tgt


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_dir, device, pixel_criterion=None, perceptual_criterion=None,
             perceptual_weight=0.0, crop_size=512):
    """
    Validate on held-out frames with larger crops than training.
    Returns dict with PSNR, pixel loss, perceptual loss, and combined loss.
    """
    model.eval()
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    psnrs = []
    pixel_losses = []
    percep_losses = []

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

        # PSNR
        mse = ((out_t - tgt_t) ** 2).mean().item()
        psnrs.append(10 * math.log10(1.0 / mse) if mse > 0 else 100.0)

        # Pixel loss
        if pixel_criterion is not None:
            pixel_losses.append(pixel_criterion(out_t, tgt_t).item())

        # Perceptual loss (fp32)
        if perceptual_criterion is not None:
            p_loss = perceptual_criterion(out_t.float(), tgt_t.float())
            percep_losses.append(p_loss.item())

    model.train()
    result = {"psnr": np.mean(psnrs) if psnrs else 0.0, "n_frames": len(psnrs)}
    if pixel_losses:
        result["pixel_loss"] = np.mean(pixel_losses)
    if percep_losses:
        result["percep_loss"] = np.mean(percep_losses)
        result["combined_loss"] = result.get("pixel_loss", 0) + perceptual_weight * result["percep_loss"]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint saving
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, scheduler, scaler, iteration, best_psnr, args, ckpt_dir):
    """Save a full training checkpoint (model + optimizer + scheduler state).

    Filters callable attributes from args to avoid serialization issues.
    """
    ckpt_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
    }
    return ckpt_data


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
    enc_blk_nums = [int(x) for x in args.enc_blk_nums.split(",")]
    dec_blk_nums = [int(x) for x in args.dec_blk_nums.split(",")]
    model = NAFNet(
        img_channel=3, width=args.width,
        middle_blk_num=args.middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
    )

    # Load pretrained weights (skip if resuming from a training checkpoint)
    if args.resume and os.path.exists(args.resume):
        pass  # will load model + optimizer + scheduler below
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))
        # Use strict=False to handle architecture mismatches (e.g. fewer middle blocks)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  Skipped {len(unexpected)} unexpected keys (arch mismatch):")
            for k in unexpected[:10]:
                print(f"    {k}")
            if len(unexpected) > 10:
                print(f"    ... and {len(unexpected) - 10} more")
        if missing:
            print(f"  {len(missing)} missing keys (will be randomly initialized):")
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if not missing and not unexpected:
            print("  Loaded pretrained weights (exact match)")
    else:
        print("WARNING: Training from scratch (no pretrained weights)")

    model = model.to(device)
    model.train()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"NAFNet w={args.width} mid={args.middle_blk_num} enc={enc_blk_nums} dec={dec_blk_nums}: {params_m:.2f}M parameters")
    if device.type == "cuda":
        print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # ---- Dataset ----
    cache_in_ram = getattr(args, 'cache_in_ram', False)
    dataset = PairedFrameDataset(
        args.data_dir,
        crop_size=args.crop_size,
        augment=True,
        cache_in_ram=cache_in_ram,
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

    # ---- Perceptual loss (optional) ----
    perceptual_criterion = None
    if args.perceptual_weight > 0:
        perceptual_criterion = VGGPerceptualLoss().to(device)
        print(f"Perceptual loss: VGG19, weight={args.perceptual_weight}")
        if device.type == "cuda":
            print(f"  VGG VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.9),  # NAFNet uses (0.9, 0.9)
        fused=device.type == "cuda",  # single fused kernel for optimizer step
    )

    # ---- LR Scheduler: cosine with linear warmup ----
    warmup_iters = args.warmup_iters
    def lr_lambda(step):
        if step < warmup_iters:
            return max(step / max(warmup_iters, 1), 1e-8)  # avoid exact 0 at step 0
        progress = (step - warmup_iters) / max(args.max_iters - warmup_iters, 1)
        return max(args.eta_min / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
    best_val_loss = None
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
    print(f"  perceptual:  {args.perceptual_weight}")
    print(f"  grad_clip:   {args.grad_clip}")
    print(f"  amp:         {use_amp}")
    print(f"  checkpoints: {ckpt_dir}")
    print()

    # ---- cuDNN benchmark warmup ----
    torch.backends.cudnn.benchmark = True

    # Run one full training step (including perceptual loss) to trigger
    # cudnn.benchmark algorithm selection, then free scratch memory.
    print("  cuDNN benchmark warmup...")
    _wb = next(iter(dataloader))
    _wi, _wt = _wb[0].to(device, non_blocking=True), _wb[1].to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        _wo = model(_wi)
        _wl = criterion(_wo, _wt)
    if perceptual_criterion is not None:
        with torch.amp.autocast("cuda", enabled=False):
            _pl = perceptual_criterion(_wo.float(), _wt.float())
        _wl = _wl + args.perceptual_weight * _pl
    _wl.backward()
    optimizer.zero_grad(set_to_none=True)
    del _wb, _wi, _wt, _wo, _wl
    if perceptual_criterion is not None:
        del _pl
    torch.cuda.empty_cache()
    gc.collect()
    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    curr_gb = torch.cuda.memory_reserved() / 1024**3
    print(f"  cuDNN benchmark done. Peak: {peak_gb:.1f}GB, settled: {curr_gb:.1f}GB")
    torch.cuda.reset_peak_memory_stats()

    # ---- Training state ----
    data_iter = iter(dataloader)
    loss_sum = 0.0
    pixel_loss_sum = 0.0
    percep_loss_sum = 0.0
    loss_count = 0
    percep_count = 0
    data_time_sum = 0.0
    compute_time_sum = 0.0
    start_time = time.time()
    t_data_start = time.time()

    stop_check = getattr(args, 'stop_check', None)
    stop_check_freq = getattr(args, 'stop_check_freq', 50)
    percep_freq = getattr(args, 'perceptual_freq', 10)

    # CUDA event profiling — measures GPU time per phase, logged every print_freq
    profile_on = device.type == "cuda"
    if profile_on:
        ev_fwd_start = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_vgg_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

    # ---- Signal handler for graceful interrupts ----
    def _save_emergency(it):
        """Save checkpoint on interrupt/signal."""
        ckpt_data = _save_checkpoint(model, optimizer, scheduler, scaler,
                                     it, best_psnr, args, ckpt_dir)
        path = os.path.join(ckpt_dir, "nafnet_latest.pth")
        torch.save(ckpt_data, path)
        torch.save({"params": model.state_dict()},
                    os.path.join(ckpt_dir, "nafnet_final.pth"))
        print(f"  Saved checkpoint at iter {it}: {path}")

    _current_iter = start_iter
    interrupted = False
    def _handle_interrupt(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print(f"\n  SIGINT received at iter {_current_iter}. "
                  f"Saving checkpoint...")
            _save_emergency(_current_iter)
            print("  Checkpoint saved. Exiting.")
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handle_interrupt)

    # ---- Main training loop ----
    for iteration in range(start_iter, args.max_iters):
        _current_iter = iteration
        # Graceful stop check
        if stop_check and (iteration + 1) % stop_check_freq == 0:
            if stop_check():
                print(f"\n  STOP signal at iter {iteration + 1}.")
                _save_emergency(iteration + 1)
                return

        # Get batch (restart iterator if exhausted)
        try:
            inp_batch, tgt_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inp_batch, tgt_batch = next(data_iter)

        inp_batch = inp_batch.to(device, non_blocking=True)
        tgt_batch = tgt_batch.to(device, non_blocking=True)
        t_compute_start = time.time()
        data_time_sum += t_compute_start - t_data_start

        # CUDA event profiling (on print iters)
        is_print_iter = (iteration + 1) % args.print_freq == 0
        do_profile = profile_on and is_print_iter
        if do_profile:
            torch.cuda.synchronize()
            ev_fwd_start.record()

        # Forward with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(inp_batch)
            pixel_loss = criterion(pred, tgt_batch)

        if do_profile:
            ev_fwd_end.record()

        # Perceptual loss runs in fp32 (VGG is numerically unstable in fp16)
        # Only compute every N iters to save GPU time (scale up to compensate)
        if perceptual_criterion is not None and (iteration % percep_freq == 0):
            with torch.amp.autocast("cuda", enabled=False):
                p_loss = perceptual_criterion(pred.float(), tgt_batch.float())
            loss = pixel_loss + args.perceptual_weight * percep_freq * p_loss
        else:
            p_loss = None
            loss = pixel_loss

        if do_profile:
            ev_vgg_end.record()

        # Backward with scaler
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping (prevents explosion during fine-tuning)
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if do_profile:
            ev_bwd_end.record()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if do_profile:
            ev_opt_end.record()

        t_data_start = time.time()
        compute_time_sum += t_data_start - t_compute_start

        loss_sum += loss.item()
        pixel_loss_sum += pixel_loss.item()
        if p_loss is not None:
            percep_loss_sum += p_loss.item()
            percep_count += 1
        loss_count += 1

        # ---- Logging ----
        if (iteration + 1) % args.print_freq == 0:
            avg_loss = loss_sum / loss_count
            elapsed = time.time() - start_time
            iters_per_sec = (iteration + 1 - start_iter) / elapsed
            eta = (args.max_iters - iteration - 1) / max(iters_per_sec, 0.01)
            lr_now = optimizer.param_groups[0]["lr"]
            vram = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0
            total_t = data_time_sum + compute_time_sum
            data_pct = data_time_sum / total_t * 100 if total_t > 0 else 0
            loss_detail = f"loss={avg_loss:.6f}"
            if perceptual_criterion is not None and percep_count > 0:
                avg_px = pixel_loss_sum / loss_count
                avg_pc = percep_loss_sum / percep_count
                loss_detail = f"loss={avg_loss:.6f} (px={avg_px:.6f} perc={avg_pc:.4f})"
            profile_str = ""
            if do_profile:
                torch.cuda.synchronize()
                t_fwd = ev_fwd_start.elapsed_time(ev_fwd_end)
                t_vgg = ev_fwd_end.elapsed_time(ev_vgg_end)
                t_bwd = ev_vgg_end.elapsed_time(ev_bwd_end)
                t_opt = ev_bwd_end.elapsed_time(ev_opt_end)
                t_total = t_fwd + t_vgg + t_bwd + t_opt
                profile_str = (
                    f"\n    fwd={t_fwd:.0f} vgg={t_vgg:.0f} "
                    f"bwd={t_bwd:.0f} opt={t_opt:.0f} "
                    f"total={t_total:.0f}ms")
            print(
                f"  {iteration + 1:5d}/{args.max_iters} | "
                f"{loss_detail} | lr={lr_now:.2e} | "
                f"{iters_per_sec:.1f}it/s "
                f"ETA:{eta / 60:.0f}m | "
                f"{vram:.1f}GB | data:{data_pct:.0f}%"
                f"{profile_str}")
            loss_sum = 0.0
            pixel_loss_sum = 0.0
            percep_loss_sum = 0.0
            loss_count = 0
            percep_count = 0
            data_time_sum = 0.0
            compute_time_sum = 0.0

        # ---- Validation ----
        if (iteration + 1) % args.val_freq == 0:
            val_dir = args.val_dir if args.val_dir else args.data_dir
            val = validate(model, val_dir, device,
                          pixel_criterion=criterion,
                          perceptual_criterion=perceptual_criterion,
                          perceptual_weight=args.perceptual_weight,
                          crop_size=512)
            psnr = val["psnr"]
            # Use total loss (pixel + perceptual) to decide best when
            # perceptual loss is active, otherwise fall back to PSNR
            val_loss = val.get("combined_loss", None)
            if val_loss is not None:
                is_best = best_val_loss is None or val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_psnr = psnr
            else:
                is_best = psnr > best_psnr
                if is_best:
                    best_psnr = psnr
            val_detail = f"PSNR={psnr:.2f} dB"
            if "pixel_loss" in val:
                val_detail += f" | px={val['pixel_loss']:.6f}"
            if "percep_loss" in val:
                val_detail += f" perc={val['percep_loss']:.4f}"
            if "combined_loss" in val:
                val_detail += f" total={val['combined_loss']:.6f}"
            best_str = (f"best_loss={best_val_loss:.6f}"
                        if best_val_loss is not None
                        else f"best_psnr={best_psnr:.2f}")
            print(f"  VAL {iteration + 1} ({val['n_frames']}f): "
                  f"{val_detail} {'(BEST)' if is_best else ''} "
                  f"[{best_str}]")
            model.train()

            # Save best model
            if is_best:
                best_path = os.path.join(ckpt_dir, "nafnet_best.pth")
                torch.save({"params": model.state_dict()}, best_path)
                print(f"  Saved best model: {best_path}")

        # ---- Checkpoint ----
        if (iteration + 1) % args.save_freq == 0:
            ckpt_data = _save_checkpoint(model, optimizer, scheduler, scaler,
                                         iteration + 1, best_psnr, args, ckpt_dir)
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
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Validation pairs directory (separate from training data)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")

    # Architecture
    parser.add_argument("--width", type=int, default=64,
                        help="NAFNet channel width (default: 64)")
    parser.add_argument("--middle-blk-num", type=int, default=12,
                        help="Number of middle blocks (default: 12)")
    parser.add_argument("--enc-blk-nums", type=str, default="2,2,4,8",
                        help="Encoder block counts, comma-separated (default: 2,2,4,8)")
    parser.add_argument("--dec-blk-nums", type=str, default="2,2,2,2",
                        help="Decoder block counts, comma-separated (default: 2,2,2,2)")

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
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled, default: 1.0)")
    parser.add_argument("--loss", type=str, default="charbonnier",
                        choices=["charbonnier", "psnr", "l1"],
                        help="Loss function")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use AMP mixed precision (default: on, essential for 6GB VRAM)")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable AMP mixed precision")
    parser.add_argument("--perceptual-weight", type=float, default=0.0,
                        help="VGG perceptual loss weight (0 = disabled, try 0.05)")
    parser.add_argument("--perceptual-freq", type=int, default=10,
                        help="Compute perceptual loss every N iters (default: 10, aligned with print_freq)")

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
    # Ensure enc/dec blk nums are strings (parse_args gives strings, but
    # when constructed programmatically they might already be strings)
    if not isinstance(args.enc_blk_nums, str):
        args.enc_blk_nums = ",".join(str(x) for x in args.enc_blk_nums)
    if not isinstance(args.dec_blk_nums, str):
        args.dec_blk_nums = ",".join(str(x) for x in args.dec_blk_nums)
    train(args)
