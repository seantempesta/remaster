"""
Unified training script for all denoising model architectures.

Supports: NAFNet, PlainDenoise, UNetDenoise (via --model flag).

Features:
    - Charbonnier / PSNRLoss / L1 pixel loss + DISTS perceptual + Focal Frequency
    - EMA weights (--ema), GPU dataset caching (--cache-on-gpu)
    - Intensity scaling augmentation, small initialization for tiny models
    - CUDA event profiling, graceful stop (SIGINT + Modal Dict)
    - Validation with PSNR + all losses + sample comparison images + loss curves
    - Checkpoint save/resume with full optimizer + scheduler + EMA state

Usage:
    # NAFNet (original)
    python training/train_nafnet.py --model nafnet --width 32 --middle-blk-num 4

    # UNetDenoise (INT8-native, 67 fps TRT INT8)
    python training/train_nafnet.py --model unet --nc 64 --nb-mid 2 --ema --cache-on-gpu

    # PlainDenoise (sequential CNN)
    python training/train_nafnet.py --model plain --nc 64 --nb 15

    # Environment variables for Modal: DATA_DIR, CHECKPOINT_DIR, MAX_ITERS, etc.
"""
import os
import sys
import gc
import glob
import time
import math
import argparse
import signal
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.nafnet_arch import NAFNet
from lib.plainnet_arch import PlainDenoise, UNetDenoise, count_params as _count_params
from lib.paths import PROJECT_ROOT, REFERENCE_CODE, CHECKPOINTS_DIR
from training.losses import (
    CharbonnierLoss, PSNRLoss, FocalFrequencyLoss, DISTSPerceptualLoss,
    build_pixel_criterion,
)
from training.dataset import PairedFrameDataset, GPUCachedDataset
from training.viz import TrainingLogger, save_val_samples


# ──────────────────────────────────────────────────────────────────────────────
# Model builder — supports NAFNet, PlainDenoise, UNetDenoise
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args):
    """Build model based on --model flag."""
    model_type = getattr(args, 'model', 'nafnet')

    if model_type == 'nafnet':
        enc_blk_nums = [int(x) for x in args.enc_blk_nums.split(",")]
        dec_blk_nums = [int(x) for x in args.dec_blk_nums.split(",")]
        model = NAFNet(
            img_channel=3, width=args.width,
            middle_blk_num=args.middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )
        desc = (f"NAFNet w={args.width} mid={args.middle_blk_num} "
                f"enc={enc_blk_nums} dec={dec_blk_nums}")
    elif model_type == 'plain':
        model = PlainDenoise(
            in_nc=3, nc=getattr(args, 'nc', 64),
            nb=getattr(args, 'nb', 15), use_bn=True, deploy=False,
        )
        desc = f"PlainDenoise nc={args.nc} nb={args.nb}"
    elif model_type == 'unet':
        nb_enc = tuple(int(x) for x in getattr(args, 'nb_enc', '2,2').split(","))
        nb_dec = tuple(int(x) for x in getattr(args, 'nb_dec', '2,2').split(","))
        model = UNetDenoise(
            in_nc=3, nc=getattr(args, 'nc', 64),
            nb_enc=nb_enc, nb_dec=nb_dec,
            nb_mid=getattr(args, 'nb_mid', 2),
            use_bn=True, deploy=False,
        )
        desc = f"UNetDenoise nc={args.nc} mid={args.nb_mid}"
    else:
        raise ValueError(f"Unknown model: {model_type}")

    params = sum(p.numel() for p in model.parameters())
    print(f"{desc}: {params/1e6:.2f}M parameters")
    return model, desc


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) — from SPAN/BasicSR
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Small weight init — from XLSR (helps tiny models avoid early instability)
# ──────────────────────────────────────────────────────────────────────────────

def init_weights_small(model, scale=0.1):
    """Initialize conv weights with scaled Kaiming normal."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# 2:4 Structured Sparsity (APEX ASP)
# From https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/
# Pipeline: train dense → prune 2:4 + fine-tune → QAT → TRT --int8 --sparsity=enable
# ──────────────────────────────────────────────────────────────────────────────

def prepare_sparsity(model, optimizer):
    """Initialize 2:4 structured sparsity via APEX ASP.

    Prunes a pretrained dense model to 2:4 pattern (2 of every 4 weights zeroed).
    Ampere tensor cores skip zero multiplications in hardware → ~1.3-1.4x speedup
    on top of INT8.

    Requires: apex with sparsity support (pip install apex or build from source).
    Call AFTER loading pretrained weights, BEFORE fine-tuning.
    """
    from apex.contrib.sparsity import ASP
    ASP.prune_trained_model(model, optimizer)
    print("  2:4 structured sparsity applied via APEX ASP")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quantization-Aware Training (QAT)
# From XLSR QAT recipe: train dense → QAT fine-tune at lower LR → export with Q/DQ nodes
# ──────────────────────────────────────────────────────────────────────────────

def prepare_qat(model):
    """Prepare model for Quantization-Aware Training.

    Inserts FakeQuantize modules that simulate INT8 rounding during forward pass.
    Uses symmetric per-channel quantization for weights, per-tensor for activations
    — matches TensorRT's INT8 scheme.

    QAT trains in FP32 (fake quant is a FP32 operation). Disable AMP when using QAT.
    """
    from torch.ao.quantization import (
        QConfig, FakeQuantize,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )

    act_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    weight_fq = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
    )

    model.qconfig = QConfig(activation=act_fq, weight=weight_fq)
    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)
    print("  QAT prepared: FakeQuantize nodes inserted")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Intensity scaling augmentation — from XLSR
# ──────────────────────────────────────────────────────────────────────────────

def apply_intensity_aug(inp, tgt, scales=(0.5, 0.7, 1.0)):
    """Randomly scale intensity of input+target pair (same scale for both)."""
    idx = torch.randint(len(scales), (1,)).item()
    scale = scales[idx]
    if scale != 1.0:
        return inp * scale, tgt * scale
    return inp, tgt


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_dir, device, pixel_criterion=None,
             perceptual_criterion=None, fft_criterion=None,
             perceptual_weight=0.0, fft_weight=0.0, crop_size=512):
    """
    Validate on held-out frames with larger crops than training.
    Returns dict with PSNR, pixel loss, perceptual loss, FFT loss, and combined loss.
    """
    import cv2

    model.eval()
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    psnrs = []
    pixel_losses = []
    percep_losses = []
    fft_losses = []

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

        # FFT loss
        if fft_criterion is not None:
            f_loss = fft_criterion(out_t.float(), tgt_t.float())
            fft_losses.append(f_loss.item())

    model.train()
    result = {"psnr": np.mean(psnrs) if psnrs else 0.0, "n_frames": len(psnrs)}
    if pixel_losses:
        result["pixel_loss"] = np.mean(pixel_losses)
    if percep_losses:
        result["percep_loss"] = np.mean(percep_losses)
    if fft_losses:
        result["fft_loss"] = np.mean(fft_losses)

    # Combined loss for best-model selection
    total = result.get("pixel_loss", 0)
    if percep_losses:
        total += perceptual_weight * result["percep_loss"]
    if fft_losses:
        total += fft_weight * result["fft_loss"]
    result["combined_loss"] = total

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint saving
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, scheduler, scaler, iteration, best_psnr, args, ckpt_dir, ema=None):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    ckpt_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_psnr": best_psnr,
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
    }
    if ema is not None:
        ckpt_data["ema"] = ema.state_dict()
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
    model, model_desc = build_model(args)

    # Load pretrained weights (skip if resuming from a training checkpoint)
    if args.resume and os.path.exists(args.resume):
        pass  # will load model + optimizer + scheduler below
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))
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
        # Small init helps tiny models (PlainDenoise/UNetDenoise) avoid instability
        if getattr(args, 'model', 'nafnet') != 'nafnet':
            init_weights_small(model, scale=0.1)
            print("  Applied small initialization (0.1x Kaiming)")

    model = model.to(device)
    model.train()

    if device.type == "cuda":
        print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # ---- EMA ----
    use_ema = getattr(args, 'ema', False)
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=getattr(args, 'ema_decay', 0.999))
        print(f"EMA: ON (decay={ema.decay})")

    # ---- Dataset ----
    from training.dataset import GPUCachedDataset
    cache_on_gpu = getattr(args, 'cache_on_gpu', False)
    cache_in_ram = getattr(args, 'cache_in_ram', False)
    gpu_dataset = None
    dataloader = None

    if cache_on_gpu and device.type == "cuda":
        gpu_dataset = GPUCachedDataset(
            args.data_dir, crop_size=args.crop_size, device=device,
        )
        print(f"Data pipeline: GPU-cached ({gpu_dataset.n} pairs in VRAM)")
    else:
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

    # ---- Losses ----
    criterion = build_pixel_criterion(args.loss)
    print(f"Pixel loss: {args.loss}")

    perceptual_criterion = None
    if args.perceptual_weight > 0:
        perceptual_criterion = DISTSPerceptualLoss().to(device)
        print(f"Perceptual loss: DISTS, weight={args.perceptual_weight}")
        if device.type == "cuda":
            print(f"  DISTS VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    fft_criterion = None
    if args.fft_weight > 0:
        fft_criterion = FocalFrequencyLoss(alpha=args.fft_alpha)
        print(f"FFT loss: Focal Frequency (alpha={args.fft_alpha}), weight={args.fft_weight}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.9),  # NAFNet uses (0.9, 0.9)
        fused=device.type == "cuda",  # single fused kernel for optimizer step
    )

    # ---- 2:4 Sparsity (after optimizer, before scheduler) ----
    use_sparse = getattr(args, 'sparse', False)
    if use_sparse:
        print("Applying 2:4 structured sparsity (APEX ASP)...")
        model = prepare_sparsity(model, optimizer)

    # ---- QAT (after optimizer, before scheduler) ----
    use_qat = getattr(args, 'qat', False)
    if use_qat:
        print("Preparing Quantization-Aware Training...")
        model = prepare_qat(model)
        use_amp = False  # AMP and QAT don't mix
        print("  AMP disabled (incompatible with QAT)")

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

    # ---- Training logger + viz ----
    logger = TrainingLogger(os.path.join(ckpt_dir, "training_log.json"))

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
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])
            print("  Restored EMA weights")
        print(f"  Resumed at iter {start_iter}, best_psnr={best_psnr:.2f}")

    # ---- Print config ----
    print(f"\nTraining config:")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  crop_size:   {args.crop_size}")
    print(f"  lr:          {args.lr}")
    print(f"  max_iters:   {args.max_iters}")
    print(f"  warmup:      {warmup_iters}")
    print(f"  pixel loss:  {args.loss}")
    print(f"  perceptual:  {args.perceptual_weight}")
    print(f"  fft:         {args.fft_weight} (alpha={args.fft_alpha})")
    print(f"  grad_clip:   {args.grad_clip}")
    print(f"  amp:         {use_amp}")
    print(f"  checkpoints: {ckpt_dir}")
    print()

    # ---- cuDNN benchmark warmup ----
    torch.backends.cudnn.benchmark = True

    # Run one full training step (including all losses) to trigger
    # cudnn.benchmark algorithm selection, then free scratch memory.
    print("  cuDNN benchmark warmup...")
    if gpu_dataset is not None:
        _wi, _wt = gpu_dataset.sample_batch(min(args.batch_size, 4))
    else:
        _wb = next(iter(dataloader))
        _wi, _wt = _wb[0].to(device, non_blocking=True), _wb[1].to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        _wo = model(_wi)
        _wl = criterion(_wo, _wt)
    if perceptual_criterion is not None:
        with torch.amp.autocast("cuda", enabled=False):
            _pl = perceptual_criterion(_wo.float(), _wt.float())
        _wl = _wl + args.perceptual_weight * _pl
    if fft_criterion is not None:
        _fl = fft_criterion(_wo.float(), _wt.float())
        _wl = _wl + args.fft_weight * _fl
    _wl.backward()
    optimizer.zero_grad(set_to_none=True)
    del _wi, _wt, _wo, _wl
    if perceptual_criterion is not None:
        del _pl
    if fft_criterion is not None:
        del _fl
    torch.cuda.empty_cache()
    gc.collect()
    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    curr_gb = torch.cuda.memory_reserved() / 1024**3
    print(f"  cuDNN benchmark done. Peak: {peak_gb:.1f}GB, settled: {curr_gb:.1f}GB")
    torch.cuda.reset_peak_memory_stats()

    # ---- Training state ----
    data_iter = iter(dataloader) if dataloader is not None else None
    loss_sum = 0.0
    pixel_loss_sum = 0.0
    percep_loss_sum = 0.0
    fft_loss_sum = 0.0
    loss_count = 0
    percep_count = 0
    fft_count = 0
    data_time_sum = 0.0
    compute_time_sum = 0.0
    start_time = time.time()
    t_data_start = time.time()

    stop_check = getattr(args, 'stop_check', None)
    stop_check_freq = getattr(args, 'stop_check_freq', 50)
    percep_freq = getattr(args, 'perceptual_freq', 1)

    # CUDA event profiling — measures GPU time per phase, logged every print_freq
    profile_on = device.type == "cuda"
    if profile_on:
        ev_fwd_start = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_aux_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

    # ---- Signal handler for graceful interrupts ----
    def _save_emergency(it):
        """Save checkpoint on interrupt/signal."""
        ckpt_data = _save_checkpoint(model, optimizer, scheduler, scaler,
                                     it, best_psnr, args, ckpt_dir, ema=ema)
        path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(ckpt_data, path)
        torch.save({"params": model.state_dict()},
                    os.path.join(ckpt_dir, "final.pth"))
        logger.flush()
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

        # Get batch
        if gpu_dataset is not None:
            inp_batch, tgt_batch = gpu_dataset.sample_batch(args.batch_size)
        else:
            try:
                inp_batch, tgt_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inp_batch, tgt_batch = next(data_iter)
            inp_batch = inp_batch.to(device, non_blocking=True)
            tgt_batch = tgt_batch.to(device, non_blocking=True)

        # Intensity scaling augmentation (from XLSR)
        if getattr(args, 'intensity_aug', False):
            inp_batch, tgt_batch = apply_intensity_aug(inp_batch, tgt_batch)

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

        # Auxiliary losses run in fp32 (numerically unstable in fp16)
        loss = pixel_loss
        p_loss = None
        f_loss = None

        if perceptual_criterion is not None and (iteration % percep_freq == 0):
            with torch.amp.autocast("cuda", enabled=False):
                p_loss = perceptual_criterion(pred.float(), tgt_batch.float())
            loss = loss + args.perceptual_weight * percep_freq * p_loss

        if fft_criterion is not None:
            f_loss = fft_criterion(pred.float(), tgt_batch.float())
            loss = loss + args.fft_weight * f_loss

        if do_profile:
            ev_aux_end.record()

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

        # EMA update
        if ema is not None:
            ema.update(model)

        if do_profile:
            ev_opt_end.record()

        t_data_start = time.time()
        compute_time_sum += t_data_start - t_compute_start

        loss_sum += loss.item()
        pixel_loss_sum += pixel_loss.item()
        if p_loss is not None:
            percep_loss_sum += p_loss.item()
            percep_count += 1
        if f_loss is not None:
            fft_loss_sum += f_loss.item()
            fft_count += 1
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

            # Build loss detail string
            avg_px = pixel_loss_sum / loss_count
            parts = [f"px={avg_px:.6f}"]
            if percep_count > 0:
                parts.append(f"perc={percep_loss_sum / percep_count:.4f}")
            if fft_count > 0:
                parts.append(f"fft={fft_loss_sum / fft_count:.2e}")
            loss_detail = f"loss={avg_loss:.6f} ({' '.join(parts)})"

            # Log to training logger
            logger.log_train(
                iteration + 1,
                pixel_loss=avg_px,
                perceptual_loss=percep_loss_sum / percep_count if percep_count > 0 else None,
                fft_loss=fft_loss_sum / fft_count if fft_count > 0 else None,
                total_loss=avg_loss,
                lr=lr_now,
            )

            profile_str = ""
            if do_profile:
                torch.cuda.synchronize()
                t_fwd = ev_fwd_start.elapsed_time(ev_fwd_end)
                t_aux = ev_fwd_end.elapsed_time(ev_aux_end)
                t_bwd = ev_aux_end.elapsed_time(ev_bwd_end)
                t_opt = ev_bwd_end.elapsed_time(ev_opt_end)
                t_total = t_fwd + t_aux + t_bwd + t_opt
                profile_str = (
                    f"\n    fwd={t_fwd:.0f} aux={t_aux:.0f} "
                    f"bwd={t_bwd:.0f} opt={t_opt:.0f} "
                    f"total={t_total:.0f}ms")
            samples_per_sec = iters_per_sec * args.batch_size
            print(
                f"  {iteration + 1:5d}/{args.max_iters} | "
                f"{loss_detail} | lr={lr_now:.2e} | "
                f"{iters_per_sec:.1f}it/s ({samples_per_sec:.0f}samp/s) "
                f"ETA:{eta / 60:.0f}m | "
                f"{vram:.1f}GB | data:{data_pct:.0f}%"
                f"{profile_str}")
            loss_sum = 0.0
            pixel_loss_sum = 0.0
            percep_loss_sum = 0.0
            fft_loss_sum = 0.0
            loss_count = 0
            percep_count = 0
            fft_count = 0
            data_time_sum = 0.0
            compute_time_sum = 0.0

        # ---- Validation ----
        if (iteration + 1) % args.val_freq == 0:
            val_dir = args.val_dir if args.val_dir else args.data_dir
            val_model = ema.model if ema is not None else model
            val = validate(val_model, val_dir, device,
                          pixel_criterion=criterion,
                          perceptual_criterion=perceptual_criterion,
                          fft_criterion=fft_criterion,
                          perceptual_weight=args.perceptual_weight,
                          fft_weight=args.fft_weight,
                          crop_size=512)
            psnr = val["psnr"]
            val_loss = val.get("combined_loss", None)

            # Best model by total loss when auxiliary losses are active
            if val_loss is not None and (perceptual_criterion is not None or fft_criterion is not None):
                is_best = best_val_loss is None or val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_psnr = psnr
            else:
                is_best = psnr > best_psnr
                if is_best:
                    best_psnr = psnr

            # Log validation
            logger.log_val(
                iteration + 1,
                psnr=psnr,
                pixel_loss=val.get("pixel_loss"),
                perceptual_loss=val.get("percep_loss"),
                fft_loss=val.get("fft_loss"),
                total_loss=val_loss,
            )
            logger.flush()

            # Print validation summary
            val_detail = f"PSNR={psnr:.2f} dB"
            if "pixel_loss" in val:
                val_detail += f" | px={val['pixel_loss']:.6f}"
            if "percep_loss" in val:
                val_detail += f" perc={val['percep_loss']:.4f}"
            if "fft_loss" in val:
                val_detail += f" fft={val['fft_loss']:.2e}"
            if "combined_loss" in val:
                val_detail += f" total={val['combined_loss']:.6f}"
            best_str = (f"best_loss={best_val_loss:.6f}"
                        if best_val_loss is not None
                        else f"best_psnr={best_psnr:.2f}")
            print(f"  VAL {iteration + 1} ({val['n_frames']}f): "
                  f"{val_detail} {'(BEST)' if is_best else ''} "
                  f"[{best_str}]")

            # Save sample comparison images
            save_val_samples(val_model, val_dir, ckpt_dir, iteration + 1,
                           device, num_samples=3, crop_size=512)

            # Update loss curves chart
            try:
                logger.plot_curves(os.path.join(ckpt_dir, "training_curves.png"))
            except Exception as e:
                print(f"  (chart error: {e})")

            model.train()

            # Save best model
            if is_best:
                best_params = ema.state_dict() if ema is not None else model.state_dict()
                best_path = os.path.join(ckpt_dir, "best.pth")
                torch.save({"params": best_params}, best_path)
                print(f"  Saved best model: {best_path}")

        # ---- Checkpoint ----
        if (iteration + 1) % args.save_freq == 0:
            ckpt_data = _save_checkpoint(model, optimizer, scheduler, scaler,
                                         iteration + 1, best_psnr, args, ckpt_dir, ema=ema)
            ckpt_path = os.path.join(ckpt_dir, f"iter{iteration + 1:06d}.pth")
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Also save latest (for easy resume)
            latest_path = os.path.join(ckpt_dir, "latest.pth")
            torch.save(ckpt_data, latest_path)
            logger.flush()

    # ---- Final save ----
    final_path = os.path.join(ckpt_dir, "final.pth")
    torch.save({"params": model.state_dict()}, final_path)
    logger.flush()

    # Final loss curves
    try:
        logger.plot_curves(os.path.join(ckpt_dir, "training_curves.png"))
    except Exception:
        pass

    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best PSNR: {best_psnr:.2f} dB")

    best_ckpt = os.path.join(ckpt_dir, "best.pth")
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

    # Model type
    parser.add_argument("--model", type=str, default="nafnet",
                        choices=["nafnet", "plain", "unet"],
                        help="Model architecture (nafnet, plain, unet)")

    # NAFNet architecture
    parser.add_argument("--width", type=int, default=64,
                        help="NAFNet channel width (default: 64)")
    parser.add_argument("--middle-blk-num", type=int, default=12,
                        help="Number of middle blocks (default: 12)")
    parser.add_argument("--enc-blk-nums", type=str, default="2,2,4,8",
                        help="Encoder block counts, comma-separated (default: 2,2,4,8)")
    parser.add_argument("--dec-blk-nums", type=str, default="2,2,2,2",
                        help="Decoder block counts, comma-separated (default: 2,2,2,2)")

    # PlainDenoise / UNetDenoise architecture
    parser.add_argument("--nc", type=int, default=64,
                        help="Base channel count (plain/unet)")
    parser.add_argument("--nb", type=int, default=15,
                        help="Number of conv layers (plain)")
    parser.add_argument("--nb-enc", type=str, default="2,2",
                        help="Encoder blocks per level (unet)")
    parser.add_argument("--nb-dec", type=str, default="2,2",
                        help="Decoder blocks per level (unet)")
    parser.add_argument("--nb-mid", type=int, default=2,
                        help="Middle blocks (unet)")

    # EMA
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Enable EMA weights")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    # GPU dataset caching
    parser.add_argument("--cache-on-gpu", action="store_true", default=False,
                        help="Load entire dataset into GPU VRAM")
    parser.add_argument("--intensity-aug", action="store_true", default=False,
                        help="Intensity scaling augmentation (from XLSR)")

    # Sparsity + QAT (advanced optimization stages)
    parser.add_argument("--sparse", action="store_true", default=False,
                        help="Apply 2:4 structured sparsity (requires APEX, pretrained checkpoint)")
    parser.add_argument("--qat", action="store_true", default=False,
                        help="Enable Quantization-Aware Training (INT8 fake quantize)")

    # Model
    parser.add_argument("--pretrained", type=str,
                        default=os.environ.get("PRETRAINED_PATH", ""),
                        help="Pretrained checkpoint path (empty = train from scratch)")
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
                        help="Pixel loss function")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use AMP mixed precision (default: on)")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable AMP mixed precision")

    # Perceptual loss (DISTS)
    parser.add_argument("--perceptual-weight", type=float, default=0.0,
                        help="DISTS perceptual loss weight (0 = disabled, try 0.05)")
    parser.add_argument("--perceptual-freq", type=int, default=1,
                        help="Compute perceptual loss every N iters (default: 1)")

    # FFT loss
    parser.add_argument("--fft-weight", type=float, default=0.0,
                        help="Focal frequency loss weight (0 = disabled, try 0.1)")
    parser.add_argument("--fft-alpha", type=float, default=1.0,
                        help="Focal frequency loss alpha (focal exponent, default: 1.0)")

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
