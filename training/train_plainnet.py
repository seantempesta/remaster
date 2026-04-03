"""
Train PlainDenoise / UNetDenoise for video denoising via distillation.

Features (learned from NAFNet, ECBSR, SPAN, XLSR, DnCNN reference implementations):
- EMA (Exponential Moving Average) weights — free +0.1-0.3 dB PSNR (from SPAN/BasicSR)
- PSNRLoss option — NAFNet's official loss, better than Charbonnier for PSNR
- AdamW with beta2=0.9 — more responsive optimizer (from NAFNet)
- Intensity scaling augmentation — brightness robustness (from XLSR)
- INT8 Quantization-Aware Training (QAT) — native INT8 training
- 2:4 structured sparsity support via APEX ASP — additional 1.3x on TRT
- Reparameterizable training blocks — multi-branch train, single conv infer (ECBSR/SPAN)
- RAM cache, CUDA event profiling, graceful stop, training curves, sample viz

Usage:
    # Local test
    python training/train_plainnet.py --arch unet --nc 64 --nb-mid 2 \
        --data-dir data/train_pairs --max-iters 100 --batch-size 8

    # Full training on Modal (via cloud/modal_train_plainnet.py)
    modal run cloud/modal_train_plainnet.py --arch unet --nc 64 --nb-mid 2

    # With QAT (INT8-aware training)
    python training/train_plainnet.py --arch unet --nc 64 --nb-mid 2 --qat

    # With 2:4 sparsity (requires pretrained dense checkpoint)
    python training/train_plainnet.py --arch unet --nc 64 --nb-mid 2 \
        --sparse --pretrained checkpoints/plainnet/plainnet_best.pth

    # Environment variables for Modal: DATA_DIR, CHECKPOINT_DIR, MAX_ITERS, etc.
"""
import copy
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.plainnet_arch import PlainDenoise, UNetDenoise, count_params
from lib.paths import PROJECT_ROOT, CHECKPOINTS_DIR
from training.losses import (
    CharbonnierLoss, PSNRLoss, FocalFrequencyLoss, DISTSPerceptualLoss,
    build_pixel_criterion,
)
from training.dataset import PairedFrameDataset, GPUCachedDataset
from training.viz import TrainingLogger, save_val_samples


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Prefetcher — overlap data transfer with GPU compute
# ──────────────────────────────────────────────────────────────────────────────

class CUDAPrefetcher:
    """Prefetches data to GPU on a separate CUDA stream.

    Overlaps CPU→GPU data transfer with the previous batch's GPU compute.
    This hides transfer latency and keeps the GPU fed continuously.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self._iter = None
        self.inp = None
        self.tgt = None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            inp, tgt = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            inp, tgt = next(self._iter)

        with torch.cuda.stream(self.stream):
            self.inp = inp.to(self.device, non_blocking=True)
            self.tgt = tgt.to(self.device, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inp, tgt = self.inp, self.tgt
        self._preload()
        return inp, tgt


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# From SPAN/BasicSR — maintains a shadow copy of weights that tracks the
# exponential moving average. Validated by both SPAN and SwinIR (KAIR) at 0.999.
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # after each optimizer step:
        ema.update(model)
        # for validation/export:
        ema.model  # the EMA model
    """

    def __init__(self, model, decay=0.999):
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
# Weight initialization
# From XLSR: smaller-than-Kaiming init helps tiny models avoid early instability.
# From KAIR (DnCNN): orthogonal init with small gain.
# ──────────────────────────────────────────────────────────────────────────────

def init_weights_small(model, scale=0.1):
    """Initialize conv weights with scaled Kaiming normal (from XLSR).

    Standard Kaiming normal has std = sqrt(2/fan_out). We multiply by `scale`
    (default 0.1) so initial outputs are small, preventing early instability
    in tiny models. The residual connection means the model starts as near-identity.
    """
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
# QAT Setup
# ──────────────────────────────────────────────────────────────────────────────

def prepare_qat(model):
    """Prepare model for Quantization-Aware Training.

    Inserts FakeQuantize modules that simulate INT8 rounding during forward.
    Uses symmetric per-channel quantization for weights, per-tensor for activations
    — matches TensorRT's INT8 scheme.
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
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2:4 Structured Sparsity (APEX ASP)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_sparsity(model, optimizer):
    """Initialize 2:4 structured sparsity via APEX ASP.

    Prunes a pretrained dense model to 2:4 pattern (50% of weights zeroed).
    Ampere tensor cores skip zero computations → ~1.3-1.4x speedup on top of INT8.

    Requires: apex installed with sparsity support.
    Call AFTER loading pretrained weights, BEFORE training.
    """
    from apex.contrib.sparsity import ASP
    ASP.prune_trained_model(model, optimizer)
    print("  2:4 structured sparsity applied via APEX ASP")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args):
    """Build model from args."""
    if args.arch == "plain":
        model = PlainDenoise(
            in_nc=3, nc=args.nc, nb=args.nb,
            use_bn=True, deploy=False,
        )
    elif args.arch == "unet":
        nb_enc = tuple(int(x) for x in args.nb_enc.split(","))
        nb_dec = tuple(int(x) for x in args.nb_dec.split(","))
        model = UNetDenoise(
            in_nc=3, nc=args.nc,
            nb_enc=nb_enc, nb_dec=nb_dec, nb_mid=args.nb_mid,
            use_bn=True, deploy=False,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Intensity scaling augmentation (from XLSR)
# ──────────────────────────────────────────────────────────────────────────────

def apply_intensity_aug(inp, tgt, scales=(0.5, 0.7, 1.0)):
    """Randomly scale intensity of input+target pair (same scale for both).

    From XLSR: simulates brightness variation. Cheap augmentation that
    improves robustness to different scene brightnesses in video content.
    Applied on GPU tensors for zero CPU overhead.
    """
    if len(scales) <= 1:
        return inp, tgt
    idx = torch.randint(len(scales), (1,)).item()
    scale = scales[idx]
    if scale != 1.0:
        inp = inp * scale
        tgt = tgt * scale
    return inp, tgt


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_dir, device, pixel_criterion=None,
             perceptual_criterion=None, fft_criterion=None,
             perceptual_weight=0.0, fft_weight=0.0, crop_size=512):
    """Validate on held-out frames."""
    import cv2

    model.eval()
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    psnrs, pixel_losses, percep_losses, fft_losses = [], [], [], []

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
        cs = min(crop_size, h, w)
        top, left = (h - cs) // 2, (w - cs) // 2
        inp = inp[top:top+cs, left:left+cs]
        tgt = tgt[top:top+cs, left:left+cs]

        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(tgt.transpose(2, 0, 1)).unsqueeze(0).to(device)

        out_t = model(inp_t).clamp(0, 1)

        mse = ((out_t - tgt_t) ** 2).mean().item()
        psnrs.append(10 * math.log10(1.0 / mse) if mse > 0 else 100.0)

        if pixel_criterion is not None:
            pixel_losses.append(pixel_criterion(out_t, tgt_t).item())
        if perceptual_criterion is not None:
            percep_losses.append(perceptual_criterion(out_t.float(), tgt_t.float()).item())
        if fft_criterion is not None:
            fft_losses.append(fft_criterion(out_t.float(), tgt_t.float()).item())

    result = {"psnr": np.mean(psnrs) if psnrs else 0.0, "n_frames": len(psnrs)}
    if pixel_losses:
        result["pixel_loss"] = np.mean(pixel_losses)
    if percep_losses:
        result["percep_loss"] = np.mean(percep_losses)
    if fft_losses:
        result["fft_loss"] = np.mean(fft_losses)

    total = result.get("pixel_loss", 0)
    if percep_losses:
        total += perceptual_weight * result["percep_loss"]
    if fft_losses:
        total += fft_weight * result["fft_loss"]
    result["combined_loss"] = total
    return result


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
    model = build_model(args)
    params_k = count_params(model) / 1e3
    arch_desc = (f"{args.arch} nc={args.nc} "
                 + (f"nb={args.nb}" if args.arch == "plain" else f"mid={args.nb_mid}"))
    print(f"Architecture: {arch_desc}, {params_k:.1f}K params")

    # Small-scale initialization (from XLSR) — helps tiny models
    if not args.pretrained and not args.resume:
        init_weights_small(model, scale=0.1)
        print("  Applied XLSR-style small initialization (0.1x Kaiming)")

    # ---- Load pretrained ----
    if args.resume and os.path.exists(args.resume):
        pass  # loaded below with optimizer state
    elif args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  Skipped {len(unexpected)} unexpected keys")
        if missing:
            print(f"  {len(missing)} missing keys (randomly initialized)")
        if not missing and not unexpected:
            print("  Loaded pretrained weights (exact match)")
    else:
        print("Training from scratch (no pretrained weights)")

    # ---- QAT ----
    use_qat = getattr(args, 'qat', False)
    if use_qat:
        print("Preparing Quantization-Aware Training (INT8 fake quantize)...")
        model = prepare_qat(model)
        print("  QAT prepared: FakeQuantize nodes inserted")
        model = model.to(device)
        use_amp = False  # AMP and QAT don't mix
    else:
        model = model.to(device)
        use_amp = args.amp and device.type == "cuda"

    model.train()
    if device.type == "cuda":
        print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # ---- EMA (from SPAN/BasicSR, validated by SwinIR) ----
    use_ema = getattr(args, 'ema', True)
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=getattr(args, 'ema_decay', 0.999))
        print(f"EMA: ON (decay={ema.decay})")
    else:
        print("EMA: OFF")

    # ---- Dataset ----
    cache_on_gpu = getattr(args, 'cache_on_gpu', False)
    cache_in_ram = getattr(args, 'cache_in_ram', False)
    gpu_dataset = None
    dataloader = None

    if cache_on_gpu and device.type == "cuda":
        # GPU-resident dataset: entire dataset in VRAM, no DataLoader needed
        gpu_dataset = GPUCachedDataset(
            args.data_dir, crop_size=args.crop_size, device=device,
        )
        print(f"Data pipeline: GPU-cached ({gpu_dataset.n} pairs in VRAM)")
    else:
        # CPU DataLoader path (for local testing or CPU training)
        dataset = PairedFrameDataset(
            args.data_dir, crop_size=args.crop_size,
            augment=True, cache_in_ram=cache_in_ram,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            drop_last=True, persistent_workers=args.num_workers > 0,
        )
        print(f"Data pipeline: CPU DataLoader (workers={args.num_workers}, "
              f"cache_ram={cache_in_ram})")

    # ---- Losses ----
    criterion = build_pixel_criterion(args.loss)
    print(f"Pixel loss: {args.loss}")

    perceptual_criterion = None
    if args.perceptual_weight > 0:
        perceptual_criterion = DISTSPerceptualLoss().to(device)
        print(f"Perceptual loss: DISTS, weight={args.perceptual_weight}")

    fft_criterion = None
    if args.fft_weight > 0:
        fft_criterion = FocalFrequencyLoss(alpha=args.fft_alpha)
        print(f"FFT loss: Focal Frequency, weight={args.fft_weight}")

    # ---- Optimizer (beta2=0.9 from NAFNet — more responsive for small models) ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.9),
        fused=device.type == "cuda" and not use_qat,
    )

    # ---- 2:4 Sparsity (must be after optimizer creation, before LR scheduler) ----
    use_sparse = getattr(args, 'sparse', False)
    if use_sparse:
        print("Applying 2:4 structured sparsity (APEX ASP)...")
        model = prepare_sparsity(model, optimizer)

    # ---- LR Scheduler: cosine with warmup (from NAFNet) ----
    warmup_iters = args.warmup_iters
    def lr_lambda(step):
        if step < warmup_iters:
            return max(step / max(warmup_iters, 1), 1e-8)
        progress = (step - warmup_iters) / max(args.max_iters - warmup_iters, 1)
        return max(args.eta_min / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Checkpoint dir ----
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training logger ----
    logger = TrainingLogger(os.path.join(ckpt_dir, "training_log.json"))

    # ---- AMP scaler ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {'ON' if use_amp else 'OFF'}")
    if use_qat:
        print("QAT: ON")
    if use_sparse:
        print("Sparsity: ON (2:4 structured)")

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
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])
            print("  Restored EMA weights")
        start_iter = ckpt["iteration"]
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"  Resumed at iter {start_iter}")

    # ---- Print config ----
    print(f"\nTraining config:")
    print(f"  arch:        {arch_desc}")
    print(f"  params:      {params_k:.1f}K")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  crop_size:   {args.crop_size}")
    print(f"  lr:          {args.lr}  betas=(0.9, 0.9)")
    print(f"  max_iters:   {args.max_iters}")
    print(f"  intensity_aug: {getattr(args, 'intensity_aug', True)}")
    print()

    # ---- cuDNN benchmark warmup ----
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        print("  cuDNN warmup...")
        if gpu_dataset is not None:
            _wi, _wt = gpu_dataset.sample_batch(min(args.batch_size, 4))
        else:
            _wb = next(iter(dataloader))
            _wi = _wb[0].to(device, non_blocking=True)
            _wt = _wb[1].to(device, non_blocking=True)
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
            try:
                del _pl
            except NameError:
                pass
        if fft_criterion is not None:
            try:
                del _fl
            except NameError:
                pass
        torch.cuda.empty_cache()
        gc.collect()
        peak_gb = torch.cuda.max_memory_reserved() / 1024**3
        curr_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"  cuDNN done. Peak: {peak_gb:.1f}GB, settled: {curr_gb:.1f}GB")
        torch.cuda.reset_peak_memory_stats()

    # ---- Data iterator setup ----
    if gpu_dataset is not None:
        data_iter = None  # not needed, we call gpu_dataset.sample_batch()
    elif device.type == "cuda" and dataloader is not None:
        prefetcher = CUDAPrefetcher(dataloader, device)
        data_iter = iter(prefetcher)
        print("CUDA prefetcher: ON")
    else:
        data_iter = iter(dataloader)
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
    do_intensity_aug = getattr(args, 'intensity_aug', True)

    # CUDA event profiling — per-phase GPU time on print iters
    profile_on = device.type == "cuda"
    if profile_on:
        ev_fwd_start = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

    # ---- Signal handler for graceful interrupts ----
    _current_iter = start_iter
    interrupted = False

    def _save_emergency(it):
        ckpt_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "iteration": it,
            "best_psnr": best_psnr,
            "args": {k: v for k, v in vars(args).items() if not callable(v)},
        }
        if ema is not None:
            ckpt_data["ema"] = ema.state_dict()
        path = os.path.join(ckpt_dir, "plainnet_latest.pth")
        torch.save(ckpt_data, path)
        # Save EMA weights as the "best" inference weights
        params_key = ema.state_dict() if ema is not None else model.state_dict()
        torch.save({"params": params_key},
                    os.path.join(ckpt_dir, "plainnet_final.pth"))
        logger.flush()
        print(f"  Saved checkpoint at iter {it}")

    def _handle_interrupt(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print(f"\n  SIGINT at iter {_current_iter}. Saving...")
            _save_emergency(_current_iter)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handle_interrupt)

    # ---- Main training loop ----
    for iteration in range(start_iter, args.max_iters):
        _current_iter = iteration

        # Graceful stop check (Modal Dict)
        if stop_check and (iteration + 1) % stop_check_freq == 0:
            if stop_check():
                print(f"\n  STOP signal at iter {iteration + 1}.")
                _save_emergency(iteration + 1)
                return

        # Get batch
        if gpu_dataset is not None:
            inp_batch, tgt_batch = gpu_dataset.sample_batch(args.batch_size)
        else:
            inp_batch, tgt_batch = next(data_iter)
            if device.type != "cuda":
                inp_batch = inp_batch.to(device)
                tgt_batch = tgt_batch.to(device)

        # Intensity scaling augmentation (from XLSR)
        if do_intensity_aug:
            inp_batch, tgt_batch = apply_intensity_aug(inp_batch, tgt_batch)

        t_compute_start = time.time()
        data_time_sum += t_compute_start - t_data_start

        is_print_iter = (iteration + 1) % args.print_freq == 0

        # Profile on print iterations for per-phase breakdown
        do_profile = profile_on and is_print_iter
        if do_profile:
            torch.cuda.synchronize()
            ev_fwd_start.record()

        # Forward
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(inp_batch)
            pixel_loss = criterion(pred, tgt_batch)

        if do_profile:
            ev_fwd_end.record()

        # Auxiliary losses (fp32)
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

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if do_profile:
            ev_bwd_end.record()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA update (after optimizer step)
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
        if is_print_iter:
            avg_loss = loss_sum / loss_count
            elapsed = time.time() - start_time
            iters_per_sec = (iteration + 1 - start_iter) / elapsed
            samples_per_sec = iters_per_sec * args.batch_size
            eta = (args.max_iters - iteration - 1) / max(iters_per_sec, 0.01)
            lr_now = optimizer.param_groups[0]["lr"]
            vram_gb = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0

            # Wall-clock time breakdown
            total_wall = data_time_sum + compute_time_sum
            data_pct = data_time_sum / total_wall * 100 if total_wall > 0 else 0
            compute_pct = compute_time_sum / total_wall * 100 if total_wall > 0 else 0

            avg_px = pixel_loss_sum / loss_count
            parts = [f"px={avg_px:.6f}"]
            if percep_count > 0:
                parts.append(f"perc={percep_loss_sum/percep_count:.4f}")
            if fft_count > 0:
                parts.append(f"fft={fft_loss_sum/fft_count:.2e}")
            loss_detail = f"loss={avg_loss:.6f} ({' '.join(parts)})"

            # Per-iteration GPU time from CUDA events (this iteration only)
            gpu_detail = ""
            if do_profile:
                torch.cuda.synchronize()
                t_fwd = ev_fwd_start.elapsed_time(ev_fwd_end)
                t_bwd = ev_fwd_end.elapsed_time(ev_bwd_end)
                t_opt = ev_bwd_end.elapsed_time(ev_opt_end)
                t_gpu_total = t_fwd + t_bwd + t_opt
                wall_per_iter_ms = total_wall / loss_count * 1000 if loss_count > 0 else 1
                iter_gpu_pct = t_gpu_total / wall_per_iter_ms * 100 if wall_per_iter_ms > 0 else 0
                gpu_detail = (f"  fwd={t_fwd:.0f} bwd={t_bwd:.0f} "
                             f"opt={t_opt:.0f} = {t_gpu_total:.0f}ms/iter "
                             f"({iter_gpu_pct:.0f}% of {wall_per_iter_ms:.0f}ms wall)")

            logger.log_train(
                iteration + 1, pixel_loss=avg_px,
                perceptual_loss=percep_loss_sum/percep_count if percep_count > 0 else None,
                fft_loss=fft_loss_sum/fft_count if fft_count > 0 else None,
                total_loss=avg_loss, lr=lr_now,
            )

            print(f"  {iteration+1:5d}/{args.max_iters} | "
                  f"{loss_detail} | lr={lr_now:.2e} | "
                  f"{iters_per_sec:.1f}it/s ({samples_per_sec:.0f}samp/s) "
                  f"ETA:{eta/60:.0f}m | {vram_gb:.1f}GB | data:{data_pct:.0f}%"
                  f"{gpu_detail}")

            loss_sum = pixel_loss_sum = percep_loss_sum = fft_loss_sum = 0.0
            loss_count = percep_count = fft_count = 0
            data_time_sum = compute_time_sum = 0.0

        # ---- Validation (use EMA model if available) ----
        if (iteration + 1) % args.val_freq == 0:
            val_model = ema.model if ema is not None else model
            val_dir = args.val_dir if args.val_dir else args.data_dir
            val = validate(val_model, val_dir, device,
                          pixel_criterion=criterion,
                          perceptual_criterion=perceptual_criterion,
                          fft_criterion=fft_criterion,
                          perceptual_weight=args.perceptual_weight,
                          fft_weight=args.fft_weight, crop_size=512)
            psnr = val["psnr"]
            val_loss = val.get("combined_loss")

            if val_loss is not None and (perceptual_criterion or fft_criterion):
                is_best = best_val_loss is None or val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_psnr = psnr
            else:
                is_best = psnr > best_psnr
                if is_best:
                    best_psnr = psnr

            logger.log_val(iteration + 1, psnr=psnr,
                          pixel_loss=val.get("pixel_loss"),
                          perceptual_loss=val.get("percep_loss"),
                          fft_loss=val.get("fft_loss"),
                          total_loss=val_loss)
            logger.flush()

            val_detail = f"PSNR={psnr:.2f} dB"
            if "combined_loss" in val:
                val_detail += f" total={val['combined_loss']:.6f}"
            ema_tag = " (EMA)" if ema is not None else ""
            print(f"  VAL{ema_tag} {iteration+1} ({val['n_frames']}f): "
                  f"{val_detail} {'(BEST)' if is_best else ''}")

            save_val_samples(val_model, val_dir, ckpt_dir, iteration + 1,
                           device, num_samples=3, crop_size=512)
            try:
                logger.plot_curves(os.path.join(ckpt_dir, "training_curves.png"))
            except Exception:
                pass

            model.train()

            if is_best:
                # Save EMA weights as the inference model (better than raw weights)
                best_params = ema.state_dict() if ema is not None else model.state_dict()
                best_path = os.path.join(ckpt_dir, "plainnet_best.pth")
                torch.save({"params": best_params}, best_path)
                print(f"  Saved best: {best_path}")

        # ---- Checkpoint ----
        if (iteration + 1) % args.save_freq == 0:
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "iteration": iteration + 1,
                "best_psnr": best_psnr,
                "args": {k: v for k, v in vars(args).items() if not callable(v)},
            }
            if ema is not None:
                ckpt_data["ema"] = ema.state_dict()
            ckpt_path = os.path.join(ckpt_dir, f"plainnet_iter{iteration+1:06d}.pth")
            torch.save(ckpt_data, ckpt_path)
            latest_path = os.path.join(ckpt_dir, "plainnet_latest.pth")
            torch.save(ckpt_data, latest_path)
            logger.flush()
            print(f"  Saved checkpoint: {ckpt_path}")

    # ---- Final save ----
    # Save EMA weights as the final inference model
    final_params = ema.state_dict() if ema is not None else model.state_dict()
    final_path = os.path.join(ckpt_dir, "plainnet_final.pth")
    torch.save({"params": final_params}, final_path)
    logger.flush()

    try:
        logger.plot_curves(os.path.join(ckpt_dir, "training_curves.png"))
    except Exception:
        pass

    print(f"\nTraining complete. Final: {final_path}")
    print(f"Best PSNR: {best_psnr:.2f} dB")

    # Export QAT ONNX if requested
    if use_qat and getattr(args, 'export_qat_onnx', False):
        onnx_path = os.path.join(ckpt_dir, "plainnet_int8.onnx")
        best_path = os.path.join(ckpt_dir, "plainnet_best.pth")
        if os.path.exists(best_path):
            best_model = build_model(args)
            best_model = prepare_qat(best_model)
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            best_model.load_state_dict(ckpt["params"])
            best_model.eval()
            quantized = torch.ao.quantization.convert(best_model, inplace=False)
            dummy = torch.randn(1, 3, 1088, 1920)
            torch.onnx.export(
                quantized, dummy, onnx_path, opset_version=18,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {2: "height", 3: "width"},
                              "output": {2: "height", 3: "width"}},
            )
            print(f"Exported QAT ONNX: {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PlainDenoise/UNetDenoise")

    # Architecture
    parser.add_argument("--arch", type=str, default="unet",
                        choices=["plain", "unet"])
    parser.add_argument("--nc", type=int, default=64,
                        help="Base channel count")
    parser.add_argument("--nb", type=int, default=15,
                        help="Number of conv layers (PlainDenoise)")
    parser.add_argument("--nb-enc", type=str, default="2,2",
                        help="Encoder blocks per level (UNet)")
    parser.add_argument("--nb-dec", type=str, default="2,2",
                        help="Decoder blocks per level (UNet)")
    parser.add_argument("--nb-mid", type=int, default=2,
                        help="Middle blocks (UNet)")

    # Training enhancements
    parser.add_argument("--ema", action="store_true", default=True,
                        help="Enable EMA weights (default: on)")
    parser.add_argument("--no-ema", action="store_false", dest="ema")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--intensity-aug", action="store_true", default=True,
                        help="Intensity scaling augmentation (from XLSR)")
    parser.add_argument("--no-intensity-aug", action="store_false", dest="intensity_aug")
    parser.add_argument("--cache-on-gpu", action="store_true", default=False,
                        help="Load entire dataset into GPU VRAM (needs ~16GB, best for cloud)")
    parser.add_argument("--cache-in-ram", action="store_true", default=False,
                        help="Cache all images in RAM (fallback if GPU cache disabled)")

    # QAT / Sparsity
    parser.add_argument("--qat", action="store_true",
                        help="Enable Quantization-Aware Training (INT8)")
    parser.add_argument("--export-qat-onnx", action="store_true",
                        help="Export QAT model to ONNX after training")
    parser.add_argument("--sparse", action="store_true",
                        help="Enable 2:4 structured sparsity (requires APEX, pretrained)")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("DATA_DIR", "data/train_pairs"))
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("BATCH_SIZE", "32")))
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    # Model
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    # Training
    parser.add_argument("--max-iters", type=int,
                        default=int(os.environ.get("MAX_ITERS", "25000")))
    parser.add_argument("--lr", type=float,
                        default=float(os.environ.get("LR", "2e-4")))
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="charbonnier",
                        choices=["charbonnier", "psnr", "l1"])
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")

    # Perceptual + FFT losses
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--perceptual-freq", type=int, default=1)
    parser.add_argument("--fft-weight", type=float, default=0.0)
    parser.add_argument("--fft-alpha", type=float, default=1.0)

    # Logging
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.environ.get("CHECKPOINT_DIR",
                            str(CHECKPOINTS_DIR / "plainnet")))
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--val-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
