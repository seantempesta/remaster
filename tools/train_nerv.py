"""
HNeRV training for video denoising via spectral bias.

Fits an HNeRV model to a directory of video frames. The network's spectral
bias causes it to learn clean structure before high-frequency noise, producing
denoised output when capacity/regularization is tuned properly.

Logs per-epoch metrics for noise-fit curve analysis:
  - PSNR vs input (reconstruction quality -- how well the model fits)
  - High-frequency energy ratio (FFT-based -- detects when noise is being memorized)
  - Late-layer weight norm (tracks regularization effectiveness)

Saves checkpoints at intervals for post-hoc analysis of the denoising sweet spot.

Usage:
  python tools/train_nerv.py --data-dir E:/upscale-data/nerv-test/clip_02 --epochs 300
  python tools/train_nerv.py --data-dir E:/upscale-data/nerv-test/clip_02 --epochs 300 --model-size 0.75
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Dataset
# =============================================================================

class FrameDataset(Dataset):
    """Load frames from a directory of images."""

    def __init__(self, frame_dir, height=1080, width=1920):
        self.paths = sorted([
            os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.height = height
        self.width = width
        self.transform = transforms.ToTensor()
        assert len(self.paths) > 0, f"No images found in {frame_dir}"
        # Probe first frame for resolution
        first = Image.open(self.paths[0]).convert("RGB")
        self.height, self.width = first.height, first.width
        print(f"Dataset: {len(self.paths)} frames at {self.width}x{self.height}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)  # [3, H, W] in [0, 1]
        norm_idx = float(idx) / len(self.paths)
        return img, norm_idx, idx


# =============================================================================
# Model: Simplified HNeRV (no quantization, no SFT, no boost)
# =============================================================================

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm, FP16-safe."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x.to(orig_dtype)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V1 block for the encoder."""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtEncoder(nn.Module):
    """Lightweight ConvNeXt encoder for HNeRV."""

    def __init__(self, in_chans=3, dims=(64, 64), strides=(5, 2, 2, 2, 2),
                 blocks_per_stage=1):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = in_chans
        for i, (dim, stride) in enumerate(zip(
            [dims[0]] * (len(strides) - 1) + [dims[-1]], strides
        )):
            ds = nn.Sequential(
                nn.Conv2d(ch, dim, kernel_size=stride + 1, stride=stride,
                          padding=stride // 2),
                LayerNorm2d(dim),
            )
            self.downsamples.append(ds)
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim) for _ in range(blocks_per_stage)]
            )
            self.stages.append(stage)
            ch = dim

    def forward(self, x, return_features=False):
        features = []
        for ds, stage in zip(self.downsamples, self.stages):
            x = ds(x)
            x = stage(x)
            if return_features:
                features.append(x)
        if return_features:
            return x, features
        return x


class UpConvBlock(nn.Module):
    """Upsample via Conv + PixelShuffle + Norm + Activation."""

    def __init__(self, in_ch, out_ch, stride, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * stride * stride, kernel_size,
                              1, kernel_size // 2)
        self.ps = nn.PixelShuffle(stride)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.ps(self.conv(x))))


# -- SFT (Spatial Feature Transform) for conditional decoding -----------------

class SFTLayer(nn.Module):
    """Per-frame scale+shift modulation: output = x * (scale + 1) + shift."""

    def __init__(self, cond_ch, out_ch):
        super().__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv2d(cond_ch, cond_ch, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cond_ch, out_ch, 1),
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(cond_ch, cond_ch, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cond_ch, out_ch, 1),
        )

    def forward(self, x, cond):
        return x * (self.scale_conv(cond) + 1) + self.shift_conv(cond)


class SFTUpConvBlock(nn.Module):
    """UpConv + SFT: upsample then modulate with temporal embedding."""

    def __init__(self, in_ch, out_ch, stride, kernel_size=3, cond_ch=32):
        super().__init__()
        self.upconv = UpConvBlock(in_ch, out_ch, stride, kernel_size)
        self.sft = SFTLayer(cond_ch, out_ch)

    def forward(self, x, cond):
        return self.sft(self.upconv(x), cond)


class PositionEncoding(nn.Module):
    """Sinusoidal positional encoding for frame index [0, 1]."""

    def __init__(self, base=1.25, levels=40):
        super().__init__()
        freqs = base ** torch.arange(levels) * math.pi
        self.register_buffer("freqs", freqs)
        self.embed_length = levels * 2

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        vals = t * self.freqs
        return torch.cat([torch.sin(vals), torch.cos(vals)], dim=-1)


# -- Model ---------------------------------------------------------------------

class HNeRVSimple(nn.Module):
    """
    HNeRV with SFT conditional decoding.

    Each decoder stage is modulated by a per-frame temporal embedding
    (from the frame's position in the video). Shared decoder weights handle
    general visual vocabulary; SFT personalizes each frame's reconstruction.
    """

    def __init__(self, height=1080, width=1920, enc_strides=(5, 3, 2, 2, 2),
                 dec_strides=(5, 3, 2, 2, 2), dec_blks=(1, 1, 2, 2, 2),
                 enc_dim=16, fc_dim=170, reduce=1.2, lower_width=12,
                 grad_checkpoint=False, cond_ch=32, enc_blocks=1):
        super().__init__()
        self.height = height
        self.width = width
        self.grad_checkpoint = grad_checkpoint

        total_enc = math.prod(enc_strides)
        total_dec = math.prod(dec_strides)
        self.fc_h = total_enc // total_dec
        self.fc_w = self.fc_h
        assert total_enc >= total_dec

        # Encoder
        self.encoder = ConvNeXtEncoder(
            in_chans=3, dims=(64, enc_dim), strides=enc_strides,
            blocks_per_stage=enc_blocks,
        )

        # Temporal conditioning
        self.pe = PositionEncoding(base=1.25, levels=40)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.pe.embed_length, cond_ch * 2),
            nn.GELU(),
            nn.Linear(cond_ch * 2, cond_ch),
        )
        self.cond_ch = cond_ch

        # Projection
        proj_out = fc_dim * self.fc_h * self.fc_w
        self.proj = nn.Sequential(
            nn.Conv2d(enc_dim, proj_out, 1),
            nn.GELU(),
        )

        # Decoder: SFT-conditioned UpConv blocks
        # Kernel pattern: min 3x3 for all blocks (was 1x1 for first block)
        # 3x3 head for spatial mixing in final output
        self.decoder = nn.ModuleList()
        ch = fc_dim
        for i, stride in enumerate(dec_strides):
            new_ch = max(int(ch / reduce), lower_width)
            ks = min(3 + 2 * i, 5)  # [3, 5, 5, 5, 5] -- all have spatial mixing
            n_blks = dec_blks[i] if i < len(dec_blks) else 1
            for j in range(n_blks):
                s = stride if j == 0 else 1  # only first block upsamples
                self.decoder.append(SFTUpConvBlock(ch, new_ch, s, ks, cond_ch))
                ch = new_ch

        self.head = nn.Conv2d(ch, 3, 3, 1, 1)  # 3x3 with padding

    def forward(self, img, norm_idx=None):
        B = img.shape[0]

        # Encode
        embed = self.encoder(img)

        # Temporal conditioning
        if norm_idx is None:
            norm_idx = torch.zeros(B, device=img.device)
        cond = self.cond_mlp(self.pe(norm_idx))  # (B, cond_ch)
        cond = cond.view(B, self.cond_ch, 1, 1)

        # Project
        proj = self.proj(embed)
        _, _, eh, ew = proj.shape
        if self.fc_h > 1 or self.fc_w > 1:
            proj = proj.view(B, -1, self.fc_h, self.fc_w, eh, ew)
            proj = proj.permute(0, 1, 4, 2, 5, 3).reshape(
                B, -1, self.fc_h * eh, self.fc_w * ew
            )

        # Decode with SFT conditioning
        x = proj
        for block in self.decoder:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, cond, use_reentrant=False)
            else:
                x = block(x, cond)

        out = torch.tanh(self.head(x)) * 0.5 + 0.5  # [0, 1] with steeper gradients
        return out[:, :, :self.height, :self.width]

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def encoder_params(self):
        return sum(p.numel() for p in self.encoder.parameters())

    @property
    def decoder_params(self):
        dec = sum(p.numel() for p in self.decoder.parameters())
        dec += sum(p.numel() for p in self.head.parameters())
        dec += sum(p.numel() for p in self.proj.parameters())
        return dec


# =============================================================================
# Loss functions
# =============================================================================

def compute_loss(pred, target, loss_type="l1_freq", pixel_weight=10.0):
    """Compute training loss.

    l1_freq: L1 pixel loss + FFT frequency loss. pixel_weight controls balance.
    Default pixel_weight=10 makes L1 and FFT roughly equal magnitude.
    Lower pixel_weight emphasizes frequency reconstruction (sharper output).
    """
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    elif loss_type == "l2":
        return F.mse_loss(pred, target)
    elif loss_type == "l1_freq":
        # L1 pixel loss
        l1 = F.l1_loss(pred, target)
        # FFT frequency loss (encourages high-frequency detail reconstruction)
        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1); del pred_fft
        target_freq = torch.stack([target_fft.real, target_fft.imag], -1); del target_fft
        freq_loss = F.l1_loss(pred_freq, target_freq)
        del pred_freq, target_freq
        return pixel_weight * l1 + freq_loss
    elif loss_type == "fusion6":
        # Reference HNeRV loss: 0.7*L1 + 0.3*(1-SSIM)
        from pytorch_msssim import ssim
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim(pred.float(), target.float(), data_range=1, size_average=True)
        return 0.7 * l1 + 0.3 * (1 - ssim_val)
    elif loss_type == "l1_ssim_freq":
        # L1 + SSIM + FFT frequency loss (combines structural + frequency awareness)
        from pytorch_msssim import ssim
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim(pred.float(), target.float(), data_range=1, size_average=True)
        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1); del pred_fft
        target_freq = torch.stack([target_fft.real, target_fft.imag], -1); del target_fft
        freq_loss = F.l1_loss(pred_freq, target_freq)
        del pred_freq, target_freq
        return 10 * (0.7 * l1 + 0.3 * (1 - ssim_val)) + freq_loss
    elif loss_type == "fusion10_freq":
        from pytorch_msssim import ms_ssim
        l1 = F.l1_loss(pred, target)
        msssim = ms_ssim(pred.float(), target.float(), data_range=1, size_average=True)
        spatial = 0.7 * l1 + 0.3 * (1 - msssim)
        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        pred_freq = torch.stack([pred_fft.real, pred_fft.imag], -1); del pred_fft
        target_freq = torch.stack([target_fft.real, target_fft.imag], -1); del target_fft
        freq_loss = F.l1_loss(pred_freq, target_freq)
        del pred_freq, target_freq
        return 60 * spatial + freq_loss
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(pred, target):
    """PSNR between two [0,1] tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return -10 * math.log10(mse.item())


def compute_hf_energy(img_tensor):
    """High-frequency energy ratio via FFT.

    Returns the fraction of total spectral energy above the median frequency.
    Higher = more high-frequency content (detail or noise).
    Computed on the luminance channel for efficiency.
    """
    # Convert to grayscale (FP32 required for non-power-of-2 cuFFT)
    img_tensor = img_tensor.float()
    gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
    # FFT
    fft = torch.fft.fft2(gray)
    magnitude = torch.abs(fft)
    # Shift zero-frequency to center
    magnitude = torch.fft.fftshift(magnitude)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Create distance map from center
    y = torch.arange(h, device=magnitude.device).float() - cy
    x = torch.arange(w, device=magnitude.device).float() - cx
    dist = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    # Median frequency as threshold
    max_freq = min(cy, cx)
    threshold = max_freq * 0.25  # top 75% of frequencies = "high frequency"
    total_energy = (magnitude ** 2).sum()
    hf_energy = (magnitude[dist > threshold] ** 2).sum()
    return (hf_energy / (total_energy + 1e-8)).item()


def compute_late_layer_norm(model):
    """L2 norm of the last 2 decoder UpConv layer weights.

    Tracks whether late layers are growing (memorizing noise) or staying
    small (regularization is working).
    """
    norms = []
    for block in list(model.decoder)[-2:]:
        for p in block.parameters():
            norms.append(p.data.norm(2).item())
    return sum(norms) / len(norms) if norms else 0.0


# =============================================================================
# Visualization
# =============================================================================

def _to_uint8(t):
    """(3, H, W) float [0,1] tensor -> (H, W, 3) uint8 numpy."""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _apply_colormap(arr_2d, vmin=None, vmax=None):
    """Apply plasma colormap to a 2D numpy array. Returns (H, W, 3) uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    if vmin is None:
        vmin = arr_2d.min()
    if vmax is None:
        vmax = arr_2d.max()
    norm = (arr_2d - vmin) / (vmax - vmin + 1e-8)
    return (cm.plasma(norm.clip(0, 1))[:, :, :3] * 255).astype(np.uint8)


def vis_comparison(inp, out):
    """Side-by-side: Input | Reconstruction. Returns (H, W*2, 3) uint8."""
    return np.concatenate([_to_uint8(inp), _to_uint8(out)], axis=1)


def vis_residual(inp, out, gain=10.0):
    """Amplified residual (input - output). Gray=no change, color=removed content."""
    diff = (inp.float() - out.float()) * gain + 0.5
    return _to_uint8(diff)


def vis_fft_heatmap(img_tensor):
    """Log-magnitude FFT heatmap with plasma colormap."""
    gray = (0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]).float()
    fft = torch.fft.fft2(gray)
    mag = torch.log1p(torch.abs(torch.fft.fftshift(fft))).cpu().numpy()
    return _apply_colormap(mag)


def vis_temporal_strip(dataset, model, device, use_amp, row_y=None, n_frames=30, start_idx=0):
    """Temporal consistency strip: one pixel row across consecutive frames.

    Stacks row y from n_frames into a (n_frames, W, 3) image.
    Compares input strip (flickering noise) vs output strip (should be smooth).
    Returns (n_frames*2, W, 3) uint8 with input on top, output on bottom.
    """
    n_frames = min(n_frames, len(dataset) - start_idx)
    frames = torch.stack([dataset[start_idx + i][0] for i in range(n_frames)])
    if row_y is None:
        row_y = frames.shape[2] // 2  # middle row

    # Input strips
    inp_strip = frames[:, :, row_y, :].cpu()  # (N, 3, W)

    # Output strips
    model.eval()
    out_strips = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for i in range(n_frames):
            out = model(frames[i:i+1].to(device))
            out_strips.append(out[0, :, row_y, :].float().cpu())
    out_strip = torch.stack(out_strips)  # (N, 3, W)

    inp_img = (inp_strip.permute(0, 2, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    out_img = (out_strip.permute(0, 2, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    # Separator line
    sep = np.full((2, inp_img.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([inp_img, sep, out_img], axis=0)


def vis_dark_crop(inp, out, crop_size=256, zoom=4):
    """4x zoomed crop of the darkest region -- where HEVC artifacts are worst.

    Finds the darkest crop_size x crop_size region, extracts from both input
    and output, upscales with nearest-neighbor, returns side-by-side.
    """
    # Find darkest region by avg brightness in a sliding window (strided)
    gray = (0.299 * inp[0] + 0.587 * inp[1] + 0.114 * inp[2]).cpu()
    h, w = gray.shape
    stride = crop_size // 2
    best_y, best_x, best_val = 0, 0, float('inf')
    for y in range(0, h - crop_size, stride):
        for x in range(0, w - crop_size, stride):
            val = gray[y:y+crop_size, x:x+crop_size].mean().item()
            if val < best_val:
                best_val = val
                best_y, best_x = y, x

    # Extract crops
    inp_crop = inp[:, best_y:best_y+crop_size, best_x:best_x+crop_size]
    out_crop = out[:, best_y:best_y+crop_size, best_x:best_x+crop_size]

    # Nearest-neighbor upscale
    inp_up = F.interpolate(inp_crop.unsqueeze(0), scale_factor=zoom, mode='nearest')[0]
    out_up = F.interpolate(out_crop.unsqueeze(0), scale_factor=zoom, mode='nearest')[0]

    return np.concatenate([_to_uint8(inp_up), _to_uint8(out_up)], axis=1)


def vis_psnr_heatmap(inp, out, block_size=64):
    """Per-block PSNR heatmap overlaid on the frame.

    Divides into block_size x block_size regions, computes local PSNR per block,
    renders as a color-mapped overlay showing where reconstruction is strong vs weak.
    """
    inp_f, out_f = inp.float(), out.float()
    h, w = inp_f.shape[1], inp_f.shape[2]
    bh = h // block_size
    bw = w // block_size
    psnr_map = np.zeros((bh, bw), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * block_size, bx * block_size
            i_blk = inp_f[:, y0:y0+block_size, x0:x0+block_size]
            o_blk = out_f[:, y0:y0+block_size, x0:x0+block_size]
            mse = F.mse_loss(o_blk, i_blk).item()
            psnr_map[by, bx] = -10 * math.log10(mse) if mse > 0 else 60.0

    # Upsample to full resolution
    from PIL import Image as PILImage
    psnr_img = PILImage.fromarray(_apply_colormap(psnr_map, vmin=20, vmax=45))
    psnr_img = psnr_img.resize((w, h), PILImage.NEAREST)
    return np.array(psnr_img)


def log_visuals(wb, epoch, val_frames, val_output, dataset, model, device,
                use_amp, hf_ratio):
    """Log all visualization images to W&B. All vis computed on CPU to save VRAM."""
    vis_idx = 0
    inp = val_frames[vis_idx].cpu()
    out = val_output[vis_idx].float().cpu()
    frame_psnr = compute_psnr(out.unsqueeze(0), inp.unsqueeze(0))

    images = {
        "vis/input_vs_output": wb.Image(
            vis_comparison(inp, out),
            caption=f"epoch {epoch} | Left=input Right=output | {frame_psnr:.1f}dB",
        ),
        "vis/residual_10x": wb.Image(
            vis_residual(inp, out, gain=10.0),
            caption=f"epoch {epoch} | 10x amplified removed content | gray=no change",
        ),
        "vis/fft_input": wb.Image(
            vis_fft_heatmap(inp),
            caption="FFT magnitude (input, constant)",
        ),
        "vis/fft_output": wb.Image(
            vis_fft_heatmap(out),
            caption=f"epoch {epoch} | FFT output | hf_ratio={hf_ratio:.3f}",
        ),
        "vis/dark_crop_4x": wb.Image(
            vis_dark_crop(inp, out, crop_size=256, zoom=4),
            caption=f"epoch {epoch} | Darkest 256px crop 4x zoom | Left=input Right=output",
        ),
        "vis/psnr_heatmap": wb.Image(
            vis_psnr_heatmap(inp, out, block_size=64),
            caption=f"epoch {epoch} | Per-block PSNR (blue=low red=high, 20-45dB range)",
        ),
    }
    del inp, out

    # Temporal strip (only every 50 epochs -- requires running model on 30 extra frames)
    if epoch % 50 == 0:
        images["vis/temporal_strip"] = wb.Image(
            vis_temporal_strip(dataset, model, device, use_amp, n_frames=30),
            caption=f"epoch {epoch} | Row y=540 across 30 frames | Top=input Bottom=output",
        )

    wb.log(images, step=epoch)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = FrameDataset(args.data_dir)

    enc_strides = [int(x) for x in args.enc_strides.split(',')]
    dec_strides = [int(x) for x in args.dec_strides.split(',')]
    dec_blks = tuple(int(x) for x in args.dec_blks.split(','))

    model = HNeRVSimple(
        height=dataset.height,
        width=dataset.width,
        enc_strides=enc_strides,
        dec_strides=dec_strides,
        dec_blks=dec_blks,
        enc_dim=args.enc_dim,
        fc_dim=args.fc_dim,
        reduce=args.reduce,
        lower_width=args.lower_width,
        grad_checkpoint=args.grad_checkpoint,
        enc_blocks=args.enc_blocks,
    ).to(device)

    enc_p = model.encoder_params / 1e6
    dec_p = model.decoder_params / 1e6
    total_p = model.param_count / 1e6
    print(f"Model: encoder {enc_p:.2f}M + decoder {dec_p:.2f}M = {total_p:.2f}M params")
    print(f"  Encoder strides: {enc_strides}")
    print(f"  Decoder strides: {dec_strides}")
    print(f"  Base resolution: {model.fc_h}x{model.fc_w}")

    use_prodigy = args.optimizer == 'prodigy'
    if use_prodigy:
        from prodigyopt import Prodigy
        optimizer = Prodigy(
            model.parameters(),
            lr=1.0,
            d_coef=args.d_coef,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            safeguard_warmup=True,
            use_bias_correction=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0,
        )
        print(f"Optimizer: Prodigy (d_coef={args.d_coef}, safeguard_warmup=True)")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        warmup_epochs = min(10, args.epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Optimizer: Adam (lr={args.lr}, cosine warmup)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        ckpt_path = args.resume
        if os.path.isfile(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt["epoch"] + 1
            if "metrics" in ckpt:
                best_psnr = ckpt["metrics"].get("val_psnr", 0)
            if args.fresh_optimizer:
                print(f"  Fresh optimizer (model weights from epoch {ckpt['epoch']}, optimizer reset)")
            else:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print(f"  Optimizer state restored")
                except (ValueError, KeyError):
                    print(f"  WARNING: Could not restore optimizer (different type?), starting fresh")
            del ckpt
            gc.collect()
            # Advance scheduler to match resumed epoch
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}, best_psnr={best_psnr:.2f}")

    # Metrics log (append mode for resume)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    # Holdout validation: remove every 12th frame from training entirely
    all_indices = list(range(len(dataset)))
    val_indices = list(range(0, len(dataset), 12))[:10]  # ~10 held-out frames
    train_indices = [i for i in all_indices if i not in val_indices]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    # Keep val frames on CPU, load one at a time during validation
    val_frames_cpu = [dataset[i][0] for i in val_indices]  # list of (3, H, W) tensors
    print(f"  Train: {len(train_indices)} frames, Holdout: {len(val_indices)} frames")

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # W&B
    wb = None
    if args.wandb:
        import wandb
        wb = wandb
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = f"hnerv-{total_p:.1f}M-enc{args.enc_dim}-fc{args.fc_dim}"
            if args.late_layer_decay > 0:
                run_name += f"-decay{args.late_layer_decay}"
        wb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": "hnerv",
                "params_M": round(total_p, 3),
                "encoder_params_M": round(enc_p, 3),
                "decoder_params_M": round(dec_p, 3),
                "enc_strides": enc_strides,
                "dec_strides": dec_strides,
                "enc_dim": args.enc_dim,
                "enc_blocks": args.enc_blocks,
                "fc_dim": args.fc_dim,
                "reduce": args.reduce,
                "lower_width": args.lower_width,
                "late_layer_decay": args.late_layer_decay,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "n_frames": len(dataset),
                "resolution": f"{dataset.width}x{dataset.height}",
                "loss": args.loss,
                "pixel_weight": args.pixel_weight,
                "dec_blks": list(dec_blks),
                "data_dir": args.data_dir,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "resumed_from_epoch": start_epoch if args.resume else 0,
            },
            resume="allow",
        )

    # Graceful interrupt: save checkpoint on Ctrl+C instead of losing progress
    _current_epoch = start_epoch
    _interrupted = False

    def _handle_interrupt(signum, frame):
        nonlocal _interrupted
        if not _interrupted:
            _interrupted = True
            print(f"\n  Interrupted at epoch {_current_epoch}. Saving checkpoint...")
            ckpt = {
                "epoch": _current_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            path = os.path.join(args.output_dir, "interrupted.pth")
            torch.save(ckpt, path)
            print(f"  Saved to {path}. Resume with --resume {path}")
        raise KeyboardInterrupt

    import signal
    signal.signal(signal.SIGINT, _handle_interrupt)

    training_start_time = time.perf_counter()
    print(f"\nTraining epochs {start_epoch}-{args.epochs}, ckpt every {args.ckpt_interval}")
    if args.max_time > 0:
        print(f"Wall-clock timeout: {args.max_time}s ({args.max_time/60:.0f} min)")
    print(f"Late-layer decay: {args.late_layer_decay}, AMP: {use_amp}, W&B: {wb is not None}")
    print()

    for epoch in range(start_epoch, args.epochs):
        _current_epoch = epoch

        # Wall-clock timeout
        if args.max_time > 0:
            elapsed = time.perf_counter() - training_start_time
            if elapsed > args.max_time:
                print(f"\n  Timeout ({args.max_time}s) reached at epoch {epoch}. Saving and exiting.")
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(args.output_dir, "latest.pth"))
                break
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        t0 = time.perf_counter()

        for batch_img, batch_norm_idx, batch_idx in loader:
            batch_img = batch_img.to(device)
            batch_norm_idx = batch_norm_idx.to(device).float()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_img, norm_idx=batch_norm_idx)
                loss = compute_loss(output, batch_img, args.loss, args.pixel_weight)

                if args.late_layer_decay > 0:
                    reg = 0
                    for block in list(model.decoder)[-2:]:
                        for p in block.parameters():
                            reg = reg + p.pow(2).sum()
                    loss = loss + args.late_layer_decay * reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"\n  ERROR: Loss is {loss_val} at epoch {epoch}. Training unstable.")
                    print(f"  Saving emergency checkpoint and exiting.")
                    torch.save({
                        "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, os.path.join(args.output_dir, "nan_crash.pth"))
                    if wb:
                        wb.finish(exit_code=1)
                    return
                epoch_loss += loss_val
                epoch_psnr += compute_psnr(output.float(), batch_img)
            del output, loss, batch_img, batch_norm_idx

        scheduler.step()
        n_batches = len(loader)
        epoch_loss /= n_batches
        epoch_psnr /= n_batches
        dt = time.perf_counter() - t0

        # Validation: run model on ONE holdout frame for PSNR (lightweight)
        model.eval()
        with torch.no_grad():
            vf = val_frames_cpu[0]
            vf_gpu = vf.unsqueeze(0).to(device)
            vi_idx = torch.tensor([val_indices[0] / len(dataset)], device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                vo = model(vf_gpu, norm_idx=vi_idx)
            vo_cpu = vo[0].float().cpu()
            vf_cpu = vf.float()
            val_psnr = compute_psnr(vo_cpu.unsqueeze(0), vf_cpu.unsqueeze(0))
            del vf_gpu, vo
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # HF energy only at vis epochs (FFT is expensive)
        is_vis_epoch = (epoch + 1) % args.ckpt_interval == 0
        if is_vis_epoch:
            hf_input = compute_hf_energy(vf_cpu)
            hf_output = compute_hf_energy(vo_cpu)
            hf_ratio = hf_output / (hf_input + 1e-8)
        else:
            hf_input = hf_output = hf_ratio = 0.0
        late_norm = compute_late_layer_norm(model)

        d_val = optimizer.param_groups[0].get("d", None)
        effective_lr = scheduler.get_last_lr()[0] * d_val if d_val else scheduler.get_last_lr()[0]
        iters_per_sec = n_batches / dt
        metrics = {
            "epoch": epoch,
            "loss": round(epoch_loss, 6),
            "train_psnr": round(epoch_psnr, 2),
            "val_psnr": round(val_psnr, 2),
            "hf_energy_input": round(hf_input, 4),
            "hf_energy_output": round(hf_output, 4),
            "hf_ratio": round(hf_ratio, 4),
            "late_layer_norm": round(late_norm, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "effective_lr": round(effective_lr, 8),
            "prodigy_d": round(d_val, 8) if d_val else None,
            "batch_size": args.batch_size,
            "iters_per_sec": round(iters_per_sec, 2),
            "time": round(dt, 1),
        }
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

        # W&B scalar logging (every epoch)
        if wb:
            wb_log = {
                "train/loss": epoch_loss,
                "train/psnr": epoch_psnr,
                "val/psnr": val_psnr,
                "noise/hf_ratio": hf_ratio,
                "noise/hf_energy_output": hf_output,
                "noise/hf_energy_input": hf_input,
                "noise/late_layer_norm": late_norm,
                "train/lr": scheduler.get_last_lr()[0],
                "train/effective_lr": effective_lr,
                "train/epoch_time": dt,
                "train/iters_per_sec": iters_per_sec,
                "train/batch_size": args.batch_size,
            }
            if d_val is not None:
                wb_log["train/prodigy_d"] = d_val
            wb.log(wb_log, step=epoch)

        # Visual logging (at checkpoint intervals only -- vf_cpu/vo_cpu from val above)
        if is_vis_epoch:
            vis_dir = os.path.join(args.output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            from PIL import Image as PILImage

            comp = vis_comparison(vf_cpu, vo_cpu)
            resid = vis_residual(vf_cpu, vo_cpu)
            PILImage.fromarray(comp).save(os.path.join(vis_dir, f"compare_e{epoch:04d}.png"))
            PILImage.fromarray(resid).save(os.path.join(vis_dir, f"residual_e{epoch:04d}.png"))

            if wb:
                wb.log({
                    "vis/input_vs_output": wb.Image(comp,
                        caption=f"epoch {epoch} | Left=input Right=output | {val_psnr:.1f}dB"),
                    "vis/residual_10x": wb.Image(resid,
                        caption=f"epoch {epoch} | 10x amplified removed content"),
                }, step=epoch)
            del comp, resid
            gc.collect()

        # Print
        if epoch % args.print_interval == 0 or epoch == args.epochs - 1:
            print(f"  [{epoch:4d}/{args.epochs}] loss={epoch_loss:.5f} "
                  f"psnr={epoch_psnr:.1f}dB val={val_psnr:.1f}dB "
                  f"hf_ratio={hf_ratio:.3f} late_norm={late_norm:.2f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} ({dt:.1f}s)")

        # Save checkpoint
        if (epoch + 1) % args.ckpt_interval == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            }
            path = os.path.join(args.output_dir, f"ckpt_epoch_{epoch:04d}.pth")
            torch.save(ckpt, path)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))

        # Save latest (every 10 epochs for resume)
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(args.output_dir, "latest.pth"))

    metrics_file.close()
    if wb:
        wb.finish()
    print(f"\nDone. Best val PSNR: {best_psnr:.2f} dB")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoints: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train HNeRV on video frames for denoising analysis"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory of PNG/JPG frames")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/nerv/<dirname>)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-time", type=int, default=0,
                        help="Max wall-clock training time in seconds (0=unlimited). "
                             "Saves checkpoint and exits cleanly when exceeded.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 for 6GB VRAM at 1080p)")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Architecture
    parser.add_argument("--enc-strides", default="5,3,2,2,2",
                        help="Encoder downsample strides (comma-separated)")
    parser.add_argument("--dec-strides", default="5,3,2,2,2",
                        help="Decoder upsample strides (comma-separated)")
    parser.add_argument("--enc-dim", type=int, default=16,
                        help="Encoder embedding channels (smaller = more denoising)")
    parser.add_argument("--fc-dim", type=int, default=170,
                        help="Decoder base channel width (reference: 170 for 5M model)")
    parser.add_argument("--reduce", type=float, default=1.2,
                        help="Channel reduction per decoder stage (reference: 1.2)")
    parser.add_argument("--lower-width", type=int, default=12,
                        help="Minimum channel width in decoder")
    parser.add_argument("--dec-blks", default="1,1,2,2,2",
                        help="Blocks per decoder stage (more at high-res)")
    parser.add_argument("--enc-blocks", type=int, default=1,
                        help="ConvNeXt blocks per encoder stage (1 or 2)")
    parser.add_argument("--loss", default="l1_freq",
                        choices=["l1", "l2", "l1_freq", "fusion6", "l1_ssim_freq", "fusion10_freq"],
                        help="Loss function (default: l1_freq = L1 + FFT frequency)")
    parser.add_argument("--pixel-weight", type=float, default=10.0,
                        help="Weight for pixel loss in l1_freq (default: 10, lower = more freq emphasis)")

    # Optimizer
    parser.add_argument("--optimizer", default="prodigy", choices=["adam", "prodigy"],
                        help="Optimizer: prodigy (auto LR, default) or adam (manual LR)")
    parser.add_argument("--d-coef", type=float, default=1.0, dest="d_coef",
                        help="Prodigy d_coef: scales auto-tuned LR (0.5=conservative, 2.0=aggressive)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (AdamW-style for Prodigy, L2 for Adam)")

    # Memory
    parser.add_argument("--grad-checkpoint", action="store_true", default=False,
                        dest="grad_checkpoint",
                        help="Enable gradient checkpointing (saves VRAM, 2x slower)")

    # Denoising controls
    parser.add_argument("--late-layer-decay", type=float, default=0.0,
                        help="L2 weight decay on last 2 decoder layers (0=off, try 1e-4 to 1e-2)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    parser.add_argument("--fresh-optimizer", action="store_true", default=False,
                        dest="fresh_optimizer",
                        help="Resume model weights but reset optimizer/scheduler (for switching optimizers)")

    # Logging
    parser.add_argument("--ckpt-interval", type=int, default=30,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--print-interval", type=int, default=10,
                        help="Print metrics every N epochs")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="remaster",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (default: from env/config)")
    parser.add_argument("--run-name", default=None,
                        help="W&B run name (also used in console output)")

    args = parser.parse_args()

    if args.output_dir is None:
        dirname = os.path.basename(args.data_dir.rstrip("/\\"))
        args.output_dir = os.path.join(PROJECT_ROOT, "output", "nerv", dirname)

    train(args)


if __name__ == "__main__":
    main()
