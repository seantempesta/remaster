"""Benchmark NAFNet vs SCUNet on the mid-episode 1080p clip.

Clones NAFNet if needed, downloads SIDD-width64 checkpoint, runs both
models on frames from data/frames_mid_1080p/, and compares quality + speed.

Usage:
    conda activate remaster
    python bench/bench_nafnet.py [--frames N] [--skip-scunet]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import subprocess
import glob
import time

import numpy as np
import cv2
import torch

from lib.paths import PROJECT_ROOT, DATA_DIR, REFERENCE_CODE
from lib.nafnet_arch import NAFNet
from lib.metrics import compute_psnr, compute_ssim
from lib.ffmpeg_utils import get_ffmpeg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NAFNET_DIR = str(REFERENCE_CODE / "NAFNet")
INPUT_DIR = str(DATA_DIR / "frames_mid_1080p")
NAFNET_OUT_DIR = str(DATA_DIR / "frames_mid_nafnet")
SCUNET_OUT_DIR = str(DATA_DIR / "frames_mid_scunet")
NAFNET_CKPT_DIR = os.path.join(NAFNET_DIR, 'experiments', 'pretrained_models')
NAFNET_CKPT = os.path.join(NAFNET_CKPT_DIR, 'NAFNet-SIDD-width64.pth')

NAFNET_REPO = 'https://github.com/megvii-research/NAFNet.git'
# Google Drive file ID for NAFNet-SIDD-width64
GDRIVE_FILE_ID = '14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR'

DEVICE = 'cuda'


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def clone_nafnet():
    """Clone NAFNet repo if not present."""
    if os.path.isdir(os.path.join(NAFNET_DIR, 'basicsr')):
        print(f"NAFNet already cloned at {NAFNET_DIR}")
        return
    print(f"Cloning NAFNet -> {NAFNET_DIR} ...")
    subprocess.run(['git', 'clone', NAFNET_REPO, NAFNET_DIR], check=True)
    print("  Done.")


def download_checkpoint():
    """Download SIDD-width64 checkpoint from Google Drive via gdown."""
    if os.path.isfile(NAFNET_CKPT):
        sz = os.path.getsize(NAFNET_CKPT) / 1024**2
        print(f"Checkpoint already exists: {NAFNET_CKPT} ({sz:.1f} MB)")
        return

    os.makedirs(NAFNET_CKPT_DIR, exist_ok=True)

    # Try gdown first (pip install gdown if missing)
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive download...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'], check=True)
        import gdown

    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    print(f"Downloading NAFNet-SIDD-width64 checkpoint...")
    gdown.download(url, NAFNET_CKPT, quiet=False)

    if not os.path.isfile(NAFNET_CKPT):
        raise RuntimeError(f"Download failed — checkpoint not found at {NAFNET_CKPT}")
    sz = os.path.getsize(NAFNET_CKPT) / 1024**2
    print(f"  Saved: {NAFNET_CKPT} ({sz:.1f} MB)")


def extract_frames_if_needed():
    """Extract frames from clip_mid_1080p.mp4 if frames_mid_1080p is empty."""
    existing = glob.glob(os.path.join(INPUT_DIR, '*.png'))
    if existing:
        print(f"Found {len(existing)} input frames in {INPUT_DIR}")
        return

    clip = str(DATA_DIR / 'clip_mid_1080p.mp4')
    if not os.path.isfile(clip):
        raise FileNotFoundError(f"No input frames and no source clip at {clip}")

    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"Extracting frames from {clip} ...")
    subprocess.run([
        get_ffmpeg(), '-hide_banner', '-i', clip,
        '-start_number', '1',
        os.path.join(INPUT_DIR, 'frame_%05d.png')
    ], check=True, capture_output=True)
    n = len(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    print(f"  Extracted {n} frames.")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_nafnet():
    """Load NAFNet-SIDD-width64 in fp16."""
    model = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(NAFNET_CKPT, map_location='cpu', weights_only=False)
    # BasicSR checkpoints wrap state_dict under 'params' or 'params_ema'
    if 'params_ema' in ckpt:
        ckpt = ckpt['params_ema']
    elif 'params' in ckpt:
        ckpt = ckpt['params']
    elif 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE).half()
    return model


def load_scunet():
    """Load SCUNet color-real-psnr in fp16."""
    from lib.paths import add_scunet_to_path, resolve_scunet_dir
    add_scunet_to_path()
    from models.network_scunet import SCUNet as net

    scunet_ckpt = str(resolve_scunet_dir() / "model_zoo" / "scunet_color_real_psnr.pth")
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(
        torch.load(scunet_ckpt, map_location='cpu', weights_only=True), strict=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE).half()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def read_frame(path):
    """Read a frame, return (H,W,3) BGR uint8 numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def to_tensor(bgr_img):
    """BGR uint8 -> (1,3,H,W) fp16 CUDA tensor in [0,1]."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).half().unsqueeze(0) / 255.0
    return t.to(DEVICE)


def to_image(tensor):
    """(1,3,H,W) tensor -> BGR uint8 numpy."""
    out = (tensor.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def pad_to_multiple(t, multiple=64):
    """Pad tensor so H and W are multiples of `multiple`. Returns (padded, (pad_h, pad_w))."""
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t, (pad_h, pad_w)


def run_model(model, frames, output_dir, name, need_pad=False):
    """Run a model on a list of frame paths, save outputs, return timing info."""
    os.makedirs(output_dir, exist_ok=True)
    # Clear old outputs
    for f in glob.glob(os.path.join(output_dir, '*.png')):
        os.remove(f)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    print(f"\n{'='*60}")
    print(f"  Running {name} on {len(frames)} frames")
    print(f"{'='*60}")

    for i, frame_path in enumerate(frames):
        img = read_frame(frame_path)
        t = to_tensor(img)

        if need_pad:
            t, (ph, pw) = pad_to_multiple(t, 64)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model(t)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if need_pad and (ph > 0 or pw > 0):
            _, _, oh, ow = out.shape
            out = out[:, :, :oh - ph if ph > 0 else oh, :ow - pw if pw > 0 else ow]

        out_img = to_image(out)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i+1:05d}.png'), out_img)

        # Skip warmup frame for timing
        if i > 0:
            times.append(elapsed)

        del t, out

        if i == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Frame 1 (warmup): {elapsed:.3f}s, peak VRAM: {peak:.0f} MB")
        elif (i + 1) % 10 == 0:
            avg = np.mean(times)
            fps = 1.0 / avg
            print(f"  [{i+1}/{len(frames)}] avg {avg:.3f}s/frame, {fps:.2f} fps")

    avg_time = np.mean(times) if times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n  {name} results:")
    print(f"    Avg time/frame: {avg_time:.3f}s  ({fps:.2f} fps)")
    print(f"    Peak VRAM: {peak_vram:.0f} MB")
    print(f"    Output dir: {output_dir}")

    return {'name': name, 'avg_time': avg_time, 'fps': fps, 'peak_vram': peak_vram}


def compare_outputs(frames, nafnet_dir, scunet_dir, n_frames):
    """Compute PSNR/SSIM between original, NAFNet output, and SCUNet output."""
    print(f"\n{'='*60}")
    print("  Quality Comparison (PSNR / SSIM)")
    print(f"{'='*60}")

    metrics = {
        'SCUNet vs Original': {'psnr': [], 'ssim': []},
        'NAFNet vs Original': {'psnr': [], 'ssim': []},
        'NAFNet vs SCUNet':   {'psnr': [], 'ssim': []},
    }

    for i in range(n_frames):
        fname = f'frame_{i+1:05d}.png'
        orig = cv2.imread(frames[i], cv2.IMREAD_COLOR)
        nafnet_path = os.path.join(nafnet_dir, fname)
        scunet_path = os.path.join(scunet_dir, fname)

        if not os.path.isfile(nafnet_path) or not os.path.isfile(scunet_path):
            continue

        naf = cv2.imread(nafnet_path, cv2.IMREAD_COLOR)
        scu = cv2.imread(scunet_path, cv2.IMREAD_COLOR)

        # Ensure same shape (should be, but just in case)
        if naf.shape != orig.shape:
            naf = cv2.resize(naf, (orig.shape[1], orig.shape[0]))
        if scu.shape != orig.shape:
            scu = cv2.resize(scu, (orig.shape[1], orig.shape[0]))

        metrics['SCUNet vs Original']['psnr'].append(compute_psnr(orig, scu))
        metrics['SCUNet vs Original']['ssim'].append(compute_ssim(orig, scu))
        metrics['NAFNet vs Original']['psnr'].append(compute_psnr(orig, naf))
        metrics['NAFNet vs Original']['ssim'].append(compute_ssim(orig, naf))
        metrics['NAFNet vs SCUNet']['psnr'].append(compute_psnr(scu, naf))
        metrics['NAFNet vs SCUNet']['ssim'].append(compute_ssim(scu, naf))

    print(f"\n  {'Comparison':<28} {'PSNR (dB)':>12} {'SSIM':>12} {'Frames':>8}")
    print(f"  {'-'*64}")
    for label, m in metrics.items():
        if m['psnr']:
            psnr_mean = np.mean(m['psnr'])
            ssim_mean = np.mean(m['ssim'])
            print(f"  {label:<28} {psnr_mean:>8.2f}     {ssim_mean:>8.4f}     {len(m['psnr']):>5}")
        else:
            print(f"  {label:<28} {'N/A':>12} {'N/A':>12}")

    # Save a few side-by-side comparisons
    compare_dir = str(DATA_DIR / 'comparisons_nafnet')
    os.makedirs(compare_dir, exist_ok=True)

    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    for idx in sample_indices:
        fname = f'frame_{idx+1:05d}.png'
        orig = cv2.imread(frames[idx], cv2.IMREAD_COLOR)
        nafnet_path = os.path.join(nafnet_dir, fname)
        scunet_path = os.path.join(scunet_dir, fname)

        if not os.path.isfile(nafnet_path) or not os.path.isfile(scunet_path):
            continue

        naf = cv2.imread(nafnet_path, cv2.IMREAD_COLOR)
        scu = cv2.imread(scunet_path, cv2.IMREAD_COLOR)

        # Crop to a 960x540 center region for readable comparison
        h, w = orig.shape[:2]
        cy, cx = h // 2, w // 2
        crop = (slice(cy - 270, cy + 270), slice(cx - 480, cx + 480))

        panels = [orig[crop], scu[crop], naf[crop]]
        labels = ['Original', 'SCUNet', 'NAFNet']
        for panel, label in zip(panels, labels):
            cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        comparison = np.hstack(panels)
        out_path = os.path.join(compare_dir, f'compare_{idx+1:05d}.png')
        cv2.imwrite(out_path, comparison)

    print(f"\n  Side-by-side crops saved to {compare_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Benchmark NAFNet vs SCUNet')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames to process (default: 30)')
    parser.add_argument('--skip-scunet', action='store_true',
                        help='Skip SCUNet if outputs already exist')
    args = parser.parse_args()

    # 1. Setup
    print("=" * 60)
    print("  NAFNet vs SCUNet Benchmark")
    print("=" * 60)

    clone_nafnet()
    download_checkpoint()
    extract_frames_if_needed()

    frames = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    n = min(args.frames, len(frames))
    frames = frames[:n]
    print(f"\nUsing {n} frames from {INPUT_DIR}")

    # 2. Run NAFNet
    print("\nLoading NAFNet-SIDD-width64...")
    nafnet_model = load_nafnet()
    model_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"  Model VRAM: {model_vram:.0f} MB")

    nafnet_stats = run_model(nafnet_model, frames, NAFNET_OUT_DIR, 'NAFNet-SIDD-w64', need_pad=True)
    del nafnet_model
    torch.cuda.empty_cache()

    # 3. Run SCUNet (or reuse existing outputs)
    scunet_existing = sorted(glob.glob(os.path.join(SCUNET_OUT_DIR, '*.png')))
    if args.skip_scunet and len(scunet_existing) >= n:
        print(f"\nSkipping SCUNet — reusing {len(scunet_existing)} existing outputs")
        scunet_stats = {'name': 'SCUNet (cached)', 'avg_time': 0, 'fps': 0, 'peak_vram': 0}
    else:
        print("\nLoading SCUNet (color-real-psnr)...")
        scunet_model = load_scunet()
        model_vram = torch.cuda.memory_allocated() / 1024**2
        print(f"  Model VRAM: {model_vram:.0f} MB")

        scunet_stats = run_model(scunet_model, frames, SCUNET_OUT_DIR, 'SCUNet-color-real')
        del scunet_model
        torch.cuda.empty_cache()

    # 4. Compare
    compare_outputs(frames, NAFNET_OUT_DIR, SCUNET_OUT_DIR, n)

    # 5. Summary
    print(f"\n{'='*60}")
    print("  Performance Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'FPS':>8} {'ms/frame':>10} {'Peak VRAM':>12}")
    print(f"  {'-'*58}")
    for s in [nafnet_stats, scunet_stats]:
        ms = s['avg_time'] * 1000 if s['avg_time'] > 0 else 0
        fps_str = f"{s['fps']:.2f}" if s['fps'] > 0 else 'N/A'
        ms_str = f"{ms:.1f}" if ms > 0 else 'N/A'
        vram_str = f"{s['peak_vram']:.0f} MB" if s['peak_vram'] > 0 else 'N/A'
        print(f"  {s['name']:<25} {fps_str:>8} {ms_str:>10} {vram_str:>12}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
