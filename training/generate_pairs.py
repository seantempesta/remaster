"""
Generate training pairs for NAFNet distillation from SCUNet.

Takes input frames (compressed video frames) and generates SCUNet-denoised outputs
as pseudo ground truth. Saves full-resolution frames so we can random-crop during training.

Crash-safe: saves each frame individually, skips already-processed frames on restart.

Usage:
    # Local (slow, ~0.6 fps on RTX 3060)
    python training/generate_pairs.py --input-dir data/frames_mid_1080p --output-dir data/train_pairs --max-frames 5

    # Full dataset
    python training/generate_pairs.py --input-dir data/frames_mid_1080p --output-dir data/train_pairs

    # Use existing SCUNet outputs (skip generation)
    python training/generate_pairs.py --input-dir data/frames_mid_1080p --scunet-dir data/frames_mid_scunet --output-dir data/train_pairs
"""
import sys
import os
import glob
import time
import argparse
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import add_scunet_to_path, resolve_scunet_dir

try:
    SCUNET_DIR = add_scunet_to_path()
except FileNotFoundError:
    SCUNET_DIR = None

import numpy as np
import cv2
import torch


def load_scunet(model_name="scunet_color_real_psnr", device="cuda", fp16=True):
    """Load SCUNet model for inference."""
    from models.network_scunet import SCUNet as net
    model_path = os.path.join(SCUNET_DIR, "model_zoo", f"{model_name}.pth")
    print(f"Loading SCUNet ({model_name}) from {model_path}")
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    if fp16:
        model = model.half()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {params_m:.1f}M params, VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")
    return model


def process_frame_scunet(model, frame_bgr, device="cuda", fp16=True):
    """Run SCUNet on a single BGR frame, return denoised BGR frame."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    if fp16:
        img_t = img_t.half()
    img_t = img_t.to(device)

    with torch.no_grad():
        out_t = model(img_t)

    out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    del img_t, out_t
    return out_bgr


def main():
    parser = argparse.ArgumentParser(description="Generate SCUNet training pairs for NAFNet distillation")
    parser.add_argument("--input-dir", required=True, help="Directory with input frames (compressed)")
    parser.add_argument("--output-dir", required=True, help="Output directory for training pairs")
    parser.add_argument("--scunet-dir", default=None,
                        help="Directory with pre-existing SCUNet outputs (skip generation)")
    parser.add_argument("--model", default="scunet_color_real_psnr",
                        choices=["scunet_color_real_psnr", "scunet_color_real_gan"])
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 for all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    input_out_dir = os.path.join(output_dir, "input")
    target_out_dir = os.path.join(output_dir, "target")
    os.makedirs(input_out_dir, exist_ok=True)
    os.makedirs(target_out_dir, exist_ok=True)

    # Get input frames
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not frames:
        raise FileNotFoundError(f"No PNG/JPG frames found in {input_dir}")

    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Input: {len(frames)} frames from {input_dir}")

    # If pre-existing SCUNet outputs are provided, just copy/symlink pairs
    if args.scunet_dir:
        scunet_dir = os.path.abspath(args.scunet_dir)
        print(f"Using pre-existing SCUNet outputs from {scunet_dir}")
        copied = 0
        for frame_path in frames:
            fname = os.path.basename(frame_path)
            scunet_path = os.path.join(scunet_dir, fname)
            if not os.path.exists(scunet_path):
                print(f"  SKIP: {fname} not found in SCUNet dir")
                continue
            dst_input = os.path.join(input_out_dir, fname)
            dst_target = os.path.join(target_out_dir, fname)
            if not os.path.exists(dst_input):
                shutil.copy2(frame_path, dst_input)
            if not os.path.exists(dst_target):
                shutil.copy2(scunet_path, dst_target)
            copied += 1
        print(f"Copied {copied} pairs to {output_dir}")
        return

    # Generate SCUNet outputs
    if SCUNET_DIR is None:
        raise RuntimeError("SCUNet not found. Use --scunet-dir to point to pre-existing outputs.")

    fp16 = not args.fp32
    model = load_scunet(args.model, device=args.device, fp16=fp16)

    # Find already-processed frames (crash-safe resume)
    existing = set(os.path.basename(f) for f in glob.glob(os.path.join(target_out_dir, "*.png")))
    remaining = [(i, p) for i, p in enumerate(frames) if os.path.basename(p) not in existing]
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")

    start = time.time()
    for idx, (i, frame_path) in enumerate(remaining):
        fname = os.path.basename(frame_path)
        t0 = time.time()

        # Read input frame
        frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"  ERROR: Could not read {frame_path}")
            continue

        # Run SCUNet
        try:
            denoised = process_frame_scunet(model, frame_bgr, device=args.device, fp16=fp16)
        except RuntimeError as e:
            print(f"  ERROR on {fname}: {e}")
            torch.cuda.empty_cache()
            continue

        # Save pair
        cv2.imwrite(os.path.join(input_out_dir, fname), frame_bgr)
        cv2.imwrite(os.path.join(target_out_dir, fname), denoised)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start
        fps = (idx + 1) / total_elapsed
        eta = (len(remaining) - idx - 1) / max(fps, 0.01)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(remaining)}] {fname} {elapsed:.2f}s, "
                  f"{fps:.2f} fps, ETA: {eta / 60:.1f}min, "
                  f"VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    total = time.time() - start
    print(f"\nDONE: {len(remaining)} frames in {total / 60:.1f} min ({len(remaining) / max(total, 1):.2f} fps)")
    print(f"Output: {output_dir}")
    print(f"  input/  — {len(glob.glob(os.path.join(input_out_dir, '*.png')))} frames")
    print(f"  target/ — {len(glob.glob(os.path.join(target_out_dir, '*.png')))} frames")


if __name__ == "__main__":
    main()
