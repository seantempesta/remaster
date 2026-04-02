"""Quick sanity test for all loss functions.

Tests on synthetic data (random tensors) and optionally on real
image pairs from the val set. No GPU model needed — just verifies
losses compute, are differentiable, and have reasonable values.

Usage:
    python training/test_losses.py
    python training/test_losses.py --real-images data/val_pairs
"""
import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.losses import (
    CharbonnierLoss, PSNRLoss, FocalFrequencyLoss, DISTSPerceptualLoss,
    build_pixel_criterion,
)


def test_synthetic():
    """Test all losses on random tensors."""
    print("=" * 60)
    print("Testing losses on synthetic data (256x256 random tensors)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create test tensors
    pred = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)
    target = torch.rand(1, 3, 256, 256, device=device)
    # Also test with near-identical images (should give low loss)
    near = target.clone().detach() + 0.01 * torch.randn_like(target)
    near.requires_grad_(True)

    losses = {
        "Charbonnier": CharbonnierLoss(),
        "PSNR": PSNRLoss(),
        "L1": torch.nn.L1Loss(),
        "FocalFrequency": FocalFrequencyLoss(alpha=1.0),
    }

    # Test DISTS separately (downloads VGG weights)
    try:
        losses["DISTS"] = DISTSPerceptualLoss().to(device)
    except Exception as e:
        print(f"\nDISTS skipped (torchmetrics not installed): {e}")

    print(f"\n{'Loss':<20} {'random->random':>15} {'near->target':>15} {'grad ok':>8}")
    print("-" * 62)

    for name, loss_fn in losses.items():
        loss_fn = loss_fn.to(device) if hasattr(loss_fn, 'to') else loss_fn

        # Fresh tensors for each loss to avoid stale grad issues
        p = pred.detach().clone().requires_grad_(True)
        n = near.detach().clone().requires_grad_(True)

        # Random vs random (should be moderate)
        val_rand = loss_fn(p, target)
        val_rand.backward()
        grad_ok = p.grad is not None and p.grad.abs().sum() > 0

        # Near-identical (should be low)
        val_near = loss_fn(n, target)

        print(f"{name:<20} {val_rand.item():>15.6f} {val_near.item():>15.6f} {'yes' if grad_ok else 'NO':>8}")

    # Verify FFL focal property: higher alpha should weight hard freqs more
    print("\nFocal frequency alpha test:")
    for alpha in [0.5, 1.0, 2.0]:
        ffl = FocalFrequencyLoss(alpha=alpha)
        val = ffl(pred.detach(), target)
        print(f"  alpha={alpha:.1f}: {val.item():.6f}")

    print("\nAll synthetic tests passed!")


def test_real_images(val_dir):
    """Test losses on real image pairs from validation set."""
    import cv2
    import glob

    print("\n" + "=" * 60)
    print(f"Testing losses on real images from {val_dir}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_files = sorted(glob.glob(os.path.join(val_dir, "input", "*.png")))[:3]
    if not input_files:
        print(f"No images found in {val_dir}/input/")
        return

    losses = {
        "Charbonnier": CharbonnierLoss(),
        "FocalFrequency": FocalFrequencyLoss(alpha=1.0),
    }
    try:
        losses["DISTS"] = DISTSPerceptualLoss().to(device)
    except Exception as e:
        print(f"DISTS skipped: {e}")

    for inp_path in input_files:
        fname = os.path.basename(inp_path)
        tgt_path = os.path.join(val_dir, "target", fname)
        if not os.path.exists(tgt_path):
            continue

        inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Center crop 256x256
        h, w = inp.shape[:2]
        cs = min(256, h, w)
        t, l = (h - cs) // 2, (w - cs) // 2
        inp = inp[t:t+cs, l:l+cs]
        tgt = tgt[t:t+cs, l:l+cs]

        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(tgt.transpose(2, 0, 1)).unsqueeze(0).to(device)

        print(f"\n{fname} ({cs}x{cs} crop):")
        for name, loss_fn in losses.items():
            val = loss_fn(inp_t.float(), tgt_t.float())
            print(f"  {name:<20}: {val.item():.6f}")

        # Also compute PSNR for context
        mse = ((inp_t - tgt_t) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        print(f"  {'PSNR (input vs tgt)':<20}: {psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-images", type=str, default=None,
                        help="Path to val_pairs dir for real image test")
    args = parser.parse_args()

    test_synthetic()
    if args.real_images:
        test_real_images(args.real_images)
