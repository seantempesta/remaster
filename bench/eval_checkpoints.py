"""
Evaluate multiple training checkpoints on the validation set and graph results.

Runs each checkpoint on the held-out val_pairs, computes PSNR, pixel loss,
and perceptual loss. Outputs a comparison chart.

Usage:
    python bench/eval_checkpoints.py
    python bench/eval_checkpoints.py --val-dir data/val_pairs --ckpt-dir checkpoints/nafnet_distill/history
"""
import sys
import os
import glob
import math
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.nafnet_arch import NAFNet


def load_model(checkpoint_path, device="cpu"):
    """Load NAFNet from a checkpoint file."""
    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("params", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return model


@torch.no_grad()
def evaluate(model, val_dir, device="cpu", crop_size=512):
    """Evaluate model on validation set. Returns dict of metrics."""
    input_dir = os.path.join(val_dir, "input")
    target_dir = os.path.join(val_dir, "target")
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    psnrs = []
    l1_losses = []
    mse_losses = []

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
        top = (h - cs) // 2
        left = (w - cs) // 2
        inp = inp[top:top + cs, left:left + cs]
        tgt = tgt[top:top + cs, left:left + cs]

        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tgt_t = torch.from_numpy(tgt.transpose(2, 0, 1)).unsqueeze(0).to(device)

        out_t = model(inp_t).clamp(0, 1)

        mse = ((out_t - tgt_t) ** 2).mean().item()
        l1 = (out_t - tgt_t).abs().mean().item()
        psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 100.0

        psnrs.append(psnr)
        l1_losses.append(l1)
        mse_losses.append(mse)

    return {
        "psnr": np.mean(psnrs),
        "l1_loss": np.mean(l1_losses),
        "mse_loss": np.mean(mse_losses),
        "n_frames": len(psnrs),
    }


@torch.no_grad()
def generate_comparisons(checkpoints, val_dir, output_dir, num_frames=3, crop_size=512):
    """Generate side-by-side comparison images: Input | Target | Checkpoint outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_dir = os.path.join(val_dir, "input")
    target_dir = os.path.join(val_dir, "target")
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    # Pick evenly spaced frames
    indices = np.linspace(0, len(input_files) - 1, num_frames, dtype=int)
    selected = [input_files[i] for i in indices]

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx, inp_path in enumerate(selected):
        fname = os.path.basename(inp_path)
        tgt_path = os.path.join(target_dir, fname)
        if not os.path.exists(tgt_path):
            continue

        inp_img = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        tgt_img = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        inp_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        tgt_rgb = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        h, w, _ = inp_rgb.shape
        cs = min(crop_size, h, w)
        top = (h - cs) // 2
        left = (w - cs) // 2
        inp_crop = inp_rgb[top:top+cs, left:left+cs]
        tgt_crop = tgt_rgb[top:top+cs, left:left+cs]

        # Run each checkpoint
        n_cols = 2 + len(checkpoints)  # input + target + each checkpoint
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        axes[0].imshow(inp_crop)
        axes[0].set_title("Input", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(tgt_crop)
        axes[1].set_title("Target (SCUNet)", fontsize=10)
        axes[1].axis("off")

        inp_t = torch.from_numpy(inp_crop.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)

        for i, (ckpt_path, label) in enumerate(checkpoints):
            model = load_model(ckpt_path)
            out_t = model(inp_t).clamp(0, 1)
            out_np = (out_t.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            axes[i + 2].imshow(out_np)
            axes[i + 2].set_title(label, fontsize=10)
            axes[i + 2].axis("off")
            del model

        plt.suptitle(fname, fontsize=11, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"compare_{frame_idx}_{fname}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Comparison: {out_path}")


def plot_results(results, output_path="bench/checkpoint_eval.png"):
    """Plot metrics across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [r["iteration"] for r in results]
    psnrs = [r["psnr"] for r in results]
    l1s = [r["l1_loss"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR
    ax1.plot(iters, psnrs, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Validation PSNR (dB)")
    ax1.set_title("PSNR on Held-Out Validation Set")
    ax1.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(iters, psnrs)):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # L1 Loss
    ax2.plot(iters, l1s, "r-o", linewidth=2, markersize=8)
    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Validation L1 Loss")
    ax2.set_title("L1 Loss on Held-Out Validation Set")
    ax2.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(iters, l1s)):
        ax2.annotate(f"{y:.5f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on validation set")
    parser.add_argument("--val-dir", default="data/val_pairs")
    parser.add_argument("--ckpt-dir", default="checkpoints/nafnet_distill/history")
    parser.add_argument("--also-best", action="store_true", default=True,
                        help="Also evaluate nafnet_best.pth from parent dir")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--output", default="bench/checkpoint_eval.png")
    args = parser.parse_args()

    # Find checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "nafnet_iter*.pth")))
    if args.also_best:
        best = os.path.join(os.path.dirname(args.ckpt_dir), "nafnet_best.pth")
        if os.path.exists(best):
            ckpts.append(best)

    if not ckpts:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    print(f"Evaluating {len(ckpts)} checkpoints on {args.val_dir}")
    print()

    results = []
    for ckpt_path in ckpts:
        name = os.path.basename(ckpt_path)
        # Extract iteration number
        if "iter" in name:
            # Extract digits after "iter" — handles nafnet_iter005000.pth and nafnet_iter001000_best.pth
            import re
            m = re.search(r'iter(\d+)', name)
            iteration = int(m.group(1)) if m else 0
        elif "best" in name:
            # Try to figure out which iter it came from
            iteration = 1000  # known from training history
        else:
            iteration = 0

        print(f"  {name} (iter {iteration})...", end=" ", flush=True)
        model = load_model(ckpt_path)
        metrics = evaluate(model, args.val_dir, crop_size=args.crop_size)
        metrics["iteration"] = iteration
        metrics["checkpoint"] = name
        results.append(metrics)
        print(f"PSNR={metrics['psnr']:.2f} dB, L1={metrics['l1_loss']:.6f}")
        del model

    # Sort by iteration
    results.sort(key=lambda r: r["iteration"])

    # Print table
    print(f"\n{'Checkpoint':<30} {'Iter':>8} {'PSNR':>8} {'L1 Loss':>10} {'MSE Loss':>10} {'Frames':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['checkpoint']:<30} {r['iteration']:>8} {r['psnr']:>8.2f} "
              f"{r['l1_loss']:>10.6f} {r['mse_loss']:>10.8f} {r['n_frames']:>6}")

    # Plot
    if len(results) > 1:
        plot_results(results, args.output)

    # Side-by-side comparisons (pick best, worst, and a middle checkpoint)
    compare_dir = args.output.replace(".png", "_comparisons")
    compare_ckpts = []
    for r in results:
        label = f"iter {r['iteration']//1000}K ({r['psnr']:.1f}dB)"
        compare_ckpts.append((os.path.join(args.ckpt_dir, r["checkpoint"])
                              if r["checkpoint"] != "nafnet_best.pth"
                              else os.path.join(os.path.dirname(args.ckpt_dir), r["checkpoint"]),
                              label))
    print(f"\nGenerating side-by-side comparisons...")
    generate_comparisons(compare_ckpts, args.val_dir, compare_dir,
                        num_frames=4, crop_size=args.crop_size)

    # Save raw data
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    main()
