"""
Compare SCUNet teacher variants and NAFNet student on validation frames.

Generates side-by-side comparison images:
  Original | SCUNet PSNR | SCUNet GAN | PSNR→GAN (stacked) | NAFNet 25K

Runs locally on GPU, one model at a time to minimize RAM usage.

Usage:
    python bench/compare_teachers.py --num-frames 4
    python bench/compare_teachers.py --num-frames 4 --no-stacked  # skip slow PSNR→GAN
"""
import sys
import os
import gc
import glob
import math
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.nafnet_arch import NAFNet


def load_scunet(model_name, device="cuda"):
    """Load SCUNet model. Imports SCUNet from reference code."""
    scunet_dir = str(Path(__file__).resolve().parent.parent / "reference-code" / "SCUNet")
    sys.path.insert(0, scunet_dir)
    from models.network_scunet import SCUNet as net

    model_path = os.path.join(scunet_dir, "model_zoo", f"{model_name}.pth")
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.half().to(device)
    return model


def load_nafnet(checkpoint_path, device="cuda"):
    """Load NAFNet from checkpoint."""
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("params", ckpt.get("model", ckpt))
    del ckpt
    gc.collect()
    model.load_state_dict(state)
    del state
    gc.collect()
    model = model.half().to(device).eval()
    return model


@torch.no_grad()
def process_frame(model, frame_rgb, device="cuda"):
    """Run a model on a single RGB float32 [0,1] frame. Returns RGB float32 [0,1]."""
    inp_t = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0).half().to(device)
    out_t = model(inp_t).clamp(0, 1)
    result = out_t.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)
    del inp_t, out_t
    return result


def compute_psnr(a, b):
    """PSNR between two float32 [0,1] images."""
    mse = ((a - b) ** 2).mean()
    return 10 * math.log10(1.0 / mse) if mse > 0 else 100.0


def main():
    parser = argparse.ArgumentParser(description="Compare SCUNet PSNR vs GAN vs NAFNet")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--val-dir", default="data/val_pairs")
    parser.add_argument("--nafnet-ckpt", default="checkpoints/nafnet_distill/safe/nafnet_best.pth")
    parser.add_argument("--output-dir", default="bench/teacher_comparison")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--no-stacked", action="store_true", help="Skip PSNR→GAN stacked pass")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(args.val_dir, "input", "*.png")))
    indices = np.linspace(0, len(input_files) - 1, args.num_frames, dtype=int)
    selected = [input_files[i] for i in indices]

    print(f"Comparing on {len(selected)} frames from {args.val_dir}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Load and process frames through each model ONE AT A TIME
    cs = args.crop_size
    frames_data = []

    # Step 0: Load all input frames
    for inp_path in selected:
        fname = os.path.basename(inp_path)
        tgt_path = os.path.join(args.val_dir, "target", fname)
        inp_img = cv2.cvtColor(cv2.imread(inp_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt_img = cv2.cvtColor(cv2.imread(tgt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = inp_img.shape[:2]
        crop_sz = min(cs, h, w)
        t, l = (h - crop_sz) // 2, (w - crop_sz) // 2
        frames_data.append({
            "fname": fname,
            "input": inp_img[t:t+crop_sz, l:l+crop_sz],
            "target_psnr": tgt_img[t:t+crop_sz, l:l+crop_sz],  # existing SCUNet PSNR output
            "full_input": inp_img,  # full res for model processing
            "crop": (t, l, crop_sz),
        })
    print(f"Loaded {len(frames_data)} frames")

    # Step 1: SCUNet GAN
    print("\nLoading SCUNet GAN...", flush=True)
    model = load_scunet("scunet_color_real_gan", device)
    vram = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    print(f"  VRAM: {vram:.0f}MB")
    for fd in frames_data:
        out = process_frame(model, fd["full_input"], device)
        t, l, crop_sz = fd["crop"]
        fd["scunet_gan"] = out[t:t+crop_sz, l:l+crop_sz]
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  Done, model unloaded")

    # Step 2: PSNR→GAN stacked (reload GAN, process the PSNR output)
    if not args.no_stacked:
        print("\nLoading SCUNet GAN for stacked pass (PSNR→GAN)...", flush=True)
        # We need full-res PSNR output. Load from target dir.
        model = load_scunet("scunet_color_real_gan", device)
        for fd in frames_data:
            tgt_path = os.path.join(args.val_dir, "target", fd["fname"])
            psnr_full = cv2.cvtColor(cv2.imread(tgt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out = process_frame(model, psnr_full, device)
            t, l, crop_sz = fd["crop"]
            fd["stacked"] = out[t:t+crop_sz, l:l+crop_sz]
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("  Done, model unloaded")

    # Step 3: NAFNet 25K
    print(f"\nLoading NAFNet from {args.nafnet_ckpt}...", flush=True)
    model = load_nafnet(args.nafnet_ckpt, device)
    vram = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    print(f"  VRAM: {vram:.0f}MB")
    for fd in frames_data:
        out = process_frame(model, fd["full_input"], device)
        t, l, crop_sz = fd["crop"]
        fd["nafnet"] = out[t:t+crop_sz, l:l+crop_sz]
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  Done, model unloaded")

    # Step 4: Generate comparison images
    print("\nGenerating comparisons...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_stacked = not args.no_stacked and "stacked" in frames_data[0]
    n_cols = 5 if has_stacked else 4

    for i, fd in enumerate(frames_data):
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))

        variants = [
            ("Input\n(compressed)", fd["input"]),
            ("SCUNet PSNR\n(teacher)", fd["target_psnr"]),
            ("SCUNet GAN", fd["scunet_gan"]),
        ]
        if has_stacked:
            variants.append(("PSNR→GAN\n(stacked)", fd["stacked"]))
        variants.append(("NAFNet 25K\n(student)", fd["nafnet"]))

        for j, (title, img) in enumerate(variants):
            # Compute PSNR vs SCUNet PSNR (teacher) for each
            if j > 0:
                psnr = compute_psnr(img, fd["target_psnr"])
                title += f"\n({psnr:.1f} dB vs PSNR)"
            axes[j].imshow(np.clip(img, 0, 1))
            axes[j].set_title(title, fontsize=9)
            axes[j].axis("off")

        plt.suptitle(fd["fname"], fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"compare_{i}_{fd['fname']}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")

    # Summary metrics
    print("\n" + "=" * 60)
    print("Average PSNR vs SCUNet PSNR teacher:")
    for name, key in [("SCUNet GAN", "scunet_gan"), ("NAFNet 25K", "nafnet")] + \
                     ([("PSNR→GAN stacked", "stacked")] if has_stacked else []):
        psnrs = [compute_psnr(fd[key], fd["target_psnr"]) for fd in frames_data]
        print(f"  {name:20s}: {np.mean(psnrs):.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
