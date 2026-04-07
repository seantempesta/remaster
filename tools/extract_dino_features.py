"""Extract DINOv3 ViT-S features from video frames.

Step 1: Extract and cache feature vectors for all frames.
Step 2 (separate script): Experiment with cross-frame matching, averaging, etc.

Uses timm for model loading (no auth needed).
VRAM: ~300MB for ViT-S at 1080p FP16. Fine for RTX 3060 (6GB).
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import timm


# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(model_name="vit_small_patch16_dinov3", device="cuda"):
    """Load DINOv3 ViT-S (21M params) via timm."""
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device).half().eval()

    n_params = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  {n_params/1e6:.1f}M params, {vram_mb:.0f}MB VRAM")

    # Get model config
    data_config = timm.data.resolve_model_data_config(model)
    print(f"  Input size: {data_config.get('input_size', 'unknown')}")
    print(f"  Patch size: {getattr(model, 'patch_embed', None) and model.patch_embed.patch_size}")
    return model


def preprocess(img_bgr, device="cuda"):
    """Preprocess image for DINOv3 ViT. Pad to patch_size=16 multiple."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    # Pad to multiple of 16 (patch size)
    _, _, h, w = tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor.to(device).half()


@torch.no_grad()
def extract_features(model, tensor):
    """Extract patch-level features from DINOv3 ViT.

    Returns:
        patches: (1, num_patches, embed_dim) - spatial feature map
        cls: (1, embed_dim) - global image descriptor
        grid_h, grid_w: spatial dimensions of patch grid
    """
    B, C, H, W = tensor.shape
    patch_size = 16
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Use timm's forward_features to get intermediate representation
    features = model.forward_features(tensor)
    # features shape: (B, 1 + num_patches + n_register, embed_dim)
    # For DINOv3: [CLS] + [4 register tokens] + [patch tokens]

    cls_token = features[:, 0]  # (B, embed_dim)

    # DINOv3 has 4 register/storage tokens after CLS
    n_register = 4
    patch_tokens = features[:, 1 + n_register:]  # (B, num_patches, embed_dim)

    return {
        "patches": patch_tokens,  # (B, H*W, embed_dim)
        "cls": cls_token,  # (B, embed_dim)
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features from frames")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with PNG frames")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save feature .npz files")
    parser.add_argument("--model", type=str, default="vit_small_patch16_dinov3",
                        help="timm model name")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        print(f"No PNG files found in {input_dir}")
        return
    print(f"Found {len(frames)} frames in {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    total_time = 0
    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  WARNING: could not read {frame_path.name}")
            continue

        tensor = preprocess(img, device)

        t0 = time.time()
        feats = extract_features(model, tensor)
        dt = time.time() - t0
        total_time += dt

        # Save features
        save_dict = {
            "patches": feats["patches"].float().cpu().numpy(),  # (1, N, 384)
            "cls": feats["cls"].float().cpu().numpy(),  # (1, 384)
            "grid_h": np.array(feats["grid_h"]),
            "grid_w": np.array(feats["grid_w"]),
        }
        out_path = output_dir / f"{frame_path.stem}.npz"
        np.savez_compressed(str(out_path), **save_dict)

        if i == 0:
            patches = save_dict["patches"]
            print(f"\nFeature shapes for {img.shape[1]}x{img.shape[0]} input:")
            print(f"  patches: {patches.shape} (grid {feats['grid_h']}x{feats['grid_w']}, {patches.shape[-1]}-dim)")
            print(f"  cls: {save_dict['cls'].shape}")
            print(f"  File size: {out_path.stat().st_size/1024:.0f}KB")
            vram = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak VRAM: {vram:.0f}MB\n")

        fps = (i + 1) / total_time
        print(f"  [{i+1}/{len(frames)}] {frame_path.name} ({dt*1000:.0f}ms, {fps:.1f} fps)")

    print(f"\nDone! {len(frames)} frames in {total_time:.1f}s ({len(frames)/total_time:.1f} fps)")
    print(f"Features saved to: {output_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
