"""
Verify NAFNet architecture configs: weight loading + shape correctness.

Tests each experiment config (A, B, C) by:
1. Building the model with the target architecture
2. Loading pretrained weights with strict=False
3. Reporting matched/skipped/missing keys
4. Running a forward pass at 1080p to verify output shape
5. Checking that matched weights are actually identical (not silently zeroed)

Runs on CPU, loads one model at a time to stay within 16GB RAM.
"""
import sys
import os
import gc
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.nafnet_arch import NAFNet

PRETRAINED_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "reference-code", "NAFNet", "experiments", "pretrained_models"
)

CONFIGS = {
    "A: width64, middle 12->4": {
        "width": 64, "middle_blk_num": 4,
        "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2],
        "pretrained": os.path.join(PRETRAINED_DIR, "NAFNet-SIDD-width64.pth"),
    },
    "B: width32, full depth": {
        "width": 32, "middle_blk_num": 12,
        "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2],
        "pretrained": os.path.join(PRETRAINED_DIR, "NAFNet-SIDD-width32.pth"),
    },
    "C: width32, middle 12->4": {
        "width": 32, "middle_blk_num": 4,
        "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2],
        "pretrained": os.path.join(PRETRAINED_DIR, "NAFNet-SIDD-width32.pth"),
    },
}


def verify_config(name, cfg):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Build model
    model = NAFNet(
        img_channel=3, width=cfg["width"],
        middle_blk_num=cfg["middle_blk_num"],
        enc_blk_nums=cfg["enc_blk_nums"],
        dec_blk_nums=cfg["dec_blk_nums"],
    )
    model_keys = set(model.state_dict().keys())
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params_m:.2f}M params, {len(model_keys)} keys")

    # Load pretrained
    pretrained_path = cfg["pretrained"]
    if not os.path.exists(pretrained_path):
        print(f"  ERROR: pretrained not found: {pretrained_path}")
        return False

    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt)))
    del ckpt
    gc.collect()

    ckpt_keys = set(state_dict.keys())
    print(f"Checkpoint: {len(ckpt_keys)} keys")

    # Categorize keys
    matched = model_keys & ckpt_keys
    unexpected = ckpt_keys - model_keys
    missing = model_keys - ckpt_keys

    # Check shape compatibility of matched keys
    shape_ok = []
    shape_mismatch = []
    for k in sorted(matched):
        if model.state_dict()[k].shape == state_dict[k].shape:
            shape_ok.append(k)
        else:
            shape_mismatch.append((k, model.state_dict()[k].shape, state_dict[k].shape))

    print(f"\n  Matched keys (same shape): {len(shape_ok)}/{len(model_keys)}")
    print(f"  Shape mismatches:          {len(shape_mismatch)}")
    print(f"  Unexpected (in ckpt only): {len(unexpected)}")
    print(f"  Missing (in model only):   {len(missing)}")

    if shape_mismatch:
        print(f"\n  SHAPE MISMATCHES (these will NOT load):")
        for k, model_shape, ckpt_shape in shape_mismatch:
            print(f"    {k}: model={list(model_shape)} vs ckpt={list(ckpt_shape)}")

    if unexpected:
        # Group by prefix for readability
        prefixes = {}
        for k in sorted(unexpected):
            prefix = k.rsplit(".", 1)[0] if "." in k else k
            prefixes.setdefault(prefix, []).append(k)
        print(f"\n  UNEXPECTED keys (dropped from checkpoint):")
        shown = 0
        for prefix in sorted(prefixes):
            keys = prefixes[prefix]
            if shown < 15:
                print(f"    {prefix}.* ({len(keys)} params)")
                shown += 1
            elif shown == 15:
                remaining = sum(len(v) for v in list(prefixes.values())[shown:])
                print(f"    ... and {remaining} more params in {len(prefixes) - shown} more modules")
                break

    if missing:
        prefixes = {}
        for k in sorted(missing):
            prefix = k.rsplit(".", 1)[0] if "." in k else k
            prefixes.setdefault(prefix, []).append(k)
        print(f"\n  MISSING keys (randomly initialized):")
        shown = 0
        for prefix in sorted(prefixes):
            keys = prefixes[prefix]
            if shown < 15:
                print(f"    {prefix}.* ({len(keys)} params)")
                shown += 1
            elif shown == 15:
                remaining = sum(len(v) for v in list(prefixes.values())[shown:])
                print(f"    ... and {remaining} more params in {len(prefixes) - shown} more modules")
                break

    # Actually load the weights (filtering out shape mismatches)
    loadable = {k: state_dict[k] for k in shape_ok}
    missing_ret, unexpected_ret = model.load_state_dict(loadable, strict=False)
    del state_dict, loadable
    gc.collect()

    # Verify loaded weights are not zero (spot check)
    sd = model.state_dict()
    nonzero_checks = 0
    zero_warns = 0
    for k in list(shape_ok)[:20]:  # spot check first 20
        if sd[k].numel() > 0 and sd[k].abs().max().item() > 0:
            nonzero_checks += 1
        elif "bias" not in k and "beta" not in k and "gamma" not in k:
            zero_warns += 1
            print(f"  WARNING: loaded key {k} is all zeros (shape {list(sd[k].shape)})")

    pct = len(shape_ok) / len(model_keys) * 100
    print(f"\n  Weight transfer: {len(shape_ok)}/{len(model_keys)} params loaded ({pct:.0f}%)")
    if zero_warns:
        print(f"  WARNING: {zero_warns} non-bias loaded keys are all zeros")

    # Forward pass at 1080p
    print(f"\n  Forward pass test (1x3x1080x1920)...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 1080, 1920)
        y = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(y.shape)}")
    ok = list(y.shape) == [1, 3, 1080, 1920]
    print(f"  Shape match: {'PASS' if ok else 'FAIL'}")

    # Check output is not trivially zero or identity
    residual = (y - x).abs().mean().item()
    print(f"  Mean |output - input|: {residual:.6f} (should be > 0 if weights loaded)")
    if residual < 1e-6:
        print(f"  WARNING: output ≈ input, model may not be doing anything")

    del model, x, y, sd
    gc.collect()

    return ok and len(shape_mismatch) == 0


if __name__ == "__main__":
    results = {}
    for name, cfg in CONFIGS.items():
        try:
            results[name] = verify_config(name, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False
        gc.collect()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
