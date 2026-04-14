"""HYPIR 4K remaster comparison grid test.

Runs HYPIR at 2x upscale (1080p -> 4K) across multiple timestep configurations
on 10 diverse test frames. Applies wavelet color correction, downscales to 1080p,
and generates 7-panel comparison grids.

All outputs go to data/hypir-test/grid/ (non-destructive, no training data modified).

Usage:
    python tools/hypir_grid_test.py              # Run all 5 configs
    python tools/hypir_grid_test.py --run-hypir   # Only run HYPIR (skip post-processing)
    python tools/hypir_grid_test.py --post-only   # Only post-process + grids (skip HYPIR)
    python tools/hypir_grid_test.py --grids-only  # Only regenerate grids
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
HYPIR_ROOT = PROJECT_ROOT / "reference-code" / "HYPIR"

TRAIN_INPUT = DATA_ROOT / "training" / "train" / "input"
TRAIN_TARGET = DATA_ROOT / "training" / "train" / "target"

GRID_ROOT = DATA_ROOT / "hypir-test" / "grid"
INPUT_DIR = GRID_ROOT / "input"
SCUNET_DIR = GRID_ROOT / "scunet"
COMPARE_DIR = GRID_ROOT / "compare"

# ---------------------------------------------------------------------------
# Test frames (2 per show, verified to exist)
# ---------------------------------------------------------------------------
TEST_FRAMES = [
    "foundation_S03E01_00000.png",
    "dune2_Dune.Part.Two.2024.1_00050.png",
    "squidgame_S02E01_00000.png",
    "squidgame_S02E04_00050.png",
    "expanse_S02E01_00000.png",
    "expanse_S02E07_00050.png",
    "firefly_S01E05_00000.png",
    "firefly_S01E12_00050.png",
    "onepiece_S01E02_00000.png",
    "onepiece_S01E05_00050.png",
]

# ---------------------------------------------------------------------------
# HYPIR configurations: (name, model_t, coeff_t, upscale, source)
#   source = "input" means originals, "scunet" means SCUNet targets
#   Priority order: most useful configs first
# ---------------------------------------------------------------------------
CONFIGS = [
    ("hypir_2x_t25_noprompt",         25,  25, 2, "input"),
    ("hypir_2x_t50_noprompt",         50,  50, 2, "input"),
    ("hypir_2x_t100_noprompt",       100, 100, 2, "input"),
    ("hypir_1x_t25_scunet_noprompt",  25,  25, 1, "scunet"),
    ("hypir_2x_t25_scunet_noprompt",  25,  25, 2, "scunet"),
]

USE_EMPTY_CAPTION = True

# HYPIR config values from sd2_gradio.yaml
LORA_RANK = 256
LORA_MODULES = "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj"
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
WEIGHT_PATH = str(HYPIR_ROOT / "weights" / "HYPIR_sd2.pth")


# ---------------------------------------------------------------------------
# Wavelet color correction (from Upscale-A-Video, inlined)
# ---------------------------------------------------------------------------
def wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def wavelet_decomposition(image: torch.Tensor, levels: int = 5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Transfer color/lighting from source to target via wavelet decomposition.

    target: HYPIR output (has detail we want to keep)
    source: original frame (has color/lighting we want to keep)
    Returns: target's high-freq detail + source's low-freq color
    """
    content_high, _ = wavelet_decomposition(target)
    _, style_low = wavelet_decomposition(source)
    return content_high + style_low


# ---------------------------------------------------------------------------
# Step 1: Setup directories and copy test frames
# ---------------------------------------------------------------------------
def setup_dirs():
    print("=== Setting up directories ===")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCUNET_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_name, *_ in CONFIGS:
        (GRID_ROOT / cfg_name).mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in TEST_FRAMES:
        src_input = TRAIN_INPUT / fname
        src_target = TRAIN_TARGET / fname
        dst_input = INPUT_DIR / fname
        dst_target = SCUNET_DIR / fname

        if not src_input.exists():
            print(f"  WARNING: missing input {fname}")
            continue
        if not src_target.exists():
            print(f"  WARNING: missing target {fname}")
            continue

        if not dst_input.exists():
            shutil.copy2(src_input, dst_input)
        if not dst_target.exists():
            shutil.copy2(src_target, dst_target)
        copied += 1

    print(f"  {copied} frames ready in {GRID_ROOT}")


# ---------------------------------------------------------------------------
# Step 2: Run HYPIR for each configuration
# ---------------------------------------------------------------------------
def run_hypir_config(cfg_name: str, model_t: int, coeff_t: int, upscale: int, source: str):
    """Run HYPIR test.py for one configuration on all 10 frames."""
    output_dir = GRID_ROOT / cfg_name
    result_dir = output_dir / "result"

    # Check if already done
    if result_dir.exists():
        existing = list(result_dir.glob("*.png"))
        if len(existing) >= len(TEST_FRAMES):
            print(f"  {cfg_name}: already have {len(existing)} outputs, skipping")
            return True

    lq_dir = INPUT_DIR if source == "input" else SCUNET_DIR

    cmd = [
        sys.executable, str(HYPIR_ROOT / "test.py"),
        "--base_model_type", "sd2",
        "--base_model_path", BASE_MODEL,
        "--model_t", str(model_t),
        "--coeff_t", str(coeff_t),
        "--lora_rank", str(LORA_RANK),
        "--lora_modules", LORA_MODULES,
        "--weight_path", WEIGHT_PATH,
        "--patch_size", "256",
        "--stride", "128",
        "--upscale", str(upscale),
        "--lq_dir", str(lq_dir),
        "--output_dir", str(output_dir),
        "--captioner", "empty",
        "--seed", "42",
    ]

    print(f"\n  Running: {cfg_name} (model_t={model_t}, upscale={upscale}x, source={source})")
    print(f"  Command: {' '.join(cmd[-8:])}")

    result = subprocess.run(
        cmd,
        cwd=str(HYPIR_ROOT),
        env={**dict(__import__("os").environ), "PYTHONUTF8": "1"},
    )

    if result.returncode != 0:
        print(f"  ERROR: {cfg_name} failed with return code {result.returncode}")
        return False

    # Verify outputs
    outputs = list(result_dir.glob("*.png")) if result_dir.exists() else []
    print(f"  {cfg_name}: {len(outputs)} outputs generated")
    return len(outputs) >= len(TEST_FRAMES)


def run_all_hypir():
    print("\n=== Running HYPIR configurations ===")
    results = {}
    for cfg_name, model_t, coeff_t, upscale, source in CONFIGS:
        ok = run_hypir_config(cfg_name, model_t, coeff_t, upscale, source)
        results[cfg_name] = ok
        if not ok:
            print(f"  WARNING: {cfg_name} may be incomplete")

    print("\n=== HYPIR run summary ===")
    for name, ok in results.items():
        status = "OK" if ok else "INCOMPLETE"
        print(f"  {name}: {status}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Post-process (wavelet color correction + 1080p downscale)
# ---------------------------------------------------------------------------
def postprocess_config(cfg_name: str, source: str):
    """Apply wavelet color correction and 1080p downscale for one config."""
    result_dir = GRID_ROOT / cfg_name / "result"
    color_dir = GRID_ROOT / cfg_name / "color_corrected"
    down_dir = GRID_ROOT / cfg_name / "1080p"

    color_dir.mkdir(parents=True, exist_ok=True)
    down_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        print(f"  {cfg_name}: no results to post-process")
        return

    for fname in TEST_FRAMES:
        hypir_path = result_dir / fname
        if not hypir_path.exists():
            print(f"  {cfg_name}/{fname}: HYPIR output missing, skipping")
            continue

        color_path = color_dir / fname
        down_path = down_dir / fname

        # Skip if already done
        if color_path.exists() and down_path.exists():
            continue

        # Load HYPIR output and original
        hypir_img = Image.open(hypir_path).convert("RGB")
        # Color source is always the original (we want original's color/lighting)
        orig_img = Image.open(INPUT_DIR / fname).convert("RGB")

        # For color correction, resize original to match HYPIR output size
        orig_resized = orig_img.resize(hypir_img.size, Image.LANCZOS)

        # Convert to tensors
        hypir_t = torch.from_numpy(np.array(hypir_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        orig_t = torch.from_numpy(np.array(orig_resized)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Wavelet color correction
        corrected_t = wavelet_color_fix(hypir_t, orig_t)
        corrected_t = corrected_t.clamp(0, 1).squeeze(0).permute(1, 2, 0)
        corrected_np = (corrected_t.numpy() * 255).astype(np.uint8)
        corrected_img = Image.fromarray(corrected_np)

        # Save color-corrected full-res
        corrected_img.save(color_path)

        # Save 1080p downscale (match original height)
        orig_w, orig_h = orig_img.size
        down_img = corrected_img.resize((orig_w, orig_h), Image.LANCZOS)
        down_img.save(down_path)

    count = len(list(down_dir.glob("*.png")))
    print(f"  {cfg_name}: {count} frames color-corrected + downscaled")


def postprocess_all():
    print("\n=== Post-processing HYPIR outputs ===")
    for cfg_name, _, _, _, source in CONFIGS:
        postprocess_config(cfg_name, source)


# ---------------------------------------------------------------------------
# Step 4: Generate comparison grids
# ---------------------------------------------------------------------------
def get_font(size: int):
    """Get a font for labels, falling back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


GRID_PANELS = [
    ("Original", lambda fname: INPUT_DIR / fname),
    ("SCUNet GAN + USM", lambda fname: SCUNET_DIR / fname),
    ("HYPIR 2x t=25", lambda fname: GRID_ROOT / "hypir_2x_t25_noprompt" / "1080p" / fname),
    ("HYPIR 2x t=50", lambda fname: GRID_ROOT / "hypir_2x_t50_noprompt" / "1080p" / fname),
    ("HYPIR 2x t=100", lambda fname: GRID_ROOT / "hypir_2x_t100_noprompt" / "1080p" / fname),
    ("HYPIR 1x t=25 (SCUNet)", lambda fname: GRID_ROOT / "hypir_1x_t25_scunet_noprompt" / "1080p" / fname),
    ("HYPIR 2x t=25 (SCUNet)", lambda fname: GRID_ROOT / "hypir_2x_t25_scunet_noprompt" / "1080p" / fname),
]


def _draw_grid(panels, thumb_w, thumb_h, label_height, padding, font):
    """Draw a grid from (label, image_or_None) panels. 4 cols top, 3 cols bottom."""
    cols = 4
    rows = 2
    grid_w = cols * thumb_w + (cols + 1) * padding
    grid_h = rows * (thumb_h + label_height) + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), (32, 32, 32))
    draw = ImageDraw.Draw(grid)

    for idx, (label, img) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + label_height + padding)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (thumb_w - text_w) // 2, y + 4), label,
                  fill=(255, 255, 255), font=font)

        if img is not None:
            thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            grid.paste(thumb, (x, y + label_height))
        else:
            draw.rectangle(
                [x, y + label_height, x + thumb_w, y + label_height + thumb_h],
                fill=(64, 64, 64),
            )
            draw.text(
                (x + thumb_w // 2 - 30, y + label_height + thumb_h // 2),
                "(missing)", fill=(128, 128, 128), font=font,
            )
    return grid


def make_comparison_grid(fname: str):
    """Create a labeled 7-panel comparison grid for one frame.

    Row 1: Original | SCUNet | HYPIR 2x t=25 | HYPIR 2x t=100
    Row 2: HYPIR 2x t=200 | HYPIR 1x t=25 SCUNet | HYPIR 2x t=25 SCUNet
    """
    panels = []
    for label, path_fn in GRID_PANELS:
        path = path_fn(fname)
        img = Image.open(path).convert("RGB") if path.exists() else None
        panels.append((label, img))

    available = sum(1 for _, img in panels if img is not None)
    if available < 3:
        print(f"  {fname}: only {available} panels available, skipping grid")
        return

    ref_img = panels[0][1] or panels[1][1]
    if ref_img is None:
        return
    panel_w, panel_h = ref_img.size

    scale = min(1.0, 960 / panel_w)
    thumb_w = int(panel_w * scale)
    thumb_h = int(panel_h * scale)

    grid = _draw_grid(panels, thumb_w, thumb_h, 32, 4, get_font(20))
    grid_path = COMPARE_DIR / f"grid_{Path(fname).stem}.png"
    grid.save(grid_path)
    return grid_path


def make_detail_crop(fname: str, crop_size: int = 480):
    """Create a detail crop comparison -- center crops for texture comparison."""
    panels = []
    for label, path_fn in GRID_PANELS:
        path = path_fn(fname)
        if path.exists():
            img = Image.open(path).convert("RGB")
            w, h = img.size
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            left, top = max(0, cx - half), max(0, cy - half)
            crop = img.crop((left, top, min(w, left + crop_size), min(h, top + crop_size)))
            panels.append((label, crop))
        else:
            panels.append((label, None))

    if sum(1 for _, c in panels if c is not None) < 3:
        return

    grid = _draw_grid(panels, crop_size, crop_size, 28, 4, get_font(18))
    detail_path = COMPARE_DIR / f"detail_{Path(fname).stem}.png"
    grid.save(detail_path)
    return detail_path


def generate_grids():
    print("\n=== Generating comparison grids ===")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    for fname in TEST_FRAMES:
        grid_path = make_comparison_grid(fname)
        if grid_path:
            print(f"  Grid: {grid_path.name}")

        detail_path = make_detail_crop(fname)
        if detail_path:
            print(f"  Detail: {detail_path.name}")

    print(f"\n  Grids saved to: {COMPARE_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HYPIR comparison grid test")
    parser.add_argument("--run-hypir", action="store_true",
                        help="Only run HYPIR (skip post-processing and grids)")
    parser.add_argument("--post-only", action="store_true",
                        help="Only post-process and generate grids (skip HYPIR)")
    parser.add_argument("--grids-only", action="store_true",
                        help="Only regenerate comparison grids")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (e.g. hypir_2x_t25,hypir_2x_t100)")
    args = parser.parse_args()

    # Filter configs if specified
    global CONFIGS
    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        CONFIGS = [c for c in CONFIGS if c[0] in names]
        print(f"Running configs: {[c[0] for c in CONFIGS]}")

    if args.grids_only:
        generate_grids()
        return

    if not args.post_only:
        setup_dirs()

    if not args.post_only:
        run_all_hypir()

    if not args.run_hypir:
        postprocess_all()
        generate_grids()

    print("\n=== Done ===")
    print(f"  Results: {GRID_ROOT}")
    print(f"  Grids:   {COMPARE_DIR}")


if __name__ == "__main__":
    main()
