"""
Create before/after comparison visualizations for remastered video.
Extracts frames from original and remastered, finds the most interesting
differences, and creates publication-quality comparison images.
"""
import subprocess
import sys
import os
import numpy as np
import cv2
from pathlib import Path

FFMPEG = "C:/Users/sean/src/upscale-experiment/bin/ffmpeg.exe"
ORIGINAL = "E:/plex/tv/Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)/Firefly (2002) - S01E04 - Shindig (1080p BluRay x265 Silence).mkv"
REMASTERED = "C:/Users/sean/src/upscale-experiment/data/archive/Firefly_S01E04_student_remaster.mkv"
ASSETS = Path("C:/Users/sean/src/upscale-experiment/assets")
ASSETS.mkdir(exist_ok=True)

# Timestamps to scan (MM:SS format) - extended range for better content
TIMESTAMPS = [
    "0:30", "1:00", "1:30", "2:00", "3:00", "4:00",
    "5:00", "6:00", "7:00", "8:00", "9:00", "10:00",
    "11:00", "12:00", "13:00", "14:00", "15:00",
    "16:00", "18:00", "20:00", "22:00", "25:00",
]


def extract_frame(video_path, timestamp):
    """Extract a single frame at given timestamp using ffmpeg."""
    cmd = [
        FFMPEG, "-ss", timestamp, "-i", video_path,
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "error", "pipe:1"
    ]
    # First get resolution
    probe_cmd = [
        FFMPEG, "-ss", timestamp, "-i", video_path,
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png",
        "-v", "error", "pipe:1"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.decode('utf-8', errors='replace')}")
        return None
    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def compute_interest_score(orig, remaster):
    """Score how visually interesting the difference is between frames."""
    # Convert to float
    o = orig.astype(np.float32)
    r = remaster.astype(np.float32)
    diff = np.abs(o - r)

    # Mean absolute difference
    mad = np.mean(diff)

    # Edge sharpness change (Laplacian variance)
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    rem_gray = cv2.cvtColor(remaster, cv2.COLOR_BGR2GRAY)
    orig_lap = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    rem_lap = cv2.Laplacian(rem_gray, cv2.CV_64F).var()
    sharpness_gain = rem_lap - orig_lap

    # Penalize very dark frames (low detail, boring)
    brightness = np.mean(orig_gray)
    brightness_factor = min(brightness / 80.0, 1.0)  # penalize below avg brightness 80

    # Bonus for edge-rich content (faces, textures, text)
    edge_bonus = min(orig_lap / 500.0, 1.0)

    # Combine: visible differences + sharpness improvement + interesting content
    score = (mad * 0.5 + max(sharpness_gain, 0) * 0.001) * brightness_factor * (0.5 + 0.5 * edge_bonus)
    return score, mad, sharpness_gain


def find_best_crop(orig, remaster, crop_size=512):
    """Find the crop region with the most interesting difference."""
    h, w = orig.shape[:2]
    best_score = -1
    best_pos = (0, 0)

    o = orig.astype(np.float32)
    r = remaster.astype(np.float32)
    diff = np.abs(o - r)

    # Also weight towards face-like regions (center-ish, upper half)
    step = crop_size // 4
    for y in range(0, h - crop_size, step):
        for x in range(0, w - crop_size, step):
            crop_diff = diff[y:y+crop_size, x:x+crop_size]
            # Mean difference in this crop
            score = np.mean(crop_diff)

            # Slight bonus for upper-center crops (faces tend to be there)
            cy = (y + crop_size // 2) / h
            cx = (x + crop_size // 2) / w
            center_bonus = 1.0 + 0.3 * (1.0 - abs(cx - 0.5)) * (1.0 - cy)
            score *= center_bonus

            # Bonus for edge-rich regions (textures, faces)
            orig_crop_gray = cv2.cvtColor(orig[y:y+crop_size, x:x+crop_size], cv2.COLOR_BGR2GRAY)
            edge_score = cv2.Laplacian(orig_crop_gray, cv2.CV_64F).var()
            score += edge_score * 0.002

            if score > best_score:
                best_score = score
                best_pos = (x, y)

    return best_pos


def add_label(img, text, position="top-left", font_scale=0.8, bg_alpha=0.7):
    """Add a text label with semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10

    if position == "top-left":
        x, y = padding, padding
    elif position == "top-right":
        x = img.shape[1] - tw - 3 * padding
        y = padding
    elif position == "top-center":
        x = (img.shape[1] - tw) // 2
        y = padding
    else:
        x, y = padding, padding

    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x - padding, y - padding),
                  (x + tw + padding, y + th + padding + baseline),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    # Draw text
    cv2.putText(img, text, (x, y + th), font, font_scale, (255, 255, 255), thickness)
    return img


def create_side_by_side(orig_crop, rem_crop, label="comparison", crop_size=512):
    """Create a side-by-side comparison with divider and labels."""
    divider_width = 3
    canvas = np.zeros((crop_size, crop_size * 2 + divider_width, 3), dtype=np.uint8)
    canvas[:, :crop_size] = orig_crop
    canvas[:, crop_size:crop_size + divider_width] = [200, 200, 200]  # gray divider
    canvas[:, crop_size + divider_width:] = rem_crop

    add_label(canvas, "Original", "top-left", 0.7)
    add_label(canvas, "Remastered", "top-right", 0.7)
    return canvas


def create_diff_map(orig, remaster, crop_size=512):
    """Create a color-coded difference map.
    Blue channel = removed noise/artifacts, Orange = added sharpness."""
    o = orig.astype(np.float32)
    r = remaster.astype(np.float32)

    # What was removed (original brighter than remastered = noise removed)
    removed = np.clip(o - r, 0, 255)
    # What was added (remastered brighter than original = detail added)
    added = np.clip(r - o, 0, 255)

    # Amplify for visibility (strong amplification to make differences obvious)
    amp = 15.0
    removed_amp = np.clip(removed * amp, 0, 255)
    added_amp = np.clip(added * amp, 0, 255)

    # Create color-coded map on dark background
    removed_gray = np.mean(removed_amp, axis=2)
    added_gray = np.mean(added_amp, axis=2)

    # Blue = removed noise, Orange = added detail
    canvas = np.zeros_like(orig, dtype=np.float32)
    # Blue channel for removed
    canvas[:, :, 0] = removed_gray * 1.0   # B
    canvas[:, :, 1] = removed_gray * 0.4   # G
    canvas[:, :, 2] = removed_gray * 0.1   # R
    # Orange for added
    canvas[:, :, 2] += added_gray * 1.0    # R
    canvas[:, :, 1] += added_gray * 0.6    # G
    canvas[:, :, 0] += added_gray * 0.1    # B

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # Add a subtle grid overlay
    for i in range(0, crop_size, crop_size // 8):
        canvas[i, :] = np.clip(canvas[i, :].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        canvas[:, i] = np.clip(canvas[:, i].astype(np.int16) + 20, 0, 255).astype(np.uint8)

    add_label(canvas, "Difference (15x amplified)", "top-left", 0.6)
    # Legend
    h = canvas.shape[0]
    cv2.putText(canvas, "Blue = noise removed", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 50), 1)
    cv2.putText(canvas, "Orange = detail added", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 150, 255), 1)
    return canvas


def create_removed_heatmap(orig, remaster, crop_size=512):
    """Create a heatmap of what the model removed (noise/artifacts)."""
    o = orig.astype(np.float32)
    r = remaster.astype(np.float32)

    # Total absolute change per pixel
    diff = np.mean(np.abs(o - r), axis=2)

    # Normalize and amplify (strong amplification for visibility)
    diff_norm = np.clip(diff * 12.0, 0, 255).astype(np.uint8)

    # Apply colormap (magma is brighter than inferno for small values)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_MAGMA)

    add_label(heatmap, "Artifact removal heatmap (12x)", "top-left", 0.6)
    return heatmap


def create_hero_montage(orig_full, rem_full, orig_crop, rem_crop, diff_crop, name):
    """Create the hero montage image (~1600x900)."""
    target_w = 1600

    # Scale full frames to fit width
    h, w = orig_full.shape[:2]
    scale = target_w / w
    small_h = int(h * scale * 0.5)
    small_w = target_w

    orig_small = cv2.resize(orig_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
    rem_small = cv2.resize(rem_full, (small_w, small_h), interpolation=cv2.INTER_AREA)

    row_gap = 4

    # Row 1: Original full + Remastered full side by side
    half_w = (target_w - row_gap) // 2
    orig_half = cv2.resize(orig_full, (half_w, small_h), interpolation=cv2.INTER_AREA)
    rem_half = cv2.resize(rem_full, (half_w, small_h), interpolation=cv2.INTER_AREA)
    top_row = np.zeros((small_h, target_w, 3), dtype=np.uint8)
    top_row[:, :half_w] = orig_half
    top_row[:, half_w:half_w + row_gap] = 80
    top_row[:, half_w + row_gap:half_w + row_gap + half_w] = rem_half
    add_label(top_row, "Original", "top-left", 0.9)
    overlay = top_row.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = half_w + row_gap + 10
    cv2.rectangle(overlay, (tx - 10, 0), (tx + 250, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, top_row, 0.3, 0, top_row)
    cv2.putText(top_row, "Remastered", (tx, 30), font, 0.9, (255, 255, 255), 2)

    # Row 2: Three panels that fill the width exactly
    # Each panel gets 1/3 of the width minus gaps
    panel_w = (target_w - row_gap * 2) // 3
    panel_h = panel_w  # square panels

    orig_crop_resized = cv2.resize(orig_crop, (panel_w, panel_h))
    rem_crop_resized = cv2.resize(rem_crop, (panel_w, panel_h))
    diff_resized = cv2.resize(diff_crop, (panel_w, panel_h))

    bottom_row = np.zeros((panel_h, target_w, 3), dtype=np.uint8)
    panels = [orig_crop_resized, rem_crop_resized, diff_resized]
    labels = ["Original (crop)", "Remastered (crop)", "Difference map"]
    x = 0
    for panel, label in zip(panels, labels):
        bottom_row[:, x:x + panel_w] = panel
        add_label(bottom_row[:, x:x + panel_w], label, "top-left", 0.6)
        x += panel_w + row_gap

    # Combine rows
    total_h = small_h + row_gap + panel_h
    montage = np.zeros((total_h, target_w, 3), dtype=np.uint8)
    montage[:small_h, :] = top_row
    montage[small_h:small_h + row_gap, :] = 80
    montage[small_h + row_gap:small_h + row_gap + panel_h, :] = bottom_row

    return montage


def main():
    print("=== Remaster Comparison Visualization ===")
    print(f"Output: {ASSETS}")

    # Step 1: Extract and score frames
    print("\n--- Step 1: Extracting and scoring frames ---")
    frames = []
    for ts in TIMESTAMPS:
        print(f"  Extracting at {ts}...")
        orig = extract_frame(ORIGINAL, ts)
        rem = extract_frame(REMASTERED, ts)
        if orig is None or rem is None:
            print(f"    SKIP (extraction failed)")
            continue
        if orig.shape != rem.shape:
            # Resize remaster to match original if needed
            rem = cv2.resize(rem, (orig.shape[1], orig.shape[0]))

        score, mad, sharpness = compute_interest_score(orig, rem)
        print(f"    MAD={mad:.2f}, sharpness_gain={sharpness:.1f}, score={score:.2f}")
        frames.append({
            "ts": ts, "orig": orig, "rem": rem,
            "score": score, "mad": mad, "sharpness": sharpness
        })

    # Sort by interest score
    frames.sort(key=lambda f: f["score"], reverse=True)
    print(f"\nRanking:")
    for i, f in enumerate(frames):
        marker = " <-- SELECTED" if i < 5 else ""
        print(f"  {i+1}. ts={f['ts']} score={f['score']:.2f} mad={f['mad']:.2f} sharp={f['sharpness']:.1f}{marker}")

    selected = frames[:5]

    # Step 2: Create visualizations for each selected frame
    print("\n--- Step 2: Creating visualizations ---")
    crop_size = 512
    categories = ["face", "texture", "dark", "detail", "wide"]

    hero_data = None

    for i, frame in enumerate(selected):
        ts = frame["ts"]
        orig = frame["orig"]
        rem = frame["rem"]
        cat = categories[i] if i < len(categories) else f"frame{i}"

        print(f"\n  Frame {i+1}: ts={ts} -> shindig_{cat}")

        # Find best crop
        cx, cy = find_best_crop(orig, rem, crop_size)
        print(f"    Best crop: ({cx}, {cy})")

        orig_crop = orig[cy:cy+crop_size, cx:cx+crop_size]
        rem_crop = rem[cy:cy+crop_size, cx:cx+crop_size]

        # a) Side-by-side
        sbs = create_side_by_side(orig_crop, rem_crop, cat, crop_size)
        path_sbs = ASSETS / f"shindig_{cat}_comparison.png"
        cv2.imwrite(str(path_sbs), sbs)
        print(f"    Saved: {path_sbs}")

        # b) Difference map
        diff = create_diff_map(orig_crop, rem_crop, crop_size)
        path_diff = ASSETS / f"shindig_{cat}_diff.png"
        cv2.imwrite(str(path_diff), diff)
        print(f"    Saved: {path_diff}")

        # c) Removed heatmap
        heatmap = create_removed_heatmap(orig_crop, rem_crop, crop_size)
        path_heat = ASSETS / f"shindig_{cat}_removed.png"
        cv2.imwrite(str(path_heat), heatmap)
        print(f"    Saved: {path_heat}")

        # Save the best one for hero montage
        if i == 0:
            hero_data = (orig, rem, orig_crop, rem_crop, diff, cat)

    # Step 3: Hero montage
    print("\n--- Step 3: Creating hero montage ---")
    if hero_data:
        orig_full, rem_full, orig_crop, rem_crop, diff_crop, cat = hero_data
        montage = create_hero_montage(orig_full, rem_full, orig_crop, rem_crop, diff_crop, cat)
        path_hero = ASSETS / "shindig_hero.png"
        cv2.imwrite(str(path_hero), montage)
        print(f"    Saved: {path_hero}")

    print("\n=== Done! ===")
    print(f"All images saved to: {ASSETS}")


if __name__ == "__main__":
    main()
