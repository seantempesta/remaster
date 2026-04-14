"""Generate before/after comparison images for README.

Scans video clips, finds interesting regions with high artifact visibility,
and creates labeled side-by-side crops.
"""

import sys
import os
import numpy as np
import cv2
import av

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(PROJECT_ROOT, "data", "archive")
ASSETS = os.path.join(PROJECT_ROOT, "assets")

# Clip pairs: (original, remastered, label)
CLIPS = [
    (f"{ARCHIVE}/firefly_s01e08_30s.mkv",
     f"{ARCHIVE}/firefly_s01e08_30s_student_finetuned.mkv",
     "firefly_e08"),
    (f"{ARCHIVE}/firefly_s01e04_30s.mkv",
     f"{ARCHIVE}/firefly_s01e04_30s_student_TRT.mkv",
     "firefly_e04"),
    (f"{ARCHIVE}/onepiece_s01e03_30s.mkv",
     f"{ARCHIVE}/onepiece_s01e03_30s_student_NOCOLORFIX.mkv",
     "onepiece"),
]


def decode_frame(path, frame_idx):
    """Decode a single frame from a video file."""
    container = av.open(path)
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img
    container.close()
    return None


def get_frame_count(path):
    container = av.open(path)
    stream = container.streams.video[0]
    # Estimate from duration
    n = stream.frames or int(stream.duration * stream.time_base * stream.average_rate) if stream.duration else 750
    container.close()
    return min(n, 900)  # cap


def score_region_interest(orig_crop, rem_crop):
    """Score how interesting a crop region is for comparison.
    High score = big visible difference + good detail content.
    """
    # Difference magnitude
    diff = np.abs(orig_crop.astype(np.float32) - rem_crop.astype(np.float32))
    mean_diff = diff.mean()

    # Edge content (we want regions with detail, not flat areas)
    gray_orig = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray_orig, cv2.CV_64F).var()

    # Penalize very dark or very bright regions
    brightness = gray_orig.mean()
    brightness_penalty = 1.0
    if brightness < 30:
        brightness_penalty = 0.3
    elif brightness < 50:
        brightness_penalty = 0.7
    elif brightness > 240:
        brightness_penalty = 0.5

    # Combined score: want both difference and detail
    score = mean_diff * (edges ** 0.3) * brightness_penalty
    return score


def score_dark_region(orig_crop, rem_crop):
    """Score for dark scene detail recovery."""
    diff = np.abs(orig_crop.astype(np.float32) - rem_crop.astype(np.float32))
    mean_diff = diff.mean()
    gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()
    edges = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Want dark but not black, with some detail
    if brightness < 15 or brightness > 80:
        return 0
    dark_score = mean_diff * (edges ** 0.25) * (1.0 / (brightness + 1)) * 50
    return dark_score


def score_face_region(orig_crop, rem_crop):
    """Score for face/skin regions - look for skin-tone pixels and banding."""
    # Simple skin tone detection in RGB
    r, g, b = orig_crop[:,:,0].astype(float), orig_crop[:,:,1].astype(float), orig_crop[:,:,2].astype(float)
    skin_mask = (r > 80) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 15) & (r < 250)
    skin_ratio = skin_mask.mean()

    if skin_ratio < 0.05:
        return 0

    diff = np.abs(orig_crop.astype(np.float32) - rem_crop.astype(np.float32))
    mean_diff = diff.mean()
    edges = cv2.Laplacian(cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    return mean_diff * skin_ratio * 100 * (edges ** 0.2)


def score_texture_region(orig_crop, rem_crop):
    """Score for texture/fabric detail."""
    gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
    # High frequency content = texture
    edges = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Also check for medium-frequency patterns (Gabor-like)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detail = np.abs(gray.astype(float) - blur.astype(float)).mean()

    diff = np.abs(orig_crop.astype(np.float32) - rem_crop.astype(np.float32))
    mean_diff = diff.mean()

    brightness = gray.mean()
    if brightness < 25 or brightness > 240:
        return 0

    return mean_diff * detail * (edges ** 0.15)


def find_best_crops(orig_path, rem_path, clip_label, crop_size=384, stride=128,
                    frame_indices=None, n_frames=15):
    """Scan frames and find best crop regions for each comparison category."""
    total_frames = get_frame_count(orig_path)
    if frame_indices is None:
        # Sample frames spread across the clip
        frame_indices = np.linspace(30, total_frames - 30, n_frames, dtype=int).tolist()

    categories = {
        'face': {'scorer': score_face_region, 'best_score': 0, 'best_data': None},
        'dark': {'scorer': score_dark_region, 'best_score': 0, 'best_data': None},
        'texture': {'scorer': score_texture_region, 'best_score': 0, 'best_data': None},
        'general': {'scorer': score_region_interest, 'best_score': 0, 'best_data': None},
    }

    for fi, frame_idx in enumerate(frame_indices):
        print(f"  [{clip_label}] Scanning frame {frame_idx} ({fi+1}/{len(frame_indices)})...")
        orig = decode_frame(orig_path, frame_idx)
        rem = decode_frame(rem_path, frame_idx)
        if orig is None or rem is None:
            continue

        h, w = orig.shape[:2]

        # Scan crop positions
        for y in range(0, h - crop_size, stride):
            for x in range(0, w - crop_size, stride):
                orig_crop = orig[y:y+crop_size, x:x+crop_size]
                rem_crop = rem[y:y+crop_size, x:x+crop_size]

                for cat_name, cat in categories.items():
                    score = cat['scorer'](orig_crop, rem_crop)
                    if score > cat['best_score']:
                        cat['best_score'] = score
                        cat['best_data'] = {
                            'orig': orig_crop.copy(),
                            'rem': rem_crop.copy(),
                            'frame': frame_idx,
                            'x': x, 'y': y,
                            'clip': clip_label,
                            'score': score,
                        }

    return categories


def add_label(img, text, position='top-left', font_scale=0.8, thickness=2, bg_alpha=0.7):
    """Add a text label with background to an image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 8

    if position == 'top-left':
        x, y = padding, padding
    elif position == 'top-right':
        x, y = img.shape[1] - tw - padding * 3, padding
    elif position == 'top-center':
        x, y = (img.shape[1] - tw) // 2 - padding, padding
    else:
        x, y = padding, padding

    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - padding),
                  (x + tw + padding, y + th + padding + baseline),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    # Draw text
    cv2.putText(img, text, (x, y + th), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def create_side_by_side(orig_crop, rem_crop, label_left="Original", label_right="Remastered",
                        border=3):
    """Create a side-by-side comparison with labels and separator."""
    h, w = orig_crop.shape[:2]

    # Add labels
    left = add_label(orig_crop, label_left, 'top-left')
    right = add_label(rem_crop, label_right, 'top-left')

    # White separator line
    sep = np.ones((h, border, 3), dtype=np.uint8) * 200

    combined = np.concatenate([left, sep, right], axis=1)
    return combined


def create_triple_strip(orig_crop, rem_crop, border=3):
    """Create Original | Remastered | Difference (5x amplified) strip."""
    h, w = orig_crop.shape[:2]

    # Compute amplified difference
    diff = np.abs(orig_crop.astype(np.float32) - rem_crop.astype(np.float32)) * 5.0
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    left = add_label(orig_crop, "Original", 'top-left')
    mid = add_label(rem_crop, "Remastered", 'top-left')
    right = add_label(diff, "Difference (5x)", 'top-left')

    sep = np.ones((h, border, 3), dtype=np.uint8) * 200
    combined = np.concatenate([left, sep, mid, sep, right], axis=1)
    return combined


def save_png(img_rgb, path):
    """Save RGB image as PNG."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    print(f"  Saved: {path} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")


def main():
    os.makedirs(ASSETS, exist_ok=True)

    # Collect best crops across all clips
    all_categories = {}

    for orig_path, rem_path, label in CLIPS:
        if not os.path.exists(orig_path):
            print(f"SKIP: {orig_path} not found")
            continue
        if not os.path.exists(rem_path):
            print(f"SKIP: {rem_path} not found")
            continue

        print(f"\nScanning {label}...")
        cats = find_best_crops(orig_path, rem_path, label, crop_size=384, stride=96, n_frames=20)

        for cat_name, cat in cats.items():
            if cat['best_data'] is not None:
                key = cat_name
                if key not in all_categories or cat['best_score'] > all_categories[key]['best_score']:
                    all_categories[key] = cat

    # Generate comparison images
    print("\n--- Generating comparison images ---")

    category_filenames = {
        'face': 'comparison_face.png',
        'dark': 'comparison_dark.png',
        'texture': 'comparison_texture.png',
        'general': 'comparison_wide.png',
    }

    hero_crops = []

    for cat_name, cat in all_categories.items():
        data = cat['best_data']
        if data is None:
            print(f"  No good crop found for {cat_name}")
            continue

        print(f"\n{cat_name}: clip={data['clip']}, frame={data['frame']}, "
              f"pos=({data['x']},{data['y']}), score={data['score']:.1f}")

        # Side-by-side comparison
        comp = create_side_by_side(data['orig'], data['rem'])
        save_png(comp, os.path.join(ASSETS, category_filenames[cat_name]))

        # Collect for hero strip
        hero_crops.append((data['orig'], data['rem'], cat_name))

    # Hero image: pick 3 best crops for a triple strip
    if hero_crops:
        print("\n--- Generating hero strip ---")
        # Use the first 3 (or fewer)
        strips = []
        for orig, rem, name in hero_crops[:3]:
            strip = create_triple_strip(orig, rem)
            strips.append(strip)

        # Stack vertically with separator
        if len(strips) > 1:
            max_w = max(s.shape[1] for s in strips)
            padded = []
            for s in strips:
                if s.shape[1] < max_w:
                    pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), dtype=np.uint8)
                    s = np.concatenate([s, pad], axis=1)
                padded.append(s)
            sep_h = np.ones((3, max_w, 3), dtype=np.uint8) * 200
            parts = []
            for i, s in enumerate(padded):
                if i > 0:
                    parts.append(sep_h)
                parts.append(s)
            hero = np.concatenate(parts, axis=0)
        else:
            hero = strips[0]

        save_png(hero, os.path.join(ASSETS, "hero_comparison.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
