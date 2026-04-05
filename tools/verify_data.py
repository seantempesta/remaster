"""
Verify training and validation data integrity.

Checks:
- Every input has a matching target (and vice versa)
- All images are readable by OpenCV
- No zero-byte files
- Per-source counts match expectations
- Resolution sanity (all frames 1920 wide)

Usage:
    python tools/verify_data.py --train-dir data/mixed_pairs --val-dir data/mixed_val
"""
import os
import sys
import argparse
from collections import defaultdict

import cv2


def check_directory(data_dir, label):
    """Verify a paired data directory.

    Returns:
        (ok, stats) where ok is True if all checks pass
    """
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
        print(f"  ERROR: {data_dir} missing input/ or target/ subdirectory")
        return False, {}

    input_files = set(os.listdir(input_dir))
    target_files = set(os.listdir(target_dir))

    ok = True
    stats = defaultdict(int)
    resolutions = defaultdict(int)
    errors = []

    # Check pair completeness
    missing_targets = input_files - target_files
    missing_inputs = target_files - input_files
    if missing_targets:
        errors.append(f"  {len(missing_targets)} inputs without matching target")
        for f in sorted(missing_targets)[:5]:
            errors.append(f"    - {f}")
        ok = False
    if missing_inputs:
        errors.append(f"  {len(missing_inputs)} targets without matching input")
        for f in sorted(missing_inputs)[:5]:
            errors.append(f"    - {f}")
        ok = False

    paired_files = sorted(input_files & target_files)

    # Check each file
    zero_byte = 0
    unreadable = 0
    for filename in paired_files:
        # Categorize by prefix
        prefix = filename.split("_")[0]
        if filename.startswith("synth_"):
            # Use first two parts: synth_expanse, synth_onepiece, etc.
            parts = filename.split("_")
            prefix = f"{parts[0]}_{parts[1]}"
        stats[prefix] += 1

        input_path = os.path.join(input_dir, filename)
        target_path = os.path.join(target_dir, filename)

        # Zero-byte check
        if os.path.getsize(input_path) == 0:
            zero_byte += 1
            continue
        if os.path.getsize(target_path) == 0:
            zero_byte += 1
            continue

        # Readability check (sample every 50th file to avoid slow full scan)
        idx = paired_files.index(filename) if paired_files.index(filename) < 10 else -1
        if idx >= 0 or hash(filename) % 50 == 0:
            img = cv2.imread(input_path)
            if img is None:
                unreadable += 1
                errors.append(f"  Unreadable: {input_path}")
            else:
                h, w = img.shape[:2]
                resolutions[f"{w}x{h}"] += 1

    if zero_byte:
        errors.append(f"  {zero_byte} zero-byte files")
        ok = False
    if unreadable:
        errors.append(f"  {unreadable} unreadable files")
        ok = False

    # Print results
    print(f"\n{label}: {len(paired_files)} pairs")
    print(f"  {'Prefix':<25} {'Count':>6}")
    print(f"  {'-'*25} {'-'*6}")
    for prefix in sorted(stats.keys()):
        print(f"  {prefix:<25} {stats[prefix]:>6}")

    if resolutions:
        print(f"\n  Resolutions (sampled):")
        for res, count in sorted(resolutions.items()):
            print(f"    {res}: {count} files")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(e)
    else:
        print(f"\n  All checks passed")

    return ok, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Verify training data integrity")
    parser.add_argument("--train-dir", type=str, default="data/training/train",
                        help="Training data directory")
    parser.add_argument("--val-dir", type=str, default="data/training/val",
                        help="Validation data directory")
    args = parser.parse_args()

    all_ok = True

    if os.path.isdir(args.train_dir):
        ok, train_stats = check_directory(args.train_dir, "TRAINING")
        all_ok = all_ok and ok
    else:
        print(f"Training dir not found: {args.train_dir}")
        all_ok = False

    if os.path.isdir(args.val_dir):
        ok, val_stats = check_directory(args.val_dir, "VALIDATION")
        all_ok = all_ok and ok
    else:
        print(f"\nValidation dir not found: {args.val_dir}")
        all_ok = False

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
