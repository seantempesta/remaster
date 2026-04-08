"""
Create a validation split from existing training data.

Moves a random 10% of frames matching specified prefixes from the training
directory to the validation directory. Uses a fixed seed for reproducibility.

Frames are MOVED (not copied) so they don't appear in both train and val sets.

Usage:
    python tools/create_val_split.py \\
        --train-dir data/mixed_pairs \\
        --val-dir data/mixed_val \\
        --prefixes hevc_ synth_expanse_ \\
        --fraction 0.10
"""
import os
import sys
import shutil
import random
import argparse
from pathlib import Path


def split_by_prefix(train_dir, val_dir, prefix, fraction, seed):
    """Move a fraction of files matching prefix from train to val.

    Args:
        train_dir: training data directory (contains input/ and target/)
        val_dir: validation data directory (creates input/ and target/)
        prefix: filename prefix to match (e.g. 'hevc_')
        fraction: fraction to move (e.g. 0.10 for 10%)
        seed: random seed

    Returns:
        number of pairs moved
    """
    train_input = os.path.join(train_dir, "input")
    train_target = os.path.join(train_dir, "target")
    val_input = os.path.join(val_dir, "input")
    val_target = os.path.join(val_dir, "target")

    os.makedirs(val_input, exist_ok=True)
    os.makedirs(val_target, exist_ok=True)

    # Find all matching files
    files = sorted(f for f in os.listdir(train_input) if f.startswith(prefix))
    if not files:
        print(f"  {prefix}: no files found")
        return 0

    # Select random subset
    rng = random.Random(seed)
    n_val = max(1, int(len(files) * fraction))
    val_files = set(rng.sample(files, n_val))

    moved = 0
    for filename in val_files:
        src_input = os.path.join(train_input, filename)
        src_target = os.path.join(train_target, filename)
        dst_input = os.path.join(val_input, filename)
        dst_target = os.path.join(val_target, filename)

        # Only move if both input and target exist
        if not os.path.exists(src_target):
            print(f"    WARN: missing target for {filename}, skipping")
            continue

        # Don't overwrite existing val files
        if os.path.exists(dst_input):
            continue

        shutil.move(src_input, dst_input)
        shutil.move(src_target, dst_target)
        moved += 1

    return moved


def main():
    parser = argparse.ArgumentParser(
        description="Create validation split by moving frames from train to val")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Training data directory (contains input/ and target/)")
    parser.add_argument("--val-dir", type=str, required=True,
                        help="Validation data directory")
    parser.add_argument("--prefixes", nargs="+", type=str, required=True,
                        help="Filename prefixes to split (e.g. 'hevc_' 'synth_expanse_')")
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="Fraction to move to validation (default: 0.10)")
    parser.add_argument("--seed", type=int, default=9999,
                        help="Random seed (default: 9999)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be moved without actually moving")
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.train_dir, "input")):
        print(f"ERROR: {args.train_dir}/input/ not found")
        sys.exit(1)

    print(f"Train dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Fraction:  {args.fraction:.0%}")
    print(f"Seed:      {args.seed}")
    print()

    total_moved = 0
    for prefix in args.prefixes:
        if args.dry_run:
            train_input = os.path.join(args.train_dir, "input")
            files = [f for f in os.listdir(train_input) if f.startswith(prefix)]
            n_val = max(1, int(len(files) * args.fraction))
            print(f"  {prefix}: would move {n_val} of {len(files)} pairs")
        else:
            moved = split_by_prefix(
                args.train_dir, args.val_dir, prefix,
                args.fraction, args.seed,
            )
            total_moved += moved
            print(f"  {prefix}: moved {moved} pairs")

    if not args.dry_run:
        # Print final counts
        val_input = os.path.join(args.val_dir, "input")
        if os.path.isdir(val_input):
            val_count = len(os.listdir(val_input))
            train_input = os.path.join(args.train_dir, "input")
            train_count = len(os.listdir(train_input))
            print(f"\nResult: {train_count} train + {val_count} val")
        print(f"Total moved: {total_moved} pairs")


if __name__ == "__main__":
    main()
