"""
One-time cleanup of the data directory.

Moves useful comparison files into archive/, renames for clarity,
then deletes old experiment debris.

Run with --dry-run first to see what will happen.

Usage:
    python tools/cleanup_data.py --dry-run
    python tools/cleanup_data.py
"""
import os
import shutil
import argparse

DATA = "data"  # symlink to E:/upscale-data/


# --- Files/dirs to KEEP in place (active training data) ---
KEEP = {
    "mixed_pairs",
    "mixed_val",
    "output",
    "archive",
}

# --- Rename these for clarity, then move to archive/ ---
RENAME_TO_ARCHIVE = {
    # Full episode outputs (different model generations)
    "Firefly_S01E02_denoised.mp4": "Firefly_S01E02_scunet.mp4",
    "Firefly_S01E02_nafnet.mkv": "Firefly_S01E02_nafnet_w64.mkv",
    "Firefly_S01E02_w32mid4_gan.mkv": "Firefly_S01E02_nafnet_w32mid4.mkv",
    # Demo clips (same test clip, different models)
    "clip_mid_1080p.mp4": "clip_mid_1080p_original.mp4",
    "clip_mid_1080p_nafnet_gan_best.mkv": "clip_mid_1080p_nafnet_w64.mkv",
    "clip_mid_1080p_nafnet_w32mid4.mkv": "clip_mid_1080p_nafnet_w32mid4.mkv",
    "clip_mid_1080p_unetdenoise.mkv": "clip_mid_1080p_unetdenoise.mkv",
}

# --- Move to archive/ as-is ---
MOVE_TO_ARCHIVE = [
    "comparisons",
    "comparisons_nafnet",
]


def main():
    parser = argparse.ArgumentParser(description="Clean up data directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without doing it")
    args = parser.parse_args()

    archive = os.path.join(DATA, "archive")
    if not args.dry_run:
        os.makedirs(archive, exist_ok=True)

    # Step 1: Rename and archive useful files
    print("=== ARCHIVE (rename + move) ===")
    for old_name, new_name in sorted(RENAME_TO_ARCHIVE.items()):
        src = os.path.join(DATA, old_name)
        dst = os.path.join(archive, new_name)
        if os.path.exists(src):
            size_mb = os.path.getsize(src) / 1048576
            print(f"  {old_name} -> archive/{new_name} ({size_mb:.0f} MB)")
            if not args.dry_run:
                shutil.move(src, dst)
        else:
            print(f"  {old_name} (not found, skipping)")

    # Step 2: Move dirs to archive
    print("\n=== ARCHIVE (move dirs) ===")
    for name in sorted(MOVE_TO_ARCHIVE):
        src = os.path.join(DATA, name)
        dst = os.path.join(archive, name)
        if os.path.isdir(src):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(src) for f in fns
            ) / 1048576
            print(f"  {name}/ -> archive/{name}/ ({size:.0f} MB)")
            if not args.dry_run:
                shutil.move(src, dst)
        else:
            print(f"  {name}/ (not found, skipping)")

    # Step 3: Delete everything else
    print("\n=== DELETE ===")
    total_deleted = 0
    for entry in sorted(os.listdir(DATA)):
        if entry in KEEP:
            continue
        if entry in RENAME_TO_ARCHIVE:
            continue  # already moved
        path = os.path.join(DATA, entry)

        if os.path.isdir(path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(path) for f in fns
            ) / 1048576
        else:
            size = os.path.getsize(path) / 1048576

        print(f"  DELETE {entry} ({size:.0f} MB)")
        total_deleted += size

        if not args.dry_run:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    print(f"\n=== SUMMARY ===")
    if args.dry_run:
        print(f"Would free ~{total_deleted/1024:.1f} GB")
        print("Run without --dry-run to execute.")
    else:
        print(f"Freed ~{total_deleted/1024:.1f} GB")

        # Print what remains
        print("\nRemaining in data/:")
        for entry in sorted(os.listdir(DATA)):
            path = os.path.join(DATA, entry)
            if os.path.isdir(path):
                count = sum(len(fns) for _, _, fns in os.walk(path))
                print(f"  {entry}/ ({count} files)")
            else:
                size = os.path.getsize(path) / 1048576
                print(f"  {entry} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
