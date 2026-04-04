"""Download training artifacts from Modal volume.

Downloads checkpoints, sample images, loss curves, and training logs
to a local directory for inspection.

Usage:
    # Download latest run (auto-detects checkpoint dir on volume)
    python tools/download_training.py

    # Download specific run
    python tools/download_training.py drunet_nc64_nb2

    # Just list available runs
    python tools/download_training.py --list
"""
import modal
import os
import sys
import json
import argparse

VOL_NAME = "upscale-data"
LOCAL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output", "training")


def list_runs(vol):
    """List all training runs on the volume."""
    runs = []
    try:
        for entry in vol.listdir("checkpoints/"):
            name = entry.path.strip("/").split("/")[-1]
            if name and not name.startswith("."):
                runs.append(name)
    except Exception:
        pass
    return sorted(runs)


def download_run(vol, run_name, local_dir, force=False):
    """Download all artifacts for a training run."""
    remote_prefix = f"checkpoints/{run_name}"
    os.makedirs(local_dir, exist_ok=True)

    downloaded = []
    skipped = []

    def _should_download(local_path, immutable=False):
        if force:
            return True
        if not os.path.exists(local_path):
            return True
        if immutable and os.path.getsize(local_path) > 0:
            # Immutable files (samples, iter checkpoints) don't change
            return False
        # Mutable files (best.pth, latest.pth, logs, curves) always re-download
        return True

    # Core files
    for name in ["best.pth", "final.pth", "latest.pth",
                  "training_curves.png", "training_log.json"]:
        remote_path = f"/{remote_prefix}/{name}"
        local_path = os.path.join(local_dir, name)
        try:
            if not _should_download(local_path):
                size = os.path.getsize(local_path)
                skipped.append((name, size))
                continue
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(remote_path, f)
            size = os.path.getsize(local_path)
            if size == 0:
                os.remove(local_path)
                continue
            downloaded.append((name, size))
        except Exception:
            pass

    # Sample images
    samples_dir = os.path.join(local_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    try:
        for entry in vol.listdir(f"{remote_prefix}/samples/"):
            name = entry.path.split("/")[-1]
            if not name.endswith(".png"):
                continue
            local_path = os.path.join(samples_dir, name)
            if not _should_download(local_path, immutable=True):
                skipped.append((f"samples/{name}", os.path.getsize(local_path)))
                continue
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(f"/{remote_prefix}/samples/{name}", f)
            downloaded.append((f"samples/{name}", os.path.getsize(local_path)))
    except Exception:
        pass

    return downloaded, skipped


def print_summary(local_dir, run_name):
    """Print a summary of downloaded artifacts."""
    log_path = os.path.join(local_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            entries = json.load(f)

        train = [e for e in entries if e["type"] == "train"]
        val = [e for e in entries if e["type"] == "val"]

        if train:
            last = train[-1]
            print(f"\n  Latest training iter: {last['iter']}")
            print(f"    loss={last.get('total', '?'):.6f}  px={last.get('px', '?'):.6f}", end="")
            if last.get("feat") is not None:
                print(f"  feat={last['feat']:.4f}", end="")
            if last.get("perc") is not None:
                print(f"  perc={last['perc']:.4f}", end="")
            print()

        if val:
            best_val = max(val, key=lambda e: e.get("psnr", 0))
            last_val = val[-1]
            print(f"  Validations: {len(val)}")
            print(f"    Latest: iter {last_val['iter']}, PSNR={last_val.get('psnr', 0):.2f} dB")
            print(f"    Best:   iter {best_val['iter']}, PSNR={best_val.get('psnr', 0):.2f} dB")

    # Count samples
    samples_dir = os.path.join(local_dir, "samples")
    if os.path.isdir(samples_dir):
        samples = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
        print(f"  Sample images: {len(samples)}")

    # Check checkpoint sizes
    for name in ["best.pth", "latest.pth"]:
        path = os.path.join(local_dir, name)
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1024**2
            print(f"  {name}: {mb:.1f} MB")


def plot_curves(local_dir):
    """Generate training curves from the log."""
    log_path = os.path.join(local_dir, "training_log.json")
    if not os.path.exists(log_path):
        return

    with open(log_path) as f:
        entries = json.load(f)

    train = [e for e in entries if e["type"] == "train"]
    val = [e for e in entries if e["type"] == "val"]

    if not train:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping local curve plot)")
        return

    # Determine panels
    has_feat = any(e.get("feat") is not None for e in train)
    has_perc = any(e.get("perc") is not None for e in train)
    has_lr = any(e.get("lr") is not None for e in train)

    n_panels = 2  # loss + psnr
    if has_feat:
        n_panels += 1
    if has_perc:
        n_panels += 1
    if has_lr:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    panel = 0

    # Total loss
    ax = axes[panel]
    iters = [e["iter"] for e in train if e.get("total") is not None]
    vals = [e["total"] for e in train if e.get("total") is not None]
    if iters:
        ax.plot(iters, vals, "b-", alpha=0.5, linewidth=0.5, label="train")
    if val:
        vi = [e["iter"] for e in val if e.get("total") is not None]
        vv = [e["total"] for e in val if e.get("total") is not None]
        if vi:
            ax.plot(vi, vv, "ro", markersize=5, label="val")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    panel += 1

    # PSNR
    ax = axes[panel]
    if val:
        vi = [e["iter"] for e in val if e.get("psnr") is not None]
        vp = [e["psnr"] for e in val if e.get("psnr") is not None]
        if vi:
            ax.plot(vi, vp, "g-o", markersize=5, linewidth=1.5)
            for x, y in zip(vi, vp):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.grid(True, alpha=0.3)
    panel += 1

    # Feature matching
    if has_feat:
        ax = axes[panel]
        fi = [e["iter"] for e in train if e.get("feat") is not None]
        fv = [e["feat"] for e in train if e.get("feat") is not None]
        if fi:
            ax.plot(fi, fv, "b-", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feature Loss")
        ax.set_title("Feature Matching")
        ax.grid(True, alpha=0.3)
        panel += 1

    # Perceptual
    if has_perc:
        ax = axes[panel]
        pi = [e["iter"] for e in train if e.get("perc") is not None]
        pv = [e["perc"] for e in train if e.get("perc") is not None]
        if pi:
            ax.plot(pi, pv, "b-", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Perceptual Loss")
        ax.set_title("Perceptual (DISTS)")
        ax.grid(True, alpha=0.3)
        panel += 1

    # Learning rate / Prodigy D
    if has_lr:
        ax = axes[panel]
        li = [e["iter"] for e in train if e.get("lr") is not None]
        lv = [e["lr"] for e in train if e.get("lr") is not None]
        if li:
            ax.plot(li, lv, "b-", alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
        ax.grid(True, alpha=0.3)
        panel += 1

    plt.suptitle("Training Progress", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(local_dir, "curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download training artifacts from Modal")
    parser.add_argument("run", nargs="?", default=None,
                        help="Run name (checkpoint dir on volume). Auto-detects if omitted.")
    parser.add_argument("--list", action="store_true",
                        help="List available runs and exit")
    parser.add_argument("--force", action="store_true",
                        help="Re-download all files even if they exist locally")
    args = parser.parse_args()

    vol = modal.Volume.from_name(VOL_NAME)

    if args.list:
        runs = list_runs(vol)
        if runs:
            print("Available training runs:")
            for r in runs:
                print(f"  {r}")
        else:
            print("No training runs found on volume.")
        return

    # Auto-detect or use specified run
    if args.run:
        run_name = args.run
    else:
        runs = list_runs(vol)
        if not runs:
            print("No training runs found on volume.")
            return
        run_name = runs[-1]  # most recent alphabetically
        print(f"Auto-detected run: {run_name}")

    local_dir = os.path.abspath(os.path.join(LOCAL_OUTPUT, run_name))
    print(f"Downloading {run_name} -> {local_dir}")

    files, skipped = download_run(vol, run_name, local_dir, force=args.force)

    if not files and not skipped:
        print("  No files found for this run.")
        return

    if files:
        print(f"\nDownloaded {len(files)} files:")
        for name, size in files:
            if size > 1024 * 1024:
                print(f"  {name} ({size/1024**2:.1f} MB)")
            else:
                print(f"  {name} ({size/1024:.0f} KB)")
    if skipped:
        print(f"Skipped {len(skipped)} unchanged files")

    print_summary(local_dir, run_name)
    plot_curves(local_dir)

    print(f"\n{'='*60}")
    print(f"Output: {local_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
