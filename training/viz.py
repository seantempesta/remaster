"""Training visualization: sample images and loss curves.

Generates visual outputs during training to track progress:
    - Sample comparison images (input | target | prediction) at each val step
    - Loss curve charts from the training log
"""
import os
import json
import math

import numpy as np
import cv2
import torch


class TrainingLogger:
    """Logs training metrics to a JSON file for charting."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.entries = []
        # Resume from existing log if present
        if os.path.exists(log_path):
            with open(log_path) as f:
                self.entries = json.load(f)

    def log_train(self, iteration, pixel_loss, perceptual_loss=None,
                  fft_loss=None, feat_loss=None, total_loss=None, lr=None):
        self.entries.append({
            "type": "train",
            "iter": iteration,
            "px": pixel_loss,
            "perc": perceptual_loss,
            "fft": fft_loss,
            "feat": feat_loss,
            "total": total_loss,
            "lr": lr,
        })

    def log_val(self, iteration, psnr, pixel_loss=None, perceptual_loss=None,
                fft_loss=None, total_loss=None):
        self.entries.append({
            "type": "val",
            "iter": iteration,
            "psnr": psnr,
            "px": pixel_loss,
            "perc": perceptual_loss,
            "fft": fft_loss,
            "total": total_loss,
        })

    def flush(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f)

    def plot_curves(self, output_path):
        """Generate loss curve charts from logged data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = [e for e in self.entries if e["type"] == "train"]
        val = [e for e in self.entries if e["type"] == "val"]

        if not train and not val:
            return

        # Determine which loss components we have
        has_perc = any(e.get("perc") is not None for e in train + val)
        has_fft = any(e.get("fft") is not None for e in train + val)
        has_feat = any(e.get("feat") is not None for e in train + val)

        # Layout: total loss + PSNR always, then optional component panels
        n_panels = 2  # total loss + PSNR
        if has_perc:
            n_panels += 1
        if has_fft:
            n_panels += 1
        if has_feat:
            n_panels += 1

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        panel = 0

        # Panel: Total loss (train + val)
        ax = axes[panel]
        if train:
            iters = [e["iter"] for e in train if e.get("total") is not None]
            vals = [e["total"] for e in train if e.get("total") is not None]
            if iters:
                ax.plot(iters, vals, "b-", alpha=0.4, linewidth=0.5, label="train")
        if val:
            iters = [e["iter"] for e in val if e.get("total") is not None]
            vals = [e["total"] for e in val if e.get("total") is not None]
            if iters:
                ax.plot(iters, vals, "r-o", markersize=4, linewidth=1.5, label="val")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        panel += 1

        # Panel: PSNR (val only)
        ax = axes[panel]
        if val:
            iters = [e["iter"] for e in val if e.get("psnr") is not None]
            psnrs = [e["psnr"] for e in val if e.get("psnr") is not None]
            if iters:
                ax.plot(iters, psnrs, "g-o", markersize=4, linewidth=1.5)
                for x, y in zip(iters, psnrs):
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Validation PSNR")
        ax.grid(True, alpha=0.3)
        panel += 1

        # Panel: Perceptual loss
        if has_perc:
            ax = axes[panel]
            if train:
                iters = [e["iter"] for e in train if e.get("perc") is not None]
                vals = [e["perc"] for e in train if e.get("perc") is not None]
                if iters:
                    ax.plot(iters, vals, "b-", alpha=0.4, linewidth=0.5, label="train")
            if val:
                iters = [e["iter"] for e in val if e.get("perc") is not None]
                vals = [e["perc"] for e in val if e.get("perc") is not None]
                if iters:
                    ax.plot(iters, vals, "r-o", markersize=4, linewidth=1.5, label="val")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Perceptual Loss")
            ax.set_title("Perceptual (DISTS)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            panel += 1

        # Panel: FFT loss
        if has_fft:
            ax = axes[panel]
            if train:
                iters = [e["iter"] for e in train if e.get("fft") is not None]
                vals = [e["fft"] for e in train if e.get("fft") is not None]
                if iters:
                    ax.plot(iters, vals, "b-", alpha=0.4, linewidth=0.5, label="train")
            if val:
                iters = [e["iter"] for e in val if e.get("fft") is not None]
                vals = [e["fft"] for e in val if e.get("fft") is not None]
                if iters:
                    ax.plot(iters, vals, "r-o", markersize=4, linewidth=1.5, label="val")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("FFT Loss")
            ax.set_title("Focal Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            panel += 1

        # Panel: Feature matching loss
        if has_feat:
            ax = axes[panel]
            if train:
                iters = [e["iter"] for e in train if e.get("feat") is not None]
                vals = [e["feat"] for e in train if e.get("feat") is not None]
                if iters:
                    ax.plot(iters, vals, "b-", alpha=0.4, linewidth=0.5, label="train")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Feature Loss")
            ax.set_title("Feature Matching")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            panel += 1

        plt.suptitle("Training Progress", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def _compute_psnr(img_a, img_b):
    """PSNR between two uint8 RGB numpy arrays."""
    a = img_a.astype(np.float64) / 255.0
    b = img_b.astype(np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10 * math.log10(1.0 / mse) if mse > 0 else 100.0


def build_snapshot_composite(inp_t, tgt_t, pred_t, iteration,
                             has_teacher=False):
    """Build a side-by-side composite from training batch tensors (zero cost).

    Grabs the first sample from each batch tensor, converts to uint8 RGB,
    composites with labels and PSNR. No extra inference — just tensor copies.

    Args:
        inp_t: input batch (B, 3, H, W) float [0,1] on GPU
        tgt_t: target/teacher batch (B, 3, H, W) float [0,1] on GPU
        pred_t: model prediction batch (B, 3, H, W) float [0,1] on GPU
        iteration: current training iteration (for label)
        has_teacher: if True, label middle panel "Teacher" else "Target"

    Returns:
        dict with keys: composite (uint8 RGB), input, target, prediction,
        inp_psnr, pred_psnr. Returns None if tensors are empty.
    """
    if inp_t.shape[0] == 0:
        return None

    # Grab first sample, move to CPU (non-blocking since we don't need it immediately)
    inp_np = (inp_t[0].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    tgt_np = (tgt_t[0].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pred_np = (pred_t[0].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    inp_psnr = _compute_psnr(inp_np, tgt_np)
    pred_psnr = _compute_psnr(pred_np, tgt_np)

    # Build composite: input | target/teacher | prediction
    h, w = inp_np.shape[:2]
    label_h = 32
    canvas = np.full((h + label_h, w * 3, 3), 255, dtype=np.uint8)
    canvas[label_h:, 0:w] = inp_np
    canvas[label_h:, w:w*2] = tgt_np
    canvas[label_h:, w*2:w*3] = pred_np

    target_label = "Teacher" if has_teacher else "Target"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Input ({inp_psnr:.1f} dB)", (8, 22),
                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{target_label} (reference)", (w + 8, 22),
                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Model iter {iteration} ({pred_psnr:.1f} dB)", (w*2 + 8, 22),
                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.line(canvas, (w, 0), (w, h + label_h), (180, 180, 180), 1)
    cv2.line(canvas, (w*2, 0), (w*2, h + label_h), (180, 180, 180), 1)

    return {
        "composite": canvas,
        "input": inp_np,
        "target": tgt_np,
        "prediction": pred_np,
        "inp_psnr": inp_psnr,
        "pred_psnr": pred_psnr,
    }


def save_val_samples(model, val_dir, output_dir, iteration, device,
                     num_samples=3, crop_size=512, teacher_model=None,
                     teacher_needs_noise_map=False, teacher_noise_level=0.0):
    """Save comparison images: input | target/teacher | student prediction.

    When teacher_model is provided, runs teacher live and measures PSNR for
    both input and student against the teacher output (teacher = reference).

    When teacher_model is None, falls back to using target/ PNGs from disk.

    Picks evenly-spaced frames from val set, runs inference, saves a
    side-by-side PNG for each. Uses center crop to match validation eval.

    Returns:
        List of dicts with keys: fname, input, target, student, inp_psnr,
        student_psnr, composite. Each value is a uint8 RGB numpy array.
        Empty list if no samples generated.
    """
    import glob

    input_dir = os.path.join(val_dir, "input")
    target_dir = os.path.join(val_dir, "target")
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not input_files:
        return []

    # Pick evenly spaced frames
    indices = np.linspace(0, len(input_files) - 1, num_samples, dtype=int)
    selected = [input_files[i] for i in indices]

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    results = []
    model.eval()
    with torch.no_grad():
        for inp_path in selected:
            fname = os.path.basename(inp_path)

            # Load and center-crop input
            inp_img = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            inp_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)

            h, w = inp_rgb.shape[:2]
            cs = min(crop_size, h, w)
            top = (h - cs) // 2
            left = (w - cs) // 2
            inp_crop = inp_rgb[top:top+cs, left:left+cs]

            inp_t = torch.from_numpy(
                inp_crop.astype(np.float32).transpose(2, 0, 1) / 255.0
            ).unsqueeze(0).to(device)

            # Get teacher/target output (live inference or from disk)
            if teacher_model is not None:
                if teacher_needs_noise_map:
                    noise_map = torch.full(
                        (1, 1, inp_t.shape[2], inp_t.shape[3]),
                        teacher_noise_level, device=device, dtype=inp_t.dtype,
                    )
                    teacher_out = teacher_model(torch.cat([inp_t, noise_map], dim=1)).clamp(0, 1)
                else:
                    teacher_out = teacher_model(inp_t).clamp(0, 1)
                target_crop = (teacher_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            else:
                tgt_path = os.path.join(target_dir, fname)
                if not os.path.exists(tgt_path):
                    continue
                tgt_img = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
                tgt_rgb = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
                target_crop = tgt_rgb[top:top+cs, left:left+cs]

            # Student prediction
            out_t = model(inp_t).clamp(0, 1)
            student_crop = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # PSNR: both measured against target (= reference)
            inp_psnr = _compute_psnr(inp_crop, target_crop)
            student_psnr = _compute_psnr(student_crop, target_crop)

            # Composite: input | target | student
            label_h = 32
            panel_w = cs
            canvas_w = panel_w * 3
            canvas_h = cs + label_h
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

            canvas[label_h:, 0:panel_w] = inp_crop
            canvas[label_h:, panel_w:panel_w*2] = target_crop
            canvas[label_h:, panel_w*2:panel_w*3] = student_crop

            target_label = "Teacher" if teacher_model is not None else "Target"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 0, 0)
            thickness = 1
            cv2.putText(canvas, f"Input ({inp_psnr:.1f} dB)", (8, 22),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(canvas, f"{target_label} (reference)", (panel_w + 8, 22),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(canvas, f"Student iter {iteration} ({student_psnr:.1f} dB)", (panel_w*2 + 8, 22),
                        font, font_scale, color, thickness, cv2.LINE_AA)

            cv2.line(canvas, (panel_w, 0), (panel_w, canvas_h), (180, 180, 180), 1)
            cv2.line(canvas, (panel_w*2, 0), (panel_w*2, canvas_h), (180, 180, 180), 1)

            base = os.path.splitext(fname)[0]
            out_path = os.path.join(
                samples_dir, f"iter{iteration:06d}_{base}_{student_psnr:.1f}dB.png"
            )
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            results.append({
                "fname": base,
                "input": inp_crop,
                "target": target_crop,
                "student": student_crop,
                "inp_psnr": inp_psnr,
                "student_psnr": student_psnr,
                "composite": canvas,
            })

    model.train()
    return results
