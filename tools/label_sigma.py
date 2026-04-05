"""
Streamlit app for calibrating per-frame denoising parameters.

For each sample, user dials in two parameters:
  - Sigma: denoising strength (how aggressively to remove noise)
  - Detail recovery: how much edge detail to blend back from original

The detail recovery works by:
  1. Compute detail residual: original - denoised
  2. Compute edge mask from denoised frame (clean edges via Sobel)
  3. Blend back: output = denoised + residual * edge_mask * strength

This removes noise in flat areas while preserving sharpness on edges.

Usage:
    streamlit run tools/label_sigma.py
"""
import base64
import io
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

CALIBRATION_DIR = Path("data/calibration")
GRIDS_DIR = CALIBRATION_DIR / "grids"
SAMPLES_PATH = CALIBRATION_DIR / "samples.pkl"
LABELS_PATH = CALIBRATION_DIR / "labels.pkl"
SIGMA_OPTIONS = [0, 1, 2, 3, 5, 8, 10, 15]


def load_samples():
    if not SAMPLES_PATH.exists():
        st.error("No samples found. Run: python tools/calibrate_sigma.py")
        st.stop()
    with open(SAMPLES_PATH, "rb") as f:
        return pickle.load(f)


def load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save_labels(labels):
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)


def load_crop(img_path, crop_size=384):
    """Load image and take center crop."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]


DETAIL_METHODS = ["high_pass", "unsharp_mask", "laplacian_pyramid"]


def detail_high_pass(original, denoised, strength, blur_sigma=2.5):
    """High-pass injection: extract high-freq from original, add to denoised.

    Captures texture and fine detail at all locations (not just edges).
    blur_sigma controls crossover: higher = more detail transferred.
    """
    orig_f = original.astype(np.float32)
    den_f = denoised.astype(np.float32)
    # High-pass = original - blurred original
    low_pass = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=blur_sigma)
    high_pass = orig_f - low_pass
    result = den_f + strength * high_pass
    return np.clip(result, 0, 255).astype(np.uint8)


def detail_unsharp_mask(original, denoised, strength, blur_sigma=1.5):
    """Unsharp mask: amplify existing detail in denoised image.

    Safe — only amplifies what survived denoising, never reintroduces noise.
    """
    den_f = denoised.astype(np.float32)
    blurred = cv2.GaussianBlur(den_f, (0, 0), sigmaX=blur_sigma)
    detail = den_f - blurred
    result = den_f + strength * detail
    return np.clip(result, 0, 255).astype(np.uint8)


def detail_laplacian_pyramid(original, denoised, strength, levels=5):
    """Laplacian pyramid blend: fine detail from original, coarse from denoised.

    Strength controls how much fine-level detail comes from original.
    Level 0 (finest) blended at strength*0.3, level 1 at strength*0.7,
    level 2 at strength*1.0, coarser levels stay denoised.
    """
    def build_laplacian_pyr(img, n):
        pyr = []
        current = img.astype(np.float32)
        for i in range(n - 1):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            pyr.append(current - up)
            current = down
        pyr.append(current)
        return pyr

    lp_orig = build_laplacian_pyr(original, levels)
    lp_den = build_laplacian_pyr(denoised, levels)

    # Per-level blend weights: fine levels get more from original
    # Level 0 (pixel noise) = cautious, level 1-2 (texture) = aggressive
    level_weights = [0.3, 0.7, 1.0] + [0.5] * (levels - 3)

    blended = []
    for i in range(levels):
        if i < levels - 1:
            w = strength * level_weights[min(i, len(level_weights) - 1)]
            blended.append(lp_den[i] * (1 - w) + lp_orig[i] * w)
        else:
            # Coarsest level: always from denoised
            blended.append(lp_den[i])

    # Reconstruct from pyramid
    current = blended[-1]
    for i in range(levels - 2, -1, -1):
        current = cv2.pyrUp(current, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        current = current + blended[i]

    return np.clip(current, 0, 255).astype(np.uint8)


def apply_detail_recovery(original, denoised, strength, method="high_pass"):
    """Apply detail recovery using the selected method."""
    if strength <= 0:
        return denoised
    if method == "high_pass":
        return detail_high_pass(original, denoised, strength)
    elif method == "unsharp_mask":
        return detail_unsharp_mask(original, denoised, strength)
    elif method == "laplacian_pyramid":
        return detail_laplacian_pyramid(original, denoised, strength)
    return denoised


def img_to_b64(img_rgb):
    """Convert numpy RGB array to base64 PNG string."""
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def zoomable_image_row(images, captions, lens_zoom=3, lens_size=180, uid="main"):
    """Render a row of images that fill the container width, with linked hover-zoom.

    Images scale to fit side-by-side in the available width. Hovering on any
    image shows a synchronized magnified overlay on all images.
    """
    n = len(images)
    b64s = [img_to_b64(img) for img in images]
    img_h, img_w = images[0].shape[:2]
    aspect = img_h / img_w

    # Each image gets equal share of row width minus gaps
    gap = 8
    # Use CSS calc so images fill whatever the iframe width is
    cell_css_w = f"calc((100% - {gap * (n - 1)}px) / {n})"

    img_tags = ""
    for i, (b64, cap) in enumerate(zip(b64s, captions)):
        img_tags += f"""
        <div class="zoom-cell" style="width:{cell_css_w};">
          <div class="zoom-wrap" id="wrap-{uid}-{i}" style="aspect-ratio:{img_w}/{img_h};">
            <img id="img-{uid}-{i}" src="data:image/png;base64,{b64}" />
            <div class="lens" id="lens-{uid}-{i}"></div>
          </div>
          <div class="cap">{cap}</div>
        </div>
        """

    html = f"""
    <style>
      .zoom-row {{
        display: flex;
        gap: {gap}px;
        width: 100%;
      }}
      .zoom-cell {{
        flex-shrink: 0;
      }}
      .zoom-wrap {{
        position: relative;
        overflow: hidden;
        cursor: crosshair;
        border: 1px solid #444;
        border-radius: 4px;
        width: 100%;
      }}
      .zoom-wrap img {{
        width: 100%;
        height: 100%;
        display: block;
        object-fit: contain;
      }}
      .lens {{
        position: absolute;
        width: {lens_size}px;
        height: {lens_size}px;
        border: 2px solid #fff;
        border-radius: 4px;
        box-shadow: 0 0 8px rgba(0,0,0,0.6);
        background-repeat: no-repeat;
        pointer-events: none;
        display: none;
        z-index: 10;
      }}
      .cap {{
        text-align: center;
        color: #aaa;
        font-size: 13px;
        margin-top: 4px;
      }}
    </style>
    <div class="zoom-row">
      {img_tags}
    </div>
    <script>
      (function() {{
        const uid = "{uid}";
        const n = {n};
        const lensZoom = {lens_zoom};
        const lensSize = {lens_size};

        function updateAll(ratioX, ratioY, show) {{
          for (let i = 0; i < n; i++) {{
            const wrap = document.getElementById('wrap-' + uid + '-' + i);
            const img = document.getElementById('img-' + uid + '-' + i);
            const lens = document.getElementById('lens-' + uid + '-' + i);
            if (!show) {{ lens.style.display = 'none'; continue; }}
            lens.style.display = 'block';
            const imgW = wrap.clientWidth;
            const imgH = wrap.clientHeight;
            let lx = ratioX * imgW - lensSize / 2;
            let ly = ratioY * imgH - lensSize / 2;
            lx = Math.max(0, Math.min(lx, imgW - lensSize));
            ly = Math.max(0, Math.min(ly, imgH - lensSize));
            lens.style.left = lx + 'px';
            lens.style.top = ly + 'px';
            lens.style.backgroundImage = 'url(' + img.src + ')';
            lens.style.backgroundSize = (imgW * lensZoom) + 'px ' + (imgH * lensZoom) + 'px';
            const bgX = -(ratioX * imgW * lensZoom - lensSize / 2);
            const bgY = -(ratioY * imgH * lensZoom - lensSize / 2);
            lens.style.backgroundPosition = bgX + 'px ' + bgY + 'px';
          }}
        }}

        for (let i = 0; i < n; i++) {{
          const wrap = document.getElementById('wrap-' + uid + '-' + i);
          wrap.addEventListener('mousemove', function(e) {{
            const rect = wrap.getBoundingClientRect();
            updateAll((e.clientX - rect.left) / rect.width,
                      (e.clientY - rect.top) / rect.height, true);
          }});
          wrap.addEventListener('mouseleave', function() {{ updateAll(0, 0, false); }});
        }}
      }})();
    </script>
    """
    # Height: images at 1/n of container width, plus caption
    # Use a generous estimate; Streamlit iframe will scroll if needed
    row_h = int(600 * aspect / n * n) + 40  # rough
    row_h = max(300, min(row_h, 800))
    components.html(html, height=row_h, scrolling=False)


def main():
    st.set_page_config(page_title="Sigma Calibration", layout="wide")
    st.title("Sigma + Detail Recovery Calibration")

    samples = load_samples()
    total = len(samples)

    # Initialize session state
    if "labels" not in st.session_state:
        st.session_state.labels = load_labels()
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = len(st.session_state.labels)

    labels = st.session_state.labels
    idx = st.session_state.current_idx

    # Progress
    n_labeled = len(labels)
    st.progress(n_labeled / total if total > 0 else 0)
    st.write(f"**{n_labeled}/{total}** labeled")

    if idx >= total:
        st.success(f"All {total} samples labeled!")
        st.write("Run: `python tools/calibrate_sigma.py --fit`")
        if labels:
            st.subheader("Label summary")
            for l in labels:
                sigma = l.get('chosen_sigma', '?')
                detail = l.get('detail_strength', 0)
                method = l.get('detail_method', 'high_pass')
                st.write(f"  {l['filename']}: noise={l['noise_level']:.1f} "
                         f"-> sigma={sigma}, {method} {detail:.2f}")
        return

    sample = samples[idx]
    grid_dir = Path(sample.get("grid_dir", GRIDS_DIR / f"sample_{idx:03d}"))

    st.write(f"**Sample {idx + 1}/{total}** | "
             f"**Source:** {sample['source']} | "
             f"**Noise:** {sample['noise_level']:.2f} | "
             f"**Bucket:** {sample['bucket']}")

    # Controls and buttons all in one compact row at top
    col_sigma, col_method, col_detail = st.columns([2, 2, 2])
    with col_sigma:
        sigma = st.select_slider("Sigma (denoise strength)",
                                 options=SIGMA_OPTIONS, value=5,
                                 key=f"sigma_slider_{idx}")
    with col_method:
        detail_method = st.selectbox("Detail method", DETAIL_METHODS,
                                     key=f"method_{idx}",
                                     help="high_pass: injects texture from original. "
                                          "unsharp_mask: amplifies existing detail (safe). "
                                          "laplacian_pyramid: multi-scale blend (best quality).")
    with col_detail:
        detail_strength = st.slider("Detail recovery", 0.0, 2.0, 0.0, 0.05,
                                    key=f"detail_slider_{idx}",
                                    help="Strength of detail recovery (try 0.3-1.0)")

    # Action buttons right below controls
    b1, b2, b3, b4 = st.columns([3, 2, 1, 1])
    with b1:
        if st.button(f"Accept (s={sigma}, {detail_method} {detail_strength:.2f})",
                     key=f"accept_{idx}", type="primary"):
            labels.append({
                "sample_idx": idx,
                "filename": sample["filename"],
                "noise_level": sample["noise_level"],
                "bucket": sample["bucket"],
                "source": sample["source"],
                "chosen_sigma": sigma,
                "detail_strength": detail_strength,
                "detail_method": detail_method,
            })
            save_labels(labels)
            st.session_state.labels = labels
            st.session_state.current_idx = idx + 1
            st.rerun()
    with b2:
        if st.button("No processing needed", key=f"noprocess_{idx}"):
            labels.append({
                "sample_idx": idx,
                "filename": sample["filename"],
                "noise_level": sample["noise_level"],
                "bucket": sample["bucket"],
                "source": sample["source"],
                "chosen_sigma": 0,
                "detail_strength": 0.0,
            })
            save_labels(labels)
            st.session_state.labels = labels
            st.session_state.current_idx = idx + 1
            st.rerun()
    with b3:
        if st.button("Skip", key=f"skip_{idx}"):
            labels.append({
                "sample_idx": idx,
                "filename": sample["filename"],
                "noise_level": sample["noise_level"],
                "bucket": sample["bucket"],
                "source": sample["source"],
                "chosen_sigma": None,
                "detail_strength": 0.0,
            })
            save_labels(labels)
            st.session_state.labels = labels
            st.session_state.current_idx = idx + 1
            st.rerun()
    with b4:
        if st.button("Undo", key=f"undo_{idx}"):
            if labels:
                labels.pop()
                save_labels(labels)
                st.session_state.labels = labels
                st.session_state.current_idx = max(0, idx - 1)
                st.rerun()

    # Defaults for display settings (no sidebar)
    crop_size = 384
    lens_zoom = 3
    lens_size = 250

    # Load images
    orig_path = grid_dir / "original.png"
    sigma_path = grid_dir / f"sigma_{sigma}.png"

    orig_crop = load_crop(orig_path, crop_size) if orig_path.exists() else None
    denoised_crop = load_crop(sigma_path, crop_size) if sigma_path.exists() else None

    if orig_crop is not None and denoised_crop is not None:
        result = apply_detail_recovery(orig_crop, denoised_crop, detail_strength, detail_method)

        zoomable_image_row(
            [orig_crop, result, denoised_crop],
            ["Original", f"Result (s={sigma}, {detail_method} {detail_strength:.2f})",
             f"Denoised only (s={sigma})"],
            lens_zoom, lens_size, uid="main",
        )

    # Settings and detail map at bottom
    with st.expander("Display settings"):
        crop_size = st.slider("Crop size (px)", 128, 512, 384, 32, key="crop_sz")
        lens_zoom = st.slider("Lens magnification", 2, 6, 3, key="lens_z")
        lens_size = st.slider("Lens size (px)", 100, 400, 250, 20, key="lens_s")

    if orig_crop is not None and denoised_crop is not None:
        with st.expander("Detail map"):
            den_f = denoised_crop.astype(np.float32)
            orig_f = orig_crop.astype(np.float32)

            hp = orig_f - cv2.GaussianBlur(orig_f, (0, 0), sigmaX=2.5)
            hp_vis = np.clip(hp * 2 + 128, 0, 255).astype(np.uint8)

            diff = np.abs(orig_f - den_f)
            diff = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)

            zoomable_image_row(
                [hp_vis, diff],
                ["High-pass detail (what gets injected)", "Residual (what denoiser removed)"],
                lens_zoom, lens_size, uid="detail",
            )


if __name__ == "__main__":
    main()
