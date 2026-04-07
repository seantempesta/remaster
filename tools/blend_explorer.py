"""
Streamlit app for exploring blend methods between SCUNet PSNR and GAN outputs.

Loads pre-computed data from data/calibration/blend_samples.pkl:
    {filename: {'original': bgr_uint8, 'gan': bgr_uint8, 'psnr': bgr_uint8}}

Blend methods:
  - Simple blend: alpha * psnr + (1-alpha) * gan
  - LAB color transfer: L from GAN, AB from PSNR
  - LAB blend: interpolate L between PSNR/GAN, AB from PSNR
  - Chroma-only PSNR: Y from GAN, CrCb from PSNR (YCrCb space)
  - Wavelet blend: low-freq from PSNR, high-freq from GAN (Gaussian pyramid)

Usage:
    streamlit run tools/blend_explorer.py --server.headless true
"""
import base64
import io
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

BLEND_SAMPLES_PATH = Path("data/calibration/blend_samples.pkl")


@st.cache_data
def load_blend_samples():
    if not BLEND_SAMPLES_PATH.exists():
        st.error(f"Not found: {BLEND_SAMPLES_PATH}")
        st.stop()
    with open(BLEND_SAMPLES_PATH, "rb") as f:
        return pickle.load(f)


def center_crop(img, size):
    """Center crop a BGR image to size x size."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    return img[y1:y1 + size, x1:x1 + size]


# -- Blend methods ----------------------------------------------------------

def blend_simple(psnr, gan, alpha=0.5):
    """Simple linear blend: alpha * psnr + (1-alpha) * gan."""
    return cv2.addWeighted(psnr, alpha, gan, 1.0 - alpha, 0)


def blend_lab_color_transfer(psnr, gan):
    """L channel from GAN (structure/texture), AB from PSNR (color fidelity)."""
    lab_gan = cv2.cvtColor(gan, cv2.COLOR_BGR2LAB)
    lab_psnr = cv2.cvtColor(psnr, cv2.COLOR_BGR2LAB)
    result_lab = lab_gan.copy()
    result_lab[:, :, 1] = lab_psnr[:, :, 1]  # A from PSNR
    result_lab[:, :, 2] = lab_psnr[:, :, 2]  # B from PSNR
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def blend_lab_l_interpolate(psnr, gan, alpha=0.5):
    """Interpolate L channel between PSNR and GAN, keep AB from PSNR."""
    lab_gan = cv2.cvtColor(gan, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_psnr = cv2.cvtColor(psnr, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = lab_psnr.copy()
    # alpha=1 means all PSNR L, alpha=0 means all GAN L
    result_lab[:, :, 0] = alpha * lab_psnr[:, :, 0] + (1.0 - alpha) * lab_gan[:, :, 0]
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def blend_chroma_only_psnr(psnr, gan):
    """Y (luma) from GAN, CrCb (chroma) from PSNR. YCrCb space."""
    ycrcb_gan = cv2.cvtColor(gan, cv2.COLOR_BGR2YCrCb)
    ycrcb_psnr = cv2.cvtColor(psnr, cv2.COLOR_BGR2YCrCb)
    result = ycrcb_gan.copy()
    result[:, :, 1] = ycrcb_psnr[:, :, 1]  # Cr from PSNR
    result[:, :, 2] = ycrcb_psnr[:, :, 2]  # Cb from PSNR
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


def blend_wavelet(psnr, gan, sigma=10.0):
    """Low-freq from PSNR (color/tone), high-freq from GAN (texture/detail).

    Uses Gaussian blur as a simple low-pass filter. sigma controls the
    crossover frequency: higher sigma = more structure from PSNR.
    """
    psnr_f = psnr.astype(np.float32)
    gan_f = gan.astype(np.float32)
    # Low-pass from PSNR
    ksize = int(sigma * 6) | 1  # ensure odd
    ksize = max(3, ksize)
    low_psnr = cv2.GaussianBlur(psnr_f, (ksize, ksize), sigmaX=sigma)
    # High-pass from GAN
    low_gan = cv2.GaussianBlur(gan_f, (ksize, ksize), sigmaX=sigma)
    high_gan = gan_f - low_gan
    result = low_psnr + high_gan
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_gan_original_chroma(original, gan):
    """GAN luminance + original colors. YCrCb space."""
    ycrcb_gan = cv2.cvtColor(gan, cv2.COLOR_BGR2YCrCb)
    ycrcb_orig = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
    result = ycrcb_gan.copy()
    result[:, :, 1] = ycrcb_orig[:, :, 1]  # Cr from original
    result[:, :, 2] = ycrcb_orig[:, :, 2]  # Cb from original
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


def blend_gan_original_chroma_lab(original, gan):
    """GAN L channel + original AB. LAB space."""
    lab_gan = cv2.cvtColor(gan, cv2.COLOR_BGR2LAB)
    lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    result = lab_gan.copy()
    result[:, :, 1] = lab_orig[:, :, 1]
    result[:, :, 2] = lab_orig[:, :, 2]
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def unsharp_mask(img, strength=1.0, sigma=1.5):
    """Unsharp mask."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    return np.clip(f + strength * (f - blur), 0, 255).astype(np.uint8)


# -- Zoom component ---------------------------------------------------------

def img_to_b64(img_rgb):
    """Convert numpy RGB array to base64 PNG string."""
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def zoomable_image_row(images, captions, lens_zoom=3, lens_size=180, uid="main"):
    """Render a row of images with linked hover-zoom lenses."""
    n = len(images)
    b64s = [img_to_b64(img) for img in images]
    img_h, img_w = images[0].shape[:2]
    aspect = img_h / img_w

    gap = 8
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
    row_h = max(300, min(int(600 * aspect) + 40, 800))
    components.html(html, height=row_h, scrolling=False)


# -- Main app ---------------------------------------------------------------

BLEND_METHODS = [
    "GAN + Original Color (YCrCb)",
    "GAN + Original Color (LAB)",
    "Simple blend",
    "LAB color transfer",
    "LAB blend",
    "Chroma-only PSNR",
    "Wavelet blend",
]


def main():
    st.set_page_config(page_title="Blend Explorer", layout="wide")
    st.title("Target Quality Explorer")

    samples = load_blend_samples()
    filenames = sorted(samples.keys())

    if not filenames:
        st.error("No samples in blend_samples.pkl")
        return

    # -- Sidebar: display settings ------------------------------------------
    st.sidebar.header("Display")
    crop_size = st.sidebar.slider("Crop size (px)", 256, 1024, 512, 64)
    lens_zoom = st.sidebar.slider("Lens magnification", 2, 6, 3)
    lens_size = st.sidebar.slider("Lens size (px)", 100, 400, 250, 20)

    # -- Frame selection ----------------------------------------------------
    display_labels = []
    for fn in filenames:
        src = fn.split("_")[0]
        display_labels.append(f"[{src}] {fn}")

    selected_label = st.selectbox("Frame", display_labels)
    selected_idx = display_labels.index(selected_label)
    filename = filenames[selected_idx]

    entry = samples[filename]
    original_bgr = entry["original"]
    gan_bgr = entry["gan"]
    psnr_bgr = entry["psnr"]

    # Center crop all
    original_crop = center_crop(original_bgr, crop_size)
    gan_crop = center_crop(gan_bgr, crop_size)
    psnr_crop = center_crop(psnr_bgr, crop_size)

    # -- Blend method + USM controls ----------------------------------------
    col_method, col_usm = st.columns([3, 2])
    with col_method:
        method = st.selectbox("Blend method", BLEND_METHODS)
    with col_usm:
        usm_strength = st.slider("USM sharpness", 0.0, 3.0, 1.0, 0.1)

    # -- Per-method parameters and blending ---------------------------------
    if method == "GAN + Original Color (YCrCb)":
        blended = blend_gan_original_chroma(original_crop, gan_crop)
        param_str = "Y=GAN, CrCb=Original"

    elif method == "GAN + Original Color (LAB)":
        blended = blend_gan_original_chroma_lab(original_crop, gan_crop)
        param_str = "L=GAN, AB=Original"

    elif method == "Simple blend":
        alpha = st.slider("Alpha (1.0 = all PSNR, 0.0 = all GAN)",
                          0.0, 1.0, 0.5, 0.05)
        blended = blend_simple(psnr_crop, gan_crop, alpha)
        param_str = f"alpha={alpha:.2f}"

    elif method == "LAB color transfer":
        blended = blend_lab_color_transfer(psnr_crop, gan_crop)
        param_str = "L=GAN, AB=PSNR"

    elif method == "LAB blend":
        alpha = st.slider("L-channel alpha (1.0 = PSNR luminance, 0.0 = GAN luminance)",
                          0.0, 1.0, 0.5, 0.05)
        blended = blend_lab_l_interpolate(psnr_crop, gan_crop, alpha)
        param_str = f"L alpha={alpha:.2f}, AB=PSNR"

    elif method == "Chroma-only PSNR":
        blended = blend_chroma_only_psnr(psnr_crop, gan_crop)
        param_str = "Y=GAN, CrCb=PSNR"

    elif method == "Wavelet blend":
        sigma = st.slider("Blur sigma (crossover freq: higher = more PSNR structure)",
                          1.0, 40.0, 10.0, 1.0)
        blended = blend_wavelet(psnr_crop, gan_crop, sigma)
        param_str = f"sigma={sigma:.0f}"

    else:
        blended = gan_crop
        param_str = ""

    # Apply USM to the blend result
    if usm_strength > 0:
        blended = unsharp_mask(blended, usm_strength)
        param_str += f" + USM({usm_strength:.1f})"

    # Current target for reference (what we've been training on: GAN + USM 1.0)
    current_target = unsharp_mask(gan_crop, 1.0)

    # -- Convert BGR to RGB for display -------------------------------------
    original_rgb = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    current_target_rgb = cv2.cvtColor(current_target, cv2.COLOR_BGR2RGB)

    # -- Main comparison: Original | New Candidate | Current Target ---------
    st.write(f"**{method}** ({param_str})")
    zoomable_image_row(
        [original_rgb, blended_rgb, current_target_rgb],
        ["Original", f"Candidate: {method}", "Current Target (GAN+USM1.0)"],
        lens_zoom, lens_size, uid="blend",
    )

    # -- Details expander ---------------------------------------------------
    with st.expander("All outputs + diffs"):
        gan_rgb = cv2.cvtColor(gan_crop, cv2.COLOR_BGR2RGB)
        psnr_rgb = cv2.cvtColor(psnr_crop, cv2.COLOR_BGR2RGB)

        zoomable_image_row(
            [gan_rgb, psnr_rgb],
            ["GAN (raw)", "PSNR (raw)"],
            lens_zoom, lens_size, uid="raw",
        )

        # Diff: candidate vs current target
        diff = np.abs(blended.astype(np.float32) - current_target.astype(np.float32))
        diff = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        # Diff: candidate vs original
        diff_orig = np.abs(blended.astype(np.float32) - original_crop.astype(np.float32))
        diff_orig = (diff_orig / (diff_orig.max() + 1e-8) * 255).astype(np.uint8)
        diff_orig_rgb = cv2.cvtColor(diff_orig, cv2.COLOR_BGR2RGB)

        zoomable_image_row(
            [diff_rgb, diff_orig_rgb],
            ["Diff: Candidate vs Current Target", "Diff: Candidate vs Original"],
            lens_zoom, lens_size, uid="diff",
        )


if __name__ == "__main__":
    main()
