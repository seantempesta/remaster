"""
Streamlit debug/exploration app for the RAFT temporal averaging pipeline.

Lets the user interactively inspect every step of the pipeline on a single
center frame: alignment quality, temporal statistics, FFT analysis, and
results with adjustable parameters.

Usage:
    streamlit run tools/raft_debug.py --server.headless true
"""
import base64
import gc
import io
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reference-code", "RAFT", "core"))

from tools.raft_temporal_targets import (
    compute_flow_pair,
    compute_occlusion_mask,
    compute_temporal_fft_confidence,
    compute_temporal_median,
    compute_wiener_sharpen,
    extract_frames_to_numpy,
    load_raft_model,
    warp_frame,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image utilities (hover-zoom lens from label_sigma.py)
# ---------------------------------------------------------------------------
def img_to_b64(img_rgb):
    """Convert numpy RGB array to base64 PNG string."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def zoomable_image_row(images, captions, lens_zoom=3, lens_size=180, uid="main",
                       height_override=None):
    """Render a row of images with linked hover-zoom lenses.

    Images scale to fit side-by-side in the available width. Hovering on any
    image shows a synchronized magnified overlay on all images.
    """
    n = len(images)
    if n == 0:
        return
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
    if height_override:
        row_h = height_override
    else:
        row_h = int(600 * aspect / n * n) + 40
        row_h = max(300, min(row_h, 800))
    components.html(html, height=row_h, scrolling=False)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def flow_to_color(flow_cpu):
    """Convert optical flow [1,2,H,W] to HSV color visualization [H,W,3] uint8."""
    flow_np = flow_cpu.squeeze(0).numpy()  # [2, H, W]
    dx, dy = flow_np[0], flow_np[1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)

    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    # Normalize magnitude for visibility, clamp at 99th percentile
    max_mag = np.percentile(mag, 99) if mag.max() > 0 else 1.0
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def make_checkerboard_overlay(img1, img2, block_size=16):
    """Checkerboard interleave of two images for alignment comparison."""
    H, W = img1.shape[:2]
    result = img1.copy().astype(np.float32)
    img2_f = img2.astype(np.float32)
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            by = y // block_size
            bx = x // block_size
            if (by + bx) % 2 == 1:
                ye = min(y + block_size, H)
                xe = min(x + block_size, W)
                result[y:ye, x:xe] = img2_f[y:ye, x:xe]
    return np.clip(result, 0, 255).astype(np.uint8)


def make_difference_map(img1, img2, gain=3.0):
    """Absolute difference between two images, amplified for visibility."""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff = diff.mean(axis=2)  # grayscale
    diff = np.clip(diff * gain, 0, 255).astype(np.uint8)
    # Apply colormap for visibility
    colored = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def mask_to_vis(mask, label="valid"):
    """Convert boolean mask to green/red visualization."""
    H, W = mask.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask] = [0, 200, 0]    # green = valid
    vis[~mask] = [200, 0, 0]   # red = occluded
    return vis


def crop_center(img, crop_size):
    """Take a center crop of the image."""
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    half = crop_size // 2
    y1, y2 = max(0, cy - half), min(H, cy + half)
    x1, x2 = max(0, cx - half), min(W, cx + half)
    return img[y1:y2, x1:x2]


def compute_fft_analysis(center_np, warped_list, mask_list):
    """Compute FFT analysis data for visualization.

    Returns dict with magnitude spectrums, SNR map, and Wiener gain map.
    """
    H, W, C = center_np.shape
    center_f = center_np.astype(np.float32)

    # Use green channel (most perceptually important)
    center_gray = cv2.cvtColor(center_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    center_fft = np.fft.fft2(center_gray)
    center_mag = np.log1p(np.abs(np.fft.fftshift(center_fft)))

    # Collect FFTs from aligned observations
    all_obs = [center_f]
    all_masks = [np.ones((H, W), dtype=bool)]
    for w_arr, m_arr in zip(warped_list, mask_list):
        all_obs.append(w_arr)
        all_masks.append(m_arr)

    obs_grays = []
    for i, (obs, mask) in enumerate(zip(all_obs, all_masks)):
        gray = cv2.cvtColor(
            np.clip(obs, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
        if i > 0:
            # Fill invalid regions with center frame values
            center_gray_fill = cv2.cvtColor(center_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray[~mask] = center_gray_fill[~mask]
        obs_grays.append(gray)

    # Average magnitude spectrum
    all_mags_shifted = []
    all_mags_raw = []
    for gray in obs_grays:
        fft_val = np.fft.fft2(gray)
        all_mags_shifted.append(np.abs(np.fft.fftshift(fft_val)))
        all_mags_raw.append(np.abs(np.fft.rfft2(gray)))

    avg_mag_shifted = np.mean(all_mags_shifted, axis=0)
    avg_mag_vis = np.log1p(avg_mag_shifted)

    # Per-frequency SNR (using rfft2 for efficiency)
    mag_stack = np.stack(all_mags_raw, axis=0)  # [N, H, W//2+1]
    mag_mean = mag_stack.mean(axis=0)
    mag_std = mag_stack.std(axis=0)
    snr = mag_mean / (mag_std + 1e-8)

    # Wiener gain
    wiener_gain = (snr ** 2) / (snr ** 2 + 1.0)

    # Convert SNR/Wiener to full 2D for visualization (mirror rfft)
    snr_full = np.fft.fftshift(_rfft_to_full(snr, W))
    wiener_full = np.fft.fftshift(_rfft_to_full(wiener_gain, W))

    return {
        "center_mag": center_mag,
        "avg_mag": avg_mag_vis,
        "snr_map": snr_full,
        "wiener_gain": wiener_full,
    }


def _rfft_to_full(rfft_data, full_w):
    """Mirror rfft2 output back to full 2D spectrum for visualization."""
    H, rW = rfft_data.shape
    full = np.zeros((H, full_w), dtype=rfft_data.dtype)
    full[:, :rW] = rfft_data
    # Mirror (skip DC and Nyquist columns)
    if full_w % 2 == 0:
        full[:, rW:] = rfft_data[:, rW - 2:0:-1]
    else:
        full[:, rW:] = rfft_data[:, rW - 1:0:-1]
    return full


def spectrum_to_vis(mag_2d, label=""):
    """Convert log-magnitude spectrum to a displayable uint8 image."""
    normed = mag_2d - mag_2d.min()
    mx = normed.max()
    if mx > 0:
        normed = normed / mx
    vis = (normed * 255).astype(np.uint8)
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def gain_to_vis(gain_2d):
    """Convert Wiener gain map [0,1] to displayable image."""
    vis = np.clip(gain_2d * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# RAFT computation (cached in session state)
# ---------------------------------------------------------------------------
def compute_alignment_data(model, frames, center_idx, window, status_placeholder):
    """Compute RAFT flow, warp, and occlusion for all neighbors.

    Returns a dict with all alignment data, cached by session_state.
    Processes one pair at a time to fit in 6GB VRAM.
    """
    num_frames = len(frames)
    n_start = max(0, center_idx - window)
    n_end = min(num_frames - 1, center_idx + window)
    neighbors = [n for n in range(n_start, n_end + 1) if n != center_idx]

    center_frame = frames[center_idx]
    results = {
        "center_idx": center_idx,
        "center_frame": center_frame,
        "neighbors": [],
    }

    for ni, n in enumerate(neighbors):
        status_placeholder.text(
            f"Computing flow for neighbor {ni + 1}/{len(neighbors)} "
            f"(frame {n})..."
        )
        t0 = time.time()
        neighbor_frame = frames[n]

        # Forward flow: center -> neighbor
        flow_fwd = compute_flow_pair(model, center_frame, neighbor_frame)
        # Backward flow: neighbor -> center
        flow_bwd = compute_flow_pair(model, neighbor_frame, center_frame)

        # Warp neighbor to center grid
        warped, in_bounds = warp_frame(neighbor_frame, flow_fwd)

        # Occlusion mask
        valid_flow = compute_occlusion_mask(flow_fwd, flow_bwd)
        combined_mask = in_bounds & valid_flow

        elapsed = time.time() - t0
        results["neighbors"].append({
            "frame_idx": n,
            "offset": n - center_idx,
            "original": neighbor_frame,
            "warped": warped,
            "in_bounds": in_bounds,
            "valid_flow": valid_flow,
            "combined_mask": combined_mask,
            "flow_fwd": flow_fwd,
            "flow_bwd": flow_bwd,
            "time": elapsed,
        })

        # Free GPU memory between pairs
        torch.cuda.empty_cache()

    status_placeholder.text(
        f"Done -- {len(neighbors)} neighbors processed."
    )
    return results


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="RAFT Pipeline Debug", layout="wide")
    st.title("RAFT Temporal Averaging -- Debug Inspector")

    # -- Sidebar controls --
    st.sidebar.header("Settings")
    video_path = st.sidebar.text_input(
        "Video path",
        value="data/archive/firefly_s01e08_30s.mkv",
    )
    max_frames = st.sidebar.number_input(
        "Max frames to load", min_value=10, max_value=2000, value=100, step=10,
    )
    window = st.sidebar.slider("Window size (neighbors each side)", 1, 8, 4)
    crop_size = st.sidebar.slider("Crop size (px)", 128, 1024, 512, 32)
    lens_zoom = st.sidebar.slider("Lens zoom", 2, 6, 3)
    lens_size = st.sidebar.slider("Lens size (px)", 100, 400, 200, 20)
    overlay_mode = st.sidebar.radio(
        "Overlay mode", ["Checkerboard", "Difference map"], index=0,
    )
    checkerboard_block = st.sidebar.slider("Checkerboard block size", 4, 64, 16, 4)
    diff_gain = st.sidebar.slider("Difference gain", 1.0, 10.0, 3.0, 0.5)

    # -- Load video frames (cached) --
    if "frames" not in st.session_state or st.session_state.get("_video_path") != video_path:
        if not os.path.exists(video_path):
            st.error(f"Video not found: {video_path}")
            return
        with st.spinner(f"Loading frames from {video_path}..."):
            frames, w, h, fps = extract_frames_to_numpy(video_path, int(max_frames))
        st.session_state.frames = frames
        st.session_state._video_path = video_path
        st.session_state._video_info = (w, h, fps, len(frames))
        # Clear cached alignment when video changes
        st.session_state.pop("alignment_data", None)

    frames = st.session_state.frames
    w, h, fps, n_frames = st.session_state._video_info
    st.sidebar.info(f"{w}x{h} @ {fps:.2f} fps, {n_frames} frames loaded")

    # -- Load RAFT model (cached) --
    if "raft_model" not in st.session_state:
        with st.spinner("Loading RAFT-Small model..."):
            model = load_raft_model(small=True)
        st.session_state.raft_model = model

    model = st.session_state.raft_model

    # -- Center frame selection --
    center_idx = st.sidebar.slider(
        "Center frame", 0, len(frames) - 1, min(len(frames) // 2, 50),
    )

    # -- Compute alignment (cached by center_idx + window) --
    cache_key = (center_idx, window)
    needs_compute = (
        "alignment_data" not in st.session_state
        or st.session_state.get("_align_key") != cache_key
    )

    if needs_compute:
        if st.sidebar.button("Compute alignment", type="primary"):
            status = st.empty()
            data = compute_alignment_data(
                model, frames, center_idx, window, status,
            )
            st.session_state.alignment_data = data
            st.session_state._align_key = cache_key
            status.empty()
            st.rerun()
        else:
            st.info(
                f"Select center frame {center_idx} with window {window} "
                "and click 'Compute alignment' in the sidebar."
            )
            # Show the raw center frame as a preview
            preview = crop_center(frames[center_idx], crop_size)
            st.image(preview, caption=f"Center frame {center_idx} (preview)", width=512)
            return

    data = st.session_state.alignment_data
    center_frame = data["center_frame"]
    neighbors = data["neighbors"]

    # =====================================================================
    # Section 1: Alignment Quality
    # =====================================================================
    st.header("1. Alignment Quality")
    st.write(
        f"Center frame: **{center_idx}** | "
        f"Neighbors: **{len(neighbors)}** | "
        f"Window: +/- {window}"
    )

    # Neighbor selector
    neighbor_labels = [
        f"Frame {nb['frame_idx']} (offset {nb['offset']:+d}, {nb['time']:.1f}s)"
        for nb in neighbors
    ]
    selected_nb_idx = st.selectbox(
        "Select neighbor to inspect", range(len(neighbors)),
        format_func=lambda i: neighbor_labels[i],
    )
    nb = neighbors[selected_nb_idx]

    # Crop all images
    c_crop = crop_center(center_frame, crop_size)
    orig_crop = crop_center(nb["original"], crop_size)
    warped_crop = crop_center(
        np.clip(nb["warped"], 0, 255).astype(np.uint8), crop_size,
    )
    mask_crop = crop_center(nb["combined_mask"], crop_size)
    flow_vis = flow_to_color(nb["flow_fwd"])
    flow_crop = crop_center(flow_vis, crop_size)

    # Overlay: warped vs center
    if overlay_mode == "Checkerboard":
        overlay = make_checkerboard_overlay(c_crop, warped_crop, checkerboard_block)
        overlay_label = "Checkerboard (center vs warped)"
    else:
        overlay = make_difference_map(c_crop, warped_crop, diff_gain)
        overlay_label = f"Difference (gain={diff_gain:.1f})"

    mask_vis = mask_to_vis(mask_crop)

    # Row 1: Original neighbor, warped neighbor, overlay
    zoomable_image_row(
        [orig_crop, warped_crop, overlay],
        [
            f"Original neighbor (frame {nb['frame_idx']})",
            "Warped to center grid",
            overlay_label,
        ],
        lens_zoom, lens_size, uid="align_row1",
    )

    # Row 2: Center frame, occlusion mask, flow magnitude
    valid_pct = nb["combined_mask"].sum() / nb["combined_mask"].size * 100
    zoomable_image_row(
        [c_crop, mask_vis, flow_crop],
        [
            f"Center frame {center_idx}",
            f"Occlusion mask ({valid_pct:.1f}% valid)",
            "Flow magnitude (HSV)",
        ],
        lens_zoom, lens_size, uid="align_row2",
    )

    # Per-neighbor summary table
    with st.expander("All neighbors summary"):
        for i, nb_info in enumerate(neighbors):
            vpct = nb_info["combined_mask"].sum() / nb_info["combined_mask"].size * 100
            flow_mag = torch.norm(nb_info["flow_fwd"], dim=1).mean().item()
            st.write(
                f"  Frame {nb_info['frame_idx']} "
                f"(offset {nb_info['offset']:+d}) | "
                f"valid: {vpct:.1f}% | "
                f"mean flow: {flow_mag:.2f} px | "
                f"time: {nb_info['time']:.1f}s"
            )

    # =====================================================================
    # Section 2: Temporal Statistics
    # =====================================================================
    st.header("2. Temporal Statistics")

    warped_list = [nb_info["warped"] for nb_info in neighbors]
    mask_list = [nb_info["combined_mask"] for nb_info in neighbors]

    # Build observation stack
    center_f = center_frame.astype(np.float32)
    all_obs = [center_f] + warped_list
    all_masks = [np.ones(center_frame.shape[:2], dtype=bool)] + mask_list
    obs_stack = np.stack(all_obs, axis=0)   # [N, H, W, 3]
    mask_stack = np.stack(all_masks, axis=0)  # [N, H, W]

    # Per-pixel valid count
    valid_count = mask_stack.sum(axis=0)  # [H, W]

    # Masked mean
    mask_expanded = mask_stack[..., np.newaxis]  # [N, H, W, 1]
    obs_masked = np.where(mask_expanded, obs_stack, np.nan)
    with np.errstate(all="ignore"):
        pixel_mean = np.nanmean(obs_masked, axis=0)  # [H, W, 3]
        pixel_var = np.nanvar(obs_masked, axis=0).mean(axis=2)  # [H, W]

    # Fix any NaN from all-masked pixels
    nan_pixels = np.isnan(pixel_mean)
    pixel_mean[nan_pixels] = center_f[nan_pixels]
    pixel_var[np.isnan(pixel_var)] = 0

    # Visualizations
    mean_vis = np.clip(pixel_mean, 0, 255).astype(np.uint8)
    var_normed = pixel_var / (pixel_var.max() + 1e-8)
    var_vis = cv2.applyColorMap(
        (var_normed * 255).astype(np.uint8), cv2.COLORMAP_INFERNO,
    )
    var_vis = cv2.cvtColor(var_vis, cv2.COLOR_BGR2RGB)

    count_normed = valid_count.astype(np.float32) / (valid_count.max() + 1e-8)
    count_vis = cv2.applyColorMap(
        (count_normed * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS,
    )
    count_vis = cv2.cvtColor(count_vis, cv2.COLOR_BGR2RGB)

    # Crop for display
    mean_crop = crop_center(mean_vis, crop_size)
    var_crop = crop_center(var_vis, crop_size)
    count_crop = crop_center(count_vis, crop_size)

    zoomable_image_row(
        [mean_crop, var_crop, count_crop],
        [
            "Per-pixel mean (naive average)",
            f"Per-pixel variance (max={pixel_var.max():.1f})",
            f"Valid observations ({int(valid_count.min())}-{int(valid_count.max())})",
        ],
        lens_zoom, lens_size, uid="stats_row",
    )

    # =====================================================================
    # Section 3: FFT Analysis
    # =====================================================================
    st.header("3. FFT Analysis")

    # Cache FFT analysis
    if (
        "fft_analysis" not in st.session_state
        or st.session_state.get("_fft_key") != cache_key
    ):
        with st.spinner("Computing FFT analysis..."):
            fft_data = compute_fft_analysis(center_frame, warped_list, mask_list)
        st.session_state.fft_analysis = fft_data
        st.session_state._fft_key = cache_key

    fft_data = st.session_state.fft_analysis

    center_mag_vis = spectrum_to_vis(fft_data["center_mag"])
    avg_mag_vis = spectrum_to_vis(fft_data["avg_mag"])
    snr_vis = spectrum_to_vis(np.log1p(fft_data["snr_map"]))
    wiener_vis = gain_to_vis(fft_data["wiener_gain"])

    center_mag_crop = crop_center(center_mag_vis, crop_size)
    avg_mag_crop = crop_center(avg_mag_vis, crop_size)
    snr_crop = crop_center(snr_vis, crop_size)
    wiener_crop = crop_center(wiener_vis, crop_size)

    zoomable_image_row(
        [center_mag_crop, avg_mag_crop],
        ["Center frame spectrum", "Average spectrum (all aligned)"],
        lens_zoom, lens_size, uid="fft_row1",
    )
    zoomable_image_row(
        [snr_crop, wiener_crop],
        ["Per-frequency SNR (log scale)", "Wiener gain (bright = kept)"],
        lens_zoom, lens_size, uid="fft_row2",
    )

    # =====================================================================
    # Section 4: Results Comparison
    # =====================================================================
    st.header("4. Results Comparison")

    col1, col2 = st.columns(2)
    with col1:
        wiener_strength = st.slider(
            "Wiener denoise strength", 0.1, 3.0, 1.0, 0.1,
        )
    with col2:
        sharpen_strength = st.slider(
            "Sharpen strength", 0.0, 3.0, 1.0, 0.1,
        )

    # Compute results (these are fast -- no RAFT needed)
    # Cache median and FFT confidence (they don't depend on sliders)
    result_cache_key = cache_key
    if (
        "result_median" not in st.session_state
        or st.session_state.get("_result_key") != result_cache_key
    ):
        with st.spinner("Computing median and FFT confidence results..."):
            result_median = compute_temporal_median(
                center_frame, warped_list, mask_list,
            )
            result_fft = compute_temporal_fft_confidence(
                center_frame, warped_list, mask_list,
            )
        st.session_state.result_median = result_median
        st.session_state.result_fft = result_fft
        st.session_state._result_key = result_cache_key

    result_median = st.session_state.result_median
    result_fft = st.session_state.result_fft

    # Wiener + sharpen changes with sliders -- always recompute
    result_wiener = compute_wiener_sharpen(
        center_frame, warped_list, mask_list,
        wiener_strength=wiener_strength, sharpen_strength=0.0,
    )
    result_wiener_sharp = compute_wiener_sharpen(
        center_frame, warped_list, mask_list,
        wiener_strength=wiener_strength, sharpen_strength=sharpen_strength,
    )

    # Crop all
    orig_crop_r = crop_center(center_frame, crop_size)
    median_crop = crop_center(result_median, crop_size)
    fft_crop = crop_center(result_fft, crop_size)
    wiener_crop_r = crop_center(result_wiener, crop_size)
    ws_crop = crop_center(result_wiener_sharp, crop_size)

    # Row 1: Original, median, FFT confidence
    zoomable_image_row(
        [orig_crop_r, median_crop, fft_crop],
        [
            f"Original (frame {center_idx})",
            "Spatial median",
            "FFT confidence",
        ],
        lens_zoom, lens_size, uid="result_row1",
    )

    # Row 2: Wiener only, Wiener + sharpen
    zoomable_image_row(
        [orig_crop_r, wiener_crop_r, ws_crop],
        [
            f"Original (frame {center_idx})",
            f"Wiener (strength={wiener_strength:.1f})",
            f"Wiener + sharpen ({wiener_strength:.1f} / {sharpen_strength:.1f})",
        ],
        lens_zoom, lens_size, uid="result_row2",
    )

    # PSNR comparison (against median as reference, since we don't have GT)
    with st.expander("Quality metrics (all vs original center frame)"):
        for name, result in [
            ("Median", result_median),
            ("FFT confidence", result_fft),
            ("Wiener", result_wiener),
            ("Wiener + sharpen", result_wiener_sharp),
        ]:
            mse = np.mean(
                (result.astype(np.float32) - center_frame.astype(np.float32)) ** 2
            )
            if mse > 0:
                psnr = 10 * np.log10(255.0 ** 2 / mse)
            else:
                psnr = float("inf")
            st.write(f"  **{name}**: PSNR vs original = {psnr:.2f} dB (lower = more change)")

    # -- Debug info --
    with st.expander("Debug info"):
        st.write(f"Device: {DEVICE}")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            st.write(f"GPU memory: {alloc:.0f} / {reserved:.0f} MB (alloc/reserved)")
        st.write(f"Frames in memory: {len(frames)}")
        st.write(f"Frame shape: {frames[0].shape}")
        total_time = sum(nb_info["time"] for nb_info in neighbors)
        st.write(f"Total RAFT time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
