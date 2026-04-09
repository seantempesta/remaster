"""
ConvNeXt-V2 Masked Autoencoder for video enhancement.

Symmetric encoder-decoder based on ConvNeXt-V2 blocks for image-to-image
tasks (denoising, enhancement, temporal inpainting). No skip connections
between encoder and decoder -- the bottleneck carries all information.

Supports:
  - Masked training (FCMAE-style): mask random patches, reconstruct them
  - Temporal conditioning: previous cleaned frame as extra input channels
  - Loading pretrained ConvNeXt-V2 ImageNet weights into the encoder
  - Global residual: output = input + learned correction

Architecture (Atto example, 1080p input):
  Input (3ch + 3ch prev + 1ch mask = 7ch, 1080x1920)
    -> Stem (stride 4)           -> 270x480 x 40
    -> Stage 0 (2 blocks)        -> 270x480 x 40
    -> Downsample                -> 135x240 x 80
    -> Stage 1 (2 blocks)        -> 135x240 x 80
    -> Downsample                -> 68x120 x 160
    -> Stage 2 (6 blocks)        -> 68x120 x 160
    -> Downsample                -> 34x60 x 320
    -> Stage 3 (2 blocks)        -> 34x60 x 320
    -> Decoder (PixelShuffle up) -> back to 270x480 x 40
    -> Tail (PixelShuffle x4)    -> 1080x1920 x 3
    + global residual

Pretrained weights: ConvNeXt-V2 ImageNet-1K fine-tuned (CC-BY-NC-4.0).
  - Encoder loads directly (same architecture)
  - Decoder initialized randomly
  - Stem input conv re-initialized for extra channels (prev frame + mask)
"""
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Building blocks -----------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first data formats."""

    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (dim,)

    def forward(self, x):
        # Both paths compute in FP32 for numerical stability, then cast back
        # to the input dtype. This is required for FP16 inference -- without
        # it, F.layer_norm and manual mean/var upcast to FP32 and infect all
        # downstream ops with dtype mismatches.
        orig_dtype = x.dtype
        if self.data_format == "channels_last":
            out = F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
            return out.to(orig_dtype)
        # channels_first: (B, C, H, W)
        x = x.float()
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.float()[:, None, None] * x + self.bias.float()[:, None, None]
        return x.to(orig_dtype)


class GRN(nn.Module):
    """Global Response Normalization -- inter-channel feature competition."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # torch.norm upcasts FP16 to FP32 -- compute in FP32, cast back.
        orig_dtype = x.dtype
        gx = torch.norm(x.float(), p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        nx = nx.to(orig_dtype)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: DWConv7x7 -> LN -> Linear -> GELU -> GRN -> Linear + residual."""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # DropPath imported inline to avoid timm dependency at module level
        if drop_path > 0.0:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + self.drop_path(x)


# -- Encoder -------------------------------------------------------------------

class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt-V2 encoder: stride-4 stem + 4 stages with stride-2 downsamples."""

    def __init__(self, in_chans=3, depths=(2, 2, 6, 2), dims=(40, 80, 160, 320),
                 drop_path_rate=0.0):
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)

        # Stem: stride-4 patchify
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # Stride-2 downsamples between stages
        for i in range(3):
            ds = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(ds)

        # Feature stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


# -- Decoder -------------------------------------------------------------------

class ConvNeXtV2Decoder(nn.Module):
    """Symmetric decoder: ConvNeXt-V2 blocks + PixelShuffle upsampling."""

    def __init__(self, depths=(2, 2, 2, 2), dims=(40, 80, 160, 320),
                 drop_path_rate=0.0):
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)
        n_stages = len(dims)

        # Drop path rates (decreasing through decoder)
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(drop_path_rate, 0, total_blocks)]

        # Bottleneck stage (at lowest resolution, highest channel count)
        cur = 0
        self.bottleneck = nn.Sequential(
            *[ConvNeXtV2Block(dim=dims[-1], drop_path=dp_rates[cur + j])
              for j in range(depths[-1])]
        )
        cur += depths[-1]

        # Upsample + stage pairs (from deep to shallow)
        self.upsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            # Upsample: dims[i] -> dims[i-1] via PixelShuffle(2)
            up = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i - 1] * 4, kernel_size=1),
                nn.PixelShuffle(2),
            )
            self.upsample_layers.append(up)

            # Stage at the upsampled resolution
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i - 1], drop_path=dp_rates[cur + j])
                  for j in range(depths[i - 1])]
            )
            self.stages.append(stage)
            cur += depths[i - 1]

    def forward(self, x):
        x = self.bottleneck(x)
        for up, stage in zip(self.upsample_layers, self.stages):
            x = up(x)
            x = stage(x)
        return x


# -- Autoencoder ---------------------------------------------------------------

# Variant configs: (encoder_depths, decoder_depths, dims)
VARIANTS = {
    "atto":  ((2, 2, 6, 2), (2, 2, 2, 2), (40, 80, 160, 320)),
    "femto": ((2, 2, 6, 2), (2, 2, 2, 2), (48, 96, 192, 384)),
    "pico":  ((2, 2, 6, 2), (2, 2, 2, 2), (64, 128, 256, 512)),
    "nano":  ((2, 2, 8, 2), (2, 2, 2, 2), (80, 160, 320, 640)),
    "tiny":  ((3, 3, 9, 3), (2, 2, 2, 2), (96, 192, 384, 768)),
}

# Pretrained weight URLs (ImageNet-1K fine-tuned, CC-BY-NC-4.0)
PRETRAINED_URLS = {
    "atto":  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
    "femto": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt",
    "pico":  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt",
    "nano":  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt",
    "tiny":  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
}


class ConvNeXtV2Autoencoder(nn.Module):
    """
    ConvNeXt-V2 masked autoencoder for video enhancement.

    Training modes:
      - Masked reconstruction: mask random patches, reconstruct from visible
        patches + previous frame. Loss only on masked regions.
      - Full reconstruction: standard autoencoder (mask_ratio=0).

    Args:
        in_chans: Base input channels (3 for RGB).
        use_prev_frame: If True, accept previous cleaned frame as extra 3
            input channels (total 6 + 1 mask = 7ch).
        encoder_depths: Block counts per encoder stage.
        decoder_depths: Block counts per decoder stage.
        dims: Channel dimensions per stage.
        drop_path_rate: Stochastic depth rate.
        patch_size: Patch size for masking (must divide stem stride=4 evenly).
    """

    def __init__(
        self,
        in_chans=3,
        use_prev_frame=True,
        encoder_depths=(2, 2, 6, 2),
        decoder_depths=(2, 2, 2, 2),
        dims=(40, 80, 160, 320),
        drop_path_rate=0.0,
        patch_size=32,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.use_prev_frame = use_prev_frame
        self.patch_size = patch_size
        self.dims = list(dims)

        # Total input channels: current + prev_frame(opt) + mask
        total_in = in_chans  # current frame
        if use_prev_frame:
            total_in += in_chans  # previous cleaned frame
        total_in += 1  # binary mask channel
        self.total_in = total_in

        # Encoder
        self.encoder = ConvNeXtV2Encoder(
            in_chans=total_in,
            depths=encoder_depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
        )

        # Decoder
        self.decoder = ConvNeXtV2Decoder(
            depths=decoder_depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
        )

        # Tail: PixelShuffle x4 from stride-4 features to full resolution
        self.tail = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], in_chans * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4),
        )

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def encoder_param_count(self):
        return sum(p.numel() for p in self.encoder.parameters())

    @property
    def decoder_param_count(self):
        return sum(p.numel() for p in self.decoder.parameters())

    # -- Masking ---------------------------------------------------------------

    def make_mask(self, batch_size, height, width, mask_ratio, device):
        """Generate a random binary mask at image resolution.

        Returns a (B, 1, H, W) float tensor where 1 = masked, 0 = visible.
        Masking operates on patch_size x patch_size blocks.
        """
        if mask_ratio <= 0:
            return torch.zeros(batch_size, 1, height, width, device=device)
        if mask_ratio >= 1:
            return torch.ones(batch_size, 1, height, width, device=device)

        ps = self.patch_size
        ph = height // ps
        pw = width // ps
        num_patches = ph * pw
        num_mask = int(num_patches * mask_ratio)

        # Random patch selection per batch element
        noise = torch.rand(batch_size, num_patches, device=device)
        ids = torch.argsort(noise, dim=1)
        mask_flat = torch.zeros(batch_size, num_patches, device=device)
        # Mark the first num_mask patches as masked
        mask_flat.scatter_(1, ids[:, :num_mask], 1.0)

        # Reshape to spatial and upsample to image resolution
        mask = mask_flat.reshape(batch_size, 1, ph, pw)
        mask = mask.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3)

        # Handle case where H/W not perfectly divisible by patch_size
        if mask.shape[2] < height or mask.shape[3] < width:
            mask = F.pad(mask, (0, width - mask.shape[3], 0, height - mask.shape[2]))
        return mask

    # -- Forward ---------------------------------------------------------------

    def forward(self, current, prev_clean=None, mask=None, mask_ratio=0.0):
        """
        Args:
            current: (B, 3, H, W) current frame.
            prev_clean: (B, 3, H, W) previous cleaned frame, or None.
                If use_prev_frame=True and prev_clean is None, zeros are used
                (cold start / first frame).
            mask: (B, 1, H, W) pre-computed mask, or None to auto-generate.
            mask_ratio: float in [0, 1]. Only used when mask is None.

        Returns:
            output: (B, 3, H, W) reconstructed/enhanced frame.
            mask: (B, 1, H, W) the mask that was used (for computing loss).
        """
        B, C, H, W = current.shape

        # Pad to multiple of patch_size (also ensures divisible by 32 for encoder)
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            current = F.pad(current, (0, pad_w, 0, pad_h), mode="replicate")
            if prev_clean is not None:
                prev_clean = F.pad(prev_clean, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, Hp, Wp = current.shape

        # Generate or pad mask
        if mask is None:
            mask = self.make_mask(B, Hp, Wp, mask_ratio, current.device)
        elif pad_h or pad_w:
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0.0)

        # Apply mask to current frame (zero out masked patches)
        current_masked = current * (1.0 - mask)

        # Build input tensor
        if self.use_prev_frame:
            if prev_clean is None:
                prev_clean = torch.zeros_like(current)
            x = torch.cat([current_masked, prev_clean, mask], dim=1)
        else:
            x = torch.cat([current_masked, mask], dim=1)

        # Encode -> Decode -> Reconstruct
        z = self.encoder(x)
        feat = self.decoder(z)
        correction = self.tail(feat)

        # Global residual: add correction to the ORIGINAL (unmasked) current frame
        output = current + correction

        # Crop back to original size
        if pad_h or pad_w:
            output = output[:, :, :H, :W]
            mask = mask[:, :, :H, :W]

        return output, mask

    # -- Pretrained weight loading ---------------------------------------------

    def load_pretrained_encoder(self, checkpoint_path=None, variant=None):
        """Load pretrained ConvNeXt-V2 classification weights into the encoder.

        Only loads weights that match (stages + downsample_layers except stem
        input conv which has different channel count). Decoder stays random.

        Args:
            checkpoint_path: Path to .pt file, or None to auto-download.
            variant: One of 'atto', 'femto', etc. Required if checkpoint_path
                is None (to pick the right URL).

        Returns:
            dict with 'loaded', 'skipped', 'missing' key lists.
        """
        if checkpoint_path is None:
            if variant is None:
                raise ValueError("Must provide variant name to auto-download")
            url = PRETRAINED_URLS[variant]
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints", "convnextv2_pretrained",
            )
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_path = os.path.join(cache_dir, os.path.basename(url))
            if not os.path.exists(checkpoint_path):
                print(f"Downloading {variant} weights to {checkpoint_path}...")
                state = torch.hub.load_state_dict_from_url(
                    url, model_dir=cache_dir, map_location="cpu", check_hash=False,
                )
                torch.save(state, checkpoint_path)
            else:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "model" in state:
            state = state["model"]
        elif "model_ema" in state:
            state = state["model_ema"]
        elif "state_dict" in state:
            state = state["state_dict"]

        # Strip 'model.' prefix if present
        cleaned = {}
        for k, v in state.items():
            k = k.replace("model.", "") if k.startswith("model.") else k
            cleaned[k] = v
        state = cleaned

        # Map pretrained keys -> encoder keys
        # Pretrained: downsample_layers.X.Y.weight -> encoder.downsample_layers.X.Y.weight
        # Pretrained: stages.X.Y.*.weight -> encoder.stages.X.Y.*.weight
        # Skip: norm.*, head.* (classifier)
        encoder_state = self.encoder.state_dict()
        loaded, skipped = [], []

        for pk, pv in state.items():
            # Skip classifier head and final norm
            if pk.startswith("norm.") or pk.startswith("head."):
                skipped.append(pk)
                continue

            # The key should map directly to encoder
            if pk in encoder_state:
                if pv.shape == encoder_state[pk].shape:
                    encoder_state[pk] = pv
                    loaded.append(pk)
                else:
                    # Shape mismatch (e.g., stem conv with different in_chans)
                    # Partial load: copy matching channels, rest stays random
                    if pk == "downsample_layers.0.0.weight":
                        # Stem conv: pretrained has 3 in_chans, ours has more
                        min_ch = min(pv.shape[1], encoder_state[pk].shape[1])
                        encoder_state[pk][:, :min_ch] = pv[:, :min_ch]
                        loaded.append(f"{pk} (partial: {min_ch}/{encoder_state[pk].shape[1]} channels)")
                    else:
                        skipped.append(f"{pk} (shape mismatch: {pv.shape} vs {encoder_state[pk].shape})")
            else:
                skipped.append(pk)

        self.encoder.load_state_dict(encoder_state)
        del state, encoder_state
        gc.collect()

        missing = [k for k in self.encoder.state_dict() if not any(
            k == l or k == l.split(" (")[0] for l in loaded
        )]

        return {"loaded": loaded, "skipped": skipped, "missing": missing}

    # -- Factory ---------------------------------------------------------------

    @classmethod
    def from_config(cls, variant="atto", pretrained=True, **kwargs):
        """Create autoencoder from a named variant, optionally with pretrained encoder.

        Args:
            variant: One of 'atto', 'femto', 'pico', 'nano', 'tiny'.
            pretrained: If True, download and load ImageNet-1K weights into encoder.
            **kwargs: Override any constructor arg (use_prev_frame, drop_path_rate, etc.)
        """
        enc_depths, dec_depths, dims = VARIANTS[variant]
        model = cls(
            encoder_depths=enc_depths,
            decoder_depths=dec_depths,
            dims=dims,
            **kwargs,
        )

        if pretrained:
            result = model.load_pretrained_encoder(variant=variant)
            n_loaded = len(result["loaded"])
            n_skipped = len(result["skipped"])
            n_missing = len(result["missing"])
            print(f"Pretrained encoder: {n_loaded} loaded, {n_skipped} skipped, {n_missing} missing")
            if result["missing"]:
                print(f"  Missing (randomly initialized): {result['missing'][:5]}...")

        return model


# -- Loss helper ---------------------------------------------------------------

def masked_reconstruction_loss(output, target, mask, loss_fn=F.l1_loss):
    """Compute reconstruction loss only on masked regions.

    Args:
        output: (B, 3, H, W) model output.
        target: (B, 3, H, W) ground truth (original unmasked frame).
        mask: (B, 1, H, W) binary mask (1 = masked region).
        loss_fn: Loss function (default: L1).

    Returns:
        Scalar loss averaged over masked pixels. If no pixels are masked,
        returns loss over the full image.
    """
    masked_pixels = mask.sum()
    if masked_pixels == 0:
        return loss_fn(output, target)

    # Expand mask to match channels
    mask_3ch = mask.expand_as(output)
    # Compute per-pixel loss, then average over masked pixels only
    loss = (loss_fn(output, target, reduction="none") * mask_3ch).sum()
    return loss / (masked_pixels * output.shape[1])
