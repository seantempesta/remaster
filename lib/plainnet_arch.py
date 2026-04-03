"""
PlainDenoise — INT8-native denoising CNN.

Design principles:
1. ONLY uses ops that TensorRT fuses into INT8 kernels: Conv3x3, Conv1x1, BN, ReLU
2. FFDNet-style half-resolution processing (PixelUnshuffle → process at 540p → PixelShuffle)
3. Residual learning in unshuffled space
4. Reparameterizable training blocks (multi-branch → single conv at inference)
5. No LayerNorm, no channel attention, no global pooling, no GELU/SiLU

Architecture variants:
- PlainDenoise: Sequential Conv+BN+ReLU (DnCNN-style at half-res)
- UNetDenoise: Shallow 2-level U-Net with Conv+BN+ReLU blocks

Both support reparameterizable training (RepConvBlock) that collapses to plain
Conv3x3 at inference for maximum INT8 throughput.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reparameterizable Conv Block
# ---------------------------------------------------------------------------
class RepConvBlock(nn.Module):
    """Multi-branch conv block for training, fuses to single Conv3x3 at inference.

    Training: Conv3x3+BN + Conv1x1+BN + (Identity+BN if in_nc==out_nc) + ReLU
    Inference: single Conv3x3+bias + ReLU (after calling fuse_params())

    This follows RepVGG/ECBSR: richer gradient flow during training, zero overhead
    at inference. All branches are linear, so they sum into one conv.
    """

    def __init__(self, in_nc, out_nc, use_bn=True, deploy=False):
        super().__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.deploy = deploy

        if deploy:
            # Inference mode: single fused conv
            self.fused_conv = nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=True)
            self.act = nn.ReLU(inplace=True)
            return

        # Training mode: multi-branch
        if use_bn:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_nc),
            )
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_nc, out_nc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_nc),
            )
            self.identity = nn.BatchNorm2d(out_nc) if in_nc == out_nc else None
        else:
            self.conv3x3 = nn.Conv2d(in_nc, out_nc, 3, 1, 1, bias=True)
            self.conv1x1 = nn.Conv2d(in_nc, out_nc, 1, 1, 0, bias=True)
            self.identity = (nn.Identity() if in_nc == out_nc else None)

        self.act = nn.ReLU(inplace=True)
        self.use_bn = use_bn

    def forward(self, x):
        if self.deploy:
            return self.act(self.fused_conv(x))

        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + (self.identity(x) if self.use_bn else x)
        return self.act(out)

    def _fuse_bn(self, conv, bn):
        """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
        w = conv.weight
        # BN params
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = (var + eps).sqrt()

        # Fused weight and bias
        fused_w = w * (gamma / std).reshape(-1, 1, 1, 1)
        fused_b = beta - mean * gamma / std
        return fused_w, fused_b

    def _pad_1x1_to_3x3(self, w):
        """Pad 1x1 conv weight to 3x3."""
        return F.pad(w, [1, 1, 1, 1])

    def _identity_to_conv3x3(self, channels, bn=None):
        """Create a 3x3 identity convolution weight (+BN if present)."""
        w = torch.zeros(channels, channels, 3, 3, device=self._device())
        for i in range(channels):
            w[i, i, 1, 1] = 1.0
        if bn is not None:
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            std = (var + eps).sqrt()
            w = w * (gamma / std).reshape(-1, 1, 1, 1)
            b = beta - mean * gamma / std
            return w, b
        return w, torch.zeros(channels, device=self._device())

    def _device(self):
        if self.deploy:
            return self.fused_conv.weight.device
        if self.use_bn:
            return self.conv3x3[0].weight.device
        return self.conv3x3.weight.device

    def fuse_params(self):
        """Fuse multi-branch into single Conv3x3. Call before export/inference."""
        if self.deploy:
            return

        if self.use_bn:
            w3, b3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])
            w1, b1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        else:
            w3 = self.conv3x3.weight
            b3 = self.conv3x3.bias
            w1 = self.conv1x1.weight
            b1 = self.conv1x1.bias

        w1 = self._pad_1x1_to_3x3(w1)

        total_w = w3 + w1
        total_b = b3 + b1

        if self.identity is not None:
            if self.use_bn:
                wi, bi = self._identity_to_conv3x3(self.out_nc, self.identity)
            else:
                wi, bi = self._identity_to_conv3x3(self.out_nc, None)
            total_w = total_w + wi
            total_b = total_b + bi

        self.fused_conv = nn.Conv2d(
            self.in_nc, self.out_nc, 3, 1, 1, bias=True
        )
        self.fused_conv.weight.data = total_w
        self.fused_conv.bias.data = total_b

        # Remove training branches
        if hasattr(self, 'conv3x3'):
            del self.conv3x3
        if hasattr(self, 'conv1x1'):
            del self.conv1x1
        if hasattr(self, 'identity'):
            del self.identity

        self.deploy = True


# ---------------------------------------------------------------------------
# PlainDenoise: Sequential Conv+BN+ReLU at half resolution
# ---------------------------------------------------------------------------
class PlainDenoise(nn.Module):
    """FFDNet-style half-res denoiser with plain Conv+BN+ReLU.

    All INT8-friendly ops. Processes at 540p internally (4x fewer ops).

    Args:
        in_nc: input channels (3 for RGB)
        nc: internal channel count (32, 48, 64, 96)
        nb: number of conv layers (10, 12, 15, 20)
        use_bn: use BatchNorm (fuses with Conv in TRT, helps INT8 calibration)
        deploy: if True, use fused single-conv blocks (for inference/export)
    """

    def __init__(self, in_nc=3, nc=64, nb=15, use_bn=True, deploy=False):
        super().__init__()
        self.in_nc = in_nc

        # PixelUnshuffle 2x: 3ch@1080p → 12ch@540p
        self.unshuffle = nn.PixelUnshuffle(2)

        # Head: 12 → nc (plain conv, no reparam needed for channel change)
        self.head = nn.Sequential(
            nn.Conv2d(in_nc * 4, nc, 3, 1, 1, bias=not use_bn),
            nn.BatchNorm2d(nc) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Body: nc → nc × (nb - 2) RepConv blocks
        self.body = nn.Sequential(*[
            RepConvBlock(nc, nc, use_bn=use_bn, deploy=deploy)
            for _ in range(nb - 2)
        ])

        # Tail: nc → 12 (no activation — residual output)
        self.tail = nn.Conv2d(nc, in_nc * 4, 3, 1, 1, bias=True)

        # PixelShuffle 2x: 12ch@540p → 3ch@1080p
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad to even dims (PixelUnshuffle requires even H, W)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # Half-res processing with global residual
        x_down = self.unshuffle(x)         # B, 12, H/2, W/2
        feat = self.head(x_down)           # B, nc, H/2, W/2
        feat = self.body(feat)             # B, nc, H/2, W/2
        correction = self.tail(feat)       # B, 12, H/2, W/2
        x_clean = x_down + correction      # Residual in unshuffled space
        out = self.shuffle(x_clean)        # B, 3, H, W

        return out[:, :, :H, :W]

    def fuse_reparam(self):
        """Fuse all RepConvBlocks for inference. Call before export/benchmark."""
        for module in self.modules():
            if isinstance(module, RepConvBlock):
                module.fuse_params()
        return self


# ---------------------------------------------------------------------------
# UNetDenoise: Shallow U-Net at half resolution
# ---------------------------------------------------------------------------
class UNetDenoise(nn.Module):
    """Shallow 2-level U-Net denoiser at half resolution.

    Better receptive field than PlainDenoise for similar param count.
    Still 100% INT8-friendly: Conv+BN+ReLU + bilinear upsample.

    Args:
        in_nc: input channels (3 for RGB)
        nc: base channel count (doubled at each level)
        nb_enc: blocks per encoder level [level0, level1]
        nb_dec: blocks per decoder level [level1, level0]
        nb_mid: blocks at bottleneck
        use_bn: use BatchNorm
        deploy: fused mode
    """

    def __init__(self, in_nc=3, nc=48, nb_enc=(2, 2), nb_dec=(2, 2),
                 nb_mid=2, use_bn=True, deploy=False):
        super().__init__()
        self.in_nc = in_nc

        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)

        # Head: 12 → nc
        self.head = nn.Sequential(
            nn.Conv2d(in_nc * 4, nc, 3, 1, 1, bias=not use_bn),
            nn.BatchNorm2d(nc) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Encoder level 0: nc channels, original half-res (540p)
        self.enc0 = nn.Sequential(*[
            RepConvBlock(nc, nc, use_bn, deploy) for _ in range(nb_enc[0])
        ])
        # Downsample: stride-2 conv (nc → 2*nc)
        self.down0 = nn.Sequential(
            nn.Conv2d(nc, nc * 2, 2, 2, bias=not use_bn),
            nn.BatchNorm2d(nc * 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Encoder level 1: 2*nc channels, quarter-res (270p)
        self.enc1 = nn.Sequential(*[
            RepConvBlock(nc * 2, nc * 2, use_bn, deploy) for _ in range(nb_enc[1])
        ])
        # Downsample: stride-2 conv (2*nc → 4*nc)
        self.down1 = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 4, 2, 2, bias=not use_bn),
            nn.BatchNorm2d(nc * 4) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Middle: 4*nc channels, eighth-res (135p)
        self.mid = nn.Sequential(*[
            RepConvBlock(nc * 4, nc * 4, use_bn, deploy) for _ in range(nb_mid)
        ])

        # Upsample: PixelShuffle (4*nc → nc) + conv to 2*nc
        # Actually use Conv1x1 + PixelShuffle for clean upsampling
        self.up1 = nn.Sequential(
            nn.Conv2d(nc * 4, nc * 2 * 4, 1, bias=False),  # 4*nc → 8*nc
            nn.PixelShuffle(2),  # 8*nc → 2*nc at 2x resolution
        )
        # Decoder level 1: input is 2*nc (upsampled) + 2*nc (skip) = 4*nc → 2*nc
        self.dec1_reduce = nn.Sequential(
            nn.Conv2d(nc * 4, nc * 2, 1, 1, 0, bias=not use_bn),
            nn.BatchNorm2d(nc * 2) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(*[
            RepConvBlock(nc * 2, nc * 2, use_bn, deploy) for _ in range(nb_dec[0])
        ])

        # Upsample: 2*nc → nc
        self.up0 = nn.Sequential(
            nn.Conv2d(nc * 2, nc * 4, 1, bias=False),  # 2*nc → 4*nc
            nn.PixelShuffle(2),  # 4*nc → nc at 2x resolution
        )
        # Decoder level 0: input is nc (upsampled) + nc (skip) = 2*nc → nc
        self.dec0_reduce = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 1, 1, 0, bias=not use_bn),
            nn.BatchNorm2d(nc) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(*[
            RepConvBlock(nc, nc, use_bn, deploy) for _ in range(nb_dec[1])
        ])

        # Tail: nc → 12
        self.tail = nn.Conv2d(nc, in_nc * 4, 3, 1, 1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # To half-res
        x_down = self.unshuffle(x)
        feat = self.head(x_down)

        # Encoder
        e0 = self.enc0(feat)
        e1 = self.enc1(self.down0(e0))
        mid = self.mid(self.down1(e1))

        # Decoder with skip connections
        d1 = self.up1(mid)
        d1 = self.dec1(self.dec1_reduce(torch.cat([d1, e1], dim=1)))
        d0 = self.up0(d1)
        d0 = self.dec0(self.dec0_reduce(torch.cat([d0, e0], dim=1)))

        # Residual output
        correction = self.tail(d0)
        x_clean = x_down + correction
        out = self.shuffle(x_clean)

        return out[:, :, :H, :W]

    def fuse_reparam(self):
        """Fuse all RepConvBlocks for inference."""
        for module in self.modules():
            if isinstance(module, RepConvBlock):
                module.fuse_params()
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_plaindenoise(nc=64, nb=15, **kwargs):
    """Convenience constructor for PlainDenoise."""
    return PlainDenoise(in_nc=3, nc=nc, nb=nb, **kwargs)


def get_unetdenoise(nc=48, nb_enc=(2, 2), nb_dec=(2, 2), nb_mid=2, **kwargs):
    """Convenience constructor for UNetDenoise."""
    return UNetDenoise(in_nc=3, nc=nc, nb_enc=nb_enc, nb_dec=nb_dec,
                       nb_mid=nb_mid, **kwargs)


if __name__ == '__main__':
    # Quick sanity check
    for name, model in [
        ("PlainDenoise nc=64 nb=15", PlainDenoise(nc=64, nb=15)),
        ("PlainDenoise nc=48 nb=12", PlainDenoise(nc=48, nb=12)),
        ("PlainDenoise nc=32 nb=10", PlainDenoise(nc=32, nb=10)),
        ("UNetDenoise nc=48", UNetDenoise(nc=48)),
        ("UNetDenoise nc=32", UNetDenoise(nc=32)),
    ]:
        params = count_params(model)
        model.eval()  # Must be in eval mode so BN uses running stats (matches fusion)
        x = torch.randn(1, 3, 1080, 1920)
        with torch.no_grad():
            y = model(x)
        print(f"{name}: {params/1e3:.1f}K params, in={x.shape}, out={y.shape}")

        # Test reparam fusion (uses running stats, so model must be in eval mode)
        model.fuse_reparam()
        with torch.no_grad():
            y2 = model(x)
        diff = (y - y2).abs().max().item()
        print(f"  Reparam fusion error: {diff:.2e}")
