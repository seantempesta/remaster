"""Loss functions for NAFNet distillation training.

Provides:
    - CharbonnierLoss: smooth L1 pixel loss
    - PSNRLoss: PSNR-based pixel loss
    - FocalFrequencyLoss: frequency-domain loss targeting high-freq detail
    - DISTSPerceptualLoss: perceptual loss using DISTS (structure + texture)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1): sqrt((pred - target)^2 + eps^2)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class PSNRLoss(nn.Module):
    """PSNR-based loss (as used in NAFNet original training)."""
    def __init__(self):
        super().__init__()
        self.scale = 10 / math.log(10)

    def forward(self, pred, target):
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return self.scale * torch.log(mse + 1e-8).mean()


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss for image restoration (Jiang et al., ICCV 2021).

    Computes L1 distance in the frequency domain with adaptive focal
    weighting that emphasizes frequency components where the model
    struggles most. This preserves high-frequency detail (edges, texture)
    that pixel-level and perceptual losses tend to smooth away.

    Cheap to compute — just FFT + weighted L1, no extra networks.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Focal weight exponent. Higher = stronger focus on
                   hard-to-learn frequencies. 1.0 is standard.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # 2D real FFT with orthonormal normalization
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # L1 distance per frequency component
        diff = torch.abs(torch.abs(pred_fft) - torch.abs(target_fft))

        # Focal weighting: upweight frequencies with larger errors
        weight = diff.pow(self.alpha)

        return torch.mean(weight * diff)


class DISTSPerceptualLoss(nn.Module):
    """Perceptual loss using DISTS (Deep Image Structure and Texture Similarity).

    DISTS is specifically designed for image quality assessment of
    structural distortions (compression artifacts, blur, noise) while
    tolerating texture resampling. Better calibrated to human perception
    than raw VGG feature distance for restoration tasks.

    Uses the official DISTS implementation from reference-code/DISTS with
    require_grad=True so gradients flow for training. The network (VGG16
    backbone + learned alpha/beta weights) is instantiated once and reused.

    Must be called in fp32 (not inside AMP autocast) — the VGG backbone
    is numerically unstable in fp16.
    """
    def __init__(self, weights_path=None):
        super().__init__()
        import sys as _sys
        from pathlib import Path as _Path

        # Import DISTS from reference-code submodule
        dists_dir = str(_Path(__file__).resolve().parent.parent / "reference-code" / "DISTS")
        _sys.path.insert(0, dists_dir)
        from DISTS_pytorch.DISTS_pt import DISTS, L2pooling  # noqa: F401

        # Load with weights from the submodule
        if weights_path is None:
            weights_path = str(_Path(dists_dir) / "DISTS_pytorch" / "weights.pt")

        # Instantiate without loading (to avoid sys.prefix path issue)
        self._net = DISTS(load_weights=False)
        weights = torch.load(weights_path, weights_only=True)
        self._net.alpha.data = weights['alpha']
        self._net.beta.data = weights['beta']
        # VGG weights need requires_grad=True so gradients flow through
        # to our denoiser. They're NOT added to the optimizer so they
        # won't be updated — just used as a differentiable feature extractor.
        self.eval()

    def forward(self, pred, target):
        # Run pred through VGG with graph (need gradients for backprop).
        # Run target without graph (26% faster, no gradients needed).
        feats_pred = self._net.forward_once(pred)
        with torch.no_grad():
            feats_tgt = self._net.forward_once(target)

        # Compute DISTS structure (S1) and texture (S2) similarity
        dist1 = 0
        dist2 = 0
        c1, c2 = 1e-6, 1e-6
        w_sum = self._net.alpha.sum() + self._net.beta.sum()
        alpha = torch.split(self._net.alpha / w_sum, self._net.chns, dim=1)
        beta = torch.split(self._net.beta / w_sum, self._net.chns, dim=1)
        for k in range(len(self._net.chns)):
            x_mean = feats_pred[k].mean([2, 3], keepdim=True)
            y_mean = feats_tgt[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats_pred[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats_tgt[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats_pred[k] * feats_tgt[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        # Distance: 0 = identical, ~1 = very different
        score = 1 - (dist1 + dist2).squeeze()
        return score.mean()

    def train(self, mode=True):
        # Always eval mode (frozen VGG backbone)
        return super().train(False)


def build_pixel_criterion(name):
    """Build pixel-level loss by name."""
    if name == "charbonnier":
        return CharbonnierLoss(eps=1e-6)
    elif name == "psnr":
        return PSNRLoss()
    elif name == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown pixel loss: {name}")
