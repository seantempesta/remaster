"""
NAFNet architecture — extracted from megvii-research/NAFNet for standalone use.
Original: https://github.com/megvii-research/NAFNet
License: MIT

Only the core NAFNet model is included here (no basicsr dependency).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # The original LayerNormFunction custom autograd uses channel-wise
        # mean/var which loses precision catastrophically in fp16, producing
        # wildly out-of-range values.  Cast to fp32 for the normalisation
        # arithmetic (training is unaffected — backward is still correct via
        # autograd on the fp32 ops).
        if x.dtype == torch.float32:
            return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        dtype = x.dtype
        x = x.float()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / (var + self.eps).sqrt()
        x = self.weight.float().view(1, -1, 1, 1) * x + self.bias.float().view(1, -1, 1, 1)
        return x.to(dtype)


class LayerNorm2dExport(nn.Module):
    """TRT/ONNX-export-safe version of LayerNorm2d.

    Uses only standard ops (no custom autograd.Function, no dtype branching).
    Always casts to fp32 for the normalization arithmetic and back.
    Numerically identical to LayerNorm2d's fp16 path.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        x_f = x.float()
        mu = x_f.mean(1, keepdim=True)
        var = (x_f - mu).pow(2).mean(1, keepdim=True)
        x_f = (x_f - mu) / (var + self.eps).sqrt()
        x_f = self.weight.float().view(1, -1, 1, 1) * x_f + self.bias.float().view(1, -1, 1, 1)
        return x_f.half()


def _replace_modules(model, predicate, factory):
    """Replace modules matching predicate with factory(module). In-place."""
    for name, module in model.named_modules():
        if predicate(module):
            replacement = factory(module)
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            if parts[-1].isdigit():
                parent[int(parts[-1])] = replacement
            else:
                setattr(parent, parts[-1], replacement)
    return model


def swap_layernorm_for_export(model):
    """Replace all LayerNorm2d modules with LayerNorm2dExport (TRT-safe).

    Copies weights/bias from the original modules. Modifies the model in-place
    and returns it for convenience.
    """
    def make_export(m):
        ln = LayerNorm2dExport(m.weight.shape[0], m.eps)
        ln.weight = m.weight
        ln.bias = m.bias
        return ln
    return _replace_modules(model, lambda m: isinstance(m, LayerNorm2d), make_export)


class LayerNorm2dCompile(nn.Module):
    """torch.compile-friendly version of LayerNorm2d.

    Uses F.layer_norm (a standard op Inductor can fuse) instead of the custom
    autograd.Function. Normalizes over the channel dimension at each spatial
    position — identical to LayerNorm2d. Handles fp16 by casting to fp32
    internally (same as PyTorch's native layer_norm).
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # Cast to fp32 for numerical stability (same as original LayerNorm2d fp16 path)
        dtype = x.dtype
        # NCHW -> NHWC, apply layer_norm over C, NHWC -> NCHW
        return F.layer_norm(
            x.float().permute(0, 2, 3, 1), [self.channels],
            self.weight.float(), self.bias.float(), self.eps
        ).permute(0, 3, 1, 2).to(dtype)


def swap_layernorm_for_compile(model):
    """Replace all LayerNorm2d with LayerNorm2dCompile for better torch.compile fusion.

    Numerically identical to LayerNorm2d (both normalize over C at each H,W position).
    Same weights, no retraining needed. Uses F.layer_norm which is a standard op
    that Inductor can fuse, unlike the custom autograd.Function in LayerNorm2d.
    """
    def make_compile(m):
        ln = LayerNorm2dCompile(m.weight.shape[0], m.eps)
        ln.weight = m.weight
        ln.bias = m.bias
        return ln
    return _replace_modules(model, lambda m: isinstance(m, LayerNorm2d), make_compile)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
