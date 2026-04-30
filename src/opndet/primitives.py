from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from opndet.registry import register


@register("Concat")
class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)


@register("Add")
class Add(nn.Module):
    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out


@register("ResizeNearest2x")
class ResizeNearest2x(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="nearest")


@register("MaxPool")
class MaxPool(nn.Module):
    def __init__(self, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        pad = p if p is not None else k // 2
        self.pool = nn.MaxPool2d(k, stride=s, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register("Sigmoid")
class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


@register("Tanh")
class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


@register("PeakSuppress")
class PeakSuppress(nn.Module):
    """In-graph local-max via GreaterOrEqual(hm + eps, MaxPool(hm)).

    Tolerance eps absorbs FP drift PyTorch <-> ORT/OpenVINO so mask is bit-stable.
    """

    def __init__(self, k: int = 3, eps: float = 1e-3):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.k // 2)
        mask = (x + self.eps >= pooled).to(x.dtype)
        return x * mask


@register("SigmoidPeakSuppress")
class SigmoidPeakSuppress(nn.Module):
    """Sigmoid then peak-suppress. Single fused op for the peaks head."""

    def __init__(self, k: int = 3, eps: float = 1e-3):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hm = torch.sigmoid(x)
        pooled = F.max_pool2d(hm, kernel_size=self.k, stride=1, padding=self.k // 2)
        mask = (hm + self.eps >= pooled).to(hm.dtype)
        return hm * mask


@register("SigmoidScale")
class SigmoidScale(nn.Module):
    def __init__(self, scale: float = 64.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale


@register("SplitChannels")
class SplitChannels(nn.Module):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start : self.end]


@register("Conv")
class Conv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        bias: bool = True,
        bias_init: list[float] | float | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=bias)
        if bias and bias_init is not None:
            with torch.no_grad():
                if isinstance(bias_init, (int, float)):
                    self.conv.bias.fill_(float(bias_init))
                else:
                    init = torch.tensor(bias_init, dtype=self.conv.bias.dtype)
                    assert init.numel() == out_ch, f"bias_init len {init.numel()} != out_ch {out_ch}"
                    self.conv.bias.copy_(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register("Identity")
class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "sigmoid_peak": SigmoidPeakSuppress,
    "none": Identity,
}
