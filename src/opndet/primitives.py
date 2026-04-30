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


def _peak_mask(hm: torch.Tensor, k: int, eps: float, mode: str) -> torch.Tensor:
    """Local-max mask via either:
      mode='arith' (default): mask = clip((hm + eps - MaxPool(hm)) * BIG, 0, 1)
        Pure arithmetic, no comparison op. Required for OpenVINO 2022.1 Myriad
        which has a known plugin bug lowering GreaterOrEqual to a buggy
        Logical_OR stage that asserts FP16 inputs and fails for float32.
      mode='compare': mask = (hm + eps >= MaxPool(hm)). Smaller graph but breaks
        on Myriad VPU. OK for CPU/GPU OpenVINO and ORT.

    eps must be large enough to absorb the runtime's MaxPool rounding error.
    fp32: 1e-3 is fine. fp16 (Myriad): need ~5e-3 because fp16 precision at
    sigmoid-output values 0.5..1.0 is roughly 5e-4..1e-3, and MaxPool's
    rounding path can disagree with the hm path by a few units in the last
    place, pushing real peaks just below the eps margin.
    """
    pooled = F.max_pool2d(hm, kernel_size=k, stride=1, padding=k // 2)
    if mode == "compare":
        return (hm + eps >= pooled).to(hm.dtype)
    big = 1.0 / max(eps, 1e-9)
    diff = (hm + eps) - pooled
    return torch.clamp(diff * big, 0.0, 1.0)


@register("PeakSuppress")
class PeakSuppress(nn.Module):
    """In-graph local-max suppression. Default mode='arith' is Myriad-safe.

    eps must exceed the runtime's MaxPool rounding error. Default 5e-3 is
    sized for fp16 (Myriad). For fp32-only deployment 1e-3 is enough.
    """

    def __init__(self, k: int = 3, eps: float = 5e-3, mode: str = "arith"):
        super().__init__()
        self.k = k
        self.eps = eps
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * _peak_mask(x, self.k, self.eps, self.mode)


@register("SigmoidPeakSuppress")
class SigmoidPeakSuppress(nn.Module):
    """Sigmoid then peak-suppress (fused op for the obj head). Default mode='arith'."""

    def __init__(self, k: int = 3, eps: float = 5e-3, mode: str = "arith"):
        super().__init__()
        self.k = k
        self.eps = eps
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hm = torch.sigmoid(x)
        return hm * _peak_mask(hm, self.k, self.eps, self.mode)


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
