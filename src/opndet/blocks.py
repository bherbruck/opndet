from __future__ import annotations

import torch
from torch import nn


def _conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class DWSep(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s: int = 1):
        super().__init__()
        self.dw = _conv_bn_act(in_ch, in_ch, k=3, s=s, g=in_ch)
        self.pw = _conv_bn_act(in_ch, out_ch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CSPBlock(nn.Module):
    """CSP-style: split -> n DWSep chain -> concat -> 1x1 fuse.
    Implemented with 1x1 splits (Conv) instead of torch.split to keep export deterministic.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, s: int = 1):
        super().__init__()
        hidden = out_ch // 2
        self.down = _conv_bn_act(in_ch, out_ch, k=3, s=s) if (s != 1 or in_ch != out_ch) else nn.Identity()
        self.split_a = _conv_bn_act(out_ch, hidden, k=1)
        self.split_b = _conv_bn_act(out_ch, hidden, k=1)
        self.chain = nn.Sequential(*[DWSep(hidden, hidden) for _ in range(n)])
        self.fuse = _conv_bn_act(hidden * 2, out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        a = self.split_a(x)
        b = self.split_b(x)
        b = self.chain(b)
        return self.fuse(torch.cat([a, b], dim=1))
