from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from opndet.blocks import CSPBlock, _conv_bn_act
from opndet.config import ModelConfig


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c1, c2, c3, c4 = cfg.stage_ch
        n1, n2, n3, n4 = cfg.stage_n
        self.stem = _conv_bn_act(cfg.in_ch, cfg.base_ch, k=3, s=2)
        self.stage1 = CSPBlock(cfg.base_ch, c1, n=n1, s=2)
        self.stage2 = CSPBlock(c1, c2, n=n2, s=2)
        self.stage3 = CSPBlock(c2, c3, n=n3, s=2)
        self.stage4 = CSPBlock(c3, c4, n=n4, s=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return p1, p2, p3, p4


class Neck(nn.Module):
    """Top-down FPN-lite. Resize nearest + 1x1 lateral + add. Output at stride 4 (P1 level)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c1, c2, c3, c4 = cfg.stage_ch
        nc = cfg.neck_ch
        self.lat4 = _conv_bn_act(c4, nc, k=1)
        self.lat3 = _conv_bn_act(c3, nc, k=1)
        self.lat2 = _conv_bn_act(c2, nc, k=1)
        self.lat1 = _conv_bn_act(c1, nc, k=1)
        self.fuse3 = _conv_bn_act(nc, nc, k=3)
        self.fuse2 = _conv_bn_act(nc, nc, k=3)
        self.fuse1 = _conv_bn_act(nc, nc, k=3)

    @staticmethod
    def _up2(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    def forward(self, feats: tuple[torch.Tensor, ...]) -> torch.Tensor:
        p1, p2, p3, p4 = feats
        x = self.lat4(p4)
        x = self.fuse3(self._up2(x) + self.lat3(p3))
        x = self.fuse2(self._up2(x) + self.lat2(p2))
        x = self.fuse1(self._up2(x) + self.lat1(p1))
        return x


class Head(nn.Module):
    """Single-tensor head: [B, 5, H/4, W/4] = [obj, cx, cy, w, h]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.trunk = _conv_bn_act(cfg.neck_ch, cfg.head_ch, k=3)
        self.proj = nn.Conv2d(cfg.head_ch, 5, kernel_size=1, bias=True)
        nn.init.constant_(self.proj.bias, 0.0)
        nn.init.constant_(self.proj.bias[0], -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(x))


class PeakSuppress(nn.Module):
    """In-graph local-max via GreaterOrEqual(hm + eps, MaxPool(hm)).

    Tolerance eps absorbs FP drift between PyTorch and ORT/OpenVINO so the mask is
    bit-stable across runtimes. eps small enough that real non-maxima still fail.
    """

    def __init__(self, k: int = 3, eps: float = 1e-3):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, hm: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(hm, kernel_size=self.k, stride=1, padding=self.k // 2)
        mask = (hm + self.eps >= pooled).to(hm.dtype)
        return hm * mask


class OpndetBbox(nn.Module):
    """opndet-bbox variant: single output [B, 5, H/4, W/4].

    Channels (post-activation): [obj_peak, cx_rel, cy_rel, w_norm, h_norm].
    obj_peak is sigmoid-then-peak-suppressed in graph (no NMS needed client-side).
    cx, cy in [0, 1] cell-relative. w, h in [0, 1] image-normalized.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.backbone = Backbone(self.cfg)
        self.neck = Neck(self.cfg)
        self.head = Head(self.cfg)
        self.peak = PeakSuppress(self.cfg.peak_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f = self.neck(feats)
        raw = self.head(f)
        obj = torch.sigmoid(raw[:, 0:1])
        obj = self.peak(obj)
        cxy = torch.sigmoid(raw[:, 1:3])
        wh = torch.sigmoid(raw[:, 3:5])
        return torch.cat([obj, cxy, wh], dim=1)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Training-time forward: returns raw logits for first 5 channels (no peak suppress).

        Loss operates on raw logits + sigmoid; peak suppression is inference-only.
        """
        feats = self.backbone(x)
        f = self.neck(feats)
        return self.head(f)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
