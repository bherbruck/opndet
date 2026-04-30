from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def focal_heatmap_loss(pred_logit: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet/CenterNet focal loss on Gaussian heatmap.

    pred_logit: raw logits [B, 1, H, W]
    gt:         soft targets in [0,1], 1.0 at exact center, Gaussian-decayed elsewhere [B, 1, H, W]
    """
    pred = torch.sigmoid(pred_logit).clamp(1e-6, 1 - 1e-6)
    pos_mask = gt.eq(1.0).float()
    neg_mask = 1.0 - pos_mask
    neg_weight = torch.pow(1.0 - gt, beta)

    pos_loss = -torch.pow(1.0 - pred, alpha) * torch.log(pred) * pos_mask
    neg_loss = -torch.pow(pred, alpha) * torch.log(1.0 - pred) * neg_weight * neg_mask

    n_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


class OpndetBboxLoss(nn.Module):
    def __init__(
        self,
        w_hm: float = 1.0,
        w_cxy: float = 1.0,
        w_wh: float = 5.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()
        self.w_hm = w_hm
        self.w_cxy = w_cxy
        self.w_wh = w_wh
        self.alpha = focal_alpha
        self.beta = focal_beta

    def forward(self, raw: torch.Tensor, tgt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """raw: [B, 5, H, W] pre-sigmoid logits from model.forward_raw."""
        hm_logit = raw[:, 0:1]
        cxy_logit = raw[:, 1:3]
        wh_logit = raw[:, 3:5]

        l_hm = focal_heatmap_loss(hm_logit, tgt["hm"], self.alpha, self.beta)

        pos = tgt["pos"]
        n_pos = pos.sum().clamp(min=1.0)

        cxy_pred = torch.sigmoid(cxy_logit)
        l_cxy = (F.l1_loss(cxy_pred, tgt["cxy"], reduction="none") * pos).sum() / n_pos

        wh_pred = torch.sigmoid(wh_logit)
        l_wh = (F.l1_loss(wh_pred, tgt["wh"], reduction="none") * pos).sum() / n_pos

        total = self.w_hm * l_hm + self.w_cxy * l_cxy + self.w_wh * l_wh
        return {"loss": total, "l_hm": l_hm.detach(), "l_cxy": l_cxy.detach(), "l_wh": l_wh.detach()}
