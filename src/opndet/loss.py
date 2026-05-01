from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _decode_pred_xyxy(cxy_pred: torch.Tensor, wh_pred: torch.Tensor, cxy_gt: torch.Tensor, pos_mask: torch.Tensor, img_h: int, img_w: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode predicted (cxy_rel, wh_norm) to absolute xyxy at positive cells.
    Returns (pred_xyxy, gt_xyxy_image_coords) only at positive cells.
    All in pixel coords; differentiable.
    """
    B, _, H, W = cxy_pred.shape
    device = cxy_pred.device
    ys = torch.arange(H, device=device, dtype=cxy_pred.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.arange(W, device=device, dtype=cxy_pred.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    cx_px = (xs + cxy_pred[:, 0:1]) * stride
    cy_px = (ys + cxy_pred[:, 1:2]) * stride
    w_px = wh_pred[:, 0:1] * img_w
    h_px = wh_pred[:, 1:2] * img_h
    pred = torch.cat([cx_px - w_px / 2, cy_px - h_px / 2, cx_px + w_px / 2, cy_px + h_px / 2], dim=1)
    return pred  # [B,4,H,W]


def _bbox_iou(p: torch.Tensor, g: torch.Tensor, mode: str = "giou", eps: float = 1e-7) -> torch.Tensor:
    """Pairwise per-cell IoU/GIoU/CIoU. p,g: [B,4,H,W] in xyxy pixels. Returns [B,1,H,W] of (1 - IoU)-like loss."""
    px1, py1, px2, py2 = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    gx1, gy1, gx2, gy2 = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    pw = (px2 - px1).clamp(min=0); ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0); gh = (gy2 - gy1).clamp(min=0)
    p_area = pw * ph; g_area = gw * gh

    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = p_area + g_area - inter + eps
    iou = inter / union

    if mode == "iou":
        return 1.0 - iou

    cx1 = torch.min(px1, gx1); cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2); cy2 = torch.max(py2, gy2)
    c_w = (cx2 - cx1).clamp(min=0); c_h = (cy2 - cy1).clamp(min=0)

    if mode == "giou":
        c_area = c_w * c_h + eps
        giou = iou - (c_area - union) / c_area
        return 1.0 - giou

    if mode == "ciou":
        c_diag2 = c_w * c_w + c_h * c_h + eps
        p_cx = (px1 + px2) * 0.5; p_cy = (py1 + py2) * 0.5
        g_cx = (gx1 + gx2) * 0.5; g_cy = (gy1 + gy2) * 0.5
        center_d2 = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2
        v = (4 / (math.pi ** 2)) * (torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return 1.0 - iou + center_d2 / c_diag2 + alpha * v

    raise ValueError(f"unknown iou mode: {mode}")


def _nwd(p: torch.Tensor, g: torch.Tensor, c: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Normalized Wasserstein Distance loss for tiny-object regression. Treats boxes as 2D Gaussians.
    1 - exp(-W2 / c). c is a tunable normalizer (~12.8 from the paper for AI-TOD).
    """
    px1, py1, px2, py2 = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    gx1, gy1, gx2, gy2 = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    p_cx = (px1 + px2) * 0.5; p_cy = (py1 + py2) * 0.5
    g_cx = (gx1 + gx2) * 0.5; g_cy = (gy1 + gy2) * 0.5
    pw = (px2 - px1).clamp(min=eps); ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps); gh = (gy2 - gy1).clamp(min=eps)
    center_term = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2
    size_term = ((pw - gw) ** 2 + (ph - gh) ** 2) * 0.25
    w2 = center_term + size_term
    return 1.0 - torch.exp(-torch.sqrt(w2 + eps) / c)


def _iou_only(p: torch.Tensor, g: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise per-cell IoU. p,g: [B,4,H,W] xyxy. Returns [B,1,H,W] in [0,1]."""
    px1, py1, px2, py2 = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    gx1, gy1, gx2, gy2 = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    pw = (px2 - px1).clamp(min=0); ph = (py2 - py1).clamp(min=0)
    gw = (gx2 - gx1).clamp(min=0); gh = (gy2 - gy1).clamp(min=0)
    p_area = pw * ph; g_area = gw * gh
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = p_area + g_area - inter + eps
    return inter / union


def varifocal_loss(
    pred_logit: torch.Tensor,
    pos_mask: torch.Tensor,
    iou_target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Varifocal Loss (Zhang et al., 2021).

    pred_logit: [B,1,H,W] raw obj logit
    pos_mask:   [B,1,H,W] 1.0 at positive cells, 0 elsewhere
    iou_target: [B,1,H,W] IoU between predicted bbox and GT bbox at each cell
                (0 elsewhere). Detach gradients — quality target is supervisory only.

    Positive cells: BCE weighted by IoU (soft target = IoU).
    Negative cells: focal-style alpha * p^gamma * BCE, downweights easy negatives.

    Effect: confidence becomes bimodal — strong predictions push to ~IoU which is high
    (0.8-1.0), weak predictions get pushed near 0. Eliminates the squishy mid-range
    that causes detection flapping.
    """
    p = torch.sigmoid(pred_logit).clamp(eps, 1 - eps)
    q = iou_target.detach() * pos_mask
    pos = pos_mask
    neg = 1.0 - pos_mask
    weight = q * pos + alpha * p.pow(gamma) * neg
    bce = -(q * torch.log(p) + (1 - q) * torch.log(1 - p))
    n_pos = pos.sum().clamp(min=1.0)
    return (weight * bce).sum() / n_pos


def _peak_suppress(hm: torch.Tensor, k: int = 5, eps: float = 5e-3) -> torch.Tensor:
    """Differentiable arithmetic peak suppression matching the deployed in-graph op.
    Used at train time so count-aware loss sees the same sparse peak map as inference."""
    pad = k // 2
    pooled = F.max_pool2d(hm, kernel_size=k, stride=1, padding=pad)
    mask = ((hm + eps - pooled) * (1.0 / eps)).clamp(0.0, 1.0)
    return hm * mask


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
        wh_loss: str = "l1",            # l1 | giou | ciou | nwd
        cls_loss: str = "focal",        # focal | vfl
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        repulsion_weight: float = 0.0,
        nwd_c: float = 12.8,
        count_weight: float = 0.0,
        peak_kernel: int = 5,
        peak_eps: float = 5e-3,
        img_h: int = 384,
        img_w: int = 512,
        stride: int = 4,
    ):
        super().__init__()
        self.w_hm = w_hm
        self.w_cxy = w_cxy
        self.w_wh = w_wh
        self.alpha = focal_alpha
        self.beta = focal_beta
        self.wh_loss = wh_loss
        self.cls_loss = cls_loss
        self.vfl_alpha = vfl_alpha
        self.vfl_gamma = vfl_gamma
        self.rep_w = repulsion_weight
        self.nwd_c = nwd_c
        self.count_w = count_weight
        self.peak_kernel = peak_kernel
        self.peak_eps = peak_eps
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride

    def forward(self, raw: torch.Tensor, tgt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hm_logit = raw[:, 0:1]
        cxy_logit = raw[:, 1:3]
        wh_logit = raw[:, 3:5]

        pos = tgt["pos"]
        n_pos = pos.sum().clamp(min=1.0)

        cxy_pred = torch.sigmoid(cxy_logit)
        l_cxy = (F.l1_loss(cxy_pred, tgt["cxy"], reduction="none") * pos).sum() / n_pos

        wh_pred = torch.sigmoid(wh_logit)
        # decoded boxes needed for ciou/giou/nwd wh loss AND for VFL IoU target
        pred_xyxy = None
        gt_xyxy = None
        if self.wh_loss != "l1" or self.cls_loss == "vfl":
            pred_xyxy = _decode_pred_xyxy(cxy_pred, wh_pred, tgt["cxy"], pos, self.img_h, self.img_w, self.stride)
            gt_cx = (tgt["cxy"][:, 0:1] + _grid_xs(cxy_pred)) * self.stride
            gt_cy = (tgt["cxy"][:, 1:2] + _grid_ys(cxy_pred)) * self.stride
            gw = tgt["wh"][:, 0:1] * self.img_w
            gh = tgt["wh"][:, 1:2] * self.img_h
            gt_xyxy = torch.cat([gt_cx - gw/2, gt_cy - gh/2, gt_cx + gw/2, gt_cy + gh/2], dim=1)

        if self.wh_loss == "l1":
            l_wh = (F.l1_loss(wh_pred, tgt["wh"], reduction="none") * pos).sum() / n_pos
        elif self.wh_loss == "nwd":
            l_wh = (_nwd(pred_xyxy, gt_xyxy, c=self.nwd_c) * pos).sum() / n_pos
        else:
            l_wh = (_bbox_iou(pred_xyxy, gt_xyxy, mode=self.wh_loss) * pos).sum() / n_pos

        if self.cls_loss == "vfl":
            iou_target = _iou_only(pred_xyxy, gt_xyxy) * pos
            l_hm = varifocal_loss(hm_logit, pos, iou_target, alpha=self.vfl_alpha, gamma=self.vfl_gamma)
        else:
            l_hm = focal_heatmap_loss(hm_logit, tgt["hm"], self.alpha, self.beta)

        total = self.w_hm * l_hm + self.w_cxy * l_cxy + self.w_wh * l_wh
        out = {"loss": total, "l_hm": l_hm.detach(), "l_cxy": l_cxy.detach(), "l_wh": l_wh.detach()}

        if self.rep_w > 0 and pred_xyxy is not None:
            l_rep = _repulsion_loss(pred_xyxy, tgt, pos, self.img_h, self.img_w, self.stride)
            out["loss"] = out["loss"] + self.rep_w * l_rep
            out["l_rep"] = l_rep.detach()

        if self.count_w > 0:
            # Sparse peak map matches deployed graph; sum at peak cells = predicted count.
            # GT count = number of positive cells (one per object's center).
            hm_sig = torch.sigmoid(hm_logit)
            peaks = _peak_suppress(hm_sig, k=self.peak_kernel, eps=self.peak_eps)
            pred_count = peaks.flatten(1).sum(dim=1)        # [B]
            gt_count = pos.flatten(1).sum(dim=1)            # [B]
            l_count = (pred_count - gt_count).abs().mean()
            out["loss"] = out["loss"] + self.count_w * l_count
            out["l_count"] = l_count.detach()

        return out


def _grid_xs(ref: torch.Tensor) -> torch.Tensor:
    B, _, H, W = ref.shape
    return torch.arange(W, device=ref.device, dtype=ref.dtype).view(1, 1, 1, W).expand(B, 1, H, W)


def _grid_ys(ref: torch.Tensor) -> torch.Tensor:
    B, _, H, W = ref.shape
    return torch.arange(H, device=ref.device, dtype=ref.dtype).view(1, 1, H, 1).expand(B, 1, H, W)


def _repulsion_loss(pred_xyxy: torch.Tensor, tgt: dict, pos: torch.Tensor, img_h: int, img_w: int, stride: int) -> torch.Tensor:
    """RepGT-style: penalize predictions overlapping non-target neighbor GT cells.
    Approximation: at each positive cell, find the nearest other positive cell in the same image
    and penalize IoU between this prediction and that neighbor's GT box.
    """
    B = pred_xyxy.shape[0]
    total = pred_xyxy.new_zeros(())
    count = 0
    for b in range(B):
        pos_b = pos[b, 0]
        ys, xs = torch.where(pos_b > 0)
        if ys.numel() < 2:
            continue
        cy = (tgt["cxy"][b, 1, ys, xs] + ys.to(pred_xyxy.dtype)) * stride
        cx = (tgt["cxy"][b, 0, ys, xs] + xs.to(pred_xyxy.dtype)) * stride
        gw = tgt["wh"][b, 0, ys, xs] * img_w
        gh = tgt["wh"][b, 1, ys, xs] * img_h
        gt_box = torch.stack([cx - gw/2, cy - gh/2, cx + gw/2, cy + gh/2], dim=1)
        # for each positive cell, compute pairwise center distance to others; pick nearest neighbor
        d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
        d2.fill_diagonal_(float("inf"))
        nn_idx = d2.argmin(dim=1)
        neighbor_gt = gt_box[nn_idx]
        my_pred = pred_xyxy[b, :, ys, xs].t()  # [N,4]
        ix1 = torch.max(my_pred[:, 0], neighbor_gt[:, 0]); iy1 = torch.max(my_pred[:, 1], neighbor_gt[:, 1])
        ix2 = torch.min(my_pred[:, 2], neighbor_gt[:, 2]); iy2 = torch.min(my_pred[:, 3], neighbor_gt[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        nb_area = (neighbor_gt[:, 2] - neighbor_gt[:, 0]).clamp(min=0) * (neighbor_gt[:, 3] - neighbor_gt[:, 1]).clamp(min=0) + 1e-7
        ioa = inter / nb_area  # IoU-of-Area = how much of neighbor is hit by my prediction
        total = total + ioa.mean()
        count += 1
    return total / max(1, count)
