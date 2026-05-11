from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _decode_pred_ltrb_xyxy(ltrb_pred: torch.Tensor, img_h: int, img_w: int, stride: int) -> torch.Tensor:
    """Decode predicted (l, t, r, b) image-normalized to absolute xyxy in pixels.
    ltrb_pred: [B, 4, H, W] post-sigmoid in [0,1]. Cell centers are (ix+0.5, iy+0.5)*stride.
    """
    B, _, H, W = ltrb_pred.shape
    device = ltrb_pred.device
    ys = torch.arange(H, device=device, dtype=ltrb_pred.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.arange(W, device=device, dtype=ltrb_pred.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    cx_px = (xs + 0.5) * stride
    cy_px = (ys + 0.5) * stride
    l = ltrb_pred[:, 0:1] * img_w
    t = ltrb_pred[:, 1:2] * img_h
    r = ltrb_pred[:, 2:3] * img_w
    b = ltrb_pred[:, 3:4] * img_h
    return torch.cat([cx_px - l, cy_px - t, cx_px + r, cy_px + b], dim=1)


def _decode_gt_ltrb_xyxy(ltrb_gt: torch.Tensor, img_h: int, img_w: int, stride: int) -> torch.Tensor:
    """Same as _decode_pred_ltrb_xyxy but for GT ltrb tensor."""
    return _decode_pred_ltrb_xyxy(ltrb_gt, img_h, img_w, stride)


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

    if mode == "diou":
        # CIoU minus the aspect-ratio (v) term. Same center+containment
        # penalty, NO aspect blow-up at random init. Use this when CIoU is
        # destabilizing wh early in training but you still want the IoU
        # signal vs plain L1.
        c_diag2 = c_w * c_w + c_h * c_h + eps
        p_cx = (px1 + px2) * 0.5; p_cy = (py1 + py2) * 0.5
        g_cx = (gx1 + gx2) * 0.5; g_cy = (gy1 + gy2) * 0.5
        center_d2 = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2
        return 1.0 - iou + center_d2 / c_diag2

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


def probiou_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """ProbIoU rotated-box loss (Bhattacharyya distance between 2D Gaussians).

    pred, gt: [..., 5] = (cx, cy, w, h, theta_rad), all in same units (pixels).
    Returns 1 - ProbIoU per element, shape [...].

    Each rotated rectangle is approximated by a 2D Gaussian with covariance
    Σ = R · diag(w²/12, h²/12) · Rᵀ. The variance w²/12 follows from a uniform
    distribution along each axis (rectangle ≈ uniform 2D distribution).

    The Bhattacharyya distance between the two Gaussians has a closed-form
    expression in terms of the means and covariances. Hellinger ↔ ProbIoU.

    Adapted from YOLOv8-OBB's reference implementation. Numerical safeguards:
    eps in denominators, clamp on log argument, clamp on Bhattacharyya before exp.
    """
    px, py, pw, ph, pt = pred.unbind(-1)
    gx, gy, gw, gh, gt_t = gt.unbind(-1)

    cos_pt, sin_pt = torch.cos(pt), torch.sin(pt)
    cos_gt, sin_gt = torch.cos(gt_t), torch.sin(gt_t)

    pw2_12, ph2_12 = pw * pw / 12.0, ph * ph / 12.0
    gw2_12, gh2_12 = gw * gw / 12.0, gh * gh / 12.0

    # Σ = [[a, c], [c, b]] for each box
    a1 = pw2_12 * cos_pt * cos_pt + ph2_12 * sin_pt * sin_pt
    b1 = pw2_12 * sin_pt * sin_pt + ph2_12 * cos_pt * cos_pt
    c1 = (pw2_12 - ph2_12) * cos_pt * sin_pt
    a2 = gw2_12 * cos_gt * cos_gt + gh2_12 * sin_gt * sin_gt
    b2 = gw2_12 * sin_gt * sin_gt + gh2_12 * cos_gt * cos_gt
    c2 = (gw2_12 - gh2_12) * cos_gt * sin_gt

    A = a1 + a2
    B = b1 + b2
    C = c1 + c2
    detM = (A * B - C * C).clamp(min=eps)

    dx = px - gx
    dy = py - gy
    t1 = (A * dy * dy + B * dx * dx - 2.0 * C * dx * dy) / detM / 4.0

    det1 = (a1 * b1 - c1 * c1).clamp(min=eps)
    det2 = (a2 * b2 - c2 * c2).clamp(min=eps)
    t2 = 0.5 * torch.log((detM / (4.0 * torch.sqrt(det1 * det2) + eps)).clamp(min=eps))

    bd = (t1 + t2).clamp(min=eps, max=100.0)
    hd = torch.sqrt((1.0 - torch.exp(-bd)).clamp(min=0.0, max=1.0) + eps)
    iou = 1.0 - hd  # ProbIoU
    return 1.0 - iou  # loss


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


def convexity_loss(hm_pred: torch.Tensor, pos: torch.Tensor, k: int = 2, eps: float = 1e-6,
                   sigma_cells: torch.Tensor | None = None) -> torch.Tensor:
    """Per-object convexity / centroid-alignment regularizer.

    For each positive cell, extract the (2k+1)×(2k+1) heatmap patch, optionally
    weight it by a radial Gaussian whose width matches that object's size, compute
    the mass-weighted centroid offset from the cell center, and penalize it.
    Forces symmetric, centroid-aligned peaks — the convex-object prior.

    Why the per-object Gaussian: a fixed k that's too small for big objects only
    sees the blob's tip (centroid trivially centered → no signal). Sizing the
    window to the object fixes that — and the Gaussian (vs. a hard box) also
    means a *neighbor's* blob sitting inside an oversized window doesn't drag the
    centroid, which matters for crowded/touching objects.

    hm_pred:     [B,1,H,W] post-sigmoid objectness.
    pos:         [B,1,H,W] positive-cell mask (1 at GT center cell, 0 elsewhere).
    k:           MAX neighborhood radius in cells (the window cap — size it to
                 your *biggest* object; smaller ones self-narrow via sigma).
    sigma_cells: [B,1,H,W] per-cell object Gaussian σ in cells (≈ 0.5·min(w,h)/
                 stride), nonzero at positives. None → flat (2k+1) window (legacy).

    Memory: materializes a [B, (2k+1)², H·W] tensor (twice, transiently, when
    sigma is given) — so big k on a big heatmap at big batch can spike VRAM.
    """
    K = 2 * k + 1
    patches = F.unfold(hm_pred, kernel_size=K, padding=k, stride=1)  # [B, K*K, H*W]
    dev, dt = hm_pred.device, hm_pred.dtype
    coords = torch.arange(K, device=dev, dtype=dt) - float(k)  # [-k..k]
    dx = coords.repeat(K).view(1, K * K, 1)              # x offset within patch
    dy = coords.repeat_interleave(K).view(1, K * K, 1)   # y offset within patch
    if sigma_cells is not None:
        s = sigma_cells.flatten(2).clamp(min=1.0, max=float(k))            # [B,1,HW]
        r2 = dx * dx + dy * dy                                            # [1,KK,1]
        patches = patches * torch.exp(-0.5 * r2 / (s * s))               # [B,KK,HW]
    mass = patches.sum(dim=1, keepdim=True) + eps        # [B, 1, HW]
    cx_off = (patches * dx).sum(dim=1, keepdim=True) / mass
    cy_off = (patches * dy).sum(dim=1, keepdim=True) / mass
    pos_flat = pos.flatten(2)                            # [B, 1, HW]
    n_pos = pos_flat.sum().clamp(min=1.0)
    return ((cx_off.abs() + cy_off.abs()) * pos_flat).sum() / n_pos


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


def quality_focal_loss(pred_logit: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Quality Focal Loss (GFL): regress a *soft* target ∈ [0,1] (here the full
    Gaussian heatmap) with focal-style downweighting of well-predicted cells.
    Unlike center-only focal — which hard-supervises ONLY the exact peak cell and
    merely *softens* the negative penalty on its neighbours — QFL makes every cell
    a real target, so the model learns an object-*shaped* objectness dome, not a
    1-cell nipple. Normalized by the count of exact-1.0 cells (object centers) so
    the magnitude matches focal_heatmap_loss / varifocal_loss (w_hm stays
    comparable). Works as a `cls_loss: soft_hm` mode, ideally paired with
    `hm_blob_frac > 0` so the target dome actually covers the object.
    """
    p = torch.sigmoid(pred_logit).clamp(1e-6, 1 - 1e-6)
    bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    qfl = (target - p).abs().pow(beta) * bce
    n_pos = target.eq(1.0).float().sum().clamp(min=1.0)
    return qfl.sum() / n_pos


class OpndetBboxLoss(nn.Module):
    def __init__(
        self,
        w_hm: float = 1.0,
        w_cxy: float = 1.0,
        w_wh: float = 5.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        wh_loss: str = "l1",            # l1 | giou | ciou | diou | nwd | ltrb | obb
        cls_loss: str = "focal",        # focal | vfl
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        qfl_beta: float = 2.0,           # only used when cls_loss == "soft_hm"
        repulsion_weight: float = 0.0,
        nwd_c: float = 12.8,
        count_weight: float = 0.0,
        peak_kernel: int = 5,
        peak_eps: float = 5e-3,
        convexity_weight: float = 0.0,
        convexity_radius: int = 2,
        dist_weight: float = 0.5,
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
        self.qfl_beta = qfl_beta
        self.rep_w = repulsion_weight
        self.nwd_c = nwd_c
        self.count_w = count_weight
        self.peak_kernel = peak_kernel
        self.peak_eps = peak_eps
        self.convex_w = convexity_weight
        self.convex_r = convexity_radius
        self.dist_w = dist_weight
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride

    def _convex_sigma(self, w_norm: torch.Tensor, h_norm: torch.Tensor) -> torch.Tensor:
        """Per-cell object Gaussian σ for the convexity window, in heatmap cells
        (≈ object radius). w_norm/h_norm are image-normalized box extents [B,1,H,W];
        garbage/zero at non-positive cells but those don't contribute to the loss."""
        w_cells = w_norm * (self.img_w / self.stride)
        h_cells = h_norm * (self.img_h / self.stride)
        return 0.5 * torch.minimum(w_cells, h_cells)

    def forward(self, raw: torch.Tensor, tgt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hm_logit = raw[:, 0:1]

        pos = tgt["pos"]
        n_pos = pos.sum().clamp(min=1.0)

        # obb mode: raw[:, 1:6] is (cx_off, cy_off, w_norm, h_norm, θ_norm) ALL post-sigmoid.
        # ProbIoU loss: treat each rotated box as a 2D Gaussian. Loss is differentiable
        # everywhere, wrap-continuous (handles π-symmetry naturally), and directly
        # correlates with rotated IoU. NO enclosing-AABB anywhere — pure (cx, cy, w, h, θ).
        if self.wh_loss == "obb":
            B, _, Hp, Wp = raw.shape
            reg_logit = raw[:, 1:6]
            reg_pred = torch.sigmoid(reg_logit)  # cx_off, cy_off, w_norm, h_norm, θ_norm
            stride = self.stride
            # Build [B, H', W'] grid of cell origins to convert offsets to pixels.
            ys = torch.arange(Hp, device=raw.device, dtype=raw.dtype).view(1, Hp, 1)
            xs = torch.arange(Wp, device=raw.device, dtype=raw.dtype).view(1, 1, Wp)
            # Pixel coords for pred: (cell_idx + offset) * stride
            pred_cx = (xs + reg_pred[:, 0]) * stride
            pred_cy = (ys + reg_pred[:, 1]) * stride
            pred_w  = reg_pred[:, 2] * self.img_w
            pred_h  = reg_pred[:, 3] * self.img_h
            pred_th = reg_pred[:, 4] * math.pi
            gt = tgt["obb"]  # [B, 5, H', W']  (cx_off, cy_off, w_norm, h_norm, θ_norm)
            gt_cx = (xs + gt[:, 0]) * stride
            gt_cy = (ys + gt[:, 1]) * stride
            gt_w  = gt[:, 2] * self.img_w
            gt_h  = gt[:, 3] * self.img_h
            gt_th = gt[:, 4] * math.pi
            pred_box = torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_th], dim=-1)
            gt_box   = torch.stack([gt_cx,   gt_cy,   gt_w,   gt_h,   gt_th  ], dim=-1)
            # Per-cell ProbIoU loss; mask to GT-positive cells.
            iou_loss = probiou_loss(pred_box, gt_box)  # [B, H', W']
            pos2d = pos.squeeze(1) if pos.dim() == 4 else pos
            l_box = (iou_loss * pos2d).sum() / n_pos
            # Auxiliary direct-angle loss: circular L1 on θ_norm. ProbIoU's
            # angle gradient vanishes on near-square boxes (covariance becomes
            # near-isotropic, ∂B/∂θ ≈ 0). Without an aux term, models tend to
            # park θ at the sigmoid init (θ_norm ≈ 0.5 → θ = π/2) and learn
            # only via indirect-IoU pressure. Direct-L1 fixes that.
            ang_pred_norm = reg_pred[:, 4]
            ang_gt_norm = gt[:, 4]
            ang_diff = (ang_pred_norm - ang_gt_norm).abs()
            ang_diff = torch.minimum(ang_diff, 1.0 - ang_diff)  # wrap [0,1] cycle
            l_angle = (ang_diff * pos2d).sum() / n_pos
            ang_aux_w = float(getattr(self, "angle_aux_weight", 0.5))
            if self.cls_loss == "vfl":
                # VFL needs IoU as cls target — use 1 - probiou loss = ProbIoU (∈ [0, 1]).
                iou_target = (1.0 - iou_loss).detach() * pos2d
                l_hm = varifocal_loss(hm_logit, pos, iou_target.unsqueeze(1), alpha=self.vfl_alpha, gamma=self.vfl_gamma)
            elif self.cls_loss == "soft_hm":
                l_hm = quality_focal_loss(hm_logit, tgt["hm"], beta=self.qfl_beta)
            else:
                l_hm = focal_heatmap_loss(hm_logit, tgt["hm"], self.alpha, self.beta)
            total = self.w_hm * l_hm + self.w_wh * (l_box + ang_aux_w * l_angle)
            out = {"loss": total, "l_hm": l_hm.detach(),
                   "l_cxy": l_box.detach() * 0.0,
                   "l_wh": l_box.detach(),
                   "l_angle": l_angle.detach()}
            if self.count_w > 0 or self.convex_w > 0:
                hm_sig = torch.sigmoid(hm_logit)
            if self.count_w > 0:
                peaks = _peak_suppress(hm_sig, k=self.peak_kernel, eps=self.peak_eps)
                pred_count = peaks.flatten(1).sum(dim=1)
                gt_count = pos.flatten(1).sum(dim=1)
                l_count = (pred_count - gt_count).abs().mean()
                out["loss"] = out["loss"] + self.count_w * l_count
                out["l_count"] = l_count.detach()
            if self.convex_w > 0:
                sig = self._convex_sigma(gt[:, 2:3], gt[:, 3:4])  # gt = tgt["obb"]: w_norm, h_norm
                l_convex = convexity_loss(hm_sig, pos, k=self.convex_r, sigma_cells=sig)
                out["loss"] = out["loss"] + self.convex_w * l_convex
                out["l_convex"] = l_convex.detach()
            return out

        # ltrb mode: raw[:, 1:5] is (l, t, r, b) post-sigmoid.
        # The cxy term is subsumed: each ltrb cell encodes both center offset
        # and box extents jointly (cell-center-to-edge distances).
        if self.wh_loss == "ltrb":
            ltrb_logit = raw[:, 1:5]
            ltrb_pred = torch.sigmoid(ltrb_logit)
            pred_xyxy = _decode_pred_ltrb_xyxy(ltrb_pred, self.img_h, self.img_w, self.stride)
            gt_xyxy = _decode_gt_ltrb_xyxy(tgt["ltrb"], self.img_h, self.img_w, self.stride)
            # DIoU on reconstructed boxes — center+containment penalty without aspect blow-up.
            l_box = (_bbox_iou(pred_xyxy, gt_xyxy, mode="diou") * pos).sum() / n_pos
            if self.cls_loss == "vfl":
                iou_target = _iou_only(pred_xyxy, gt_xyxy) * pos
                l_hm = varifocal_loss(hm_logit, pos, iou_target, alpha=self.vfl_alpha, gamma=self.vfl_gamma)
            elif self.cls_loss == "soft_hm":
                l_hm = quality_focal_loss(hm_logit, tgt["hm"], beta=self.qfl_beta)
            else:
                l_hm = focal_heatmap_loss(hm_logit, tgt["hm"], self.alpha, self.beta)
            l_cxy = l_box.detach() * 0.0  # placeholder so downstream logging keys still exist
            total = self.w_hm * l_hm + self.w_wh * l_box
            out = {"loss": total, "l_hm": l_hm.detach(), "l_cxy": l_cxy, "l_wh": l_box.detach()}
            if self.count_w > 0 or self.convex_w > 0:
                hm_sig = torch.sigmoid(hm_logit)
            if self.count_w > 0:
                peaks = _peak_suppress(hm_sig, k=self.peak_kernel, eps=self.peak_eps)
                pred_count = peaks.flatten(1).sum(dim=1)
                gt_count = pos.flatten(1).sum(dim=1)
                l_count = (pred_count - gt_count).abs().mean()
                out["loss"] = out["loss"] + self.count_w * l_count
                out["l_count"] = l_count.detach()
            if self.convex_w > 0:
                _lt = tgt["ltrb"]  # (l, t, r, b) image-normalized → w = l+r, h = t+b
                sig = self._convex_sigma(_lt[:, 0:1] + _lt[:, 2:3], _lt[:, 1:2] + _lt[:, 3:4])
                l_convex = convexity_loss(hm_sig, pos, k=self.convex_r, sigma_cells=sig)
                out["loss"] = out["loss"] + self.convex_w * l_convex
                out["l_convex"] = l_convex.detach()
            return out

        cxy_logit = raw[:, 1:3]
        wh_logit = raw[:, 3:5]

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
        elif self.cls_loss == "soft_hm":
            l_hm = quality_focal_loss(hm_logit, tgt["hm"], beta=self.qfl_beta)
        else:
            l_hm = focal_heatmap_loss(hm_logit, tgt["hm"], self.alpha, self.beta)

        total = self.w_hm * l_hm + self.w_cxy * l_cxy + self.w_wh * l_wh
        out = {"loss": total, "l_hm": l_hm.detach(), "l_cxy": l_cxy.detach(), "l_wh": l_wh.detach()}

        if self.rep_w > 0 and pred_xyxy is not None:
            l_rep = _repulsion_loss(pred_xyxy, tgt, pos, self.img_h, self.img_w, self.stride)
            out["loss"] = out["loss"] + self.rep_w * l_rep
            out["l_rep"] = l_rep.detach()

        if self.count_w > 0 or self.convex_w > 0:
            hm_sig = torch.sigmoid(hm_logit)

        if self.count_w > 0:
            # Sparse peak map matches deployed graph; sum at peak cells = predicted count.
            # GT count = number of positive cells (one per object's center).
            peaks = _peak_suppress(hm_sig, k=self.peak_kernel, eps=self.peak_eps)
            pred_count = peaks.flatten(1).sum(dim=1)        # [B]
            gt_count = pos.flatten(1).sum(dim=1)            # [B]
            l_count = (pred_count - gt_count).abs().mean()
            out["loss"] = out["loss"] + self.count_w * l_count
            out["l_count"] = l_count.detach()

        if self.convex_w > 0:
            sig = self._convex_sigma(tgt["wh"][:, 0:1], tgt["wh"][:, 1:2])
            l_convex = convexity_loss(hm_sig, pos, k=self.convex_r, sigma_cells=sig)
            out["loss"] = out["loss"] + self.convex_w * l_convex
            out["l_convex"] = l_convex.detach()

        # Optional distance-transform aux head: raw has 6 channels, target has "dist".
        # Target-weighted L1: cells get gradient weight proportional to their target value,
        # so fg cells (target ~1 at object centers) dominate gradient mass and bg cells
        # (target=0, ~82% of pixels) barely contribute. Without this, the bg-pixel majority
        # drives the dist BIAS toward 0 globally, collapsing obj_modulated within ~1 epoch.
        if raw.shape[1] >= 6 and "dist" in tgt and self.dist_w > 0:
            dist_pred = torch.sigmoid(raw[:, 5:6])
            base_w = 0.1                                     # bg cells get 10% of fg's weight
            w = tgt["dist"] * (1.0 - base_w) + base_w
            l_dist = (torch.abs(dist_pred - tgt["dist"]) * w).mean()
            out["loss"] = out["loss"] + self.dist_w * l_dist
            out["l_dist"] = l_dist.detach()

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

    Baseline-subtracted: only the EXCESS overlap beyond what the GTs already share is
    penalized. Without this, two overlapping GTs (e.g. partially-stacked objects) push the
    regression toward shrunken boxes — a perfect prediction would still take a penalty
    just because GT_self overlaps GT_neighbor by construction.
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
        # IoA(pred, neighbor_gt): how much of neighbor's footprint my pred bleeds into
        ix1 = torch.max(my_pred[:, 0], neighbor_gt[:, 0]); iy1 = torch.max(my_pred[:, 1], neighbor_gt[:, 1])
        ix2 = torch.min(my_pred[:, 2], neighbor_gt[:, 2]); iy2 = torch.min(my_pred[:, 3], neighbor_gt[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        nb_area = (neighbor_gt[:, 2] - neighbor_gt[:, 0]).clamp(min=0) * (neighbor_gt[:, 3] - neighbor_gt[:, 1]).clamp(min=0) + 1e-7
        ioa = inter / nb_area
        # Baseline IoA(GT_self, GT_neighbor): how much of neighbor my OWN GT already covers.
        # A perfect prediction (my_pred == gt_box) hits exactly this. Subtract it so the
        # penalty is only for excess bleed.
        bx1 = torch.max(gt_box[:, 0], neighbor_gt[:, 0]); by1 = torch.max(gt_box[:, 1], neighbor_gt[:, 1])
        bx2 = torch.min(gt_box[:, 2], neighbor_gt[:, 2]); by2 = torch.min(gt_box[:, 3], neighbor_gt[:, 3])
        baseline = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0) / nb_area
        ioa_excess = (ioa - baseline).clamp(min=0)
        total = total + ioa_excess.mean()
        count += 1
    return total / max(1, count)
