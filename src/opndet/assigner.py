"""TAL (Task-Aligned Learning) and STAL (Size-aware TAL) assigners.

ROADMAP §1.8 Phase 3. Reference: docs/yolo-paper-implementation-spec.md §6-7.

Design: single-class. Runs on GPU per-batch in train.py. Drives only the
*regression-side* per-cell positive mask. Heatmap (cls) supervision stays
Gaussian per `tgt['hm']` — preserves emergent-segmentation behavior of the
peak head (see docs/engineering-decisions.md "Assigner choice").

Public API: TaskAlignedAssigner(...).__call__(pred_obj, pred_box, gt_boxes_per_image)
returns dict with `pos`, `assigned_gt`, `target_t` matching the dense grid.
"""

from __future__ import annotations

import math

import torch


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """boxes_a [M,4], boxes_b [N,4] xyxy. Returns [M, N] IoU."""
    a_area = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    b_area = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [M, N, 2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [M, N, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a_area[:, None] + b_area[None, :] - inter + eps
    return inter / union


def _decode_pred_boxes_image(pred_box: torch.Tensor, mode: str, img_h: int, img_w: int, stride: int) -> torch.Tensor:
    """Decode dense pred_box [C,H,W] (sigmoid-activated) to per-cell xyxy [HW, 4] in pixels.

    mode='ltrb': pred_box[:4] are (l, t, r, b) image-normalized.
    mode='obb' : same first 4 channels (the AABB enclosing the OBB); angle ignored
                 here — TAL ranks on the AABB IoU, which is a cheap and useful
                 proxy for true rotated IoU (documented design choice; rotated
                 IoU is non-differentiable on GPU without a custom kernel and
                 we only need a ranking signal).
    """
    C, H, W = pred_box.shape
    device = pred_box.device
    dtype = pred_box.dtype
    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    cx_px = (xs + 0.5) * stride
    cy_px = (ys + 0.5) * stride
    if mode in ("ltrb", "obb"):
        l = pred_box[0] * img_w
        t = pred_box[1] * img_h
        r = pred_box[2] * img_w
        b = pred_box[3] * img_h
        x1 = cx_px - l
        y1 = cy_px - t
        x2 = cx_px + r
        y2 = cy_px + b
    else:
        raise ValueError(f"unknown decode mode: {mode}")
    return torch.stack([x1, y1, x2, y2], dim=-1).reshape(H * W, 4)


class TaskAlignedAssigner:
    """Per-image TAL assigner. Vectorized over (N gt, HW cells).

    For each GT, find candidate cells (cell center inside the GT box), score
    them by alignment_metric = pred_obj^alpha * iou(pred_box, gt)^beta, take
    the top-k. On per-cell conflicts (selected by multiple GTs), assign to
    the GT with the highest alignment score.

    STAL: when stal=True, top-k scales per GT by sqrt(image_area / gt_area),
    clipped to [topk_min, topk_max]. Smaller objects get more positive cells.

    target_t is the per-cell alignment score normalized per-GT so that the
    max within each GT's positive set is the max IoU achieved by that GT
    (TOOD-style soft-label normalization). Logged for diagnostics; the
    classification loss path keeps consuming the Gaussian heatmap target,
    not target_t — see CLAUDE.md / engineering-decisions.md.
    """

    def __init__(
        self,
        topk: int = 10,
        alpha: float = 1.0,
        beta: float = 6.0,
        stal: bool = False,
        topk_min: int = 3,
        topk_max: int = 20,
        mode: str = "ltrb",
        img_h: int = 384,
        img_w: int = 512,
        stride: int = 4,
    ):
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stal = bool(stal)
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.mode = str(mode)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.stride = int(stride)
        self._image_area = float(img_h * img_w)

    @torch.no_grad()
    def assign_image(self, pred_obj: torch.Tensor, pred_box: torch.Tensor, gt_boxes: torch.Tensor) -> dict[str, torch.Tensor]:
        """pred_obj [1,H,W] sigmoid scores; pred_box [C,H,W] decoded; gt_boxes [N,4] xyxy."""
        H, W = pred_obj.shape[-2:]
        device = pred_obj.device
        dtype = pred_obj.dtype
        N = int(gt_boxes.shape[0])
        if N == 0:
            return dict(
                pos=torch.zeros(1, H, W, device=device, dtype=dtype),
                assigned_gt=torch.full((H, W), -1, dtype=torch.long, device=device),
                target_t=torch.zeros(1, H, W, device=device, dtype=dtype),
                ltrb=torch.zeros(4, H, W, device=device, dtype=dtype),
            )

        cell_boxes = _decode_pred_boxes_image(pred_box, self.mode, self.img_h, self.img_w, self.stride)  # [HW, 4]
        # cell pixel centers for inside-gt test
        ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W).reshape(-1)
        xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W).reshape(-1)
        cx_px = (xs + 0.5) * self.stride
        cy_px = (ys + 0.5) * self.stride
        # inside_mask [N, HW]
        gx1 = gt_boxes[:, 0:1]; gy1 = gt_boxes[:, 1:2]
        gx2 = gt_boxes[:, 2:3]; gy2 = gt_boxes[:, 3:4]
        inside = (cx_px[None, :] >= gx1) & (cx_px[None, :] <= gx2) & (cy_px[None, :] >= gy1) & (cy_px[None, :] <= gy2)
        # iou [HW, N] -> [N, HW]
        iou = _pairwise_iou(cell_boxes, gt_boxes).t().contiguous()  # [N, HW]
        s = pred_obj.reshape(-1)  # [HW]
        # alignment t = s^alpha * iou^beta, masked by inside
        align = (s.clamp(min=0).pow(self.alpha))[None, :] * iou.clamp(min=0).pow(self.beta)
        align = align * inside.to(dtype)

        # per-GT topk; STAL scales topk by sqrt(image_area / gt_area)
        if self.stal:
            gt_areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
                        * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0))
            scale = torch.sqrt(torch.tensor(self._image_area, device=device, dtype=dtype) / gt_areas)
            ks = (self.topk * scale).round().clamp(min=self.topk_min, max=self.topk_max).long()
        else:
            ks = torch.full((N,), self.topk, device=device, dtype=torch.long)
        kmax = int(ks.max().item())
        kmax = min(kmax, H * W)
        if kmax == 0:
            return dict(
                pos=torch.zeros(1, H, W, device=device, dtype=dtype),
                assigned_gt=torch.full((H, W), -1, dtype=torch.long, device=device),
                target_t=torch.zeros(1, H, W, device=device, dtype=dtype),
                ltrb=torch.zeros(4, H, W, device=device, dtype=dtype),
            )
        topk_vals, topk_idx = align.topk(kmax, dim=1)  # [N, kmax]
        # Mask out positions beyond per-GT k. Also drop anchors with zero alignment
        # (no inside cell, or all-zero predictions early in training — keep them
        # selected anyway when no inside cell has positive score so the loss still
        # gets at least one cell per GT; that's the safety net at startup).
        col = torch.arange(kmax, device=device).view(1, kmax).expand(N, kmax)
        k_mask = col < ks.view(N, 1)  # [N, kmax]
        # candidate_mask [N, HW]
        cand = align.new_zeros((N, H * W))
        # scatter alignment scores at topk positions, masked by k
        scatter_vals = topk_vals * k_mask.to(dtype)
        cand.scatter_(1, topk_idx, scatter_vals)
        # If a GT had ALL-zero alignment (no inside cells with any IoU/score),
        # fall back: pick its center cell so loss has something to supervise.
        all_zero = (cand.sum(dim=1) <= 0)
        if all_zero.any():
            cx_g = ((gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 / self.stride).long().clamp(0, W - 1)
            cy_g = ((gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 / self.stride).long().clamp(0, H - 1)
            for n in torch.nonzero(all_zero, as_tuple=False).flatten().tolist():
                cand[n, cy_g[n].item() * W + cx_g[n].item()] = 1e-6  # tiny but >0

        # Conflict resolution: each cell -> argmax over GTs. Cells with no
        # selecting GT remain unassigned.
        max_vals, max_gt = cand.max(dim=0)  # [HW], [HW]
        pos_flat = (max_vals > 0).to(dtype)
        assigned = torch.where(pos_flat > 0, max_gt, torch.full_like(max_gt, -1))

        # Normalize target_t per-GT: max within each GT's positive set scaled to
        # max IoU achieved by that GT (TOOD soft-label trick). Used only for
        # diagnostics; the cls loss reads tgt['hm'].
        target_t = pos_flat.new_zeros(H * W)
        for n in range(N):
            sel = (assigned == n) & (pos_flat > 0)
            if not sel.any():
                continue
            t_vals = cand[n, sel]
            iou_n = iou[n, sel]
            denom = t_vals.max().clamp(min=1e-9)
            iou_max = iou_n.max()
            target_t[sel] = (t_vals / denom) * iou_max

        # Per-cell ltrb GT (image-normalized distances from cell center to the
        # ASSIGNED GT's edges, clipped to [0,1]). Filled in only at positive
        # cells; zero elsewhere. Replaces the single-positive ltrb that
        # encode_targets_ltrb produced.
        ltrb_gt = pos_flat.new_zeros((4, H * W))
        if pos_flat.any():
            sel = pos_flat > 0
            sel_idx = torch.nonzero(sel, as_tuple=False).flatten()
            sel_cx = cx_px[sel_idx]
            sel_cy = cy_px[sel_idx]
            assigned_n = assigned[sel_idx]
            gt_sel = gt_boxes[assigned_n]  # [k, 4]
            l = (sel_cx - gt_sel[:, 0]).clamp(min=0) / self.img_w
            t = (sel_cy - gt_sel[:, 1]).clamp(min=0) / self.img_h
            r = (gt_sel[:, 2] - sel_cx).clamp(min=0) / self.img_w
            b = (gt_sel[:, 3] - sel_cy).clamp(min=0) / self.img_h
            ltrb_gt[0, sel_idx] = l.clamp(0.0, 1.0)
            ltrb_gt[1, sel_idx] = t.clamp(0.0, 1.0)
            ltrb_gt[2, sel_idx] = r.clamp(0.0, 1.0)
            ltrb_gt[3, sel_idx] = b.clamp(0.0, 1.0)

        return dict(
            pos=pos_flat.view(1, H, W),
            assigned_gt=assigned.view(H, W),
            target_t=target_t.view(1, H, W),
            ltrb=ltrb_gt.view(4, H, W),
        )

    @torch.no_grad()
    def __call__(
        self,
        pred_obj: torch.Tensor,
        pred_box: torch.Tensor,
        gt_boxes_per_image: list,
    ) -> dict[str, torch.Tensor]:
        """Batched. pred_obj [B,1,H,W], pred_box [B,C,H,W], gt_boxes_per_image: list of [N_b, 4] xyxy.

        Returns dict of batched tensors:
          pos          : [B, 1, H, W]
          assigned_gt  : [B, H, W] long (-1 = unassigned)
          target_t     : [B, 1, H, W] float
        """
        B = pred_obj.shape[0]
        device = pred_obj.device
        dtype = pred_obj.dtype
        outs = {"pos": [], "assigned_gt": [], "target_t": [], "ltrb": []}
        for b in range(B):
            gt = gt_boxes_per_image[b]
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt, dtype=dtype, device=device)
            else:
                gt = gt.to(device=device, dtype=dtype)
            if gt.ndim != 2 or gt.shape[-1] < 4:
                gt = gt.reshape(0, 4)
            d = self.assign_image(pred_obj[b], pred_box[b], gt[:, :4])
            for k in outs:
                outs[k].append(d[k])
        return {k: torch.stack(v, dim=0) for k, v in outs.items()}


def build_assigner(cfg: dict, *, mode: str, img_h: int, img_w: int, stride: int) -> TaskAlignedAssigner | None:
    """Factory keyed off the train-yaml's `assigner:` block.

    cfg may be:
      None or 'peak'       → returns None (caller uses existing peak path)
      'tal'                → TAL with default hyperparams
      'stal'               → TAL + size-adaptive top-k
      dict {kind: tal|stal, topk, alpha, beta, topk_min, topk_max}
    """
    if cfg is None:
        return None
    if isinstance(cfg, str):
        kind = cfg.lower()
        if kind == "peak":
            return None
        if kind not in ("tal", "stal"):
            raise ValueError(f"unknown assigner '{cfg}'; expected peak|tal|stal")
        return TaskAlignedAssigner(stal=(kind == "stal"), mode=mode, img_h=img_h, img_w=img_w, stride=stride)
    if isinstance(cfg, dict):
        kind = str(cfg.get("kind", "tal")).lower()
        if kind == "peak":
            return None
        return TaskAlignedAssigner(
            topk=int(cfg.get("topk", 10)),
            alpha=float(cfg.get("alpha", 1.0)),
            beta=float(cfg.get("beta", 6.0)),
            stal=(kind == "stal"),
            topk_min=int(cfg.get("topk_min", 3)),
            topk_max=int(cfg.get("topk_max", 20)),
            mode=mode,
            img_h=img_h,
            img_w=img_w,
            stride=stride,
        )
    raise TypeError(f"unsupported assigner cfg type: {type(cfg).__name__}")
