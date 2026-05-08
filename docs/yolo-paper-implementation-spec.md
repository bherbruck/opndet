# YOLO Paper-Implementation Frankenstein Spec for opndet

Status: design draft, single-class only, Myriad-X-compatible deployment path preserved.

This spec covers the YOLO-family ideas worth porting into opndet's existing harness (encoder + YAML graph builder + train loop + dashboard + export) without taking on the full multi-class, multi-scale, AGPL-encumbered Ultralytics codebase. Each section identifies the source paper, the actual idea, the single-class simplification, and the concrete opndet integration surface (file, function, primitive name).

Source papers consulted:

- YOLOv8: Sohan et al. (2024) *A Review on YOLOv8 and Its Advancements* + Terven et al. (2023) *A Comprehensive Review of YOLO Architectures*. The original YOLOv8 has no canonical paper — Ultralytics released code first; the algorithmic content is described post-hoc in surveys.
- YOLOv10: Wang et al. (2024) *YOLOv10: Real-Time End-to-End Object Detection* (arXiv 2405.14458). Has rigorous algorithm definitions including the consistent matching metric.
- YOLO11: Ultralytics doc + Sapkota et al. survey (arXiv 2510.09653v2). Like v8, no first-party algorithmic paper; the C3k2/C2PSA blocks are described by code and survey prose.
- YOLO26: Sapkota et al. (2025) *YOLO26: Key Architectural Enhancements* (arXiv 2509.25164). Important caveat: this paper is descriptive, not formal — ProgLoss, STAL, and MuSGD are explained conceptually but precise math is reconstructed here from prior art (TAL, focal-loss schedules, Muon).
- TOOD: Feng et al. (2021) *TOOD: Task-aligned One-stage Object Detection* — the canonical TAL paper that all "TAL-style" assigners derive from.
- Muon: Jordan (2024) *Muon: An optimizer for hidden layers in neural networks* — canonical formulation.

> Note on paper-implementation hygiene: this spec describes algorithms from the public literature only. The Ultralytics codebase is AGPL-3.0; do not read it while implementing. All design decisions below trace to the cited papers or are explicit reconstruction choices marked as such.

---

## 0. What's actually worth porting

Decision matrix — each row is a feature, the score reflects expected impact on object-detection workloads (single-class, dense, occluded, scale-varied, Myriad-X-deployed):

| Feature | Source | Impact for objects | Cost | Verdict |
|---|---|---|---|---|
| Decoupled head (cls / reg branches) | v8 | medium | YAML-only, ~30 min | **port** |
| Anchor-free dense prediction | v8 | already have | — | already done |
| C2f block | v8 | small (CSPBlock is close) | ~50 lines | optional |
| C3k2 block | v11 | small | ~50 lines | optional |
| C2PSA (partial self-attention) | v11 | medium-high on small objects | ~100 lines | port for non-Myriad targets only |
| SPPF (spatial pyramid pooling fast) | v8/v11 | medium for scale | ~30 lines | **port** |
| PAFPN neck (top-down + bottom-up) | v8 | high for scale | YAML-only | **port** |
| TAL assigner | TOOD/v8 | high for occlusion | ~150 lines | **port** |
| DFL (distribution focal loss) | v8/v11 | medium for sub-pixel | ~80 lines | optional, v26 argues against |
| Dual-assignment NMS-free | v10 | already have via PeakSuppress | ~250 lines if added | **skip** for Myriad path |
| ltrb regression instead of wh | v8/v10 | medium | ~100 lines | **port** |
| ProgLoss (progressive loss balancing) | v26 | medium | ~50 lines, plugs into existing curriculum | **port** |
| STAL (small-target-aware assignment) | v26 | high if small-object-bound | ~30 lines on top of TAL | **port** |
| MuSGD optimizer | v26 / Muon | medium | ~150 lines | port-on-eval (real win is unclear for vision) |
| Mosaic-9, copy-paste augmentation | v5+ | high for diversity | ~200 lines | already partial in opndet |

The verdict column reflects "ship-this-first" priority. The order to actually implement, justified in §11, is: SPPF + PAFPN (YAML) → ltrb regression → TAL → STAL → ProgLoss → decoupled head → MuSGD → optional DFL/dual-assign.

---

## 1. Decoupled head (YOLOv8)

**Source idea.** Pre-v8 YOLO heads used a single 1×1 conv producing `[obj, cls, box]` from one shared stem. v8 separates the head into two parallel branches off the neck: a *classification* branch (two ConvBnAct + 1×1 cls conv) and a *regression* branch (two ConvBnAct + 1×1 box conv). The motivation is gradient interference: cls and reg objectives have different optimal feature distributions, and sharing the last conv penalizes both.

**Single-class simplification.** With C=1, the cls branch outputs 1 channel (just objectness). This is closer to opndet's current head than v8's, because opndet already has `obj` as one channel and `cxy/wh` as four. The decoupling here is *physical*: the obj branch and the box branch get their own preceding ConvBnAct stack.

**opndet integration.** YAML-only, no new primitives required. Replace the current head section in `opndet-bbox-x.yaml` (lines 29-38) with two parallel branches:

```yaml
# cls branch
- {name: cls1, from: nout, module: ConvBnAct, args: {in_ch: 256, out_ch: 256, k: 3}}
- {name: cls2, from: cls1, module: ConvBnAct, args: {in_ch: 256, out_ch: 256, k: 3}}
- {name: cls,  from: cls2, module: Conv, args: {in_ch: 256, out_ch: 1, k: 1, bias_init: -2.19}}

# reg branch
- {name: reg1, from: nout, module: ConvBnAct, args: {in_ch: 256, out_ch: 256, k: 3}}
- {name: reg2, from: reg1, module: ConvBnAct, args: {in_ch: 256, out_ch: 256, k: 3}}
- {name: reg,  from: reg2, module: Conv, args: {in_ch: 256, out_ch: 4, k: 1}}

# combine
- {name: raw, from: [cls, reg], module: Concat, args: {dim: 1}}
```

The training loop already slices `raw` into channels — change is `raw[:, 0:1]` for obj and `raw[:, 1:5]` for the 4 box channels (was `raw[:, 1:3]` cxy + `raw[:, 3:5]` wh).

**Cost.** ~30 minutes. Add ~6 ConvBnAct of params relative to current single-trunk head (~600K params at neck_ch=256). Negligible on bbox-x size; significant on bbox-f.

---

## 2. SPPF — Spatial Pyramid Pooling Fast (YOLOv8)

**Source idea.** SPPF concatenates the input with three sequential MaxPool-5 outputs, giving the layer access to receptive fields {1, 5, 9, 13} on a single feature map at one cost. SPP (the original from YOLOv3) used three parallel pools with different kernel sizes; SPPF achieves the same receptive-field coverage with sequential same-size pools, ~3× faster.

**Why for objects.** SPPF widens the effective receptive field at the deepest backbone stage without adding strided convs. For mixed-scale objects in production photos, this directly helps the network see the "context" of a 200px object even when the local feature map is computed at coarse resolution. SPP/SPPF is in v8/v11/v26 and missing from opndet's current backbone.

**Definition.**
```
SPPF(x):
    y0 = ConvBnAct(x, k=1)            # halve channels
    y1 = MaxPool(y0, k=5, s=1, p=2)
    y2 = MaxPool(y1, k=5, s=1, p=2)
    y3 = MaxPool(y2, k=5, s=1, p=2)
    out = Concat([y0, y1, y2, y3], dim=1)   # 4× channels of y0
    return ConvBnAct(out, k=1)        # back to original channel count
```

**opndet integration.** New primitive in `primitives.py`:

```python
@register("SPPF")
class SPPF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        c_ = in_ch // 2
        self.cv1 = _conv_bn_act(in_ch, c_, k=1)
        self.cv2 = _conv_bn_act(c_ * 4, out_ch, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], dim=1))
```

Inserted in YAML between the deepest backbone stage and the neck:
```yaml
- {name: p4_sppf, from: p4, module: SPPF, args: {in_ch: 640, out_ch: 640}}
- {name: lat4,    from: p4_sppf, module: ConvBnAct, args: {in_ch: 640, out_ch: 256, k: 1}}
```

**Myriad note.** MaxPool-5 with stride 1 is opset-13 and Myriad-safe. Verified against opndet's existing PeakSuppress which already uses this op pattern.

---

## 3. PAFPN neck (YOLOv8 / standard since YOLOv4)

**Source idea.** Path Aggregation Network (PAN, Liu et al. 2018) adds a bottom-up path *on top of* an FPN's top-down path. FPN propagates semantic information from deep features to shallow features. PAN adds the reverse: shallow spatial information back up to the deep features. The result is a fully-connected feature pyramid where each level has access to information from all other levels.

**Why for objects.** opndet's current neck (`Neck` in model.py, or the `lat*` + `fuse*` chain in YAML) is FPN-lite — top-down only, output at stride 4. The deepest features (p4, large-receptive-field) can flow down to stride 4 via lat4 → up → add → fuse. But the inverse is missing: a small object detected with high confidence at p2 features doesn't propagate upward to influence p4's representation. PAFPN closes this loop. For multi-scale objects, this is the single cheapest scale-handling improvement.

**Definition (single-output stride-4 variant).** Standard PAFPN outputs P3, P4, P5 multi-scale; we keep opndet's single-output-at-stride-4 deployment shape, so the bottom-up path is internal — its purpose is to enrich the shared neck features that ultimately get sliced at stride 4. Topology:

```
Top-down (existing):
    lat4 = 1×1(p4)
    td3  = 3×3(up(lat4) + 1×1(p3))      # at p3 stride
    td2  = 3×3(up(td3) + 1×1(p2))       # at p2 stride
    td1  = 3×3(up(td2) + 1×1(p1))       # at p1 stride (= stride 4, output)

Bottom-up (new):
    bu1  = td1                           # alias
    bu2  = 3×3(down(bu1) + td2)          # back up to p2 stride
    bu3  = 3×3(down(bu2) + td3)          # back up to p3 stride
    bu4  = 3×3(down(bu3) + lat4)         # back up to p4 stride

Output (back down to stride 4 by reusing top-down with bottom-up features):
    out  = 3×3(up3(bu4 + lat4) + bu3)    # mix bu4 with td3-equivalent
    ...
```

For a single-output stride-4 head, the simplest correct form is:
```
PAFPN-S4(p1..p4):
    # top-down
    t4 = 1×1(p4)
    t3 = fuse(up(t4) + 1×1(p3))
    t2 = fuse(up(t3) + 1×1(p2))
    t1 = fuse(up(t2) + 1×1(p1))
    # bottom-up
    b2 = fuse(down(t1) + t2)
    b3 = fuse(down(b2) + t3)
    b4 = fuse(down(b3) + t4)
    # final output at stride 4 (combine top-down t1 with downsampled bottom-up info)
    out = fuse(t1 + up3(b4))
    return out
```

where `up3` upsamples by 8× (from p4 stride to p1 stride) via three nearest-2× ops.

**opndet integration.** YAML-only. New `Downsample2x` primitive (one ConvBnAct with stride=2). Then the neck section of the YAML grows from 9 layers to ~20. Recommended: write `opndet-bbox-x-pafpn.yaml` as a separate config so the FPN-lite version stays as a comparison baseline.

**Myriad cost.** All ops are conv/maxpool/upsample/add. Adds roughly 0.6M params and 1.5 GFLOPs at neck_ch=256 — measurable on Myriad but not prohibitive.

---

## 4. C3k2 and C2PSA (YOLO11)

**C3k2.** A YOLOv11 backbone block. C3k2 is a lighter variant of v8's C2f. It splits the input into two branches by 1×1 conv, runs `n` bottleneck blocks on one branch (where the bottleneck uses kernel size `k` — the "k2" in the name refers to a default k=3 in v11n/s and k=5 in v11m/l/x), concatenates with the skip branch, and projects back. Mathematically very close to opndet's existing CSPBlock; the differences are: (a) parametric kernel size, (b) split ratio fixed at 0.5 instead of CSPBlock's configurable `e`, (c) the bottleneck has shortcut connections.

**Verdict for opndet:** marginal value. CSPBlock in opndet's blocks.py is structurally equivalent. If you want kernel parametrization, add `k` as an arg to the existing CSPBlock and call it C3k2 in YAML for paper compatibility. Don't write a new primitive.

**C2PSA.** "C2 Position-wise Spatial Attention." A C2-style block where one of the two branches passes through a position-attention module. Defined in v11 docs as: split → branch_a stays as-is, branch_b runs a self-attention over spatial positions (Q/K/V from 1×1 convs, attention along H×W), concat, project. Adds ~5-10% mAP on small-object COCO subsets but is non-trivially expensive on Myriad VPU which lacks efficient softmax-on-large-spatial-tensors.

**Verdict for opndet:** port for non-Myriad target only. For Jetson/T4/CPU deployments where the model isn't bound by Myriad's opset, add C2PSA as a YAML block at the deepest backbone stage (p4) and the deepest neck stage. For bbox-f Myriad path, skip entirely.

**Definition (paper-implementation).**
```python
@register("PSA")
class PositionSelfAttention(nn.Module):
    """Single-head spatial self-attention. Q,K,V from 1x1 convs.
    Attention over flattened H*W. Output projected back to original shape."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = (ch // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.h, C // self.h, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # B, h, c/h, HW
        attn = (q.transpose(-2, -1) @ k) * self.scale  # B, h, HW, HW
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)   # residual

@register("C2PSA")
class C2PSA(nn.Module):
    """v11-style C2 with PSA on one branch."""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, num_heads: int = 4):
        super().__init__()
        c_ = in_ch // 2
        self.cv1 = _conv_bn_act(in_ch, c_ * 2, k=1)
        self.attn = nn.Sequential(*[PositionSelfAttention(c_, num_heads) for _ in range(n)])
        self.cv2 = _conv_bn_act(c_ * 2, out_ch, k=1)

    def forward(self, x):
        y = self.cv1(x)
        a, b = y.chunk(2, dim=1)
        return self.cv2(torch.cat([a, self.attn(b)], dim=1))
```

---

## 5. ltrb regression (YOLOv8 + descended down to v10/v26)

**Source idea.** Instead of regressing `(cx_offset, cy_offset, w_norm, h_norm)` per-cell (anchor-free wh), regress `(left, top, right, bottom)` distances from the cell's own center to the four edges of the box it belongs to. This is the FCOS-style parameterization that v8/v10/v11/v26 all use. The key insight is that ltrb is *cell-local*: each positive cell predicts distances *from itself*, no global normalization. A cell inside an object can predict an `l` distance that points off-frame, and the decoded box has `x1 = cx_cell - l < 0`, which is geometrically truthful.

**Why for objects.** Less critical than I'd thought before learning the bottleneck is scale not clipping, but still worth doing because: (a) scale-invariant. An object's regression target magnitude doesn't depend on image size, only on stride. (b) It composes with TAL — TAL's positives are cells *inside* the GT box, all of which can predict valid ltrb. With current opndet wh, only the *center* cell has a valid target; non-center cells would predict wh equal to the box size which is wrong-by-design for those positions.

**Definition.**

For a cell at output-grid position `(ix, iy)` with stride `s`, owning a GT box `(x1, y1, x2, y2)`:
```
cx_cell = (ix + 0.5) * s
cy_cell = (iy + 0.5) * s
target_l = cx_cell - x1
target_t = cy_cell - y1
target_r = x2 - cx_cell
target_b = y2 - cy_cell
```

Predicted ltrb is post-`exp` (or `softplus`) so distances are always non-negative:
```
pred_l = exp(raw_l) * s     # scale-coupled exp; or raw_l * s after sigmoid * S
pred_t = exp(raw_t) * s
pred_r = exp(raw_r) * s
pred_b = exp(raw_b) * s
```

Decoded box:
```
x1_pred = cx_cell - pred_l
y1_pred = cy_cell - pred_t
x2_pred = cx_cell + pred_r
y2_pred = cy_cell + pred_b
```

**opndet integration.**

`encode.py` — new function `encode_ltrb_targets`:
- Returns `{ltrb: [4, H', W'], pos: [1, H', W'], gt_box_idx: [1, H', W']}` where `gt_box_idx` is the index of the GT each positive cell belongs to (needed for TAL later).
- Positives are cells whose *center* falls inside *any* GT box. This already gives multi-positive-per-GT supervision, separate from TAL.
- Conflict resolution when a cell falls inside multiple GTs: assign to the GT with smallest area (Yolox/FCOS heuristic).

`primitives.py` — new primitive:
```python
@register("LTRBDecode")
class LTRBDecode(nn.Module):
    """Convert raw ltrb logits to absolute xyxy boxes per cell.
    Inputs:  raw [B, 4, H, W]  (the 4 channels are l, t, r, b)
    Outputs: xyxy [B, 4, H, W] in pixel coords.
    Uses exp activation; cell centers are pre-baked grid offsets.
    """
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, raw):
        B, _, H, W = raw.shape
        device = raw.device
        ys = torch.arange(H, device=device, dtype=raw.dtype) + 0.5
        xs = torch.arange(W, device=device, dtype=raw.dtype) + 0.5
        cy = (ys.view(1, 1, H, 1) * self.stride).expand(B, 1, H, W)
        cx = (xs.view(1, 1, 1, W) * self.stride).expand(B, 1, H, W)
        l, t, r, b = raw.exp().split(1, dim=1)
        l = l * self.stride; t = t * self.stride
        r = r * self.stride; b = b * self.stride
        return torch.cat([cx - l, cy - t, cx + r, cy + b], dim=1)
```

`loss.py` — new class `OpndetLTRBLoss` (sibling to `OpndetBboxLoss`):
- Same VFL/focal cls loss as before, but now `pos_mask` is dense (multi-cell-per-GT).
- Box regression: CIoU directly on decoded xyxy and GT xyxy (matched by `gt_box_idx`).
- Drop the `cxy` term entirely — ltrb subsumes it.

This is a sibling, not replacement. `bbox-x.yaml` keeps `OpndetBboxLoss`. New `bbox-x-ltrb.yaml` uses the new loss. Train both, compare.

---

## 6. TAL — Task-Aligned Assigner (TOOD, YOLOv8+)

**Source.** Feng et al. 2021, "TOOD: Task-aligned One-stage Object Detection."

**Algorithm.** For each GT, assign the top-k cells (k=10 in v8, k=13 in some v10 configs) by *alignment metric*. The alignment metric is:

```
t = s^α * u^β
```

where `s` is the predicted classification score for the GT's class at that cell, `u` is the IoU between the cell's predicted box and the GT box, and α, β are hyperparameters (TOOD: α=1, β=6; v8: α=0.5, β=6).

Single-class simplification: `s` is just the predicted obj score. No class indexing needed.

The top-k cells per GT become positives. Cells not in any GT's top-k are negatives. Conflict (same cell selected by multiple GTs): assign to the GT with the highest alignment score for that cell.

The metric `t` doubles as a *target weight* for the classification loss: instead of learning `obj=1` at positive cells, the cell learns `obj=t_normalized` where `t_normalized = t / max_t_for_this_gt`. This is the "soft label" trick — high-alignment cells get pushed harder than low-alignment cells.

**Why for objects.** This is the single biggest expected improvement on dense/occluded data. Currently opndet has *one* positive cell per object. With TAL k=10, ten cells around each object get gradient. On a pile of touching objects, this means 10× the gradient signal per object, distributed spatially. The alignment metric ensures the *most-aligned* cells (high cls × high IoU) get the strongest weight, so the model converges to peak-at-center while still learning from the neighborhood.

**Definition (single-class, vectorized).**

Inputs at training time, per image:
- `cls_pred` : [H, W] — predicted obj score (sigmoid'd)
- `box_pred` : [4, H, W] — decoded xyxy in pixel coords (from LTRBDecode)
- `gt_boxes` : [N, 4] — ground-truth xyxy
- `topk` : int — typically 10
- `alpha`, `beta` : floats — typically 0.5, 6.0

Step 1 — candidate filter. For each GT, only cells whose center falls inside the GT box are candidates. Build `inside_mask[N, H, W]`.

Step 2 — alignment metric. For each (GT n, cell ij) with inside_mask=1:
```
iou_nij = IoU(box_pred[:, i, j], gt_boxes[n])
t_nij   = cls_pred[i, j]^α * iou_nij^β
```

Step 3 — top-k per GT. For each GT n, find top-k cells by `t_nij` (with `t_nij = -inf` for inside_mask=0). These are the positive candidate set for GT n.

Step 4 — conflict resolution. Each cell can only belong to one GT. If cell (i,j) is selected by multiple GTs, assign it to the GT with maximum `t_nij` among them.

Step 5 — output. Build:
- `pos_mask[H, W]` : 1 at any selected cell
- `assigned_gt[H, W]` : index of the GT this cell belongs to (or -1)
- `target_t[H, W]` : the alignment score `t_nij`, normalized per-GT so max within each GT's positive set is 1.0 (this is the soft cls target)

**opndet integration.** New module `assign.py`:

```python
import torch

class TaskAlignedAssigner:
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        self.topk = topk; self.alpha = alpha; self.beta = beta

    def __call__(self, cls_pred, box_pred, gt_boxes):
        """All inputs single-image. cls_pred [H,W], box_pred [4,H,W], gt_boxes [N,4].
        Returns dict with pos_mask [H,W], assigned_gt [H,W], target_t [H,W]."""
        H, W = cls_pred.shape
        N = gt_boxes.shape[0]
        device = cls_pred.device

        if N == 0:
            return dict(
                pos_mask=torch.zeros(H, W, device=device),
                assigned_gt=torch.full((H, W), -1, dtype=torch.long, device=device),
                target_t=torch.zeros(H, W, device=device),
            )

        # cell centers (in feature-map coords; box_pred is already in pixel coords so
        # we use the pixel-coord cell centers from the decoder's grid)
        # We need cell pixel coords; pass them in or recompute. Recompute here from stride:
        # ... assume the caller provides cx_grid, cy_grid both [H,W] in pixel coords.

        # inside_mask: [N, H, W]
        # for each gt, cell is inside if x1 <= cx <= x2 and y1 <= cy <= y2
        # ... vectorized in practice with broadcasting

        # iou: [N, H, W] — IoU of each cell's predicted box against each GT
        # ... vectorized pairwise IoU

        # alignment: [N, H, W]
        # t = cls_pred[None]**alpha * iou**beta * inside_mask

        # top-k per GT
        # flat = t.view(N, H*W)
        # _, topk_idx = flat.topk(self.topk, dim=1)
        # candidate_mask: [N, H, W] = 1 at top-k positions

        # conflict: argmax over N for cells selected by multiple GTs
        # ...

        # normalize t per-GT so max=1 within each GT's positive set
        # ...

        return ...
```

(Pseudocode shown — full vectorized implementation is ~150 lines and follows the standard TAL implementation pattern published in TOOD's official repo and replicated in mmdetection's task-aligned-assigner.py, both of which are Apache-2.0 and safe to read as paper-implementation references.)

**Curriculum hook.** TAL's `topk` is a natural curriculum target — see ProgLoss in §8.

---

## 7. STAL — Small-Target-Aware Label Assignment (YOLO26)

**Source.** Sapkota et al. 2025, *YOLO26* (arXiv 2509.25164), §2.3. The paper describes STAL conceptually: "explicitly prioritizes label assignments for small objects, which are particularly difficult to detect due to their limited pixel representation and susceptibility to occlusion." Specific math is not given in the paper.

**Reconstruction (paper-implementation).** STAL is a TAL variant with size-aware top-k. Two reasonable reconstructions exist:

**Reconstruction A (size-scaled topk).** Per-GT `topk` scales with object area:
```
topk_n = max(1, round(topk_base * area_factor(gt_n)))
area_factor(gt) = clamp(sqrt(gt_area / median_area), min=0.5, max=2.0)
```
Smaller objects get more positives relative to the spatial cells available; this isn't quite right — small objects have *fewer* inside-cells available than large ones. Reverse the bias: small objects should consume *more* of the available inside-cells, so:
```
topk_n = min(inside_cells_count(gt_n), max(topk_base // 2, round(topk_base * (median_area / max(gt_area, 1)))))
```

**Reconstruction B (size-weighted alignment).** Modify the alignment metric to upweight small objects:
```
size_weight(gt_n) = 1 / sqrt(gt_area / dataset_median_area)
t_nij = (s^α * u^β) * size_weight(gt_n)
```
This raises small-GT alignment scores in conflict resolution, so when a cell is contested, the smaller object wins more often.

**Recommendation.** Implement Reconstruction A (size-scaled topk). It's closer to what the YOLO26 paper's prose implies ("prioritizes label assignments"), and doesn't distort the alignment metric's role as a soft cls target. Add as an `enable_stal: bool` flag on the TAL assigner; default off. Empirical comparison against TAL-only is the experiment that decides.

**Object-specific note.** If your bottleneck is large objects (200px) not small objects (20px), STAL won't help and may *hurt* by under-supervising the easy large-object cells. Run the per-size mAP diagnostic from §11 first.

---

## 8. ProgLoss — Progressive Loss Balancing (YOLO26)

**Source.** Sapkota et al. 2025, §2.3. Described as "dynamically adjusts the weighting of different loss components during training, ensuring that the model does not overfit to dominant object categories while underperforming on rare or small classes." No formula given.

**Reconstruction (paper-implementation).** ProgLoss is a curriculum schedule on the loss weights. The opndet train loop already has a curriculum mechanism (added on day 8 of the repo, see `_apply_curriculum`). ProgLoss fits as three curriculum entries plus an optional automatic-balancing component.

The standard reading from the paper's prose ("progressive balancing") is a schedule like:

```
Epoch 0           : w_cls = 1.0, w_box = 7.5, w_dfl = 1.5    # box-heavy early
Epoch 0.3*total   : w_cls = 1.0, w_box = 7.5, w_dfl = 1.5    # hold
Epoch 0.7*total   : w_cls = 1.5, w_box = 5.0, w_dfl = 1.0    # rebalance toward cls
Epoch total       : w_cls = 2.0, w_box = 3.0, w_dfl = 0.5    # cls-heavy late
```

The intuition: early in training, the model needs to learn *where* objects are (box loss dominates). Later, once box predictions are reasonable, learning *what they are* / *how confident* matters more, and box loss starts to overfit easy positives. The progressive shift mirrors the TAL alignment effect — early predictions have low IoU so VFL targets are low across the board, and pushing cls hard when boxes are still wrong is wasted gradient.

**opndet integration.** Add three curriculum entries to train.yaml:
```yaml
curriculum:
  - name: w_cls_progloss
    schedule: linear
    epochs: [0, total]
    values: [1.0, 2.0]
  - name: w_box_progloss
    schedule: linear
    epochs: [0, total]
    values: [7.5, 3.0]
  - name: w_dfl_progloss   # only if DFL enabled
    schedule: linear
    epochs: [0, total]
    values: [1.5, 0.5]
```

These hook into the existing `_apply_curriculum` mechanism (~5 lines of new mapping in train.py to alias these to `loss_fn.w_hm`, `loss_fn.w_wh` etc).

**Optional automatic-balance variant.** A more sophisticated reading of the paper would have ProgLoss watch the *gradient norms* of each loss term and rebalance to keep them roughly equal. This is GradNorm (Chen et al. 2018), which the paper does not cite and probably doesn't intend. Skip.

---

## 9. MuSGD optimizer (YOLO26)

**Source.** Sapkota et al. 2025, §2.4. Described as "combines the strengths of Stochastic Gradient Descent (SGD) with the recently proposed Muon optimizer, a technique inspired by optimization strategies used in large language model (LLM) training." Cites Muon's role in Kimi K2.

**Muon recap (Jordan 2024).** Muon is for matrix-valued parameters (i.e. 2D weight matrices, which in conv-nets means flattened conv weights). The update rule:
1. Compute gradient G with momentum: `M_t = β·M_{t-1} + G_t`.
2. Orthogonalize the momentum matrix: `O_t = NewtonSchulz(M_t)`. Newton-Schulz iterates a quintic polynomial 5 times to approximate the polar factor (orthogonal matrix closest to M).
3. Update: `W -= η·O_t`.

The orthogonalization gives *uniform* singular values — every direction gets the same step size, which is desirable when training matrices that have been initialized with unit-spectrum (which conv weights effectively are after batch-norm normalization).

Muon does NOT replace AdamW for everything. The standard recipe ("Muon hybrid"): Muon for matrix params (conv weight, linear weight), AdamW for everything else (biases, BN affine, embeddings, anything 1-D).

**MuSGD reconstruction.** The paper says "combines SGD with Muon." The plausible reading: Muon is used for matrix params with SGD-style momentum (no AdamW second moment), and SGD-with-momentum for non-matrix params. This is essentially Muon as Jordan defines it but explicitly using SGD-momentum for the non-matrix branch instead of AdamW. The "Mu" is matrix-Muon, the "SGD" is the non-matrix branch.

```
for each parameter group:
    if param.ndim >= 2 and is_conv_or_linear_weight:
        use Muon update (Newton-Schulz on momentum)
    else:
        use SGD-with-momentum
```

**opndet integration.** New module `optim/musgd.py`. Define a `MuSGD` torch.optim.Optimizer subclass that internally splits params at construction time:

```python
class MuSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']
                m.mul_(group['momentum']).add_(p.grad)
                if p.ndim >= 2:
                    # Muon path: orthogonalize momentum
                    update = self._newton_schulz(m, steps=group['ns_steps'])
                else:
                    # SGD-momentum path
                    update = m
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.add_(update, alpha=-group['lr'])

    @staticmethod
    def _newton_schulz(G, steps=5, eps=1e-7):
        """Quintic Newton-Schulz iteration for matrix orthogonalization.
        Reshapes 4D conv weight to 2D for the iteration, then back."""
        original_shape = G.shape
        if G.ndim == 4:
            G2d = G.reshape(G.shape[0], -1)
        else:
            G2d = G
        # normalize to spectral norm <= 1 (approx via Frobenius)
        X = G2d / (G2d.norm() + eps)
        if X.shape[0] > X.shape[1]:
            X = X.T
        # quintic iteration: X <- a*X + b*(X X^T) X + c*(X X^T)^2 X
        a, b, c = 3.4445, -4.7750, 2.0315  # Jordan's coefficients
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if G2d.shape[0] > G2d.shape[1]:
            X = X.T
        return X.reshape(original_shape)
```

The Newton-Schulz coefficients (a, b, c) and quintic structure are from Jordan's blog post — that's the canonical formulation, public, citable, not Ultralytics-derived.

**Honest caveat.** Muon's empirical wins are most clearly demonstrated in transformer training. For convnet vision tasks, the literature is thinner. The YOLO26 paper claims faster convergence; whether this translates to your target dataset is genuinely unknown. Keep this as a *late-stage* port — only worth implementing once everything else is bedded in, and only if AdamW-with-cosine-schedule is showing convergence pain.

**Cost.** ~150 lines for the optimizer plus 5 lines in train.py to switch on a config flag. The Newton-Schulz iteration adds roughly 1-3% per-step overhead at the model sizes opndet operates at (small — Muon's overhead is dominated by matrix size, and conv weights flattened to 2D are typically a few thousand × a few thousand at most).

---

## 10. Dual-assignment NMS-free (YOLOv10) — for non-Myriad targets

**Source.** Wang et al. 2024, arXiv 2405.14458. Has explicit math.

**Algorithm.** Two parallel heads share backbone+neck:
- **One-to-many (o2m) head**: trained with TAL k=10. Provides rich gradient signal. Used during training only.
- **One-to-one (o2o) head**: trained with TAL k=1, i.e., each GT picks exactly one positive cell. Used at inference. No NMS needed because there's exactly one positive per GT by construction.

The two heads share a *consistent matching metric* — the alignment metric `t = s^α * u^β` is the same for both, ensuring they don't push the shared backbone toward conflicting objectives.

**Why for objects.** opndet's PeakSuppress already achieves NMS-free output. The question is whether v10's o2o branch is better than PeakSuppress. The arguments:

- PeakSuppress is a *postprocessing* step on a *dense* heatmap. It works because the cls heatmap is trained to peak at object centers. If two objects are very close (touching), their heatmap peaks may merge, and PeakSuppress collapses them to one detection. This is a *failure mode* on dense data.
- v10 o2o is *built into the loss*. It explicitly trains exactly one positive per GT, with the constraint enforced through the assignment. Two touching objects will have two distinct positive cells (each GT gets exactly one positive, the assignments are disjoint by construction). At inference, the o2o head produces two predictions, no merging.

**Verdict for opndet.** For Myriad-X-deployed bbox-f, keep PeakSuppress and skip dual-assignment. For higher-end deployments (Jetson, T4, server) where Ultralytics-style multi-output graphs are fine, dual-assign is genuinely better for dense scenes. Build it as a separate head family `head_family: v10_dual` in the YAML, sharing backbone+neck, doubling head params.

**Skip the implementation in this spec** — it's a separate ~250-line port (TAL with k=1 variant + dual-output train loop), and the bbox-f deployment path doesn't need it. Reference: arXiv 2405.14458 §3.1, equation for the consistent matching metric is `m(α,β) = s^α · IoU(b̂, b)^β · 1[anchor ∈ instance]`.

---

## 11. Ordering of experiments (the actually-important section)

The features above are independently valuable but have very different costs and very different expected impacts on object data. The right order is dictated by *cheapest diagnosis first*:

**Step 0 (1 hour). Add per-size mAP to eval.py.** Bin GTs into small (area < 32²), medium (32²-96²), large (>96²). Report per-bin mAP. Run on current bbox-x checkpoint. This tells you what kind of scale problem you have:
- If small-bin mAP is the worst → small-object problem, STAL/SPPF are worth it.
- If large-bin mAP is the worst → large-object receptive-field problem, PAFPN + SPPF are critical, STAL won't help.
- If all bins are roughly equal but all mediocre → backbone capacity, augmentation, or assignment is the issue, scale isn't the bottleneck.

**Step 1 (30 min). PAFPN neck + SPPF.** YAML-only changes. New config `bbox-x-pafpn.yaml`. Train it. Compare per-size mAP to baseline. This is the cheapest scale fix.

**Step 2 (half day). ltrb regression + new encoder + new loss.** Together because they're inseparable. New config `bbox-x-ltrb.yaml` reusing the PAFPN neck from step 1. Train. Compare. ltrb on its own may not move numbers much because positives are still center-only — that's fine, the next step needs ltrb to be in place.

**Step 3 (half day). TAL assigner.** Replaces center-only positive assignment with TAL's dense top-k. New config `bbox-x-tal.yaml`. Train. Compare. **This is the step most likely to close the gap to your v11 number** because TAL is what gives v8/v11/v26 their dense-scene competence.

**Step 4 (30 min). STAL switch.** Add `enable_stal: true` to the TAL assigner. Train. Compare. Only worth the run if step 0 said small objects are a problem.

**Step 5 (30 min). ProgLoss curriculum.** Add the three curriculum entries. Train. Compare. Modest expected lift, easy to validate.

**Step 6 (30 min). Decoupled head.** New YAML config `bbox-x-tal-decoupled.yaml`. Train. Compare. Modest expected lift.

**Step 7 (full day). MuSGD.** Implement, swap optimizer in config. Train. Compare. May lift, may not. Worth knowing.

**Steps 8+: optional.** C2PSA for non-Myriad targets. Dual-assignment for dense scenes if PeakSuppress is failing.

Total elapsed: 2-3 days of you+Claude-Code time, plus GPU time for ~7 training runs. If steps 1-3 already match or beat your v11 number on object data, stop there and ship.

---

## 12. What this spec deliberately doesn't include

- **Multi-class anything.** All assigners, losses, encoders here assume C=1. Adding multi-class is mostly straightforward (cls dim in the channel layout, class-conditioned alignment metric) but it's not what you want and it adds tensor-shape complexity that pays no dividend for objects.
- **Multi-scale heads.** Single-output stride-4 head is preserved throughout. Multi-scale heads (P3/P4/P5 outputs) are fundamentally what modern YOLO does and provide the cleanest scale-handling, but they break opndet's "one tensor, no postprocessing" deployment story. The PAFPN neck change in §3 captures most of the multi-scale benefit while preserving single-output deployment. If after all of §11 the scale problem isn't solved, multi-scale heads are the last resort — that's a separate spec.
- **DFL (Distribution Focal Loss).** v8/v11 have it, v26 dropped it. The v26 paper's argument is that with good assignment (TAL+STAL), the bin-distribution sub-pixel benefit isn't worth the export complexity. Skip unless §11 step 7 reveals a sub-pixel-precision bottleneck.
- **Mosaic-9 / copy-paste augmentation.** opndet already has mosaic and cutout. Mosaic-9 (3×3 instead of 2×2) and copy-paste are real wins but they're augmentation work, not architecture, and they belong in a separate augment.py spec.
- **Training-from-pretrained-weights.** Pulling Ultralytics' pretrained backbones into opndet would require state-dict-key mapping, and is a paper-implementation legality question. Skip — train from scratch is fine for single-class, dataset-specific training.
- **OBB (oriented bounding boxes).** Already on the opndet roadmap. Out of scope here.

---

## 13. References

- Wang, A. et al. (2024). YOLOv10: Real-Time End-to-End Object Detection. arXiv:2405.14458.
- Sapkota, R. et al. (2025). YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection. arXiv:2509.25164.
- Sapkota, R. & Karkee, M. (2025). Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8, and YOLOv5. arXiv:2510.09653.
- Feng, C. et al. (2021). TOOD: Task-aligned One-stage Object Detection. ICCV 2021.
- Tian, Z. et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV 2019. *(canonical ltrb regression source)*
- Liu, S. et al. (2018). Path Aggregation Network for Instance Segmentation. CVPR 2018. *(PAN/PAFPN)*
- Li, X. et al. (2020). Generalized Focal Loss. NeurIPS 2020. *(DFL canonical)*
- Zhang, H. et al. (2021). VarifocalNet: An IoU-aware Dense Object Detector. CVPR 2021. *(VFL)*
- Jordan, K. (2024). Muon: An optimizer for hidden layers in neural networks. https://kellerjordan.github.io/posts/muon/
- Liu, Z. et al. (2025). Muon is Scalable for LLM Training. arXiv:2502.16982. *(Moonlight, the actual benchmarking of Muon at scale)*
