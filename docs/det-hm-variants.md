# Heatmap-only detection variants — design

Two new presets staged at the bbox-x size (10.36M params) for evaluation:

| Preset | Channels | Output content | Decode | OpenVINO |
|---|---|---|---|---|
| `bbox-x-hm2` | 2 | `(obj, radius)` | peak suppress + radius lookup | ✓ opset 13 |
| `bbox-x-flow` | 4 | `(cell_prob, flow_y, flow_x, log_radius)` | flow integration in PyTorch | ✗ by design |

Both presets are wired in `src/opndet/configs/`. They build, forward, and (for hm2) ONNX-export with parity max-abs-diff < 1e-5. **Loss + target encoder + decoder are not yet implemented** — checkpoint of the architectural choice before committing to the training pipeline.

## bbox-x-hm2 — 2-channel HM

Replaces the existing 5-channel `(obj, cx, cy, w, h)` head with 2 channels: `obj` (peak-suppressed sigmoid, same as bbox-x) and `radius` (sigmoid, in [0,1] of `max(img_h, img_w)`). Decoder reads peak position from `obj`, radius value at that cell, recovers a bbox.

Same input pipeline as bbox-x. Same backbone, same neck. Only the head + loss + target encoder change.

### Targets (encode.py needs an `encode_hm2_targets`)

For each GT box `(x1, y1, x2, y2)` at center `(cx, cy)` with size `(w, h)`:

- `obj_target[cy/stride, cx/stride] = 1` placed via Gaussian with σ from CornerNet rule (existing `_gaussian_radius` helper)
- `radius_target[cy/stride, cx/stride] = sqrt(w² + h²) / 2 / max(img_h, img_w)`

Returns `{"obj": (Hs,Ws), "radius": (Hs,Ws), "radius_mask": (Hs,Ws) in {0,1}}` where the mask is 1 only at GT-center cells (radius supervision is sparse, only those cells have meaningful targets).

### Loss (loss.py needs an `Hm2Loss`)

```python
loss_obj    = focal_heatmap_loss(pred_obj, target_obj)              # existing helper
loss_radius = (pred_radius - target_radius).abs() * radius_mask     # L1 only at GT centers
loss        = w_obj * loss_obj + w_radius * loss_radius.sum() / max(1, n_gt)
```

`w_obj` defaults to 1.0, `w_radius` defaults to 1.0. Tune later.

### Decode (decode.py needs `decode_hm2`)

```python
peaks = np.argwhere(obj > threshold)         # post-peak-suppress, sparse
for y, x in peaks:
    cx_img, cy_img = x * stride, y * stride
    r_norm = radius[y, x]
    r_px   = r_norm * max(img_h, img_w)
    box    = (cx_img - r_px, cy_img - r_px, cx_img + r_px, cy_img + r_px)
```

`r_px` is half-diagonal — squared boxes. If you need rectangular boxes, the head needs a 2-channel radius (separate `radius_x, radius_y`); minor extension.

### What's still TODO
- [ ] `encode_targets` branch keyed on output channel count (2 vs 5)
- [ ] `Hm2Loss` class + train.py route based on preset
- [ ] `decode_hm2` + predict.py routing
- [ ] Smoke training run → mAP@.5:.95 vs bbox-x

## bbox-x-flow — CellPose-style 4-channel head

`(cell_prob, flow_y, flow_x, log_radius)`. PyTorch-only decode by design — no opset constraint, freedom to use whatever ops work. The model is a regular convnet; the cleverness is at decode and target generation.

### Targets — REQUIRES SAM-derived per-instance segmentation masks

Bbox-only annotations don't carry enough info. For each training image:

1. Run SAM (or any instance segmentation that you trust on eggs) over the image. Use the existing GT bbox set as point prompts → one mask per egg.
2. For each instance mask:
   - Compute the signed distance transform inside the mask (positive = inside, decreasing toward edge)
   - Take the gradient of that distance transform → vectors pointing toward the centroid
   - Normalize per-pixel: `flow_y[mask] = -∂dist/∂y`, `flow_x[mask] = -∂dist/∂x` (negate so flow points TOWARD center, not away from it)
3. `cell_prob_target` = union of all instance masks (binary, dilated by 1 px to soften edges)
4. `log_radius_target` at each instance's centroid = `log(sqrt(mask_area / pi))`

Save to disk once per dataset; fast to load at training time.

### Loss

```python
loss_prob = focal_loss(pred_prob, target_prob)                          # focal_alpha=2, focal_beta=4
loss_flow = ((pred_flow - target_flow) ** 2).sum(dim=1) * target_prob   # L2 inside mask only
loss_radius = (pred_radius - target_radius).abs() * centroid_mask
loss = w_prob * loss_prob + w_flow * loss_flow.mean() + w_radius * ...
```

### Decode — flow-field integration

```python
def decode_flow(prob, flow_y, flow_x, prob_thresh=0.5, n_steps=32):
    # 1. find candidate pixels
    H, W = prob.shape
    ys, xs = np.where(prob > prob_thresh)
    pts = np.stack([ys.astype(float), xs.astype(float)], axis=1)  # (N, 2)

    # 2. Euler-integrate the flow field
    for _ in range(n_steps):
        ys_i = np.clip(pts[:, 0].astype(int), 0, H-1)
        xs_i = np.clip(pts[:, 1].astype(int), 0, W-1)
        pts[:, 0] += flow_y[ys_i, xs_i]
        pts[:, 1] += flow_x[ys_i, xs_i]
        pts[:, 0] = np.clip(pts[:, 0], 0, H-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, W-1)

    # 3. cluster by terminal attractor (round to int, group by hash)
    keys = (pts[:, 0].round().astype(int) * W + pts[:, 1].round().astype(int))
    instances = {}
    for i, k in enumerate(keys):
        instances.setdefault(k, []).append((ys[i], xs[i]))

    # 4. per-instance bbox from pixel set extents
    boxes = []
    for k, pixel_list in instances.items():
        ys_arr, xs_arr = zip(*pixel_list)
        y0, y1 = min(ys_arr), max(ys_arr)
        x0, x1 = min(xs_arr), max(xs_arr)
        boxes.append((x0 * stride, y0 * stride, x1 * stride, y1 * stride))
    return boxes
```

Touching objects separate naturally: each pixel belongs to whichever instance's flow basin it falls into.

### Why it works on touching cases

- bbox-x's peak suppression on `obj`: two adjacent eggs both peak; the smaller one's peak gets zeroed by MaxPool of the bigger one's neighbor → false negative.
- flow-field: each pixel gets its OWN flow vector toward its OWN instance's center. Two adjacent eggs have flow fields that diverge across the boundary between them → distinct attractors → distinct instances. No max-pool-driven merging.

### What's still TODO
- [ ] SAM target generation script (one-time per dataset)
- [ ] `encode_flow_targets` reading the cached mask + flow files
- [ ] `FlowLoss` class
- [ ] `decode_flow` + predict.py routing
- [ ] Validation: per-instance F1 on a touching-objects test set

## Suggested order of operations

1. **bbox-x-hm2 first.** Mechanical loss + encode + decode. Verify it matches bbox-x's mAP@.5:.95 (proves the simpler head doesn't lose anything). If it ties, stay on it as the new opset-13 baseline.
2. **SAM target generation script** before flow. Cached per-dataset, used by both flow training and any future seg/distance-aware variant.
3. **bbox-x-flow** last. Bigger lift, but the touching-object case is what justifies it.

Both variants reuse the existing per-preset training defaults (LR, batch, aug, calib). No `training_defaults.py` entries needed unless we want different LR/aug for the flow loss specifically.
