# opndet — Design Draft

Tiny single-class object detector. OpenVINO 2022 opset compatible. No NMS, no postprocessing.

## Goals

- **Single-class** detector. Generalizes across "simple" rigid-ish objects.
- **Target object profile**: geometrically simple **convex** shapes (ellipsoid/blob/rigid), low intra-class variance, slight occlusion expected (touching/overlapping objects, partial fixture edges). NOT cluttered natural scenes.
- **Density**: up to **100+ objects per frame**, must detect every one. Industrial counting accuracy.
- **Image size**: small. Reference: **512×384** (4:3, non-square). Native sensor/ROI resolution; do NOT upsample.
- **Lighting**: arbitrary. Industrial-grade generalization — works under flicker, mixed lighting, shadows. Photometric robustness via heavy augmentation, not lab-conditioned data.
- **Size/FLOPs budget: match YOLOv8n**. Target ~3M params, ~4 GFLOPs at 512×384 (≈ YOLOv8n's 8.7 GFLOPs @ 640² scaled by pixel ratio). Edge/CPU inference, sub-10ms on modest CPU.
- **Industrial-grade**: deterministic outputs, robust to lighting/scale variance, reproducible training.
- **OpenVINO 2022 opset compat**: ONNX opset ≤ 13 safe. Avoid: ScatterND quirks, GridSample (opset 16), recent Resize modes, dynamic shapes where avoidable.
- **No NMS / no postprocessing**: model output is directly consumable. Either:
  - (A) single-instance assumption → regress one bbox + confidence, OR
  - (B) dense heatmap → argmax / threshold only (still no NMS), OR
  - (C) fixed-grid dense regression with peak-pick by argmax per cell.
- Optional add-ons: orientation (angle regression), heatmap output, count.

## Open questions for deliberation

1. **Multi-instance or single?** Many objects per frame w/ slight occlusion. Strong lean: heatmap (CenterNet-style). Single-instance regression rejected — won't survive occlusion or crowded scenes.
2. **Bbox parameterization**: AABB only, or oriented bbox (OBB) for rotation? Elongated convex objects benefit from orientation.
3. **Output tensor shape**: heatmap `[B, 1, H/s, W/s]` + size regression `[B, 2, H/s, W/s]` + (optional) angle `[B, 2, H/s, W/s]` (sin/cos). Stride `s` = 4 typical.
4. **Backbone**: MobileNetV2 blocks? Tiny custom CSP? Repvgg-style for speed? Constraint: opset 13 ops only (Conv, BN, ReLU/ReLU6, Add, Concat, Resize-nearest, MaxPool).
5. **Peak extraction inside graph**: `MaxPool(k=3,s=1) == heatmap` mask trick is opset 13 safe and counts as "no NMS" since it's deterministic local-max. Acceptable?
6. **Training framework**: PyTorch → ONNX export. Loss: focal heatmap + L1 size + (smooth-L1 or 1-cos) angle.
7. **Input resolution**: fixed (e.g., 256×256 or 320×320) for industrial determinism.
8. **Negative class / no-object case**: heatmap naturally handles. For single-instance variant need explicit confidence head.

## Non-goals

- Multi-class (single-class only, by design).
- COCO-style benchmarks. Industrial fixtures, not natural images.
- Anchor boxes. Anchor-free only.

## Deliverable from deliberation

Converged spec:
- Output tensor layout (exact)
- Backbone block list
- Loss formulation
- Export contract (input shape, output names, opset, IR version)
- Postproc-free decoding rule (the one client-side op allowed, e.g. argmax)

---

## Converged Architecture (v1)

Three architects deliberated: heatmap (A), dense regression (B), segmentation-w-distance-field (C). Convergence:
- B conceded heatmap is safer for touching objects.
- C is geometrically richer for ≥50% occlusion but requires connected-components for bbox = postproc violation.
- **v1 = CenterNet-lite (A) with B's exp-free size encoding. C's distance head reserved for v2 if peak-collapse observed.**

### Input

- Shape `[1, 3, 384, 512]` (NCHW, **H=384, W=512**), fp32
- Both dims divisible by 32 → clean stride-2 output (192×256) and stride-4 output (96×128)
- Pre-normalization: ImageNet mean/std, baked into preprocessing (NOT in graph)
- Fixed shape (no dynamic axes) — industrial determinism
- **Grayscale-friendly**: support 1ch input variant (replicate-to-3ch in preprocessing) for mono industrial cameras

### Output (4 tensors, stride 2 → 192×256)

**Stride 2 chosen over stride 4:** at 100+ small convex objects in 512×384, stride 4 (96×128 = 12288 cells) risks peak collapse for tightly-packed clusters where centers can be <8px (2 cells) apart. Stride 2 (192×256 = 49152 cells) gives 4× more cells, comfortable separation for objects with centers ≥4px apart.

| Name | Shape | Dtype | Activation | Meaning |
|---|---|---|---|---|
| `peaks`    | `[1, 1, 192, 256]` | fp32 | Sigmoid → in-graph local-max | Center confidence (peak-suppressed) |
| `size`     | `[1, 2, 192, 256]` | fp32 | Sigmoid × `MAX_PX` | (w, h) pixels. No `exp` in graph |
| `angle`    | `[1, 2, 192, 256]` | fp32 | Tanh | (sin θ, cos θ) |
| `dist`†    | `[1, 1, 192, 256]` | fp32 | Sigmoid | **Auxiliary** distance-to-nearest-center field (C's idea, promoted to v1) |

† `dist` head primarily for training supervision (helps separate touching peaks). At inference, client may fuse `peaks ⊙ dist` for higher-density scenes.

**Decoding (client):** threshold `peaks > T` → gather (cy, cx) → lookup `size`, `angle` → `(x, y) = ((cx, cy) + 0.5) × 2`. No NMS, no IoU, no sort.

### Backbone — CSP-lite, ~2.0M params

Channels scaled up to absorb the ~3M budget. CSP-style fused blocks (CSP-stage = split → conv chain → concat → fuse), all opset-13 safe (Conv, BN, ReLU6, Add, Concat, Split via Slice).

```
Stem:    Conv 3→32  k3 s2  BN ReLU6                    (192×256)   ~900
Stage1:  CSP 32→64  n=1   s2 (DWSep inside)            (96×128)    ~30K
Stage2:  CSP 64→128 n=2   s2                           (48×64)     ~150K
Stage3:  CSP 128→192 n=3  s2                           (24×32)     ~600K
Stage4:  CSP 192→256 n=2  s2                           (12×16)     ~900K
         (Stage4 replaces SPP with simple 5×5 MaxPool concat — opset 13 safe)
```

ReLU6 throughout (not SiLU — SiLU = `x · sigmoid(x)` works in opset 13 via Mul+Sigmoid but ReLU6 is faster on OpenVINO CPU and equally robust at this scale).

### Neck — Bi-FPN-lite, ~500K params

Three upsample stages back to stride 2. Resize: `mode=nearest, coordinate_transformation_mode=asymmetric` (pinned — half_pixel buggy in some OV2022 IRs).

```
P4 (12×16, 256ch)  → 1×1 → Resize×2 → Add P3 (24×32, 192ch→1×1 to 128ch)
P3' (24×32, 128ch) → 1×1 → Resize×2 → Add P2 (48×64, 128ch→1×1 to 96ch)
P2' (48×64, 96ch)  → 1×1 → Resize×2 → Add P1 (96×128, 64ch→1×1 to 64ch)
P1' (96×128, 64ch) → 3×3 Conv → Resize×2 → 3×3 Conv → F (192×256, 48ch)
```

### Heads — shared trunk + 4 1×1 (~400K params)

Shared 3×3 Conv 48→64 BN ReLU6, then 4 parallel 1×1 convs:
- `peaks_raw` → 1ch → Sigmoid → in-graph peak op
- `size`      → 2ch → Sigmoid × `MAX_PX` (e.g. 64)
- `angle`     → 2ch → Tanh
- `dist`      → 1ch → Sigmoid

**Total: ~2.9M params. ~3.8 GFLOPs at 512×384.** Matches YOLOv8n budget.

### In-graph peak extraction (the "no NMS" trick)

```
heatmap   = Sigmoid(peaks_raw)                          # [1,1,192,256]
hm_pooled = MaxPool(heatmap, k=3, s=1, pads=1)
peak_mask = Equal(heatmap, hm_pooled)                   # ONNX opset 11+, OV 2022 ✓
peak_f    = Cast(peak_mask, float32)
peaks     = Mul(heatmap, peak_f)
```

Deterministic local-max in 3×3 window. Not NMS — no IoU, no sorting.

### Loss

```
L_total = L_heatmap + 0.1·L_size + 0.05·L_angle·M_ar + 0.5·L_dist
```

- `L_heatmap`: CornerNet focal (α=2, β=4); GT Gaussian σ = max(r/3, 1.0) cells (stride 2 means 1 cell = 2px).
- `L_size`: L1 on (w, h) at GT-positive cells.
- `L_angle`: cosine distance `1 − (sin·sin_gt + cos·cos_gt)`.
- `M_ar`: gate, apply L_angle only where `max(w,h)/min(w,h) > 1.15` (near-circular = no orientation).
- `L_dist`: L1 on foreground pixels, target = normalized Euclidean distance to nearest GT center (capped at object radius). Trains the aux head.

Training graph free (any ops). Opset constraint applies to **export only**.

### Industrial-grade lighting / generalization

Achieved via training-time augmentation, NOT in-graph normalization (in-graph would freeze stats):

- **Photometric**: brightness ±50%, contrast ±50%, gamma 0.5–2.0, hue ±20°, saturation ±50%, channel shuffle (5%), grayscale (20%)
- **Synthetic shadows**: random low-frequency multiplicative noise + occasional gradient overlays
- **Sensor sim**: Gaussian noise σ ∈ [0, 0.05], Poisson shot noise, JPEG compression q ∈ [50, 100], motion blur (small)
- **Geometric**: rotation ±180° (objects are convex/symmetric — angle aug must rotate sin/cos labels too), scale 0.7–1.3, mosaic (combines tray crops)
- **Domain randomization**: train across collected datasets + synthetic renders if available
- **Color-space invariance**: optionally train on grayscale-replicated inputs at 30% probability so model handles mono cameras

The point: model never sees "lab conditions" — distribution shift is built into training so deployment on new sites needs no recalibration.

### Export contract

- Framework: PyTorch → `torch.onnx.export`
- Opset: **13** (OpenVINO 2022 supports through 13 cleanly)
- IR version: 7
- Static shapes only, no dynamic axes
- Inputs: `image` `[1,3,384,512]` fp32
- Outputs: `peaks`, `size`, `angle`, `dist`
- Allowed ops: Conv, BN, ReLU/ReLU6, Add, Mul, Sub, Concat, Split/Slice, Resize (nearest, asymmetric), MaxPool, Sigmoid, Tanh, Equal, Cast
- **Forbidden ops**: GridSample, ScatterND, GroupNorm, GELU, Resize half_pixel, dynamic shapes, SiLU as fused op (use Sigmoid+Mul if needed)

### Top risks (carried from all 3 architects + new constraints)

1. **Peak collapse, dense clusters of 100+ small objects.** Two centers <2 cells apart at stride 2 (≤4px input space) merge. Mitigations: stride 2 (chosen), min Gaussian σ=1.0 cell, tight-pack mosaic augmentation, `dist` aux head primed for v2 inference fusion.
2. **Resize coord-transform mode.** Pin `asymmetric` in export, regression test vs calibration fixture.
3. **Generalization to unseen lighting.** Mitigation: aggressive photometric aug + grayscale-mix training (above). Validate on held-out site data before deployment, not just train-distribution split.
4. **FLOP budget overrun.** ~3.8 GFLOPs estimated, but Bi-FPN-lite stride-2 head is the heaviest part. If profiling shows >5 GFLOPs, drop to stride 4 with denser σ=2.0 GT and rely on `dist` aux head for separation.
5. **Near-circular angle instability.** AR<1.15 mask (already in spec).

### v2 deferred features

- Promote `dist` to inference-fused output (`peaks ⊙ dist` peak refinement) if v1 recall insufficient at >150 obj/frame
- Multi-class head (bump `peaks` to Nch) — only if needed
- INT8 quantization via OpenVINO POT (architecture is already quant-friendly: ReLU6 + no exp/atan2 in graph)

---

## Opset 13 / OpenVINO 2022 op compatibility check

| Op | Min opset | OV 2022 | Notes |
|---|---|---|---|
| Conv | 1 | ✓ | |
| BatchNormalization | 1 | ✓ | Folded by export in eval mode (PyTorch default) |
| ReLU | 1 | ✓ | |
| ReLU6 → Clip(0,6) | 1 | ✓ | `nn.ReLU6` exports as `Clip` cleanly |
| Add / Mul / Sub | 1 | ✓ | |
| Concat | 1 | ✓ | |
| Split | 1 / **13** | ✓ | Opset 13 moved `split` from attr to input — exports OK from PyTorch |
| Slice | 1 / **10** | ✓ | Opset 10+ moved indices to inputs; opset 13 stable |
| Resize | 11 | ✓ | Pin `mode=nearest`, `coordinate_transformation_mode=asymmetric` |
| MaxPool | 1 | ✓ | |
| Sigmoid | 1 | ✓ | |
| Tanh | 1 | ✓ | |
| Equal | 11 | ✓ | Returns `bool`, must `Cast` |
| Cast | 1 | ✓ | bool → fp32 for `Mul` with heatmap |

**No SiLU/HardSwish** in v1. SiLU = `Sigmoid + Mul` works opset 13 but ReLU6 chosen for OpenVINO CPU latency.
**No GELU, GroupNorm, GridSample, ScatterND, LayerNorm.** None in spec.

**Export sanity test (CI gate):**
1. `torch.onnx.export(..., opset_version=13, do_constant_folding=True, dynamic_axes=None)`
2. `onnx.checker.check_model(model)`
3. `mo.convert_model(...)` (OpenVINO Model Optimizer 2022.x) returns IR with no warnings about unsupported ops or coord-transform fallbacks
4. fp32 IR vs ONNX cosine-sim ≥ 0.9999 on calibration batch
5. INT8 IR (POT default) vs fp32 IR cosine-sim ≥ 0.99
