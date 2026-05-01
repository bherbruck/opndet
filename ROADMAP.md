# opndet — Industrial-Grade Improvement Roadmap

This document is a working plan for the next phase of opndet. The goal is to take what is already a working tiny single-class detector (28K–2.4M params, ONNX opset-13 safe, no NMS, deployed-clean on Myriad X / Ethos-U / Neural-ART / RT1062) and make it **industrial-grade**: stable confidence, no flapping, deterministic counting, robust to lighting/scale variance, and exceptionally good at the convex-object regime that real industrial inspection actually deals with.

This is not a research wishlist. It is an engineering plan organized so that each item can be picked up independently, has a clear definition of done, and ships value to deployed users.

---

## Context: what already exists

Before proposing changes, the working state as of this writing:

- **Architecture**: single output tensor `[1, 5, H/4, W/4]` = `(obj, cx, cy, w, h)`. CenterNet-family lineage. Anchor-free, single-scale, single-class.
- **Peak suppression**: arithmetic in-graph, `clip((hm + eps - MaxPool(hm)) * (1/eps), 0, 1)`. Myriad-safe. The `eps=5e-3` is sized for fp16 rounding error and is a load-bearing constant.
- **Five presets** (bbox-f/p/n/s/m) all sharing the same YAML DSL, registered modules, layer graph.
- **Loss**: focal heatmap + L1 cxy + L1/GIoU/CIoU/NWD wh, with optional Varifocal Loss (VFL) for confidence quality and optional repulsion loss for crowded scenes.
- **Encoding**: CornerNet Gaussian radius heuristic, σ = max(r/3, 1.0), `pos` mask at peak cells only.
- **Augmentation**: photometric (brightness/contrast/gamma/hue/saturation/grayscale/blur/noise), geometric (hflip/vflip/rotate90/scale_jitter/translate), cutout, mosaic. `min_visible_frac` drops boxes that fall below visibility threshold.
- **Training**: AdamW + cosine LR + AMP, EMA, early stopping by patience on configurable metric (f1/map50/map_50_95), TensorBoard scalars + image grids. **Auto-split** (deterministic seeded train/val/test) and **multi-source merge** with **single-class collapse** are already in `dataset.py` — users do not pre-split their data, they point at COCO json + image dir and the system handles it.
- **Export**: opset 13, static shapes, `--bake-input-norm` baked-normalization variant, parity-tested fp32 PyTorch vs ORT.
- **Quantization**: static int8 PTQ via onnxruntime with optional verification.

What is missing or under-developed: confidence stability, touching-object handling, the convex prior, distillation, QAT, multi-scale support, and the broader platform story.

---

## Part 1 — Industrial-grade reliability (highest priority)

These are the items that determine whether a real customer can deploy opndet on a line and trust the output. None are flashy. All are required.

### 1.1 Confidence stability (no flapping)

The single most common failure mode in deployed detectors is **flapping**: a confidence that hovers around the threshold, causing the same object to be counted, then not counted, then counted again across consecutive frames. This is what makes integrators tear their hair out. opndet is *less* susceptible than YOLO (peak suppression alone removes one class of flapping) but the residual problem is in the heatmap itself.

**Current state**: focal heatmap loss with α=2, β=4. VFL is implemented but not on by default. No explicit anti-flapping mechanism. **The training/eval data is non-sequential** — random snapshots, not video frames — so true frame-to-frame temporal stability cannot be measured directly. We have to use perturbation-based proxies.

**Plan**:

1. **Make VFL the default** for bbox-s and bbox-m presets. The whole point of VFL is to push confidence toward bimodal (high-IoU predictions → near-IoU score, low-quality predictions → near-zero), eliminating the squishy mid-range. The implementation already exists in `loss.py`. Validate that switching from focal to VFL reduces *perturbation-induced* confidence variance (see #2).

2. ✅ **Perturbation-stability metric** (proxy for temporal stability). Since the dataset is non-sequential, synthesize "adjacent frames" by applying small augmentations to each test image: ±2 px translations, ±2° rotations, ±5% brightness/contrast jitter. For each (image, perturbation) pair, match predictions to GT and measure the confidence variance for the *same physical object* across perturbations. Report mean and 95th-percentile confidence stddev. A model that's robust under perturbation is robust frame-to-frame in deployment, even though we can't validate that directly. This is also the right metric to optimize: a model that doesn't flap under tiny augmentations won't flap under tiny inter-frame motion either. *(Shipped: `opndet eval --stability [--n-perturbations N]` runs each test image through K small perturbations (±2px translate, ±5% brightness, ±5% contrast), Hungarian-matches preds to GT per version, transforms pred centers back to the original frame, and reports mean / p95 / max stddev of (score, cx, cy, w, h) across versions. New "perturbation stability" section in eval-report.md and `eval/stability/*` scalars in TB. Off by default — costs ~Nx forward passes. Rotation deferred since axis-aligned-bbox supervision plays poorly with rotated GT.)*

3. ✅ **Calibration** of confidence values so that "0.7 confidence" actually means "70% of the time, this is a real object." Most detectors are wildly miscalibrated; the score is a ranking signal, not a probability. Add an optional Platt-scaling or temperature-scaling step at the end of training that fits the confidence-to-precision curve on the validation set. Export the temperature as part of the ONNX graph (a single Mul before the sigmoid). This is one extra op, hardware-safe everywhere. *(Shipped: `opndet calibrate --ckpt --config [--split]` fits a Platt-style scalar T via scipy.minimize_scalar, writes it back into the ckpt under key `temperature`. SigmoidPeakSuppress carries an optional `temperature` attr; predict / eval / export all auto-apply it on load. ONNX trace folds T as a Constant — extra Mul before Sigmoid, opset-13 safe. Verified pt-vs-ort parity at T=2.5 within 1e-5.)*

4. **Hysteresis-aware threshold guidance** (deployment-side, separate from training data). For deployment on actual video streams, document the recommended pattern: use a high threshold (e.g., 0.7) to *enter* a detection track, a low threshold (e.g., 0.3) to *maintain* it, and require N-of-M consecutive frames to confirm a count. This is application-layer logic, not model logic — it's what users layer on top of opndet for always-on counters. Add a reference implementation as a separate `opndet-stream` example or as part of `predict.py --stream`. Note explicitly in docs that opndet itself is frame-stateless; track-and-count logic is the user's responsibility (or a future companion package).

5. **Future temporal data collection**. If/when sequential video datasets become available (deployment recordings, dedicated capture sessions), add true frame-to-frame stability as a first-class metric. Until then, perturbation-stability is the honest proxy.

**Definition of done**: perturbation-stability metric is computed and reported in TensorBoard. VFL is the default for bbox-s/m. Calibration step is optional and disabled by default but documented. Reference stream-mode logic ships as an example.

### 1.2 Touching-object disambiguation

The DESIGN.md already identified this as the biggest risk: "peak collapse, dense clusters of 100+ small objects." Two centers within 4 px (2 cells at stride 4) merge into a single peak after MaxPool with k=5. For touching pills, eggs in a flat, fish on a sorter, this is *the* problem.

**Current state**: single peak suppression with k=3 or k=5 MaxPool. CornerNet σ heuristic for GT encoding, min σ = 1.0 cell. No touching-object-specific supervision.

**Plan**:

1. ✅ **Add the auxiliary distance head** that DESIGN.md v1 already specced but isn't in the current single-tensor output. Output becomes `[1, 6, H/4, W/4]` with an extra `dist` channel — a distance-to-nearest-center scalar field. Train with L1 loss against the normalized Euclidean distance to the nearest GT center, capped at object radius. At inference, multiply `obj * dist` to push down peaks that are between two centers (where `dist` is low) and preserve peaks at true centers (where `dist` is high). This is a v1-deferred feature in DESIGN.md; promote to v2 default for bbox-s/m. *(Shipped: new preset `bbox-m-dist` with internal 6-ch raw -> Sigmoid · dist · PeakSuppress in-graph, deployment output stays [1, 5, H/4, W/4] for backward compat. Target = inscribed-ellipse linear ramp (1 at center, 0 at boundary), aggregated with elementwise max — touching objects' ramps decay to 0 at midplanes by construction. New `Mul` and `SigmoidT` ops in primitives.py. encode.py auto-includes `dist` target when model has a `dist` alias. l_dist logged to TB. ONNX-13 parity verified at 1.33e-6. Patterns for bbox-s-dist / bbox-n-dist follow trivially.)*

2. **Smaller suppression neighborhood for dense scenes**. Add a per-preset `peak_kernel` setting that the YAML controls explicitly. For dense scenes (eggs in a 6×8 flat, pills on a tray), k=3 is correct; for sparse scenes (one fish at a time), k=5 is fine. Currently k is set in the YAML but the trade-off isn't documented and the default (k=5 for bbox-m) is wrong for dense scenes.

3. **Tight-pack mosaic augmentation**. The current mosaic combines 4 images. Add a "tight-pack" mosaic mode that *deliberately* creates touching-object boundaries by reducing inter-image padding to zero or even negative (overlapping crops). This forces the model to see touching objects at training time, not just at deployment time.

4. **Separating-surface auxiliary supervision** (longer-term). For each pair of adjacent objects, the midplane between centers carries information. Add a binary "is-this-a-separating-surface" head supervised at the inter-object midpoints. Helps the model learn that intensity gradients across midplanes are real signal, not noise. Optional, opt-in via YAML.

**Definition of done**: `dist` head is on by default in bbox-s/m. Mosaic has a `tight_pack_prob` config knob. Documented decoding rule is `obj * dist` for dense scenes, `obj` alone for sparse. Validation on a test set with deliberately-touching synthetic objects shows >5% recall improvement vs current.

### 1.3 Determinism and reproducibility

Industrial customers need bit-for-bit reproducibility. If they re-train a model with the same config and same data, they need the same weights. If they run inference on the same image twice, they need identical output to the LSB.

**Current state**: training uses `np.random.default_rng()` with no seed in augmentation; PyTorch DataLoader workers may or may not be seeded; cuDNN nondeterminism not pinned.

**Plan**:

1. **Seed everything**. Augmentation RNG, DataLoader worker seeds, PyTorch global seed, cuDNN deterministic mode. Add `seed:` to the training config (already partially present); make it the source of truth for every RNG in the pipeline.

2. **Document the determinism contract**: same seed + same data + same code = same weights. Same model + same input = same output (within fp32 epsilon, which is essentially zero for inference). For ONNX exports, byte-for-byte identical given identical PyTorch state dict. Make this a CI gate.

3. **Pin operator behavior**. AMP/fp16 training is non-deterministic across hardware. Document that bit-exact reproducibility requires fp32 training; AMP is for speed during R&D. The exported model is fp32 unless explicitly quantized.

**Definition of done**: re-running training twice with the same seed produces identical `best.pt`. CI runs the determinism check on every commit.

### 1.4 Robust evaluation: per-object metrics, not just mAP — **SHIPPED**

mAP is a ranking-based metric and hides problems that matter for industrial deployment. Specifically: it doesn't care about miscount, it doesn't care about confidence stability, it averages over IoU thresholds you may not deploy at, and it doesn't decompose error into the categories that matter (missed detections vs false positives vs duplicate detections).

**Status**: validation suite shipped via `opndet eval --ckpt PATH --config CFG [--split val|test]`. Writes `eval-report.md` + `eval-report.json` + four diagnostic PNGs (reliability, P/R/F1-vs-threshold, conf-IoU 2D hist, count-error histogram) to `<ckpt_dir>/eval_<split>/`. Training-loop `evaluate()` now uses Hungarian matching internally (same return shape, no caller change).

**Plan**:

1. ✅ **Counting accuracy distribution** (not just mean). Report median, 95th percentile, 99th percentile, and max of `|pred_count - gt_count|`. The tail kills you in production — a model that's exact on 99% of images and off by 100 on 1% has the same mean error as a model that's off by ±1 everywhere, but wildly different deployment characteristics. *(Shipped: mean / median / p95 / p99 / max + signed bias + exact-count fraction.)*

2. ✅ **Hungarian-matched evaluation** (replace greedy matching). Most evaluation code uses greedy IoU matching: sort by confidence, match highest IoU first. For "must detect every one" industrial counting, **Hungarian matching is strictly better** — it finds the globally optimal assignment, so a single weak detection doesn't get "stolen" by a stronger neighbor and counted as a false positive. Implementing Hungarian for eval is small (scipy has it). The difference vs greedy shows up exactly in your worst-case scenario: dense scenes with touching objects. *(Shipped: `metrics.hungarian_match` via `scipy.optimize.linear_sum_assignment`. Used by both `train.evaluate` and the new `opndet eval`.)*

3. ✅ **Per-error-type breakdown**: missed objects (recall miss), false positives (precision miss), duplicate detections (peak collapse failure mode), localization error (IoU < 0.5 but match found). Report each separately. Currently they're mushed into mAP@0.5. *(Shipped: TP / FP_localization / FP_duplicate / FP_background / FN_missed table.)*

4. ✅ **Size-stratified metrics**. Small objects fail differently than large ones. Stratify metrics by object size (small / medium / large in pixel area) and report each tier separately. This catches the "model works great on big eggs but misses small ones" failure that mAP averages away. *(Shipped: COCO-style strata — small <32², medium <96², large ≥96² — recall and precision per tier.)*

5. ✅ **Localization error decomposition** (matters for sizing applications). For sizing, *localization bias* matters more than IoU. A bbox that's offset 2 pixels in +x direction across all detections won't show up in IoU stats (they'll all be > 0.9) but will systematically bias your size estimates. Decompose into: **systematic bias** (mean of pred_center − gt_center, should be near zero), **random scatter** (stddev, tells you precision), **scale bias** (mean of pred_size − gt_size relative to gt_size, should be near zero). These never get reported and they're exactly what you need for industrial sizing. *(Shipped: `loc_bias` reports center bias x/y, center scatter x/y, scale bias w/h, scale scatter w/h.)*

6. ✅ **Confidence calibration plot** (reliability diagram). Bin predictions by confidence (0.0–0.1, 0.1–0.2, etc.), plot empirical precision in each bin against the bin midpoint. A perfectly-calibrated model has all points on the y=x diagonal. Most detectors are dramatically miscalibrated. Knowing this lets you choose deployment thresholds that mean what they say. *(Shipped: 10-bin reliability diagram + ECE scalar.)*

7. ✅ **Confidence-stratified P-R curve**. Plot precision and recall as functions of confidence threshold on the same axes. A model with a well-defined "knee" — where both metrics are high in a wide threshold range — is deployable. A sharp knee means small threshold changes cause big metric swings (fragile). This diagnoses "can I pick a threshold and stick with it." *(Shipped: P/R/F1 sweep across 19 thresholds, plotted on shared axes.)*

8. ✅ **Confidence-IoU 2D histogram** (the single most useful diagnostic). For each detection, record (confidence, IoU-with-GT). Stack across the test set into a 2D density plot. A well-calibrated, well-localized model has all density on the high-conf high-IoU diagonal. A flapping model has density spread across mid-confidence ranges. A miscalibrated model has density off the diagonal. *(Shipped: 20×20 log-density plot.)*

9. ⏸ **Per-domain stratified metrics** (you said the dataset is varied for generalization — measure that). Split the validation set by domain (lighting, camera angle, source dataset, whatever variation axis exists) and report metrics per-domain. Compute a robustness score: `mean(per_domain_mAP) - 2 * std(per_domain_mAP)`. Penalizes models that work great on average but fail on some domains. This is the metric that catches "trained too long on the easy data and forgot the hard data." *(Deferred: data sources don't carry domain tags yet. Plumb a `domain:` key through `data.sources` and a `--domain-key` CLI flag when needed.)*

10. ✅ **Validation loss alongside metrics**. Most pipelines report training loss and validation mAP, then act surprised when they disagree. Report validation loss too. Training loss going down while validation loss goes up is the canonical overfitting signal, and it shows up earlier than any metric does. *(Already in train loop: `train/loss`, `train/l_hm`, `train/l_cxy`, `train/l_wh` + per-epoch `val/*` metrics in TB. Validation loss as its own scalar can be added — small.)*

**Definition of done**: every `eval` run produces a structured report covering all ten items, with TensorBoard scalars for the numerical ones and PNGs for the diagnostic plots. The report is also written to `runs/expN/eval-report.md` so it can be shared with customers. *(Status: report.md + report.json + 4 PNGs all ship. `eval.write_tb_scalars()` helper exists; auto-call from training-loop epoch eval is a one-line follow-up if/when richer per-epoch TB views are wanted.)*

### 1.5 Industrial-grade loss formulation

Standard detection losses optimize per-cell BCE/focal + per-cell box regression. For "convex, single-class, varied dataset, must count and size correctly," the loss can be sharpened in ways the academic CV community has no incentive to explore (because their benchmarks don't measure these things).

**Current state**: focal/VFL heatmap + L1/GIoU/CIoU/NWD wh + L1 cxy. Optional repulsion loss for crowded scenes. All terms are *local* (per-cell).

**Plan**:

1. ✅ **Count-aware loss term**. Add `λ * |sum(peaks) - n_gt|` to the total loss, where `peaks` is the post-suppression heatmap. The model learns to make the *total mass* of the heatmap equal the GT count, not just to be locally correct. This addresses a failure mode where the model has high mAP but consistently miscounts by 1-2 because it dropped a low-confidence detection or hallucinated a duplicate. λ ≈ 0.05-0.1 (small enough not to dominate, big enough to bias). This is a **non-local** loss term — most detection losses are per-cell. Adjacent prior art: crowd-counting density-map regression, but applied here to a sparse peak map. *(Shipped: `loss.count_weight` (default 0, disabled), `loss.peak_kernel`, `loss.peak_eps` knobs in train.yaml. `_peak_suppress` mirrors the deployed in-graph op so train and inference see the same sparse map. `train/l_count` logged to TB. Validate against `eval-report.md` count_stats.)*

2. **Hungarian-matched VFL targets**. VFL uses IoU between predicted box and GT box at each positive cell as the supervisory target. Currently this is done greedy (each cell matches its own GT). Use Hungarian matching to assign predictions to GTs globally before computing IoU targets. Reduces the case where two predictions both match the same GT and one of them gets a spuriously low IoU target.

3. **Asymmetric count loss for deployment safety**. In some industrial contexts, **over-counting and under-counting have different costs**. Counting eggs for grading: missing one is yield loss; phantom egg breaks the packaging machine. Add an asymmetric count term: `α * max(0, pred - gt) + β * max(0, gt - pred)` with α and β configurable per-application. Lets users say "I want a recall-biased model" (small α, large β) or "I want a precision-biased model" (large α, small β).

4. **Per-domain loss balancing**. If the dataset has domain labels (lighting, camera, source), oversample under-represented domains or weight their loss higher. Prevents the model from optimizing easy domains at the expense of hard ones. This pairs with the per-domain validation metric in 1.4.

5. **Localization-bias regularizer**. For sizing applications, penalize systematic offset directly. Add a small regularization term that penalizes `|mean(pred_cx - gt_cx)|` and `|mean(pred_cy - gt_cy)|` across the batch. Pushes the model toward unbiased localization (random scatter is fine, systematic offset is not). Tiny weight (1e-4 ish), only matters at the end of training.

6. **Curriculum metric annealing** (the multi-phase training idea). Train phase 1 with mAP@0.5 as the patience metric and a forgiving loss (focal + L1). Phase 2 with F1 as the patience metric and VFL + L1, init from phase 1, LR 0.3x. Phase 3 with mAP@0.5:0.95 as the patience metric and VFL + CIoU, init from phase 2, LR 0.1x. Each phase optimizes a different aspect (propose well → threshold well → localize tightly), letting tiny models specialize each phase's full capacity. Opt-in via `--curriculum metric`. Tiny models (bbox-f, bbox-p) probably benefit most because they can't be globally optimal across all metrics simultaneously.

**Definition of done**: count-aware loss is opt-in via YAML and validated to improve count accuracy without hurting mAP. Asymmetric count loss is documented with example use cases. Per-domain balancing is a config option. Curriculum mode is an opt-in flag with prescribed phase configs in each preset.

---

## Part 2 — The convex prior (the differentiator)

The single biggest architectural insight for opndet's actual application domain is that **industrial objects are convex or near-convex**. Eggs, pills, bottles, fish, fasteners, lumber cross-sections, kernels, tablets, fruit. This is a real architectural constraint that nobody else is exploiting because general-purpose detectors can't make the assumption.

If opndet commits to "convex objects only" as a positioning choice, several optimizations become available that would be wrong for general detection.

### 2.1 Ellipse output format (alternative to AABB)

For convex objects, the bounding ellipse is a *better* shape descriptor than the bounding box. It captures size and orientation honestly, doesn't include the empty corners that bounding boxes add, and makes downstream measurement (sizing, grading, weight estimation) more accurate.

**Plan**:

1. **Add a `--shape ellipse` mode** to model export. Output channels become `(obj, cx, cy, r_major, r_minor, sin_2θ, cos_2θ)` — 7 channels instead of 5. The 2θ encoding handles the π-periodicity of orientation correctly. Bounding box can be computed from ellipse parameters at decode time for compatibility with downstream code.

2. **Loss for ellipse mode**: L1 on (r_major, r_minor), cosine distance on (sin 2θ, cos 2θ), gated by aspect ratio (no orientation supervision when r_major / r_minor < 1.15, since orientation is undefined for circles).

3. **Decoder**: client-side decode produces both AABB (for backwards compatibility) and ellipse parameters (for accurate measurement). The browser demo gets a "show ellipses" toggle.

4. **GT encoding** for ellipse mode: fit an ellipse to each GT bounding box (assuming the bbox is tight to a convex object, the inscribed ellipse is a reasonable proxy). For datasets with mask annotations, fit ellipse directly to mask. Document both paths.

**Definition of done**: bbox-s and bbox-m support `output_format: ellipse` in YAML. Loss handles both cases. Export and decode tested. Validation on the egg dataset shows that ellipse output reduces sizing error compared to AABB-derived size estimates.

### 2.2 Distance-transform output (replaces or augments objectness heatmap)

For touching convex objects, classical watershed segmentation works because the **distance transform** has clean local maxima at object centers and clean valleys between objects. Training a model to *output* a learned distance transform — a scalar field that increases monotonically from the boundary to the center — is a strictly better representation than a Gaussian heatmap for the touching-object case.

This is genuinely different from CenterNet. CenterNet's heatmap is a blurry Gaussian with no shape information. A distance transform is a shape-aware scalar field where peaks are unambiguous and valleys are clean.

**Plan**:

1. **GT encoding**: at training time, compute the distance transform of the foreground mask (union of all GT bounding boxes, or actual masks if available), normalize per-object by object radius, clamp to [0, 1]. Each pixel's GT value is its distance to the nearest object boundary, normalized.

2. **Loss**: L1 on foreground pixels. Background gets 0. The model learns to predict the distance transform directly.

3. **Inference**: the distance transform IS the objectness signal. Peak suppression on the distance transform is provably correct for non-overlapping convex objects. For overlapping objects, the in-graph arithmetic peak op still works — and the smaller overlap regions correctly produce smaller peaks.

4. **Backwards compat**: keep the Gaussian-heatmap mode as an option. Distance transform is opt-in via YAML.

5. **Synthetic supervision**: for convex-only datasets, the distance transform of an ellipse has a closed form. This means perfect synthetic supervision is available for free during convex pre-training (see 2.4).

**Definition of done**: `objectness_format: distance_transform` is a YAML option. Validation on touching-pill synthetic dataset shows ≥10% recall improvement vs Gaussian heatmap at the same param budget.

### 2.3 Convexity prior in the loss

Even without changing the output format, we can add a regularization term that enforces "the predicted box should contain a convex blob, not a complex shape."

**Plan**:

1. **Convexity-aware classification** as an auxiliary loss. For each predicted bounding box, compute the segmentation mask (using a tiny U-Net-style decoder head, or just thresholded feature activation), then compute the ratio `(mask_area / convex_hull_area)`. Penalize ratios below 0.9. This forces the model to learn that valid detections have convex foreground regions.

2. **Cheaper alternative**: use the distance transform consistency check. For a true convex object, the distance transform has a single local maximum, the gradient field points uniformly toward it, and the level sets are nested convex curves. Add a loss term that penalizes distance-transform predictions that don't have these properties.

3. **Defect detection as a feature**: explicit deviation-from-convexity is a *signal*, not a problem. A pill that doesn't fit the convex prior well is probably broken. Add an optional `--detect-defects` mode that reports per-detection convexity scores; below-threshold scores flag defects without requiring a separate defect classifier. This is a *feature for free* if we have the convexity term.

**Definition of done**: convexity loss is an opt-in YAML config. Defect detection is a documented inference mode. Test on a dataset with deliberately-broken pills shows that low-convexity predictions correlate with broken samples.

### 2.4 Synthetic convex pre-training

This is the biggest win for the femto tier. Models with 28K-92K params have so little capacity that *any* prior knowledge injection pays off enormously. Convex objects are easy to render — you don't need photorealism, just correct shape, lighting, scale, occlusion, and packing.

**Plan**:

1. **Build a synthetic data generator** as a separate package or sub-module. Renders random convex objects (ellipsoids, capsules, cylinders, rounded rectangles, superellipses) with:
   - Random texture (procedural noise, gradient, solid color, learned natural-image patches)
   - Random lighting (point sources, ambient, shadows)
   - Random scale (matching the deployment distribution)
   - Random packing (sparse, dense, touching, overlapping)
   - Random occlusion (random rectangles or other convex shapes drawn on top)
   - Random sensor noise (Gaussian, Poisson, JPEG compression, motion blur)

2. **Pre-train every preset on synthetic convex data** as a default first stage. The pre-trained checkpoints become part of the bundled distribution. Fine-tuning on real data starts from these instead of from scratch.

3. **Validate the gain**: pre-train + fine-tune vs scratch-train, compare mAP at fixed epochs. The hypothesis is that bbox-f and bbox-p see large gains (5-10 mAP), bbox-s sees moderate gains (1-3 mAP), bbox-m might break even or lose slightly because it has enough capacity to learn from real data alone.

4. **Document the prior**: customers should know that the bundled checkpoints come with a convex bias built in. For non-convex objects (e.g., rectilinear parts, complex shapes), they should train from scratch and the docs should say so.

**Definition of done**: `opndet pretrain --synthetic` produces a checkpoint. Bundled checkpoints for all five presets are pre-trained and downloadable. Documentation explains when to use them and when not to.

### 2.5 Real-world unit output

For convex objects of known type, pixel dimensions can be converted to real-world units (mm, grams) with high accuracy given camera calibration. opndet should make this trivial.

**Plan**:

1. **Add a `--units` flag** to predict and decode that takes a calibration object (a JSON or YAML with camera intrinsics, mounting height, and per-class size-to-weight mappings).

2. **Output schema for predictions** becomes a structured JSON with both pixel and physical units: `{x1, y1, x2, y2, score, width_mm, height_mm, mass_g}`. Optional, off by default.

3. **Reference calibrations** for common configurations (overhead conveyor at H mm, with a known fiducial in frame for scale). Ships as part of the docs.

**Definition of done**: predict CLI has `--calibration cal.yaml` flag. Reference calibrations for at least three common configurations ship with the repo. Documented end-to-end pipeline from camera setup to mass output.

---

## Part 3 — Architecture and training improvements

These are general-purpose improvements that apply regardless of the convex specialization. Most are independent of each other; pick them up in any order.

### 3.1 Knowledge distillation — **SHIPPED**

bbox-m teaches bbox-f. The big model produces soft targets, the small model learns from them. For tiny detectors this is genuinely transformational — published distillation papers regularly show 5-10 mAP improvements at fixed param count.

**Status**: shipped via `opndet train --teacher PATH` (or `--self-distill`). Teacher arch is read from its saved `config.model_config` so the student command is self-describing — no `--teacher-arch` flag needed. Path lives only on the CLI to keep yaml reproducible across machines (Colab won't have your local `/content/runs` paths). Hyperparams in `distill: { hm_weight, reg_weight, conf_gate }` yaml block. `train/l_kd_hm` and `train/l_kd_reg` logged to TB.

**Plan**:

1. ✅ **Distillation loss**: KL divergence between teacher's heatmap and student's heatmap, plus L2 between teacher's regression outputs and student's. Weight scales with confidence (high-confidence teacher predictions are higher-weighted in the student loss). *(Shipped: BCE on student `sigmoid(obj_logit)` against teacher's post-peak `obj` at gated cells (where `t_obj > conf_gate`); L1 on (cxy, wh) at the same gated cells. Teacher's calibration temperature, if present, is auto-applied on load.)*

2. ✅ **Training loop**: optional `--teacher PATH` flag in `opndet train`. Teacher runs in eval mode, no gradients. Student trains normally with the additional loss term. *(Shipped via opndet/distill.py.)*

3. **Recipe**: standard practice is teacher = bbox-m, student = anything smaller. Document the recipe and bundle distilled checkpoints alongside the from-scratch checkpoints. *(Recipe: train teacher → calibrate teacher → `opndet train --teacher` for the student. Bundling pre-distilled checkpoints is a release-time concern.)*

4. ✅ **Self-distillation**: bbox-s teaches bbox-s with EMA of itself as the teacher. Often gives 1-2 mAP for free, especially on small datasets. Add as `--self-distill` option. *(Shipped: `--self-distill` reuses the existing EMA shadow as the teacher, no extra memory. Mutually exclusive with `--teacher`.)*

**Definition of done**: distillation is a documented training mode. Distilled bbox-f has ≥3 mAP improvement over scratch-trained bbox-f at the same training budget on a benchmark dataset.

### 3.2 Quantization-aware training (QAT)

The current quantization path is post-training PTQ via onnxruntime. PTQ works fine for bbox-m (parameter slack absorbs quantization error) but loses real accuracy at the femto tier where every parameter counts. QAT closes that gap.

**Plan**:

1. **Integrate Brevitas or torch.ao.quantization** for QAT during training. The architecture is already quant-friendly (ReLU6, no exp, no GroupNorm).

2. **Two-phase training**: float pre-training, then QAT fine-tuning. Add `--qat` flag to enable the second phase.

3. **Export from QAT**: produces an int8 ONNX directly, no separate PTQ step. Validates against the float model on the validation set.

4. **Hardware-specific quantization schemes**: Ethos-U55 wants symmetric per-tensor; Hailo wants symmetric per-channel; Myriad wants asymmetric. Add per-target QAT presets so the resulting int8 model deploys cleanly without further re-quantization.

**Definition of done**: `opndet train --qat` produces an int8-ready model. Per-target presets (`--target ethos-u55`, etc.) produce models that deploy without further quantization tweaks. bbox-f QAT model has ≥2 mAP improvement over PTQ at the same int8 size.

### 3.3 Optional multi-scale head

Single-scale H/4 is a real limitation for scenes with high size variance. The clean way to add multi-scale without breaking the "identical output layout" guarantee for users who don't need it is to make it opt-in.

**Plan**:

1. **Add an optional H/8 head** that produces a separate `[1, 5, H/8, W/8]` output tensor. Models with the multi-scale option have two outputs; models without have one. The output contract for single-scale models is unchanged.

2. **Decoder handles both**: if multi-scale, decode each scale independently and merge (no NMS still — peak suppression at each scale, plus a cross-scale dedup based on center proximity).

3. **GT encoding**: assign each GT to the scale that best matches its size (small objects to H/8, large to H/4). Standard FPN-style assignment.

4. **YAML option**: `multi_scale: true` in the preset config. Default is false (single-scale, current behavior).

**Definition of done**: bbox-s and bbox-m have `multi_scale` variants. Test on a dataset with bimodal size distribution shows that multi-scale variants outperform single-scale by ≥5 mAP on small objects without hurting large-object performance.

### 3.4 Larger preset(s): bbox-l and bbox-x

Discussed in conversation. The current lineup tops out at 2.4M params (bbox-m), which is appropriate for MCU deployment but leaves a gap for OAK / Hailo / Jetson / desktop users.

**Plan**:

1. **bbox-l** at ~6M params, ~6 MB int8. Targets OAK (Myriad X has lots of headroom) and Hailo-8L. Same architecture, scaled-up channels.

2. **bbox-x** at ~11M params, ~11 MB int8. Targets Hailo-8 / Jetson Orin Nano / desktop CPU. Same architecture, more scaled.

3. **Optional SiLU/Swish variant** for bbox-l/x: at this scale, SiLU costs a small accuracy gain over ReLU6, and these targets all support it. Add a `activation: silu` YAML option that's only validated on non-Myriad targets.

4. **Document that bbox-l/x are NOT for MCU**: the README and CLAUDE.md should be explicit that the femto-medium tier is for embedded and the large-extra tier is for accelerator hardware. Different deployment story, different validation requirements.

**Definition of done**: bbox-l and bbox-x presets exist, are tested on at least one accelerator each, and are documented with appropriate "use this for X" guidance.

### 3.5 Training pipeline improvements

A grab-bag of practical improvements that don't fit elsewhere.

1. **Mixup augmentation**: in addition to mosaic, add mixup (linear blend of two images and their labels). Often gives 1-2 mAP on small datasets.

2. **CopyPaste augmentation**: cut objects from one image and paste into another. Especially effective for crowded-scene training. Industrial-relevant.

3. **EMA with bias correction**: the current EMA is plain exponential. Add Karras-style bias-corrected EMA that performs better in early training.

4. **Gradient clipping**: not currently in the training loop. Add `grad_clip_norm` config. Prevents loss spikes at the start of training.

5. **Learning rate finder**: a `--find-lr` mode that runs an LR sweep and reports the optimal LR for the dataset. Saves users from guessing.

6. **Reproducible data splits**: the current split is seeded but document the exact split (which images go in train/val/test) as a JSON manifest in the run directory. Customers re-running need to reproduce splits exactly.

7. **Multi-GPU training via DDP**: currently single-GPU only. Add DDP support for users with multiple GPUs. Standard PyTorch DDP wrapper around the existing loop.

**Definition of done**: each item is independently mergeable. Document each in CLAUDE.md.

---

## Part 4 — Deployment and tooling

These items are about making opndet *easier to use*, not making the model itself better. Equally important for adoption.

### 4.1 Migration story (mostly free once Part 8 ships)

Once Part 8's format auto-detection ships, there's no separate "YOLO migration" work needed — users with YOLO data point `--data` at their directory and it works. What remains is documentation and positioning:

**Plan**:

1. **README "switching from YOLO" section**: side-by-side commands. `yolo train data=data.yaml model=yolov8n.pt epochs=100` becomes `uvx opndet train --data ./my-data`. Make the parallel structure explicit.

2. **Single-class collapse for multi-class datasets**: add `--target-class N` (or `--target-class name`) for users with multi-class annotations who want to detect only one. Filter at load time, no data conversion required.

3. **Worked example in docs**: take a real Roboflow-exported YOLO dataset, run opndet on it directly, ship the ONNX. End-to-end, no manual conversion step anywhere.

**Definition of done**: a YOLO user can switch to opndet with one command. Documented with a worked example. No `convert` subcommand needed.

### 4.2 Roboflow and Edge Impulse integration

These are where the buyers already are. Integration removes the largest activation barrier.

**Plan**:

1. **Roboflow direct download**: `opndet train --roboflow-url <project-url>` pulls and unpacks Roboflow datasets directly. No manual export step.

2. **Edge Impulse export target**: produce a model file that imports cleanly into Edge Impulse Studio for downstream deployment. EI accepts ONNX; the integration is mostly documentation.

3. **Reference notebooks**: Colab notebooks that take a Roboflow dataset, train opndet, export to ONNX, run in-browser. End-to-end in one notebook.

**Definition of done**: Roboflow URL works. EI export documented. Colab notebook exists.

### 4.3 Hardware target validation matrix

The README compatibility table currently says "should work, untested" for several targets. Fill it in.

**Targets to validate**:

- **OpenVINO 2022 Myriad X** (already validated, keep current)
- **OpenVINO 2024+ CPU/GPU** — needs revalidation, opset compatibility may have shifted
- **Ethos-U55 via Vela** — convert ONNX → TFLite → Vela, document any ops that fall back to host CPU
- **Ethos-U85** (newer Arm NPU) — same path as U55, document differences
- **Neural-ART (STM32N6) via X-CUBE-AI** — convert ONNX → TFLite → ST tools, document
- **Hailo-8 / Hailo-8L via Hailo SDK** — direct ONNX import
- **Coral Edge TPU via TFLite** — likely needs full int8 quantization, document path
- **Kendryte K210** (low-cost RISC-V NPU) — popular hobbyist target, worth validating
- **TFLite Micro on Cortex-M (RT1062)** — for comparison with the CMSIS-NN path

**Definition of done**: each row in the compatibility matrix has either a "✓ tested" or a documented reason it doesn't work. Per-target validation script in `scripts/validate_<target>.sh`.

### 4.4 The "industrial opset" specification

This is the meta-artifact that ties everything together. opndet's architectural choices weren't arbitrary — they were dictated by the intersection of what NPUs can run fast. Document that intersection as a standalone spec.

**Plan**:

1. **A markdown document** (`docs/industrial-opset.md`) listing the operators that compile to fast paths on every major edge NPU. With a table of which ops are supported on which targets, with notes on quirks.

2. **A validation script**: `industrial-opset-check model.onnx` that takes any ONNX model and reports which ops will run fast on which targets. Useful to anyone deploying ML at the edge, not just opndet users.

3. **Versioned**: industrial opset v1 is what opndet currently targets. Future versions can add ops as they become widely supported (e.g., when Ethos-U85 brings new ops, document a v2 that uses them).

**Definition of done**: spec document exists, validation script works on arbitrary ONNX files, both are linked from the README.

### 4.5 Documentation overhaul

The current README is solid for a developer audience but assumes too much. The DESIGN.md is gold but invisible. Break out user-focused docs.

**Plan**:

1. **Quickstart that produces a working model in 10 minutes**: synthetic data, train, predict, export. End-to-end. Linked from the top of the README.

2. **"Choosing a preset" guide**: decision tree based on hardware target and accuracy requirements. Currently buried in the table.

3. **"Deploying opndet" guide**: per-target deployment instructions (OAK, Hailo, RT1062, AE3, N6, Coral, Jetson, browser). Each target gets its own short doc.

4. **"Training tips" guide**: how to choose augmentation parameters, when to use mosaic vs not, how to handle imbalanced size distributions, when to enable VFL, when to use the convexity prior. The hard-won knowledge that's currently in headers and comments.

5. **A FAQ**: "why no NMS?", "why opset 13?", "why single-class?", "how do I handle multiple classes?", "how does this compare to YOLO?". The questions that come up every time.

**Definition of done**: docs/ directory restructured, each guide is its own page, README links to the guides.

---

## Part 5 — The platform play (longer-horizon)

Discussed at length in conversation. opndet is one model. The unifying insight — opset-constrained model design for industrial single-class deployment — generalizes. This is the platform direction; treat it as a separate product roadmap, not part of opndet itself.

Adjacent models that fit the same opset constraint and the same single-class industrial niche:

- **opndet-seg**: single-class semantic segmentation. Same opset, same training infrastructure, different head. For binary defect-vs-not segmentation.
- **opndet-anom**: single-class anomaly detection. Reconstruction-based, trained on "normal" examples. For predictive maintenance.
- **opndet-key**: keypoint localization. Detects N keypoints per object. For robotic pick-and-place fiducials.
- **opndet-ocr**: known-font industrial OCR. For lot codes, expiration dates, serial numbers.
- **opndet-flow**: fixed-camera optical flow. For motion detection, tripwire counting.

Each is its own project. Each shares opndet's opset, training pipeline, license, and Beige Systems voice. The unifying brand is "permissively-licensed industrial vision stack for edge NPUs."

This is not for tomorrow. This is the framing that keeps the whole roadmap coherent: opndet is the first piece, not the only piece. Decisions about opndet (license, opset, voice, naming convention) should be made with this in mind.

---

## Part 6 — Sequential data and self-supervised fine-tuning

The current dataset is non-sequential (random snapshots). For an always-on counter, the natural deployment is video, and there are real capability gains available from sequential data that snapshots structurally cannot provide. This work comes after the snapshot model is genuinely good (Parts 1-3) — adding temporal data before that just amplifies errors.

### 6.1 Sequential data pipeline support

Add support for video clips (MP4) and image sequences alongside the existing snapshot datasets. Same model trains on both; the data pipeline distinguishes them.

**Plan**:

1. **Storage format**: extend the existing COCO-style annotations with `video_id` (string or null) and `frame_idx` (int or null) fields. Snapshots have both null. Sequential frames share a `video_id` and have monotonic `frame_idx`. Auto-discover knows the difference.

2. **MP4 loader**: stream frames from MP4 files using `torchvision.io.read_video` or `decord`. Don't decode everything to disk — for long deployment recordings, that's prohibitive. Random-access by frame index.

3. **Image-sequence loader**: directory of numbered frames + a single annotations file with frame indices. Useful for cases where MP4 compression artifacts would hurt training.

4. **Sparse keyframe support**: for slow-moving conveyor scenes, label every Nth frame and linearly interpolate boxes between keyframes. Trivial preprocessing step. For fast or non-linear motion, denser annotation is needed.

5. **Pseudo-label propagation tool**: `opndet propagate --clip clip.mp4 --keyframes 0,30,60 --labels keyframes.json --out propagated.json`. Runs an existing model on the clip, uses keyframe labels as anchors, propagates labels to intermediate frames. Manual review on a sample before training.

**Definition of done**: opndet can train on a mixed dataset of snapshots + video clips. Sparse-keyframe interpolation works. Pseudo-label propagation produces usable labels for unlabeled frames.

### 6.2 Self-supervised fine-tuning on deployment video

Once a base model is trained on snapshots, deployment video can fine-tune it without manual labeling.

**Plan**:

1. **EMA-teacher pseudo-labels**: run the EMA teacher on unlabeled frames, use its high-confidence predictions as pseudo-labels for the student. Confidence-gate at e.g., 0.8 — only use predictions the teacher is sure about.

2. **Hard-example mining via self-disagreement**: frames where the model's predictions disagree with its own predictions on adjacent frames are exactly the hard cases. Sample these preferentially in fine-tuning.

3. **Domain adaptation pipeline**: customer points at "a folder of recordings from my line," opndet fine-tunes the base model on that footage with no manual labels needed. This is a real product feature: "opndet adapts to your specific deployment."

**Definition of done**: a 30-second deployment recording, fed to `opndet adapt`, produces a fine-tuned model that measurably outperforms the base model on that specific scene. Documented as a deployment workflow.

---

## Part 7 — Temporal prior input (center stabilization via short tails)

This is the major capability after sequential data lands. It addresses the single most common deployment complaint — frame-to-frame center jitter and confidence flapping — by feeding the model a compact heatmap of recent detection footprints.

**The goal is center stabilization, full stop.** Center stability is roughly an order of magnitude more important than any other temporal property for industrial counting and grading applications. Bbox dimensions, confidence trajectory, and orientation can all be smoothed application-side by averaging across frames; center jitter is much harder to fix downstream, because it directly affects which pixel a detection lives at. Get the center right; everything else follows.

Eggs on a conveyor don't need motion extrapolation — the snapshot detector finds them in roughly the same place every frame anyway. What it doesn't have is *consistency* — center, confidence, and bbox dimensions vary frame-to-frame even when the underlying object is stable. The temporal prior fixes the center first; everything else is downstream of that.

The architecture is deliberately simple: stamp each detection's bounding box at amplitude 1.0; fade the entire accumulator by 1/N each frame. The model gets one extra input channel (presence), trained to read it as a soft prior on where centers should lock. No long contrails, no motion vectors, no trajectory math, no size memory, no embeddings.

The model is fade-invariant by design: same checkpoint deploys at any frame rate because the accumulator parameters live in the application, not the graph.

### 7.1 The architecture

The model gains an optional 4th input channel carrying the temporal prior. Same checkpoint, two export modes:

**Snapshot mode** (single-input, fully backward compatible):
```
Input:  current_RGB         [1, 3, 384, 512]
Output: detections          [1, 5, 96, 128]
```

**Temporal mode** (dual-input, presence-only):
```
Input 1: current_RGB        [1, 3, 384, 512]
Input 2: prior_heatmap      [1, 1, 96, 128]    # presence channel
Output:  detections         [1, 5, 96, 128]
```

**The default is K=1: presence only.** This is what stabilizes the centers, which is the dominant cause of customer pain (flapping, frame-to-frame jitter, count instability). Adding more channels (size memory, rotation memory) costs measurably more compute and memory bandwidth at the stem layer. For most industrial counting and grading applications, **center stability is roughly an order of magnitude more important than dimension stability**, and dimension stability can be addressed downstream by averaging across frames in the application layer.

The prior_heatmap channel can come from any source:
- Zero tensor (first frame, or stateless mode — degrades to snapshot)
- The application's tail accumulator (default temporal use)
- A user-supplied tracker output
- A static ROI mask
- A heatmap from any other model

The model has been trained to treat it as a soft prior. Where the prior is hot, lock detection centers tightly to that location; where it's cold, detect from RGB alone.

**Architectural changes**: stem conv accepts 4 channels instead of 3. Backbone, neck, head, peak suppression — all unchanged. Single extra weight tensor at the very front. Fully opset-13 safe. No LSTMs, no recurrent state, no special ops. Pure CNN throughout.

**Optional extension: size and rotation memory (deferred)**

For applications that need explicit bbox dimension stability beyond what averaging provides — robotic pick-and-place, precise sizing-for-grading, oriented-bbox tracking — additional prior channels are possible:

- **K=3** (presence + w + h memory): adds explicit dimension stabilization
- **K=5** (presence + w + h + sin(2θ) + cos(2θ) memory): adds rotation stabilization for oriented bbox mode

Each additional channel is roughly a 1-2% inference cost (more multiply-adds in the stem conv, more memory bandwidth at the input). For K=3 vs K=1, expect ~3-5% total inference cost on small models (where the stem is a larger fraction of total compute), proportionally less on larger models.

**Defer K=3 and K=5 until validation shows real customer demand.** The pattern: ship K=1 first, validate that center stabilization solves 90%+ of customer complaints, then add K=3 if specific applications request it. Don't pay the cost upfront for a feature that may not be needed.

### 7.2 The tail accumulator (application-side)

O(1) RAM regardless of N. The accumulator is a single small fixed-size tensor (~50 KB at H/4 = 96×128 fp32); the fade-step `1/N` controls how fast detections persist, but the memory cost is constant. **N can be anywhere from 2 to 100+ depending on deployment** — frame rate, object speed, detection reliability, and ghost-tolerance all factor in.

**Reference implementation (presence-only, the default):**

```python
class TailAccumulator:
    """O(1) tail accumulator for center stabilization.

    Single-channel presence map with bbox-footprint stamping and linear fade.
    Six lines of logic. Naturally bounded in [0, 1].

    N is a deployment-time knob with no architectural cost. Default 3 is
    appropriate for fast moving conveyor lines; values up to 100+ are fine
    for slow scenes where you want long memory through detection dropouts.
    """
    def __init__(self, shape, n_frames=3):
        self.acc = np.zeros(shape, dtype=np.float32)
        self.fade_step = 1.0 / n_frames

    def update(self, detections):
        # 1. Fade entire accumulator by 1/N (clip at 0)
        self.acc = np.maximum(self.acc - self.fade_step, 0.0)
        # 2. Stamp current detections as bbox footprints (max-merge)
        new_heatmap = render_bbox_footprints(detections, self.acc.shape, peak=1.0)
        self.acc = np.maximum(self.acc, new_heatmap)
        return self.acc
```

That's the entire algorithm. Six lines. No blur, no normalization, no ring buffer.

**Why bbox-footprint rendering (not point Gaussians)**:

Stamping the bounding box (or a soft anisotropic Gaussian sized to the bbox) instead of a fixed-sigma point peak gives the model a prior structurally similar to its own output. The model produces bboxes; the prior shows recent bbox footprints. This is more directly informative than peak heatmaps and naturally adapts to object size — a 30 px egg gets a 30 px footprint, a 100 px fish gets a 100 px footprint, no per-deployment sigma tuning required.

For deployment, hard bbox stamping (a 1.0-filled rectangle inside the box) is fine and faster than the soft variant.

For training-time use of the accumulator (where differentiability matters for self-distillation), a soft anisotropic Gaussian sized to the bbox works:
```python
def render_bbox_soft(box, shape, edge_sharpness=2.0):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = x2-x1, y2-y1
    return gaussian_2d(shape, cx, cy,
                       sigma_x=w/edge_sharpness,
                       sigma_y=h/edge_sharpness)
```

**Subtract first, then stamp** is the correct order: keeps current detections always at full amplitude even at stationary positions.

**N is purely tunable, no architectural cost**:

| Scenario | Recommended N | Coverage |
|---|---|---|
| Fast conveyor at 60 fps, well-detected | 3-5 | 50-83 ms memory |
| Standard inspection at 30 fps | 5-10 | 167-333 ms memory |
| Slow line with detection dropouts | 20-50 | 0.7-1.7 s memory at 30 fps |
| Stationary objects, anti-flicker only | 50-100 | 1.7-3.3 s memory at 30 fps |

The model is trained on randomized N from 1-32 (covers most cases) and extrapolates well to higher N because the visual pattern (stable footprint with occasional re-stamping) doesn't change shape.

**Optional: size memory accumulator (K=3 mode, deferred)**

If validation shows that bbox dimension stabilization is needed beyond what application-side averaging provides, the accumulator can be extended with two additional channels carrying the (w, h) of recent detections at active pixels. Same fade behavior, same stamp logic, just more channels:

```python
# K=3 variant (sketch — implement when needed)
self.presence = np.zeros(shape, dtype=np.float32)
self.size_w = np.zeros(shape, dtype=np.float32)  # w_norm at active pixels
self.size_h = np.zeros(shape, dtype=np.float32)  # h_norm at active pixels
# fade all three; stamp w/h where new presence is high
```

The model checkpoint can support either K=1 or K=3 deployment if trained with both modes randomized. Cost is ~3-5% inference overhead on small models. **Don't ship this until customers ask for it** — center stabilization is roughly 10× more important than dimension stabilization for industrial counting and grading, and bbox dimensions can be smoothed application-side by averaging across frames.

**Subtract first, then stamp** is the correct order: keeps current detections always at full amplitude even at stationary positions.

### 7.3 What the model actually learns

With short tails and bbox-footprint stamping, the model's job is much simpler than "interpret motion contrails." It's:

1. **Read the prior**: where the accumulator has high values, recent detections happened
2. **Confirm with RGB**: is there visual evidence of an object at the prior's hot region right now?
3. **Lock the center tightly**: if both prior and RGB agree, predict with high confidence at exactly the same center as last frame (or wherever RGB indicates)
4. **Use bbox dimensions from prior**: if the prior shows a 30 px footprint, predict a 30 px bbox (don't suddenly predict 25 or 35)
5. **Fall back to RGB-only**: when the prior is cold (new object, first frame, or after long absence), detect from RGB alone

The model never has to interpret long trails or extrapolate forward. The information it needs is *local*: "is there a recent footprint near this pixel? Yes? Lock onto it. No? Detect normally."

This is well within what a CNN with a few-pixel receptive field can learn. The supervision signal naturally teaches it: training pairs are `(RGB_t, prior_built_from_GT_at_t-1, t-2, t-3, supervision = GT_t)`. The gradient pushes the model to use the prior to reduce center jitter relative to recent positions.

### 7.4 Training procedure

Same data pipeline as Part 6 (sequential video with bbox annotations). Build the prior from previous frames' GT using the same subtract-then-stamp logic the application uses:

```python
For each training sample (target frame t):
    N = random.randint(0, 6)              # randomize tail length
    fade_step = 1.0 / max(N, 1)

    # Build prior from previous N frames' GT (NOT current frame)
    prior = zeros(H/4, W/4)
    if N > 0:
        for k in range(N, 0, -1):         # iterate oldest to newest
            if t - k >= 0:
                gt_footprint = render_bbox_footprints(boxes_at[t-k], shape=(H/4, W/4))
                prior = maximum(prior - fade_step, 0)
                prior = maximum(prior, gt_footprint)

    yield (frame[t], prior, boxes_at[t])
```

**Randomized N during training is essential.** The model has to handle:
- N=0 (no prior, snapshot mode) → must still detect from RGB alone
- N=1 (single frame footprint, no fade) → simple anchor
- N=3 (typical) → small tail
- N=6 (longer tail for slower scenes) → still useful for stabilization

Range from 0 to 6 covers all reasonable deployment scenarios. By exposing the model to all these variations, the deployment-time fade rate can be anything reasonable.

**Prior-dropout for snapshot compatibility**: 30-50% of training samples use N=0 (zero prior). This ensures the same checkpoint works in pure snapshot mode without degradation, required for backward compatibility with the existing single-input deployment path.

**Critical bug to avoid**: never build the training prior from the *current* frame's GT. That teaches the model to copy the prior, which works at training time and fails catastrophically at inference. Bake assertions into the data pipeline.

### 7.5 The first-frame question

What happens on the first frame, when there's no history but objects may be present in the scene?

**They get detected normally.** The prior channel is zero everywhere, which adds no signal. The model processes RGB and produces detections from it alone — performance is identical to the snapshot baseline (assuming proper prior-dropout training).

The temporal benefit kicks in starting at frame 2, when the prior reflects frame 1's detections. Temporal mode is *strictly better than or equal to* snapshot mode at every frame, never worse.

### 7.6 Deployment knobs (all tunable without retraining)

| Parameter | Controls | Default | Tune for |
|---|---|---|---|
| `n_frames` (or `fade_step = 1/N`) | How many frames a footprint persists | 3 | Higher for slow/stationary scenes (up to 100+); lower for fast or chaotic scenes |
| `prior_weight` | Multiplier on prior before feeding to model | 1.0 | Lower if seeing ghost detections; higher if seeing flapping |

The `prior_weight` knob is the most useful at deployment: lets integrators dial the prior's influence up or down without retraining. No model changes required.

`n_frames` is set once per camera based on observed conditions and stays fixed during operation. The model is trained on a wide range so any reasonable choice works.

### 7.7 Edge cases (and why they mostly don't matter)

**Crossings (one object's tail intersects another's)**: with N=3, by the time two objects' tails would cross, the older positions have faded substantially. The crossing artifact is brief (fades in 3 frames), localized (only affects pixels actually crossed), and rarely produces phantom detections (RGB doesn't support a detection at a fading crossing pixel after both objects have moved on). For most industrial scenes this is a non-issue.

**Highways (dense moving packs)**: with N=3 and bbox-footprint stamping, adjacent objects' tails do overlap at edges but each object's recent-frame footprint dominates its center pixel. The accumulator preserves per-object structure — it's not a uniform lane. The model can read individual peaks within the highway via the same logic as sparse scenes.

**Object becomes briefly occluded (1-2 frames)**: tail amplitude drops by 1/N to 2/N at the occluded center. Still warm enough that the model treats the location as "object expected here." When RGB recovers, detection re-locks. Brief occlusion is handled gracefully.

**Object permanently leaves**: tail fades to zero over N frames. After that, accumulator forgets. Nothing accumulates over time.

**Object stops moving**: every frame, the same bbox gets stamped. The accumulator stays at 1.0 indefinitely at the object's location. This is correct — a stationary object should produce a stable, persistent prior.

### 7.8 Counting and tracking integration

The temporal prior stabilizes detections but does not assign persistent identity. For applications that need ID-stable tracking — e.g., counting objects crossing a line, where the standard pattern is "track centroid → detect line crossing → increment counter" — opndet temporal mode pairs cleanly with existing trackers.

**The recommended deployment pattern:**

1. opndet temporal mode produces stable, locked-on detections per frame
2. ByteTrack / SORT / DepthAI ObjectTracker assigns persistent IDs to those detections
3. Application-layer logic counts ID crossings of a configured line or zone

**Why this division of labor works**: tracking-by-detection algorithms work *much* better when fed stable detections than jittery ones. Center jitter, bbox dimension wobble, and confidence flapping all degrade tracker performance. The temporal prior addresses each of these directly. ByteTrack on stabilized opndet detections should significantly outperform ByteTrack on snapshot opndet detections, even though the tracker code is identical.

**DepthAI integration** specifically: DepthAI's `ObjectTracker` node consumes detections and produces tracked objects with persistent IDs. The integration is:

```python
# DepthAI pipeline (host side)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("opndet_temporal.blob")
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)

# Maintain temporal prior in Python (host side)
accumulator = TailAccumulator(shape=(96, 128), n_frames=10)
while True:
    rgb = get_frame()
    prior = accumulator.update(prev_detections)
    detections = run_nn(rgb, prior)
    tracked = tracker.process(detections)
    counter.update(tracked)  # standard line-crossing counter
    prev_detections = detections
```

The accumulator runs on the host CPU; the model runs on Myriad. The host work is trivial (a few hundred microseconds for the accumulator update). The deployment story remains "ONNX/blob file plus tiny host code."

**Future direction: in-model identity (deferred)**

Encoding tracking signals into the model itself — i.e., having the model produce identity-stable detections without an external tracker — requires per-pixel embedding outputs and identity-labeled training data. The architectural sketch:

- Add embedding output channels (~16-D) per detection
- Train with identity labels (significantly more expensive to annotate)
- Application matches new detection embeddings against recent track embeddings

This is a meaningful capability addition but a separate project. For now, the temporal prior + external tracker pattern is the right deployment story. If validation reveals that ByteTrack still struggles in dense pack scenarios with very similar-looking objects (e.g., visually identical eggs in a tight grid), then in-model identity becomes worth pursuing. Otherwise, defer.

### 7.9 Validation

Before promoting temporal mode to a default, verify the model uses the prior correctly.

**Tests**:

1. **Zero-prior == snapshot baseline**: with prior=0, temporal model performs identically to snapshot model. Validates backward compatibility.

2. **Center stabilization** (the primary success metric): across short synthetic clips (or perturbation-augmented snapshots), measure frame-to-frame variance in detection center for the same physical object. Temporal mode should reduce center jitter significantly compared to snapshot. **This is the metric we're optimizing for** — center stability is roughly 10× more important than any other temporal property for industrial counting.

3. **Confidence stability**: same setup, measure frame-to-frame variance in confidence for the same physical object. Temporal mode should produce smoother confidence trajectories (less flapping at borderline cases).

4. **Wrong prior doesn't break it**: with prior shifted by a random offset, temporal model degrades gracefully toward snapshot performance, not phantom detections.

5. **Self-feedback stability**: feeding the model's own predictions back as prior across a long video sequence should converge to stable counts, not drift.

6. **N-invariance**: same model at different fade rates (N=2, 5, 10, 30, 100) should all produce reasonable results. Validates that runtime N-tuning works without retraining across the full deployment range.

7. **Highway preservation**: synthetic dense-pack data (8+ objects packed close together, moving uniformly) should still produce per-object detections, not blob predictions. Validates that the temporal prior preserves structure in the dense-pack case.

8. **Tracker pairing**: ByteTrack on temporal-mode opndet detections should produce more stable track IDs (fewer ID switches per object lifetime) than ByteTrack on snapshot-mode detections. Validates the integration story.

**Secondary metrics (measured but not primary):**

- Bbox dimension stability (w, h variance frame-to-frame). Improvement here is welcome but not required from K=1 mode; if customers need explicit dimension stability, that's the trigger to add the K=3 size-memory variant.
- Per-frame mAP/F1: should not regress vs snapshot mode. Temporal prior should never make detection *worse* on a frame-by-frame basis.

**Definition of done for Part 7 as a whole**: model accepts optional 4-channel input (3 RGB + 1 prior), snapshot mode is bit-exact compatible with the single-input model, temporal mode measurably reduces frame-to-frame center jitter and confidence variance on representative test sequences, deployment knobs (N, prior_weight) are documented, reference accumulator and DepthAI integration example ship as part of the SDK.

This is the architectural commitment that solves the single most common deployment complaint — center jitter and confidence flapping — without the complexity of separate tracking software, and without paying for capabilities (size memory, rotation memory, in-model identity) that customers haven't yet asked for. Pair with ByteTrack or DepthAI's tracker for ID-stable counting; the model handles the per-frame center stability that makes downstream tracking work well.

---

## Part 8 — UX polish (after the model is undeniably better)

The current setup is already trivially easy: write a short yaml, point at COCO json + image dir, run `opndet train`. The auto-split, single-class collapse, and multi-source merge already work. The marginal improvement from "5 minutes of yaml" to "0 minutes of yaml" is real but small compared to the marginal improvement from "good detector" to "industrial-grade detector that doesn't flap and handles touching objects."

So this work comes *after* the capability work, not before. When the model is undeniably better than YOLO on its niche, polish the onboarding experience.

### 6.1 Format auto-detection beyond COCO

Real users show up with YOLO txt files, Pascal VOC xml, LabelMe json, CVAT and Roboflow exports. Each is a small parser; the dispatch is the only new logic.

**Plan**:

1. Add `src/opndet/formats/` with one parser per format:
   - `coco.py` (extract from existing `load_coco_single_class`)
   - `yolo.py` (txt files, one per image, normalized xywh)
   - `voc.py` (xml files, Pascal VOC)
   - `labelme.py` (json per image)

2. A `detect_format(path)` function that peeks at one label file and dispatches. Detection is trivial: `.txt` with normalized floats → YOLO; `.xml` with `<annotation>` root → VOC; `.json` with `instances.json`-style schema → COCO; `.json` per-image with `shapes` → LabelMe.

3. All parsers produce the same `list[Sample]` output. Drop-in replacement for `load_coco_single_class` everywhere.

**Definition of done**: opndet can train on a directory of YOLO-format data without conversion. Format is auto-detected and printed at startup.

### 6.2 Directory-as-data

Convenience layer over the existing `data.sources` config. Currently the config requires explicit paths; `--data path/to/directory` walks the tree and figures it out.

**Plan**:

1. Add `auto_discover(path: Path)` that:
   - If `path` is a json file: treat as COCO, find images relative to it (try common subdirs: `images/`, `imgs/`, `.`, parent dir)
   - If `path` is a directory: walk it, find images by extension, find sibling label files by stem matching, detect format
   - Returns the same `list[Sample]` as the existing loaders

2. The existing `data.sources` config still works — it's the explicit, multi-source path. `--data` is the convenience for the common single-source case.

**Definition of done**: `opndet train --data ./my-folder` works for any of the common dataset layouts, with any of the supported formats.

### 6.3 Zero-config CLI defaults

Make `--config` optional. If omitted, use the bundled `train.yaml` as-is. CLI flags override values in the loaded config.

**Plan**:

1. If `--config` is omitted, load the bundled `train.yaml`.
2. If `--data` is given, override `data.sources`.
3. If `--model` is given, override `train.model` (default: `bbox-s`).
4. Result: `uvx opndet train --data ./my-eggs` is a complete invocation. No yaml required.
5. Print every auto-decision at startup so the user sees what was detected and what was defaulted. Save the resolved config to the run directory for reproducibility.

**Definition of done**: `uvx opndet train --data ./my-eggs` works end-to-end with no config file. All auto-decisions are visible and overridable.

---

## Priorities for the next session

If picking one cluster to push hard on first, the order of impact is roughly:

1. ~~**The validation suite (Part 1.4)** — this comes first because without honest metrics, you can't tell if any other improvement actually helps. Counting accuracy, Hungarian matching, calibration, per-domain stratification, and the confidence-IoU diagnostic. Standard mAP is misleading for this use case. *Build the right scoreboard before playing the game.*~~ **SHIPPED.** Per-domain stratification deferred (no domain tags in data sources yet).
2. **Confidence stability + VFL-as-default (Part 1.1)** — VFL is one config flip and probably reduces flapping immediately. Perturbation-stability metric tells you whether it actually worked.
3. **Industrial-grade loss formulation (Part 1.5)** — count-aware loss term in particular. Most detection losses are per-cell; adding a global count term is a small change with potentially large impact for industrial counting.
4. **Touching-object disambiguation (Part 1.2)** — biggest residual accuracy gap. Promoting the `dist` head is mostly executing on existing design.
5. **Distance-transform output (Part 2.2)** — strictly better for the convex regime, addresses the most common failure mode.
6. **Synthetic convex pre-training (Part 2.4)** — biggest win for the femto tier, biggest brand differentiator.
7. **Distillation (Part 3.1)** — free accuracy improvement at the small end.
8. **Sequential data pipeline (Part 6.1)** — prerequisite for temporal mode. Mechanical work, but unlocks Part 7.
9. **Temporal prior input (Part 7)** — the major capability that turns opndet from "tiny detector" into "tiny detector + tracker in one ONNX file." Do this once snapshot model is undeniably good and sequential data is supported.
10. **UX polish (Part 8)** — only after the model is undeniably better. Setting up a yaml is already trivial.

Items in Part 4 (deployment and tooling) are urgent but different work — they're growth-driving rather than capability-driving. Run them in parallel as the model improves.

The Part 5 platform direction should be in the back of the mind for every decision but is not the immediate work.

**A note on agent pace**: most items here are agent-day-of-work scale, not multi-week sprints. The roadmap was originally sized for human pace; re-read it priced for agent pace and the whole thing looks more like a two-week sprint than a six-month plan. The actual bottleneck is human-judgment items: the rename decision, the convex commitment, the launch positioning. Those are not on this roadmap because they're not engineering — they're decisions for you.

**A note on dataset shape**: the current dataset is non-sequential (random snapshots, not video). True frame-to-frame stability cannot be measured directly; we use perturbation-based proxies. If/when sequential data becomes available, true temporal stability becomes a first-class metric. Track-and-count logic (hysteresis, N-of-M confirmation) is deployment-side and lives separately from the model.

---

## Notes on engineering style

Carry the existing codebase conventions forward:

- Terse code, no docstrings unless WHY is non-obvious
- Conventional Commits format for messages
- Roundtrip parity tests for any new graph op (PyTorch vs ORT max diff < 1e-4)
- Bundled YAML for any new preset, single source of truth in `src/opndet/configs/`
- New blocks register via `@register("Name")` in `registry.py`
- Augmentation must respect `min_visible_frac`
- Don't break the single-tensor output contract for users who depend on it; new outputs are opt-in via YAML

The opset constraint is load-bearing. Every new op proposal needs to go through `ALLOWED_OPS` in `export.py` first. If an op isn't in opset 13 or doesn't compile cleanly to Myriad/Ethos/Neural-ART, it doesn't go in. Period. This is what makes opndet what it is.
