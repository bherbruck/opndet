# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Tiny single-class object detector. **OpenVINO 2022 opset compatible (ONNX opset ≤13). No NMS, no postprocessing** — peak suppression is baked into the ONNX graph. Architecture is a CenterNet-style heatmap head + dense regression in a 5-channel single output tensor `[1, 5, H/4, W/4]` = `(obj_peak, cx, cy, w, h)`.

**The end goal is multi-object TRACKING across video frames.** Detection (and counting) are means, not ends — *count is a byproduct of tracking* (number of distinct tracks), never a primary target. So every model and design choice should optimize for what makes tracking work: **frame-to-frame stability/consistency** of detections (no flapping, no jitter — see `eval --stability`, the `-tp` temporal-prior variants) and **accurate object extent/sizing** (tight, stable boxes/masks). Get those right and count, association, and re-ID fall out naturally — especially for touching/clustered objects. When picking metrics or losses, weigh stability + size accuracy above raw count or center-only F1. (NB: keep the user's actual application domain *out* of code, docs, and commits — say "object", never name the subject; see the memory file `feedback_no_pii_in_commits.md`.)

See `DESIGN.md` for the full architectural rationale and the converged spec.

## Commands

```bash
# Install (editable, dev)
uv venv && uv pip install -e .

# Run from a fresh checkout, no install
uvx --from . opndet info

# Common subcommands
opndet info                            # list bundled presets
opndet info bbox-s                     # inspect a preset
opndet init-config --out my.yaml       # dump training config template
opndet train --config my.yaml [--run-name <name>] [--resume <path>]
opndet predict --image foo.jpg --model bbox-s --ckpt best.pt --save vis.jpg
opndet export --model bbox-s --ckpt best.pt --out opndet.onnx [--bake-input-norm]
opndet export --model bbox-s --ckpt best.pt --out diag.onnx --diagnostic   # +named-layer outputs for the webui's explain mode
opndet quantize --onnx opndet.onnx --calib data/imgs --out opndet_int8.onnx [--verify]
opndet calibrate --ckpt best.pt --config train.yaml                         # bake Platt T into the ckpt
opndet eval --ckpt best.pt --config train.yaml [--stability]                # full report; --stability runs perturbation flapping check
opndet analyze --ckpt best.pt --model bbox-s --image object.jpg                # interpretability: per-layer slider + Grad-CAM HTML
opndet dashboard --root /path/to/runs                                        # DuckDB-backed live training dashboard

# Tests
.venv/bin/pytest tests/                                  # full suite (sparse — most coverage is via integration tests)
.venv/bin/pytest tests/test_temporal_prior.py -xvs       # single test file
.venv/bin/pytest tests/test_temporal_prior.py::TestName::test_method -xvs   # single test method
```

The `--model` flag accepts a **bundled preset name** (`bbox-f`, `bbox-p`, `bbox-n`, `bbox-s`, `bbox-m`, `bbox-l`, `bbox-x`, plus variants `-dist`, `-hm2`, `-flow`, `-tp`) OR a path to a YAML.

**In-browser tester**: drop any exported ONNX into [bherbruck.github.io/opndet](https://bherbruck.github.io/opndet) (lives in `docs/index.html`, deployed via GH Pages). Runs entirely client-side via onnxruntime-web. It auto-detects the output shape: a 5-ch `[1,5,H/4,W/4]` (or 6-ch OBB) → box decode + draw; a **1-ch `[1,1,H,W]`** → treated as a `bbox-*-seg` dense dome → Turbo heatmap overlay + a JS flood-fill connected-components → per-blob centroid + pixel-area, with the threshold slider as the dome foreground cut. The "explain mode" toggle requires an ONNX exported with `--diagnostic` (works for seg presets too — the diagnostic outputs are the decoder layers).

## Architecture (big picture)

### The hard constraint that drives everything

**ONNX opset ≤13 + OpenVINO 2022 op support.** This is checked in `src/opndet/export.py::ALLOWED_OPS`. Every architectural decision filters through "does this op exist and behave correctly in opset 13?" Forbidden: GroupNorm, GridSample, ScatterND, SiLU as a fused op, Resize with `coordinate_transformation_mode=half_pixel`, dynamic shapes. Allowed: Conv, BN, ReLU/ReLU6 (Clip), Add/Mul/Sub, Concat, Resize-nearest-asymmetric, MaxPool, Sigmoid, Tanh, Equal, GreaterOrEqual, Cast, Slice/Split.

When adding any new op or block, add a roundtrip parity test: build the model, export with `opset_version=13, dynamo=False`, run both PyTorch and ORT on the same input, assert max diff <1e-4. The `verify_onnx()` helper in `export.py` does this.

### The "no NMS" trick

`PeakSuppress` (in `src/opndet/primitives.py` and `src/opndet/model.py`) is the centerpiece. It runs `mask = clip((hm + eps - MaxPoolKxK(hm)) * BIG, 0, 1)` (arith mode, default) or `(hm + eps >= MaxPool(hm))` (compare mode, smaller graph but breaks on Myriad VPU). Then `out = hm * mask` inside the graph. Local-max suppression that's ONNX-friendly. The **eps is critical** (currently 5e-3 for fp16 tolerance): without it, FP precision differences between PyTorch and ORT flip mask cells at boundaries and break parity. Don't remove it. Don't lower it without re-running parity tests across all model sizes.

**Per-preset peak_kernel:** most presets use `k=5` (radius 2, 8-px min spacing at stride=4). `bbox-x` uses `k=7` (radius 3, 12-px min spacing) since it targets quality-first deployment with more bbox-shape headroom — the wider window suppresses near-tie adjacent duplicates that the smaller kernel lets through. Each preset's `SigmoidPeakSuppress` block sets the `k` arg explicitly in `src/opndet/configs/opndet-bbox-*.yaml`.

**Two-cell ties survive any kernel size.** If two adjacent cells round to identical sigmoid output (common at saturation in fp16, or when the true object center lands on a cell boundary so the Gaussian/ellipse target is ~equal on both), they BOTH pass `>= MaxPool` since each IS the local max. Three escalating defenses, all zero inference cost: (1) **`hm_peak_moat`** (encode-side, the structural fix) — carve a `≥moat` gap into the GT around every peak cell so non-peak cells within `peak_kernel//2` are forced `≤ peak − moat`; a converged model then lands the neighbor well below the peak → PeakSuppress fully zeros it (`encode.py::_peak_moat`; two genuinely-adjacent objects keep both peaks — peaks don't moat each other). (2) **`peak_sharpen_weight`** (loss-side) — actively penalizes any neighbor of a GT-center cell within `peak_sharpen_margin` of it (`loss.py::peak_sharpen_loss`); belt-and-braces with the moat. (3) **repulsion loss** — soft pressure on the box head (no-op on the OBB/ltrb heads). If duplicates *still* persist, the data layer (hard-negative mining; see ROADMAP §1.7) is the next attack surface — pruning or larger kernels won't fix tied-pair survival.

### Two ways to define a model

1. **Hand-coded**: `src/opndet/model.py::OpndetBbox` + `src/opndet/blocks.py`. Older, kept as a reference. Uses `ModelConfig` from `config.py`.
2. **YAML DSL** (preferred): `src/opndet/yaml_build.py::build_model_from_yaml(path)`. Layer-graph syntax with named aliases and `from:` references (negative ints, names, lists for multi-input modules). New blocks register via `@register("Name")` in `src/opndet/registry.py`. All bundled presets in `src/opndet/configs/` use this path.

`YamlModel.forward()` returns `dict[str, Tensor]`. `forward_with_alias(x, name)` returns the cached output of any named layer — used by `train.py` to grab pre-activation logits at the `raw` layer for loss computation while keeping the inference graph (sigmoid + peak) separate.

### Training data flow

```
COCO json + image dir(s)
  → load_datasets        # multi-source merge, single-class collapse
  → split_samples        # deterministic seeded shuffle, train/val/test
  → OpndetDataset        # __getitem__ does: load → mosaic? → aug → letterbox → normalize → encode_targets
  → DataLoader           # workers do everything above in parallel
  → train loop           # forward_with_alias(raw) → loss → backward → step
```

GT encoding (`src/opndet/encode.py::encode_targets`) runs **inside dataloader workers**, not in the main thread. This is critical for GPU utilization. If you add new target tensors, plumb them through `__getitem__`'s return tuple AND the `collate()` fn.

The dataset returns a 3-tuple `(img_tensor, boxes_xyxy, target_dict)`. `collate()` stacks images and targets, keeps boxes as a list (variable length).

### Augmentation contract

- **Photometric** (`src/opndet/augment.py::_photometric`): brightness, contrast, gamma, hue, saturation, grayscale, blur, noise. Mutate img only.
- **Geometric** (`_geometric`): hflip, vflip, rotate90, scale_jitter, translate. Mutate img AND boxes (and OBBs). `scale_jitter`/`translate` are an affine *without rotation* — **OBB-safe** (uniform scale ⇒ θ invariant, w/h scale by it; translation shifts centers), so they can be enabled even on the `-obb` presets (which ship with them off); `hflip`/`vflip`/`rotate90` are **not** OBB-safe-yet (the θ handling for those is wired in `_geometric` but the OBB presets keep them off). Boxes that slide (mostly) out of frame after scale/translate are dropped via `min_visible_frac` (clipped-area / transformed-box-area).
- **Cutout** (`_cutout`): random gray rectangles, drops boxes whose visible area falls below `min_visible_frac`.
- **Mosaic** (in `OpndetDataset._mosaic`, not in augment.py — needs access to other samples): combines 4 images into one, transforms boxes per quadrant, drops invisible boxes.
- **`min_visible_frac` filter** is the contract: any aug that reduces a GT box's visible area below this threshold must drop the box from labels. Don't supervise the model on invisible objects.
- **Always letterbox after aug.** `rotate90` with k=1 or k=3 transposes (H,W). The `_finish` method letterboxes whenever the post-aug shape doesn't already match `(img_h, img_w)`. Don't bypass this.

### Model output contract (opndet-bbox variant)

Single tensor `[1, 5, H/4, W/4]`:

| ch  | meaning              | activation                              | range |
|-----|----------------------|-----------------------------------------|-------|
| 0   | objectness, peak-suppressed | Sigmoid → in-graph local-max mask | [0,1], sparse |
| 1   | cx (cell-relative)   | Sigmoid                                 | [0,1] |
| 2   | cy (cell-relative)   | Sigmoid                                 | [0,1] |
| 3   | w (image-normalized) | Sigmoid                                 | [0,1] |
| 4   | h (image-normalized) | Sigmoid                                 | [0,1] |

Client decoding (`src/opndet/decode.py`): threshold + `np.nonzero` + index gather. **No NMS step ever runs.** This is the deployed inference contract — don't break it.

### Deployment tiers

opndet's presets split into two tiers based on deployment-target opset constraints. Every architectural decision in a given preset must respect its tier's constraints.

**Edge tier** (`bbox-f`, `bbox-p`, `bbox-n`, `bbox-s`):
- **Hard constraint: opset-13 + Myriad VPU + Ethos-U + Neural-ART + RT1062 compatibility.**
- ALLOWED ops: Conv, BN, ReLU/ReLU6 (Clip), Add/Mul/Sub, Concat, Resize-nearest-asymmetric, MaxPool, Sigmoid, Tanh, Equal, GreaterOrEqual, Cast, Slice/Split.
- FORBIDDEN: GroupNorm, GridSample, ScatterND, SiLU as fused op, half_pixel Resize, dynamic shapes, attention (softmax + matmul + reshape can fail on Myriad), >1 output tensor.
- This is checked in `src/opndet/export.py::ALLOWED_OPS`.

**Server tier** (`bbox-m`, `bbox-l`, `bbox-x`):
- Targets Jetson / RTX / server CPU / Colab GPU. No Myriad commitment.
- Free to use SiLU, attention (C2PSA), half-pixel resize, multi-output tensors, dynamic shapes. These add `MatMul` + `Softmax` to the graph (server-tier-allowlist additions in `export.py::SERVER_TIER_EXTRA_OPS`).
- Still maintains opset-13 export for ONNX Runtime + OpenVINO 2022 CPU/GPU compatibility.
- `bbox-x` uses `peak_kernel=7` (vs k=5 elsewhere) for tighter duplicate suppression.

**Tier is declared in YAML.** Each preset's `model.tier: edge | server` field gates the op allowlist. Default if absent: `edge`. The export-time check `allowed_ops_for_tier(tier)` rejects server-only ops (MatMul, Softmax) when tier=edge; `check_resize_attrs(model, tier)` rejects half-pixel Resize on edge tier.

**`-pro` variants** (see ROADMAP §1.8) layer the YOLO-family + OBB + hard-negative-mining wins on top of each base size. Edge-tier `-pro` keeps opset-13 compat; server-tier `-pro` adds attention / SiLU / half-pixel resize. **Phase 1 shipped:** SPPF at p4, PAFPN neck (top-down + bottom-up + fuse-back-to-stride-4), decoupled head (parallel cls + reg branches off `nout`), and ltrb regression. The output tensor stays `[1, 5, H/4, W/4]` but channel semantics change to `(obj_peak, l, t, r, b)` — image-normalized cell-center-to-edge distances. Use `decode_ltrb()` (in `decode.py`) at inference. Encoded targets via `encode_targets_ltrb()` (in `encode.py`); train with `wh_loss: ltrb` in `OpndetBboxLoss` (DIoU on reconstructed boxes).

**Phase 2 (Server-tier op upgrades) shipped:** server-tier `-pro` (`m/l/x-pro`) presets now ship `C2PSA` position self-attention at p4 (post-SPPF) and at the matching neck stage (post-`b4`). New primitives: `SiLU` (`x * sigmoid(x)`, exports as Mul+Sigmoid), `ResizeBilinear2xHalfPixel` (bilinear upsample with half_pixel coord transform), and `C2PSA` (single-block multi-head spatial self-attention with residual). YAMLs declare their tier via `model.tier: edge|server` (default edge). Tier-aware export check: `allowed_ops_for_tier()` and `check_resize_attrs()` in `export.py` reject `MatMul`/`Softmax` and half_pixel Resize on edge tier. `ConvBnAct` accepts `act: silu` (default `relu6`). Edge `-pro` (`f/p/n/s-pro`) is unchanged — still 100% opset-13 + Myriad-VPU safe.

### Bundled presets and size points

| Preset | Params | Tier | Use case |
|--------|--------|------|----------|
| bbox-f | 28K    | edge   | Sub-1MB int8, MCU stunt |
| bbox-p | 92K    | edge   | TinyML / MCU            |
| bbox-n | 0.31M  | edge   | Edge SoC                |
| bbox-s | 1.27M  | edge   | Default, strong quality |
| bbox-m | 2.37M  | server | Quality-first, ≈YOLOv8n FLOPs |
| bbox-l | ~5M    | server | Mid-range server        |
| bbox-x | 10.4M  | server | Quality-first, server / Colab; uses `peak_kernel=7` |

Plus variants:
- `-dist` (e.g. `bbox-x-dist`) — distillation-aware student trained from a larger teacher
- `bbox-x-hm2` — 2-channel heatmap variant (obj + radius), opset-13 clean (see `docs/det-hm-variants.md`)
- `bbox-x-flow` — 4-channel CellPose-style flow head, server-only; planned, design doc only
- `-tp` (e.g. `bbox-f-tp`) — temporal-prior input variant, 4-channel input
- `-pro` (planned, see ROADMAP §1.8) — kitchen-sink-of-everything per size point
- `-obb` (`bbox-{f,p,n,s}-obb`) — base architecture (no SPPF/PAFPN/decoupled head) + 7-ch OBB head (`obj + ltrb + Tanh angle`). Edge tier, opset-13/Myriad-X safe. Leaner than `-pro` for tightest MCU footprint with orientation. Same OBB output contract as `-pro`; same encode/decode/loss path. Requires `opndet sam-obb` labels and curriculum `w_wh: 0.1` from ep 0 to avoid sub-cell-precision stall.
- `-seg` (`bbox-{f,p,n,s,m}-seg`) — **dense full-resolution dome SEGMENTATION head, not a detector.** Backbone = the matching base preset; the decoder upsamples all the way back to input resolution (laterals from `stem` @ /2 and the raw input @ /1, three more nearest-upsample stages /4→/2→/1, a full-res conv head) and emits a single `[1, 1, H, W]` map — 1.0 at each convex object's deepest interior point, ~linear ramp to 0 at its boundary, exactly 0 between touching objects. No boxes, **no in-graph peak op** — you threshold + connected-components for instances/area (`decode.decode_seg`). Why: a stride-4 dome (128×96 for a 512×384 input) is too coarse to read a small object's *area* off; at stride 1 it's exact. Tier edge — every op is opset-13 + Myriad-safe; the cost is the full-res conv head, which is what the size ladder is for. Detected as a seg head via the `dome` alias. Trained by the dedicated loop in `train_seg.py` (auto-dispatched from `opndet train` / `opndet.train.train()` — different metric space: Dice / IoU / per-object area & count, no calibration / curriculum / eval-threshold). GT: `encode.encode_targets_seg` renders the dense dome at `cfg.img_h/img_w // seg_stride`. Preferred source = **per-instance masks** from `opndet sam-seg` (box-prompt SAM2 with the COCO GT AABBs, dump `<stem>.png` instance-id label maps → `data.sources[*].mask_dir`; true L2-distance-transform dome, real shapes, real areas, `clip_to_box` cuts any mask that escaped the prompt). Fallback = the elliptical `_draw_rotated_dome` from `opndet sam-obb` OBBs (`obb_dir`). Profile knobs (top-level train config): `seg_dome_ramp_px` (0 = proportional ramp; >0 = flat-top plateau, ~N-px edge dropoff) and `seg_instance_gap_px` (>0, mask GT only = erode each mask by ~gap/2 px → guaranteed ≥gap 0-corridor between touchers → plain CC decode separates them, no watershed; the "repulsion", done in the GT). Masks ride through the dataloader's aug (flip/rot90/scale, NEAREST) + letterbox like OBBs; mosaic auto-disabled when masks are present. Loss: `loss.SegDomeLoss` (QFL toward the soft dome + squared-denominator soft Dice against the soft dome *ramp itself* — NOT a binarized `dome>thr` footprint, which would reward a flat-topped p≈1 over the whole object and fight the QFL ramp into a solid blob).

All standard presets produce the same `[1, 5, H/4, W/4]` output layout (except hm2 / flow / -pro / -seg variants which have different output shape). Differ in backbone widths/depths and neck/head channels.

## Conventions

- **Code style**: terse, no docstrings unless the WHY is non-obvious, no comments explaining the obvious. Match existing.
- **No emojis** in code or commits.
- **Conventional Commits format** (`feat:`, `fix:`, `perf:`, `chore:`).
- **DO NOT ADD `scipy` to opndet.** Removed in April 2026 after numpy 2.4 broke scipy 1.17's private-API chain on Colab. Use `src/opndet/_optim.py` (`linear_sum_assignment` wraps `lap.lapjv`; `minimize_scalar_bounded` is a 30-LOC golden-section search). Same rule applies to `pandas`, `xarray`, `statsmodels`, `sklearn` — sprawling research libs with huge numpy private-API surface. See `docs/engineering-decisions.md` "NO scipy" section for the why.
- **Bundled YAML** lives at `src/opndet/configs/` and ships with the wheel via `[tool.setuptools.package-data]`. The CLI's `--model` flag resolves preset names against this dir via `src/opndet/presets.py::resolve()`.
- **train.yaml is a *template*** — `init-config` dumps the bundled one for users to edit. Never assume specific paths in it.
- **Optimizer choice**: AdamW is the default for every preset. `optimizer: musgd` in the training yaml opts into MuSGD (Muon for hidden conv weights + AdamW for biases / BN / narrow heads, per YOLO26). It is intended for server-tier `-pro` presets only; train.py warns but does not block on edge-tier presets. Vision-side wins are unproven — validate on your own dataset before promoting. See `docs/engineering-decisions.md` "Optimizer choice".

## Files at a glance

- `cli.py` — argparse subcommand router; entry point for the `opndet` script.
- `train.py` — training loop. TensorBoard is **off by default** (`tensorboard: true` to enable; the dashboard reads the DuckDB store, not tfevents) — and lazy-imported with a no-op-writer fallback when import fails (Colab numpy/tensorboard mismatches). Auto-increment `out_dir`, resume, trajectory-patience, curriculum w/ alias map, cosine LR + warmup, in-process Colab `files.download()`. Reads `optimizer: adamw|musgd` from config (default adamw).
- `optim_muon.py` — Muon + MuSGD optimizers (ROADMAP §1.8 Phase 6). Opt-in via `optimizer: musgd`; routes >=2D-flattenable conv weights to Muon, everything else to AdamW.
- `assigner.py` — TAL / STAL task-aligned per-cell positive assigner for box regression (ROADMAP §1.8 Phase 3). Opt-in via `assigner: tal|stal` in train.yaml; default `peak` keeps the existing single-positive-per-GT path. Drives REGRESSION-side assignment only — cls supervision stays Gaussian heatmap (see docs/engineering-decisions.md "Assigner choice"). Default for `-pro` presets is `stal` + `curriculum: progloss` (auto-balancing loss weights).
- `model.py` / `blocks.py` — hand-coded reference model (kept for parity tests).
- `primitives.py` / `registry.py` / `yaml_build.py` — YAML DSL system. `build_model_from_yaml(path, img_h=, img_w=)` — input size lives in the model preset YAML's `model.img_h/img_w`, but train.yaml's `model: {img_h:, img_w:}` overrides it (the net is fully conv + nearest-Resize, so any size divisible by the backbone's deepest stride works — multiples of 32 are safe; e.g. 480x640). train/eval/calibrate all pass the override through.
- `encode.py` — heatmap GT encoder. `hm_target` (top-level config) picks the objectness GT shape: `gaussian` (default — CornerNet σ heuristic; or an oriented elliptical Gaussian if `hm_blob_frac > 0`) or `ellipse` (a domed ellipse fitted to the GT box: 1.0 at center, linear ramp to 0 at the inscribed-ellipse boundary, 0 in the box corners, rotated by θ — `_draw_rotated_dome` / `_render_dist_target` family). `ellipse` makes the raw heatmap read like a soft segmentation and hits exactly 0 at the object edge, so it doesn't bleed into gaps between touching objects (unlike a Gaussian tail) — the right pick for crowded scenes. `hm_ellipse_edge_margin` (default 0.1): objects whose box is within that fraction of an image edge fall back to a plain Gaussian (a clipped object's visible chunk isn't a clean ellipse). All single-peaked → peak suppression unaffected. Pair `hm_target: ellipse` with `cls_loss: soft_hm`. `hm_peak_moat` (top-level config, default 0; >0 → e.g. 0.2–0.3): after rendering, depress every non-peak cell within `peak_kernel//2` of a peak to `≤ peak − moat` (`_peak_moat`) — the GT-side fix for adjacent-cell duplicate detections (see "Two-cell ties" above); applies to all three encoders (`encode_targets` / `encode_targets_ltrb` / `encode_targets_obb`). `encode_targets_seg(cfg, obbs=, masks=)` — the dense full-res dome target for the `bbox-*-seg` head: from per-instance masks (priority) → per-instance L2-distance-transform + max-aggregated; from OBBs → `_draw_rotated_dome` per box, at `cfg.img_h/img_w // seg_stride`. `cfg.seg_dome_ramp_px` (top-level train config, default 0): `0` = proportional ramp (1.0 at the deepest interior point → 0 at the boundary over the whole inscribed radius, single-peaked); `>0` (e.g. 3) = **flat-top plateau** (1.0 across the interior, linear dropoff to 0 only over the last ~N px at the edge — trivial target, makes `seg_fg_thresh` ~irrelevant, still a true 0 at the boundary so touching objects keep a 0 valley). `cfg.seg_instance_gap_px` (default 0, mask source only): erode each instance mask by ~gap/2 px before the DT → guaranteed ≥gap-wide 0-corridor between touchers (the encode-side "repulsion" → plain CC decode separates them; costs a ~gap/2-px area shrink). Both profiles still 0 between touching instances (no bleed across the contact line).
- `loss.py` — heatmap cls (`cls_loss`: `focal` center-only | `vfl` IoU-target varifocal | `soft_hm` Quality-Focal-Loss toward the full Gaussian → object-shaped dome, pairs with `hm_blob_frac`) + L1/CIoU/DIoU/NWD/ProbIoU wh + L1 cxy + repulsion (baseline-subtracted) + count + convexity. `peak_sharpen_loss` (`peak_sharpen_weight > 0`): at each GT center cell, forces the strongest neighbor within the `peak_kernel` window to sit ≥ `peak_sharpen_margin` (≫ the PeakSuppress `eps`) below the peak — stops the model hedging two ~equal adjacent cells (true center on a cell boundary) so PeakSuppress fully zeroes the neighbor instead of attenuating it (which left a "second touching detection" that *looked* far less confident because the arith mask amplifies a tiny heatmap difference). Heatmap-side analog of repulsion, zero inference cost. `convexity_loss` is per-object-scaled: the window is Gaussian-weighted per positive cell with σ ≈ the GT box's radius in cells (so each object gets a window matched to its size, and a neighbor's blob inside an oversized window doesn't drag the centroid). `convexity_radius` is just the *cap* on the underlying `unfold` kernel — train.py auto-sizes it from the dataset's p99 GT box (`convexity_radius: auto`, the default; capped at 12 for VRAM) so it never clamps; pin a number to override. So the convexity window is effectively box-derived per object with no manual tuning. `SegDomeLoss` (used only by `bbox-*-seg` via `train_seg.py`): QFL toward the dense full-res dome target + squared-denom soft Dice against the soft dome ramp (NOT a binarized footprint — that flattens the prediction into a solid blob; soft-target Dice cooperates with the QFL ramp at any `seg_w_dice`) — no peak/repulsion/convexity terms (a dense seg map has no cell quantization to defend against).
- `decode.py` — client-side bbox decoder (no NMS). `decode_seg` / `decode_seg_batch` extract instances from a `bbox-*-seg` dome map (threshold → connected-components → per-blob centroid + pixel-area). The seg head is the one variant that *does* require postprocessing — extracting per-object area is inherently a postproc step; the dome hits 0 between touching objects so plain 4-connectivity components already separate convex blobs.
- `train_seg.py` — dedicated training loop for the `bbox-*-seg` (dense dome) head: Dice/fg-IoU/per-object area-MAPE/count-MAE metrics, soft-Dice + QFL loss, EMA + cosine-LR; no calibration / curriculum / eval-threshold / repulsion / temporal-prior (none apply). Auto-dispatched from `opndet train` and `opndet.train.train()` when the preset has `model.head: seg`. GT shapes: prefers `data.sources[*].mask_dir` (per-instance masks from `opndet sam-seg` → true distance-transform dome), falls back to `data.sources[*].obb_dir` (`opndet sam-obb` → elliptical dome) if no mask_dir. Writes the same DuckDB store as `train.py` (`metrics_db: true` default): `train/loss`+`lr` streamed ~`steps_per_epoch//8` per epoch (flushed eagerly — `MetricsDB.add_scalar` buffers to 64 before writing+CHECKPOINTing, which the seg loop would otherwise never hit), `val/*` per epoch, `test/*` on new-best epochs. **Vis only on a new-best (or first/last) epoch** (`viz_only_on_improvement`, default true — same as the detector), for `vis_samples` val *and* test samples, under tags `val/seg` / `test/seg`: `sample_<i>_rgb.png` (clean RGB, written once) + four toggleable overlays per sample — `dome_pred`/`dome_gt` (TURBO heatmap of the dome; pred per-vis-epoch, gt once) and `seg_pred`/`seg_gt` (the *decoded* instance view from `decode_seg`: each blob filled with a distinct colour + a same-hue contour + a `<N>px` area label; `_render_seg_decoded`). No box rows — it's a dense-dome head, not a box model; per-blob extent is the `<N>px` label in the seg overlay, an AABB fitted around a dome blob is meaningless. The dashboard auto-discovers the overlay kinds; its overlay-opacity slider scales them all. `seg_fg_thresh` (default 0.5) = the dome foreground cut; `seg_edge_margin` (default 0; >0 e.g. 0.05) = frame-edge-touching (clipped) blobs are excluded from area-MAPE and tagged `·E` with a thin border. `evaluate_seg()` is reused by `opndet eval` for seg ckpts.
- `dataset.py` — COCO loader, OpndetDataset, mosaic, collate.
- `augment.py` — photometric + geometric + cutout, with min_visible_frac filter.
- `visualize.py` — render predictions onto images for the DuckDB store / dashboard. `save_layered_vis` writes the clean RGB of each val/test sample ONCE to a stable `vis/<tag>/sample_<i>_rgb.png` (it's deterministic — no aug on val/test), and only the per-epoch model-output overlays (`obj_heat`, the 4-ch prior) go under `vis/<tag>/ep_<NNN>/` — so the run dir doesn't balloon with N_epochs identical RGB copies.
- `export.py` — torch.onnx.export(opset=13, dynamo=False) + opset-safety check + parity test. `--diagnostic` mode adds all named-layer outputs for the webui's explain mode.
- `predict.py` — single-image / video inference + visualization. For a `bbox-*-seg` ckpt (`opndet predict --model bbox-n-seg --ckpt … --image … --save vis.jpg`, or `--video …`) it treats the output as a dense dome: image mode un-letterboxes it to the original size, runs `decode_seg`, writes a translucent dome heatmap + per-blob contour / centroid / pixel-area overlay, returns `[{cx, cy, area_px, peak, x1,y1,x2,y2}, …]` (no `score`); video mode draws the dome heatmap + contours + blob centroids + an `n=` count per frame on the letterboxed canvas. `--threshold` becomes the dome foreground cut (default 0.5 for seg).
- `analyze.py` — postmortem CLI: per-layer activation slider + Grad-CAM HTML report on saved ckpts.
- `profile.py` — `opndet profile --ckpt … --model … --images dir/`: "what's actually doing the work?" — a self-contained HTML report with, per named layer, mean|activation|, live-channel fraction (channels whose spatial response is ≥5% of the layer's average — low = dead width), and **ablation-Δ** (normalized [0,1] change in the predicted objectness heatmap when that layer's output is zeroed — the *causal* "is it load-bearing"; only for layers with params), plus the mean-channel activation map per layer and a matplotlib bar chart down the network. Note: a sequential layer near the input shows high Δ by construction (everything downstream depends on it) — the informative comparisons are among parallel branches (the laterals) and deep-vs-shallow stages.
- `mine_negatives.py` — hard-negative mining (ROADMAP §1.7). Two entry points: `mine()` is the offline CLI (`opndet mine-negatives` — Grad-CAM-attributed ghost patches, clustered, dumped to a pool dir). `mine_into_pool()` is the in-training miner used when `auto_mine:` is set in train.yaml — every N epochs (or when `center_ghost_rate` crosses a gate) after `start_epoch`, take objectness peaks above threshold whose center isn't near any GT (`ghost_radius_frac` — lower to flag phantoms *between* real objects), crop `patch_size`-px patches (`patch_size: auto`, the default, = median GT box short-side) into `<out_dir>/hard_negatives/patches/`, FIFO-trim to `max_pool`; train.py then rebuilds the train DataLoader so workers pick up the new patches via the `augment.hard_negative_*` paste path (auto-wired). Mind `start_epoch` — mining before the cls head has converged teaches it to suppress not-yet-learned objects → recall collapse.
- `metrics.py` — Hungarian + IoU-free center matching with cell-window radius, ghost/duplicate split.
- `metrics_db.py` — DuckDB writer per-run for queryable training history.
- `dashboard.py` — FastAPI backend for the run dashboard: read-only DuckDB endpoints (`/api/*`, bulk reads are POST), per-run shadow-copy + pooled connection, vis assets under `/files/*` (immutable cache headers), and mounts the built SPA at `/`. The frontend is a Vite+React+TS app in `frontend/` (Tailwind v4, uPlot charts, @tanstack/react-query). Build it with `uv run poe build-dashboard` (or `scripts/build_dashboard.sh`) → output goes to `src/opndet/dashboard_static/`, which is **committed** and shipped as package-data (so `pip install`/Colab needs no node). The `pre-commit` hook (`.pre-commit-config.yaml`; run `pre-commit install` once) auto-rebuilds + re-stages that bundle whenever `frontend/` sources change, so it never goes stale. `uv run poe dev-dashboard` runs the Vite dev server (proxies `/api` + `/files` to a locally-running `opndet dashboard`).
- `calibrate.py` — Platt-scale T fit on val, baked into ckpt.
- `eval.py` — full validation report + perturbation stability proxy. For a `bbox-*-seg` ckpt it routes to `_run_seg_eval` (`train_seg.evaluate_seg`): Dice / fg-IoU / per-object area-MAPE / count-MAE against the same elliptical-dome GT training uses; needs OBB sidecars; writes `seg_eval_<split>.md`.
- `presets.py` — preset name resolution (`bbox-s` → bundled YAML path).

## See also

- `docs/engineering-decisions.md` — non-obvious choices and gotchas (cls_loss=focal cold-start safety, peak_kernel=7 in bbox-x, curriculum aliases, shape-mAP, trajectory-patience). Read this before changing training defaults.
- `ROADMAP.md` — plan + SHIPPED tags. Current high-priority: §2.1 OBB output, §1.7 Grad-CAM hard-negative mining.
- `docs/det-hm-variants.md` — hm2 / flow heatmap variant designs.
