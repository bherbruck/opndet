# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Tiny single-class object detector. **OpenVINO 2022 opset compatible (ONNX opset ≤13). No NMS, no postprocessing** — peak suppression is baked into the ONNX graph. Architecture is a CenterNet-style heatmap head + dense regression in a 5-channel single output tensor `[1, 5, H/4, W/4]` = `(obj_peak, cx, cy, w, h)`.

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

**In-browser tester**: drop any exported ONNX into [bherbruck.github.io/opndet](https://bherbruck.github.io/opndet) (lives in `docs/index.html`, deployed via GH Pages). Runs entirely client-side via onnxruntime-web. The "explain mode" toggle requires an ONNX exported with `--diagnostic`.

## Architecture (big picture)

### The hard constraint that drives everything

**ONNX opset ≤13 + OpenVINO 2022 op support.** This is checked in `src/opndet/export.py::ALLOWED_OPS`. Every architectural decision filters through "does this op exist and behave correctly in opset 13?" Forbidden: GroupNorm, GridSample, ScatterND, SiLU as a fused op, Resize with `coordinate_transformation_mode=half_pixel`, dynamic shapes. Allowed: Conv, BN, ReLU/ReLU6 (Clip), Add/Mul/Sub, Concat, Resize-nearest-asymmetric, MaxPool, Sigmoid, Tanh, Equal, GreaterOrEqual, Cast, Slice/Split.

When adding any new op or block, add a roundtrip parity test: build the model, export with `opset_version=13, dynamo=False`, run both PyTorch and ORT on the same input, assert max diff <1e-4. The `verify_onnx()` helper in `export.py` does this.

### The "no NMS" trick

`PeakSuppress` (in `src/opndet/primitives.py` and `src/opndet/model.py`) is the centerpiece. It runs `mask = clip((hm + eps - MaxPoolKxK(hm)) * BIG, 0, 1)` (arith mode, default) or `(hm + eps >= MaxPool(hm))` (compare mode, smaller graph but breaks on Myriad VPU). Then `out = hm * mask` inside the graph. Local-max suppression that's ONNX-friendly. The **eps is critical** (currently 5e-3 for fp16 tolerance): without it, FP precision differences between PyTorch and ORT flip mask cells at boundaries and break parity. Don't remove it. Don't lower it without re-running parity tests across all model sizes.

**Per-preset peak_kernel:** most presets use `k=5` (radius 2, 8-px min spacing at stride=4). `bbox-x` uses `k=7` (radius 3, 12-px min spacing) since it targets quality-first deployment with more bbox-shape headroom — the wider window suppresses near-tie adjacent duplicates that the smaller kernel lets through. Each preset's `SigmoidPeakSuppress` block sets the `k` arg explicitly in `src/opndet/configs/opndet-bbox-*.yaml`.

**Two-cell ties survive any kernel size.** If two adjacent cells round to identical sigmoid output (common at saturation in fp16), they BOTH pass `>= MaxPool` since each IS the local max. The repulsion loss (training-side) is the soft pressure to avoid creating these ties in the first place. If duplicates persist after training, the data layer (hard-negative mining; see ROADMAP §1.7) is the next attack surface — pruning or larger kernels won't fix tied-pair survival.

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
- **Geometric** (`_geometric`): hflip, vflip, rotate90, scale_jitter, translate. Mutate img AND boxes.
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
- Free to use SiLU, attention, half-pixel resize, multi-output tensors, dynamic shapes.
- Still maintains opset-13 export for ONNX Runtime + OpenVINO 2022 CPU/GPU compatibility.
- `bbox-x` uses `peak_kernel=7` (vs k=5 elsewhere) for tighter duplicate suppression.

**`-pro` variants** (see ROADMAP §1.8) layer the YOLO-family + OBB + hard-negative-mining wins on top of each base size. Edge-tier `-pro` keeps opset-13 compat; server-tier `-pro` adds attention / SiLU / half-pixel resize. **Phase 1 shipped:** SPPF at p4, PAFPN neck (top-down + bottom-up + fuse-back-to-stride-4), decoupled head (parallel cls + reg branches off `nout`), and ltrb regression. The output tensor stays `[1, 5, H/4, W/4]` but channel semantics change to `(obj_peak, l, t, r, b)` — image-normalized cell-center-to-edge distances. Use `decode_ltrb()` (in `decode.py`) at inference. Encoded targets via `encode_targets_ltrb()` (in `encode.py`); train with `wh_loss: ltrb` in `OpndetBboxLoss` (DIoU on reconstructed boxes).

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

All standard presets produce the same `[1, 5, H/4, W/4]` output layout (except hm2 / flow / -pro variants which have different output shape). Differ in backbone widths/depths and neck/head channels.

## Conventions

- **Code style**: terse, no docstrings unless the WHY is non-obvious, no comments explaining the obvious. Match existing.
- **No emojis** in code or commits.
- **Conventional Commits format** (`feat:`, `fix:`, `perf:`, `chore:`).
- **Bundled YAML** lives at `src/opndet/configs/` and ships with the wheel via `[tool.setuptools.package-data]`. The CLI's `--model` flag resolves preset names against this dir via `src/opndet/presets.py::resolve()`.
- **train.yaml is a *template*** — `init-config` dumps the bundled one for users to edit. Never assume specific paths in it.

## Files at a glance

- `cli.py` — argparse subcommand router; entry point for the `opndet` script.
- `train.py` — training loop. Lazy imports tensorboard with no-op fallback when import fails (Colab numpy/tensorboard mismatches). Auto-increment `out_dir`, resume, trajectory-patience, curriculum w/ alias map, cosine LR + warmup, in-process Colab `files.download()`.
- `model.py` / `blocks.py` — hand-coded reference model (kept for parity tests).
- `primitives.py` / `registry.py` / `yaml_build.py` — YAML DSL system.
- `encode.py` — Gaussian heatmap GT encoder (CornerNet σ heuristic).
- `loss.py` — focal/VFL heatmap + L1/CIoU/DIoU/NWD wh + L1 cxy + repulsion (baseline-subtracted) + count + convexity.
- `decode.py` — client-side bbox decoder (no NMS).
- `dataset.py` — COCO loader, OpndetDataset, mosaic, collate.
- `augment.py` — photometric + geometric + cutout, with min_visible_frac filter.
- `visualize.py` — render predictions onto images for TensorBoard / DuckDB / dashboard.
- `export.py` — torch.onnx.export(opset=13, dynamo=False) + opset-safety check + parity test. `--diagnostic` mode adds all named-layer outputs for the webui's explain mode.
- `predict.py` — single-image / video inference + visualization.
- `analyze.py` — postmortem CLI: per-layer activation slider + Grad-CAM HTML report on saved ckpts.
- `metrics.py` — Hungarian + IoU-free center matching with cell-window radius, ghost/duplicate split.
- `metrics_db.py` — DuckDB writer per-run for queryable training history.
- `dashboard.py` — FastAPI dashboard reading the DuckDB stores; multi-run, accordion-grouped charts, lightbox viz.
- `calibrate.py` — Platt-scale T fit on val, baked into ckpt.
- `eval.py` — full validation report + perturbation stability proxy.
- `presets.py` — preset name resolution (`bbox-s` → bundled YAML path).

## See also

- `docs/engineering-decisions.md` — non-obvious choices and gotchas (cls_loss=focal cold-start safety, peak_kernel=7 in bbox-x, curriculum aliases, shape-mAP, trajectory-patience). Read this before changing training defaults.
- `ROADMAP.md` — plan + SHIPPED tags. Current high-priority: §2.1 OBB output, §1.7 Grad-CAM hard-negative mining.
- `docs/det-hm-variants.md` — hm2 / flow heatmap variant designs.
