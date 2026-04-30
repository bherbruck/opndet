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
opndet export --model bbox-s --ckpt best.pt --out opndet.onnx

# Tests
.venv/bin/pytest tests/                # full suite (sparse — most coverage is via integration tests)
```

The `--model` flag accepts a **bundled preset name** (`bbox-f`, `bbox-p`, `bbox-n`, `bbox-s`, `bbox-m`) OR a path to a YAML.

## Architecture (big picture)

### The hard constraint that drives everything

**ONNX opset ≤13 + OpenVINO 2022 op support.** This is checked in `src/opndet/export.py::ALLOWED_OPS`. Every architectural decision filters through "does this op exist and behave correctly in opset 13?" Forbidden: GroupNorm, GridSample, ScatterND, SiLU as a fused op, Resize with `coordinate_transformation_mode=half_pixel`, dynamic shapes. Allowed: Conv, BN, ReLU/ReLU6 (Clip), Add/Mul/Sub, Concat, Resize-nearest-asymmetric, MaxPool, Sigmoid, Tanh, Equal, GreaterOrEqual, Cast, Slice/Split.

When adding any new op or block, add a roundtrip parity test: build the model, export with `opset_version=13, dynamo=False`, run both PyTorch and ORT on the same input, assert max diff <1e-4. The `verify_onnx()` helper in `export.py` does this.

### The "no NMS" trick

`PeakSuppress` (in `src/opndet/primitives.py` and `src/opndet/model.py`) is the centerpiece. It runs `mask = (hm + eps >= MaxPool3x3(hm))` then `out = hm * mask` inside the graph. This is local-max suppression that's ONNX-friendly. The **eps is critical** (currently 1e-3): without it, FP precision differences between PyTorch and ORT flip mask cells at boundaries and break parity. Don't remove it. Don't lower it without re-running parity tests across all model sizes.

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

### Bundled presets and size points

| Preset | Params | Use case |
|--------|--------|----------|
| bbox-f | 28K    | Sub-1MB int8, MCU stunt |
| bbox-p | 92K    | TinyML / MCU            |
| bbox-n | 0.31M  | Edge SoC                |
| bbox-s | 1.27M  | Default, strong quality |
| bbox-m | 2.37M  | Quality-first, ≈YOLOv8n FLOPs |

All produce identical output layout. Differ only in backbone widths/depths and neck/head channels.

## Conventions

- **Code style**: terse, no docstrings unless the WHY is non-obvious, no comments explaining the obvious. Match existing.
- **No emojis** in code or commits.
- **Conventional Commits format** (`feat:`, `fix:`, `perf:`, `chore:`).
- **Bundled YAML** lives at `src/opndet/configs/` and ships with the wheel via `[tool.setuptools.package-data]`. The CLI's `--model` flag resolves preset names against this dir via `src/opndet/presets.py::resolve()`.
- **train.yaml is a *template*** — `init-config` dumps the bundled one for users to edit. Never assume specific paths in it.

## Files at a glance

- `cli.py` — argparse subcommand router; entry point for the `opndet` script.
- `train.py` — training loop. Lazy imports tensorboard and silences TF/oneDNN env noise. Handles auto-increment `out_dir`, resume, patience, cosine LR + warmup.
- `model.py` / `blocks.py` — hand-coded reference model (kept for parity tests).
- `primitives.py` / `registry.py` / `yaml_build.py` — YAML DSL system.
- `encode.py` — Gaussian heatmap GT encoder (CornerNet σ heuristic).
- `loss.py` — focal heatmap + L1 size + L1 cxy losses.
- `decode.py` — client-side bbox decoder (no NMS).
- `dataset.py` — COCO loader, OpndetDataset, mosaic, collate.
- `augment.py` — photometric + geometric + cutout, with min_visible_frac filter.
- `visualize.py` — render predictions onto images for TensorBoard.
- `export.py` — torch.onnx.export(opset=13, dynamo=False) + opset-safety check + parity test.
- `predict.py` — single-image inference + visualization.
- `presets.py` — preset name resolution (`bbox-s` → bundled YAML path).
