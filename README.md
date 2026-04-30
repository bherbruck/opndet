# opndet

Tiny single-class object detector. **OpenVINO 2022 / Myriad X compatible.** Anchor-free, **no NMS**, peak suppression baked into the ONNX graph as pure arithmetic (no comparison ops).

Built for industrial scenes with many simple convex objects per frame. Outputs a YOLO-friendly dense tensor `[1, 5, H/4, W/4]` with channels `(obj, cx, cy, w, h)`. Decoding client-side is a threshold + index gather.

See [DESIGN.md](DESIGN.md) for the full architecture rationale.

---

## Try in browser (no install)

Drop your `.onnx` into the in-browser tester at **[bherbruck.github.io/opndet](https://bherbruck.github.io/opndet)** — runs entirely client-side via onnxruntime-web (WebGPU when available, WASM fallback). Image upload or webcam input. Useful for sanity-checking deployment ONNX before flashing to edge hardware. Works on phones — opndet is small enough to run on any modern mobile CPU.

---

## Install (no PyPI — install from git)

### Run a one-off command (no install)

```bash
uvx --from git+https://github.com/bherbruck/opndet opndet info
```

### Install as a global tool

```bash
uv tool install git+https://github.com/bherbruck/opndet
opndet --help
```

### Editable install for development

```bash
git clone https://github.com/bherbruck/opndet
cd opndet
uv venv && uv pip install -e .
```

---

## CLI

```bash
opndet info                         # list bundled model presets
opndet info bbox-s                  # inspect a preset (params, layers, IO shape)
opndet init-config --out my.yaml    # dump training config template

opndet train     --config my.yaml [--run-name RUN] [--resume PATH]
opndet predict   --image foo.jpg --model bbox-s --ckpt best.pt --save vis.jpg
opndet export    --model bbox-s --ckpt best.pt --out opndet.onnx [--bake-input-norm]
opndet quantize  --onnx opndet.onnx --calib data/imgs --out opndet_int8.onnx [--verify]
```

`--model` accepts a **bundled preset name** or a **path to a YAML** config.

---

## Bundled presets

| Preset   | Params  | int8 size | Use case                                  |
|----------|---------|-----------|-------------------------------------------|
| `bbox-f` | 28K     | ~30 KB    | Femto. Microcontroller stunt.             |
| `bbox-p` | 92K     | ~95 KB    | Pico. TinyML / MCU.                       |
| `bbox-n` | 0.31M   | ~320 KB   | Nano. Edge SoC, sub-ms latency.           |
| `bbox-s` | 1.27M   | ~1.3 MB   | Small. Solid quality.                     |
| `bbox-m` | 2.37M   | ~2.4 MB   | Medium. ≈ YOLOv8n FLOP budget.            |

All produce identical output layout: `[1, 5, H/4, W/4]`, fixed input `(3, 384, 512)`.

---

## Training

```bash
opndet init-config --out my-train.yaml
# edit data.sources to point at your COCO json + images
opndet train --config my-train.yaml
```

Auto train/val/test split (deterministic by seed). Multi-source merging is built in. AdamW + cosine LR + AMP on CUDA. Best checkpoint saved continuously by configurable metric.

**Key training-config knobs:**

```yaml
runs_dir: runs                      # parent of run dirs (auto-incremented)
name: exp1                          # run subdir; overridden by --run-name
metric_for_best: map50              # f1 | map50 | map_50_95
patience: 20                        # early stop if no improvement in N epochs
ema_decay: 0.999                    # EMA teacher; smoother val curve

cache_images: false                 # set true if you have RAM; faster epochs
num_workers: 16
amp_dtype: fp16                     # bf16 on H100/A100/L4

augment:
  brightness: 0.4
  ...
  mosaic_prob: 0.0                  # set 0.3-0.5 for crowded/diverse data
  cutout_prob: 0.0                  # synthetic occlusion
  min_visible_frac: 0.5             # drop GT boxes occluded below this fraction

loss:
  wh_loss: ciou                     # l1 | giou | ciou | nwd
  cls_loss: vfl                     # focal | vfl (varifocal — bimodal confidence)
  w_wh: 1.5                         # 1.5 for IoU-loss, 5.0 for L1
  repulsion_weight: 0.0             # >0 for crowded touching objects
```

Per-epoch eval logs precision, recall, F1, mAP@0.5, mAP@0.5:0.95. TensorBoard at `runs_dir/<name>/tb/` shows scalars + per-epoch image grids of validation predictions (GT red, pred green).

**Resume an interrupted run:**

```bash
opndet train --config my-train.yaml --resume runs/exp1   # uses last.pt
```

State preserved: model + EMA + optimizer + scaler + step + best metric.

---

## Quickstart on Colab

Use the bundled notebook: [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb). Open in Colab → GPU runtime → set `OPNDET_REPO` → run all. Trains, exports ONNX, downloads artifacts.

---

## Customizing the architecture

Models are built from a YOLO-style YAML layer graph. Each layer declares a `from:` reference (previous layer, named alias, or list of indices for multi-input modules):

```yaml
- {name: stem, from: 0,    module: ConvBnAct, args: {in_ch: 3,  out_ch: 32, k: 3, s: 2}}
- {name: p1,   from: stem, module: CSPBlock,  args: {in_ch: 32, out_ch: 64, n: 1, s: 2}}
```

New blocks register via one decorator:

```python
from opndet.registry import register
import torch.nn as nn

@register("MyBlock")
class MyBlock(nn.Module):
    ...
```

Reference `MyBlock` in any YAML. See [`src/opndet/configs/`](src/opndet/configs) for the bundled presets as examples.

---

## Output format

`[1, 5, H/4, W/4]`:

| ch  | meaning              | range                                          |
|-----|----------------------|------------------------------------------------|
| 0   | objectness, peak-suppressed | `[0, 1]`, **sparse** (zero at non-peak cells) |
| 1   | cx (cell-relative)   | `[0, 1]`                                       |
| 2   | cy (cell-relative)   | `[0, 1]`                                       |
| 3   | w (image-normalized) | `[0, 1]`                                       |
| 4   | h (image-normalized) | `[0, 1]`                                       |

**Decoding (client-side, no NMS):**

```python
mask = obj > 0.5
gy, gx = nonzero(mask)
x = (gx + cx[gy, gx]) * stride
y = (gy + cy[gy, gx]) * stride
w_px = w[gy, gx] * image_width
h_px = h[gy, gx] * image_height
boxes = [(x - w_px/2, y - h_px/2, x + w_px/2, y + h_px/2, score)]
```

Standard YOLO decoder, **minus the NMS step** — peak suppression already ran inside the graph.

---

## Why no NMS?

YOLO uses anchor boxes and multiple detection scales, so several cells fire per object and require NMS to deduplicate. opndet is anchor-free and single-scale. The in-graph peak op:

```
diff = (hm + eps) - MaxPool(hm, k=5)
mask = clip(diff * (1/eps), 0, 1)
peaks = hm * mask
```

…keeps at most one cell per local 5×5 neighborhood. One prediction per object by construction. No comparison ops in the graph (Myriad-friendly).

---

## Edge deployment

### OpenVINO 2022 + Myriad X (depthai / OAK)

opndet is designed to compile cleanly to Myriad blob:

- Only opset-13-safe ops in the graph
- Arithmetic peak suppression (no `GreaterOrEqual` / `Equal` — those have a known Myriad plugin bug)
- ReLU6 activations (efficient int8 lowering)
- No `exp`, `atan2`, `GroupNorm`, `GridSample`, dynamic shapes

**For deployment with raw camera frames** (skip Python preprocessing), bake input normalization into the graph:

```bash
opndet export --model bbox-m --ckpt best.pt --out opndet_oak.onnx --bake-input-norm
```

This adds `(x - mean*255) / (std*255)` as graph ops, so the OAK can feed raw `RGB888p` uint8 frames straight to the model.

### Quantization (int8)

```bash
opndet quantize --onnx opndet.onnx --out opndet_int8.onnx \
                --calib /path/to/calib_images --n-calib 100 --verify
```

Static int8 PTQ via onnxruntime. Architecture is quant-friendly:
- ReLU6 (efficient int8 SIMD on Cortex-M and Myriad)
- No `exp`/`atan2` in inference graph
- No `GroupNorm` (fp16/int8 unstable)

bbox-m drops 3-4× in size to int8. Smaller models drop less due to per-channel quant overhead. **Note**: int8 only speeds up inference on hardware with int8 accelerators (CMSIS-NN, OpenVINO POT on x86, Hailo). On Myriad X (fp16-native), int8 isn't faster.

---

## Compatibility

| Target                  | Status |
|-------------------------|--------|
| ONNX opset 13           | ✓      |
| OpenVINO 2022.x (CPU/GPU) | ✓    |
| OpenVINO 2022.x Myriad X (depthai) | ✓ (arith peak op, no compare) |
| OpenVINO Model Optimizer | ✓     |
| ONNXRuntime CPU         | ✓ (~150 FPS on x86 fp32 for bbox-f) |
| ONNXRuntime WebGPU      | ✓ (`docs/index.html`) |
| TensorFlow Lite (via tflite-onnx) | should work, untested |
| CMSIS-NN (Cortex-M MCUs) | should work, manual conversion |
| GPU training (PyTorch) + AMP | ✓ |

---

## License

TBD by repo owner.
