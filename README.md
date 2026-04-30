# opndet

Tiny single-class object detector. **OpenVINO 2022 opset compatible.** Anchor-free, **no NMS, no postprocessing** — peak suppression happens inside the graph via `Equal(MaxPool(hm), hm)`.

Built for industrial scenes with many simple convex objects per frame. Outputs a YOLO-friendly dense tensor `[1, 5, H/4, W/4]` with channels `(obj, cx, cy, w, h)`. Decoding client-side is a threshold + index gather.

See [DESIGN.md](DESIGN.md) for the full architecture rationale.

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
uv venv
uv pip install -e .
```

---

## CLI

```bash
opndet info                       # list bundled model presets
opndet info bbox-s                # inspect a preset (params, layers, IO shape)
opndet init-config --out my.yaml  # dump a training config template

opndet train     --config my.yaml
opndet predict   --image foo.jpg --model bbox-s --ckpt runs/exp1/best.pt --save vis.jpg
opndet export    --model bbox-s  --ckpt runs/exp1/best.pt --out opndet.onnx
```

The `--model` flag accepts a **bundled preset name** (`bbox-n`, `bbox-s`, `bbox-m`) or a **path to a YAML** config.

---

## Bundled presets

| Preset   | Params  | Notes                                   |
|----------|---------|-----------------------------------------|
| `bbox-f` | 28K     | Femto. Sub-1MB int8. Quality drops.     |
| `bbox-p` | 92K     | Pico. Microcontroller / TinyML class.   |
| `bbox-n` | 0.31M   | Nano. Edge devices.                     |
| `bbox-s` | 1.27M   | Small. Default.                         |
| `bbox-m` | 2.37M   | Medium. ≈ YOLOv8n FLOP budget.          |

All produce the same output layout: `[1, 5, H/4, W/4]`, fixed input `(3, 384, 512)`.

---

## Try in browser (no install)

Drop your `.onnx` into the in-browser tester at **[bherbruck.github.io/opndet](https://bherbruck.github.io/opndet)** — runs entirely client-side via onnxruntime-web (WebGPU when available, WASM fallback). Image upload or webcam input. Useful for sanity-checking deployment ONNX before flashing to edge hardware. Works on phones too — opndet is small enough to run on any modern mobile CPU.

## Quickstart on Colab

Use the bundled notebook: [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb). Open it in Colab → set GPU runtime → set `OPNDET_REPO` → run all. Trains, exports ONNX, downloads artifacts.

---

## Training on your own dataset

opndet trains on **any single-class detection dataset in COCO format**. Multi-source merging is built in.

1. Dump your data as one or more COCO JSON files + image directories.
2. Generate a config:

   ```bash
   opndet init-config --out my-train.yaml
   ```

3. Edit the `data.sources` list:

   ```yaml
   data:
     sources:
       - {coco: data/site_a/anno.coco.json, images: data/site_a/imgs}
       - {coco: data/site_b/anno.coco.json, images: data/site_b/imgs}
     split_ratios: [0.8, 0.1, 0.1]
   ```

4. Train:

   ```bash
   opndet train --config my-train.yaml
   ```

   Auto train/val/test split (deterministic by seed). AdamW + cosine LR + AMP on CUDA. Checkpoints saved to `out_dir/{best,last}.pt`. F1 logged per epoch on the val split.

   Logs land at `out_dir/tb/`. Launch TensorBoard:
   ```bash
   tensorboard --logdir runs/exp1/tb
   ```
   Each epoch logs scalars (loss, P/R/F1, LR) and image grids of validation predictions (GT red, preds green).

5. Export to ONNX (opset 13):

   ```bash
   opndet export --model bbox-s --ckpt runs/exp1/best.pt --out opndet.onnx
   ```

   The exporter validates that only opset-13-safe ops are used and runs a fp32 parity check vs PyTorch.

---

## Customizing the architecture

The model is built from a YOLO-style YAML layer graph. To make a custom variant:

```bash
opndet init-config --out my-train.yaml   # to get the training template
# then write your own model YAML — see src/opndet/configs/opndet-bbox-s.yaml
```

A model YAML declares each layer with a `from:` reference (previous layer, named alias, or list of indices for multi-input modules). New blocks register with one decorator:

```python
from opndet.registry import register
import torch.nn as nn

@register("MyBlock")
class MyBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ...
```

Then reference `MyBlock` in your YAML.

---

## Output format

The model returns a single tensor `[1, 5, H', W']` where `H'=H/4`, `W'=W/4`. Channel layout:

| ch  | meaning              | range                                  |
|-----|----------------------|----------------------------------------|
| 0   | objectness, peak-suppressed | `[0, 1]`, **sparse** (zero at non-peak cells) |
| 1   | cx (cell-relative)   | `[0, 1]`                               |
| 2   | cy (cell-relative)   | `[0, 1]`                               |
| 3   | w (image-normalized) | `[0, 1]`                               |
| 4   | h (image-normalized) | `[0, 1]`                               |

**Decoding (client-side, no NMS):**

```python
mask = obj > 0.3                              # boolean
gy, gx = nonzero(mask)                        # cell indices
x = (gx + cx[gy, gx]) * stride                # pixel center
y = (gy + cy[gy, gx]) * stride
w_px = w[gy, gx] * image_width
h_px = h[gy, gx] * image_height
boxes = [(x - w_px/2, y - h_px/2, x + w_px/2, y + h_px/2, score)]
```

Standard YOLO decoder, **minus the NMS step** — peak suppression already ran inside the graph.

---

## Why no NMS?

Stock YOLO uses anchor boxes and multiple detection scales, so several cells fire per object and require NMS to deduplicate. opndet is anchor-free and single-scale. The peak op (`Equal(hm, MaxPool3x3(hm))`) keeps **at most one cell per local 3×3 neighborhood**. One prediction per object by construction.

---

## Compatibility

| Target              | Status |
|---------------------|--------|
| ONNX opset 13       | ✓      |
| OpenVINO 2022.x     | ✓ (validated allowed-op list) |
| OpenVINO Model Optimizer | ✓ |
| OpenVINO POST INT8  | quant-friendly (ReLU6 + no `exp`/`atan2` in graph) |
| CPU inference       | ✓      |
| GPU training        | ✓ (PyTorch CUDA + AMP) |

---

## License

TBD by repo owner.
