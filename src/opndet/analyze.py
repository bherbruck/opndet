"""Postmortem analysis of a trained opndet checkpoint.

Runs inference on a handful of frames, captures per-layer activations and
input-gradient saliency per detection, renders a self-contained HTML report.

CLI: opndet analyze --ckpt <pt> --model <preset> --video <mp4>
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from opndet.predict import load_model, preprocess
from opndet.decode import decode
from opndet.presets import resolve


def render_heatmap(arr_2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """2D float -> TURBO BGR at target (H, W) via nearest upsample + min-max norm."""
    if arr_2d.shape != target_shape:
        arr = cv2.resize(arr_2d, (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    else:
        arr = arr_2d.copy()
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    else:
        arr = np.zeros_like(arr)
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(arr_u8, cv2.COLORMAP_TURBO)


def overlay(img_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)


def video_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def gradcam_input(model, x: torch.Tensor, cell_y: int, cell_x: int) -> np.ndarray:
    """Input-gradient saliency for obj_logit at (cell_y, cell_x). Returns (H, W) float."""
    model.zero_grad()
    x_g = x.clone().detach().requires_grad_(True)
    out = model(x_g)
    obj_t = out["output"] if isinstance(out, dict) else out
    target = obj_t[0, 0, cell_y, cell_x]
    target.backward()
    return x_g.grad[0].abs().sum(dim=0).cpu().numpy()


def denormalize(x_t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized (3, H, W) -> uint8 BGR (H, W, 3)."""
    x = x_t[:3].permute(1, 2, 0).cpu().numpy()
    x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def run(ckpt: str, model_yaml: str, out_dir: Path, device: str, threshold: float,
        video: str | None = None, n_frames: int = 6,
        images: list[str] | None = None):
    """Render a postmortem report. Source frames come from EITHER:
      - `images`: explicit list of image file paths (each becomes one sample), OR
      - `video` + `n_frames`: sample N evenly-spaced frames from the video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_yaml, ckpt, device=device)
    model.eval()
    in_ch, h, w = model.input_shape
    print(f"model: input {in_ch}x{h}x{w}, {len(model.aliases)} named layers")

    layer_names = sorted(model.aliases.items(), key=lambda kv: kv[1])

    if images:
        frames = []
        for p in images:
            img = cv2.imread(p)
            if img is None:
                print(f"  WARN: failed to read {p}")
                continue
            frames.append(img)
        if not frames:
            raise RuntimeError(f"no readable images from {images}")
        print(f"loaded {len(frames)} image(s)")
    elif video:
        frames = video_frames(video, n_frames)
        if not frames:
            raise RuntimeError(f"no frames from {video}")
        print(f"sampled {len(frames)} frames from {video}")
    else:
        raise ValueError("provide either `images=[...]` or `video=...`")

    samples = []
    for s_idx, frame in enumerate(frames):
        sd = out_dir / f"sample_{s_idx}"
        sd.mkdir(exist_ok=True)
        x_t, _meta = preprocess(frame, h, w, in_ch=in_ch)
        x_b = x_t.to(device)  # preprocess already returns (1, C, H, W)

        # Forward + capture layer cache
        with torch.no_grad():
            cache = model._run(x_b)
            out = model(x_b)
        obj_t = out["output"] if isinstance(out, dict) else out

        img_bgr = denormalize(x_t[0])
        cv2.imwrite(str(sd / "input.png"), img_bgr)

        # Final output heatmap
        obj = obj_t[0, 0].cpu().numpy()
        cv2.imwrite(str(sd / "obj_heat.png"),
                    overlay(img_bgr, render_heatmap(obj, (h, w))))

        # Per-layer activation maps (mean abs over channels)
        layers = []
        for name, idx in layer_names:
            t = cache[idx]
            if t.ndim != 4:
                continue
            act = t[0].abs().mean(dim=0).cpu().numpy()
            fname = f"layer_{idx:03d}_{name}.png"
            cv2.imwrite(str(sd / fname),
                        overlay(img_bgr, render_heatmap(act, (h, w))))
            layers.append({"name": name, "idx": idx,
                           "shape": list(t.shape), "file": fname})

        # Detections + Grad-CAM per detection
        out_np = obj_t[0].detach().cpu().numpy()
        H_out, W_out = out_np.shape[1], out_np.shape[2]
        stride = h // H_out
        dets = decode(out_np, h, w, stride, threshold=threshold)
        det_paths = []
        for d_idx, det in enumerate(dets):
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2
            cell_x = max(0, min(W_out - 1, int(cx / stride)))
            cell_y = max(0, min(H_out - 1, int(cy / stride)))
            sal = gradcam_input(model, x_b, cell_y, cell_x)
            ovr = overlay(img_bgr, render_heatmap(sal, (h, w)))
            x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
            cv2.rectangle(ovr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(ovr, f"{det.score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            fname = f"det_{d_idx:02d}_gradcam.png"
            cv2.imwrite(str(sd / fname), ovr)
            det_paths.append({"idx": d_idx, "score": float(det.score),
                              "bbox": [det.x1, det.y1, det.x2, det.y2],
                              "cell": [cell_y, cell_x], "file": fname})

        samples.append({"idx": s_idx, "layers": layers, "dets": det_paths,
                        "n_dets": len(det_paths)})
        print(f"  sample {s_idx}: {len(det_paths)} dets, {len(layers)} layers")

    (out_dir / "index.html").write_text(_html(samples, ckpt, model_yaml))
    print(f"\nwrote {out_dir / 'index.html'}")
    print(f"open: file://{(out_dir / 'index.html').absolute()}")


def _html(samples, ckpt, model_yaml) -> str:
    import json
    samples_json = json.dumps(samples)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>opndet postmortem</title>
<style>
  :root {{ color-scheme: dark; --bg:#0d1117; --fg:#c9d1d9; --mut:#7d8590;
           --acc:#58a6ff; --line:#30363d; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:14px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--fg); }}
  header {{ padding:10px 14px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--bg); z-index:10; }}
  header h1 {{ margin:0; font-size:14px; font-weight:600; }}
  header .mut {{ color:var(--mut); font-size:12px; }}
  .sample {{ border-bottom:1px solid var(--line); padding:18px 14px; }}
  .sample h2 {{ font-size:13px; color:var(--mut); margin:0 0 12px; text-transform:uppercase; letter-spacing:.5px; }}
  .row {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-start; }}
  .panel {{ flex: 1 1 320px; max-width:520px; }}
  .panel-label {{ color:var(--mut); font-size:11px; margin-bottom:4px; text-transform:uppercase; letter-spacing:.5px; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:4px; display:block; }}
  .layer-controls {{ background:#161b22; border:1px solid var(--line); border-radius:6px; padding:10px; margin-bottom:8px; }}
  .layer-controls label {{ font-size:11px; color:var(--mut); display:block; margin-bottom:4px; text-transform:uppercase; }}
  .layer-controls input[type=range] {{ width:100%; }}
  .layer-info {{ font-family:ui-monospace, Menlo, monospace; font-size:12px; color:var(--mut); margin-top:4px; }}
  .det-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px; }}
  .det {{ background:#161b22; border:1px solid var(--line); border-radius:6px; padding:8px; }}
  .det .meta {{ font-family:ui-monospace,Menlo,monospace; font-size:11px; color:var(--mut); margin-top:4px; }}
</style></head><body>
<header>
  <h1>opndet postmortem</h1>
  <div class="mut">ckpt: {ckpt} &nbsp; · &nbsp; model: {model_yaml}</div>
</header>
<div id="root"></div>
<script>
const SAMPLES = {samples_json};
function rel(s_idx, file) {{ return `sample_${{s_idx}}/${{file}}`; }}
const root = document.getElementById('root');
SAMPLES.forEach(s => {{
  const div = document.createElement('div');
  div.className = 'sample';
  const layers = s.layers;
  const layer_html = layers.length === 0 ? '' : `
    <div class="panel" style="flex-basis:520px">
      <div class="panel-label">per-layer activation (slider)</div>
      <div class="layer-controls">
        <label>layer <span id="lbl-${{s.idx}}">0 / ${{layers.length-1}}</span></label>
        <input type="range" min="0" max="${{layers.length-1}}" value="0" id="slider-${{s.idx}}"/>
        <div class="layer-info" id="info-${{s.idx}}">${{layers[0].name}} · idx ${{layers[0].idx}} · shape ${{JSON.stringify(layers[0].shape)}}</div>
      </div>
      <img id="layer-${{s.idx}}" src="${{rel(s.idx, layers[0].file)}}"/>
    </div>`;
  const det_html = s.dets.length === 0 ? '<div class="mut">no detections above threshold</div>' :
    s.dets.map(d => `
      <div class="det">
        <img src="${{rel(s.idx, d.file)}}"/>
        <div class="meta">det ${{d.idx}} · score ${{d.score.toFixed(3)}} · cell [${{d.cell.join(',')}}]</div>
      </div>`).join('');
  div.innerHTML = `
    <h2>sample ${{s.idx}} (${{s.n_dets}} dets)</h2>
    <div class="row">
      <div class="panel"><div class="panel-label">input</div><img src="${{rel(s.idx, 'input.png')}}"/></div>
      <div class="panel"><div class="panel-label">obj heat (post-suppression)</div><img src="${{rel(s.idx, 'obj_heat.png')}}"/></div>
      ${{layer_html}}
    </div>
    <h2 style="margin-top:18px">grad-cam per detection</h2>
    <div class="det-grid">${{det_html}}</div>`;
  root.appendChild(div);
  if (layers.length > 0) {{
    const slider = document.getElementById(`slider-${{s.idx}}`);
    slider.addEventListener('input', e => {{
      const i = parseInt(e.target.value);
      const L = layers[i];
      document.getElementById(`layer-${{s.idx}}`).src = rel(s.idx, L.file);
      document.getElementById(`lbl-${{s.idx}}`).textContent = `${{i}} / ${{layers.length-1}}`;
      document.getElementById(`info-${{s.idx}}`).textContent =
        `${{L.name}} · idx ${{L.idx}} · shape ${{JSON.stringify(L.shape)}}`;
    }});
  }}
}});
</script>
</body></html>"""


