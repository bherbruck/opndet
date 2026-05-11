"""opndet profile — "what's actually doing the work?" report.

For a trained ckpt, runs it over a batch of images and writes a self-contained
HTML report with, per named layer:

  - mean|activation|        — how hot the layer is
  - live-channel fraction   — channels whose spatial response actually varies
                              (a near-constant channel is dead width)
  - ablation-Δ              — relative change in the predicted objectness
                              heatmap when this layer's output is zeroed. This
                              is the *causal* "is this learned component
                              load-bearing" signal — non-zero activations ≠
                              important. Only computed for layers that have
                              parameters (i.e. learned components, not Add /
                              Concat / Resize / Sigmoid / etc).
  - channels, params        — where the capacity sits

Plus the mean-over-channels activation map per layer (a small heatmap thumbnail).

Usage:
    opndet profile --ckpt best.pt --model bbox-n-obb --images data/imgs/ [--n 8] [--out report.html]
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch

from opndet.presets import resolve as _resolve_preset
from opndet.yaml_build import build_model_from_yaml

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _letterbox(img: np.ndarray, h: int, w: int) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas[py:py + nh, px:px + nw] = r
    return canvas


def _prep(path: Path, h: int, w: int, in_ch: int, device) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    lb = _letterbox(img, h, w)
    t = torch.from_numpy(((lb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)).unsqueeze(0)
    if in_ch == 4:
        t = torch.cat([t, torch.zeros(1, 1, h, w)], dim=1)
    return t.to(device)


def _list_images(p: str | Path, n: int) -> list[Path]:
    p = Path(p)
    if p.is_file():
        return [p]
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in _IMG_EXTS)
    return files[:n] if n and n > 0 else files


def _heatmap_data_uri(arr2d: np.ndarray, long_side: int = 384) -> str:
    """Per-cell viridis heatmap as a base64 PNG, NEAREST-scaled so its long side
    is ~`long_side` px — small layer maps (e.g. p4 at 12×16) become honest big
    blocks; large ones get downscaled. Big enough to display thumbnail-sized in
    the table AND blown up in the lightbox without re-pixelating."""
    a = arr2d.astype(np.float32)
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    h, w = a.shape
    s = max(1, long_side // max(h, w)) if max(h, w) <= long_side else long_side / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    a8 = cv2.resize((a * 255).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.applyColorMap(a8, cv2.COLORMAP_VIRIDIS)
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _bar_chart_data_uri(rows: list[dict]) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    import io
    n = len(rows)
    ys = np.arange(n)[::-1]  # forward order top→bottom
    labels = [f"{r['name']}  ({r['channels']}c, {r['params'] / 1e3:.1f}K)" for r in rows]
    ma = np.array([r["mean_abs"] for r in rows])
    lf = np.array([r["live_frac"] for r in rows])
    ab = np.array([(r["ablation_delta"] if r["ablation_delta"] is not None else 0.0) for r in rows])
    has_ab = np.array([r["ablation_delta"] is not None for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(12, max(3.5, 0.32 * n)), sharey=True)
    fig.patch.set_facecolor("#0b0e13")
    titles = ["mean|activation|", "live-channel fraction", "ablation-Δ (heatmap change when zeroed)"]
    data = [ma, lf, ab]
    colors = ["#4f9dff", "#5fd35f", "#ff8c42"]
    for ax, t, d, c in zip(axs, titles, data, colors):
        ax.barh(ys, d, color=c, alpha=0.9 if t != titles[2] else None)
        if t == titles[2]:
            # gray out the bars that weren't ablated
            for i, (yy, dd, hb) in enumerate(zip(ys, d, has_ab)):
                if not hb:
                    ax.barh(yy, max(ma.max(), lf.max(), ab.max()) * 0.0, color="#333")
        ax.set_title(t, color="#d6dee6", fontsize=9)
        ax.set_facecolor("#11161d")
        ax.tick_params(colors="#8b97a3", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#232c37")
        ax.set_xlim(left=0)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels(labels, fontsize=7, color="#d6dee6")
    fig.suptitle("↓ forward order down the network", color="#8b97a3", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")


def _bar(value: float, vmax: float, color: str) -> str:
    pct = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax)) * 100.0
    return (f'<div style="background:#1a212b;border-radius:2px;height:11px;overflow:hidden">'
            f'<div style="background:{color};height:100%;width:{pct:.1f}%"></div></div>')


def _render_html(rows, montage, *, ckpt, model_path, total_params, in_ch, img_h, img_w,
                 n_images, det_summary, bar_uri) -> str:
    ma_max = max((r["mean_abs"] for r in rows), default=1.0) or 1.0
    ab_vals = [r["ablation_delta"] for r in rows if r["ablation_delta"] is not None]
    ab_max = max(ab_vals, default=1.0) or 1.0
    trs = []
    for r in rows:
        thumb = montage.get(r["name"])
        timg = (f'<img class="hm" src="{_heatmap_data_uri(thumb)}" onclick="lb(this.src)" '
                f'title="click to enlarge">') if thumb is not None else "—"
        ab = r["ablation_delta"]
        ab_cell = (f'{ab:.3f}<br>{_bar(ab, ab_max, "#ff8c42")}' if ab is not None else '<span style="color:#444">—</span>')
        # only flag *learned* layers (with params) as near-dead — the decode
        # tail (obj/cxywh/angle/...) is parameter-free and intentionally sparse.
        dead = ""
        if r["params"] > 0 and (r["live_frac"] < 0.5 or (ab is not None and ab < 0.02)):
            dead = ' style="opacity:0.5"'
        trs.append(f"""<tr{dead}>
  <td><b>{r['name']}</b><br><span style="color:#8b97a3;font-size:11px">{r['module']}</span></td>
  <td>{timg}</td>
  <td>{r['mean_abs']:.3f}<br>{_bar(r['mean_abs'], ma_max, "#4f9dff")}</td>
  <td>{r['live_frac']*100:.0f}%<br>{_bar(r['live_frac'], 1.0, "#5fd35f")}</td>
  <td>{ab_cell}</td>
  <td style="text-align:right">{r['channels']}</td>
  <td style="text-align:right">{r['params']/1e3:.2f}K</td>
</tr>""")
    bar_block = f'<img src="{bar_uri}" style="max-width:100%">' if bar_uri else "<p>(matplotlib unavailable — bar chart skipped)</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>opndet profile · {ckpt.name}</title>
<style>
 body{{background:#0b0e13;color:#d6dee6;font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:18px}}
 h1{{font-size:16px;color:#fff;margin:0 0 4px}} h2{{font-size:13px;color:#8b97a3;text-transform:uppercase;letter-spacing:.06em;margin:22px 0 8px;border-bottom:1px solid #232c37;padding-bottom:4px}}
 .meta{{color:#8b97a3;margin-bottom:14px}} .meta b{{color:#d6dee6}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #232c37;padding:5px 8px;vertical-align:top}}
 th{{background:#161c25;color:#8b97a3;text-align:left;position:sticky;top:0}}
 .legend{{color:#8b97a3;font-size:12px;margin-top:14px}} .legend b{{color:#d6dee6}}
 img.hm{{height:var(--hmh,150px);image-rendering:pixelated;border-radius:3px;cursor:zoom-in;border:1px solid #232c37;display:block}}
 .ctl{{margin:8px 0;color:#8b97a3}} .ctl input{{vertical-align:middle}}
 #lbk{{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;align-items:center;justify-content:center;z-index:99;cursor:zoom-out}}
 #lbk img{{max-width:96vw;max-height:96vh;image-rendering:pixelated;border:1px solid #2e3a47;border-radius:6px}}
</style></head><body>
<div id="lbk" onclick="this.style.display='none'"><img id="lbi"></div>
<h1>opndet profile — {ckpt.name}</h1>
<div class="meta">model: <b>{Path(model_path).name}</b> &nbsp;|&nbsp; params: <b>{total_params/1e3:.0f}K</b> &nbsp;|&nbsp; input: <b>{in_ch}×{img_h}×{img_w}</b> &nbsp;|&nbsp; averaged over <b>{n_images}</b> image{'s' if n_images != 1 else ''} &nbsp;|&nbsp; {det_summary}</div>
<h2>activity down the network</h2>
{bar_block}
<h2>per-layer detail</h2>
<div class="ctl">activation-map size <input type="range" min="80" max="480" value="150" oninput="document.documentElement.style.setProperty('--hmh', this.value + 'px')"> &nbsp;(or click any map for full-screen)</div>
<table><thead><tr><th>layer</th><th>mean activation map</th><th>mean&#124;a&#124;</th><th>live channels</th><th>ablation-Δ</th><th>chans</th><th>params</th></tr></thead><tbody>
{''.join(trs)}
</tbody></table>
<div class="legend">
 <b>mean|a|</b>: average absolute activation — "how hot". &nbsp;
 <b>live channels</b>: fraction of channels whose spatial response is at least 5% of this layer's average channel variation — low = dead width. &nbsp;
 <b>ablation-Δ</b>: normalized symmetric change (∈[0,1]) in the predicted objectness heatmap when this layer's output is zeroed — the <i>causal</i> "is it load-bearing"; near-0 = removable. Only for layers with learned parameters. <i>A sequential layer near the input shows high Δ by construction (everything downstream depends on it) — the informative comparisons are among parallel branches (laterals) and deep-vs-shallow stages.</i> &nbsp;
 Dimmed rows = learned layers that look near-dead (few live channels / ~zero ablation-Δ).
 <br>(Static report; the interactive top-down-DAG explorer is separate / TBD.)
</div>
<script>function lb(s){{document.getElementById('lbi').src=s;document.getElementById('lbk').style.display='flex';}}
document.addEventListener('keydown',function(e){{if(e.key==='Escape')document.getElementById('lbk').style.display='none';}});</script>
</body></html>"""


def profile(ckpt: str | Path, model: str | None = None, images: str | Path = ".",
            out: str | Path | None = None, n: int = 8, device: str | None = None) -> dict:
    ckpt = Path(ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = sd.get("config", {}) if isinstance(sd, dict) else {}
    model_name = model or (cfg.get("model_config") if isinstance(cfg, dict) else None)
    if not model_name:
        raise ValueError("--model required (ckpt has no saved config.model_config)")
    model_path = _resolve_preset(model_name)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model_from_yaml(model_path).to(dev).eval()
    w = (sd.get("ema") or sd["model"]) if (isinstance(sd, dict) and "model" in sd) else sd
    m.load_state_dict(w)
    in_ch, H, W = m.input_shape

    names = sorted(m.aliases, key=lambda nm: m.aliases[nm])             # forward order
    raw_pos = m.aliases.get("raw", 10 ** 9)
    obj_alias = "raw" if "raw" in m.aliases else names[-1]
    obj_idx = m.aliases[obj_alias]

    def layer_module(nm):
        i = m.aliases[nm] - 1
        return m.layers[i] if 0 <= i < len(m.layers) else None

    def n_params(mod):
        return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

    ablatable = [nm for nm in names if m.aliases[nm] < raw_pos and n_params(layer_module(nm)) > 0]

    imgs = _list_images(images, n)
    if not imgs:
        raise FileNotFoundError(f"no images found under {images}")
    xs = [_prep(p, H, W, in_ch, dev) for p in imgs]

    mean_abs = {nm: 0.0 for nm in names}
    live_frac = {nm: 0.0 for nm in names}
    channels = {nm: 0 for nm in names}
    montage: dict[str, np.ndarray] = {}
    obj_base: list[torch.Tensor] = []
    n_peaks = 0.0
    obj_max_sum = 0.0
    with torch.no_grad():
        for k, x in enumerate(xs):
            cache = m._run(x)
            for nm in names:
                a = cache[m.aliases[nm]].float()
                C = a.shape[1]
                channels[nm] = C
                mean_abs[nm] += float(a.abs().mean())
                # "live" = channel whose spatial response varies by at least 5%
                # of this layer's average channel variation (a near-constant
                # channel relative to its peers is dead width). Absolute
                # thresholds don't discriminate — every conv+BN channel has
                # *some* variance.
                ch_std = a[0].reshape(C, -1).std(dim=1)
                ref = float(ch_std.mean()) + 1e-9
                live_frac[nm] += float((ch_std > 0.05 * ref).float().mean())
                if k == len(xs) - 1:
                    montage[nm] = a[0].abs().mean(0).cpu().numpy()
            obj_logit = cache[obj_idx][:, 0:1].float()
            obj_base.append(torch.sigmoid(obj_logit).squeeze())
            res = m(x)
            o = (res["output"] if isinstance(res, dict) else res)[0, 0].float()
            n_peaks += float((o >= 0.3).sum())
            obj_max_sum += float(o.max())
    for nm in names:
        mean_abs[nm] /= len(xs)
        live_frac[nm] /= len(xs)

    # ablation-Δ: zero a layer's output, re-run, compare the predicted heatmap.
    # Normalized symmetric difference ∈ [0, 1] (0 = no change, 1 = nothing in
    # common) so it can't blow up to "10×" the way an unnormalized ratio would
    # when zeroing an early layer makes everything downstream go uniform.
    # NOTE: a sequential layer near the input shows high Δ by construction —
    # everything after it depends on it. The informative comparisons are among
    # *parallel* branches (the laterals) and deep-vs-shallow stages.
    abl: dict[str, float] = {}
    for nm in ablatable:
        mod = layer_module(nm)
        h = mod.register_forward_hook(lambda _m, _i, o: torch.zeros_like(o))
        try:
            with torch.no_grad():
                d = 0.0
                for x, base in zip(xs, obj_base):
                    cache = m._run(x)
                    new = torch.sigmoid(cache[obj_idx][:, 0].float()).squeeze()
                    d += float((new - base).abs().sum() / (new.abs().sum() + base.abs().sum() + 1e-6))
            abl[nm] = d / len(xs)
        finally:
            h.remove()

    rows = []
    for nm in names:
        mod = layer_module(nm)
        rows.append({
            "name": nm, "module": type(mod).__name__ if mod is not None else "input",
            "channels": channels[nm], "params": n_params(mod),
            "mean_abs": mean_abs[nm], "live_frac": live_frac[nm],
            "ablation_delta": abl.get(nm),
        })
    total_params = sum(p.numel() for p in m.parameters())
    det_summary = f"avg detections@0.3: <b>{n_peaks/len(xs):.1f}</b>, avg obj_max: <b>{obj_max_sum/len(xs):.3f}</b>"

    out_path = Path(out) if out else ckpt.parent / f"{ckpt.stem}_profile.html"
    bar_uri = _bar_chart_data_uri(rows)
    out_path.write_text(_render_html(rows, montage, ckpt=ckpt, model_path=model_path,
                                     total_params=total_params, in_ch=in_ch, img_h=H, img_w=W,
                                     n_images=len(xs), det_summary=det_summary, bar_uri=bar_uri))
    print(f"profile -> {out_path}  ({len(rows)} layers, {len(xs)} images, {total_params/1e3:.0f}K params)")
    return {"out": str(out_path), "layers": rows, "total_params": int(total_params), "n_images": len(xs)}
