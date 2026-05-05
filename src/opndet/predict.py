from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from opndet.dataset import letterbox
from opndet.decode import decode
from opndet.yaml_build import build_model_from_yaml


def load_model(model_config: str, ckpt: str | None, device: str = "cpu") -> torch.nn.Module:
    m = build_model_from_yaml(model_config).to(device).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        T = float(sd.get("temperature", 1.0)) if isinstance(sd, dict) else 1.0
        if T != 1.0:
            from opndet.calibrate import apply_temperature
            apply_temperature(m, T)
            print(f"applied calibration temperature T={T:.4f}")
    return m


def preprocess(
    img_bgr: np.ndarray,
    h: int,
    w: int,
    in_ch: int = 3,
    prior: np.ndarray | None = None,
    stride: int = 4,
) -> tuple[torch.Tensor, dict]:
    """RGB letterbox + ImageNet normalize. For 4-ch (temporal) models, the
    `prior` arg is a stride-coords float32 (Hs, Ws) heatmap in [0,1]; it gets
    upsampled to (h,w) via nearest and concatenated as channel 3 unnormalized.
    If in_ch==4 and prior is None, a zero prior (cold-start) is used.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, _ = letterbox(img, np.zeros((0, 4), dtype=np.float32), h, w)
    f = img_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f = (f - mean) / std
    t = torch.from_numpy(f.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    if in_ch == 4:
        if prior is None:
            prior = np.zeros((h // stride, w // stride), dtype=np.float32)
        prior_full = cv2.resize(prior.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        prior_t = torch.from_numpy(prior_full).unsqueeze(0).unsqueeze(0).contiguous()
        t = torch.cat([t, prior_t], dim=1)
    orig_h, orig_w = img.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    pad_x = (w - int(round(orig_w * scale))) // 2
    pad_y = (h - int(round(orig_h * scale))) // 2
    return t, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": orig_w, "orig_h": orig_h}


def unletterbox_box(x1: float, y1: float, x2: float, y2: float, info: dict) -> tuple[float, float, float, float]:
    s, px, py = info["scale"], info["pad_x"], info["pad_y"]
    return ((x1 - px) / s, (y1 - py) / s, (x2 - px) / s, (y2 - py) / s)


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _yt_dlp_cmd() -> list[str] | None:
    """Find a way to invoke yt-dlp: shell binary first, else python module."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        import yt_dlp   # noqa: F401
        import sys as _sys
        return [_sys.executable, "-m", "yt_dlp"]
    except ImportError:
        return None


def _download_video(url: str, out_dir: Path) -> Path:
    """Download a video URL via yt-dlp."""
    cmd = _yt_dlp_cmd()
    if cmd is None:
        raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cache: derive name from URL hash so repeated runs reuse the download
    import hashlib
    name = hashlib.sha1(url.encode()).hexdigest()[:12] + ".mp4"
    out_path = out_dir / name
    if out_path.exists():
        print(f"using cached {out_path}")
        return out_path
    print(f"downloading {url} -> {out_path}")
    subprocess.run(cmd + ["-f", "best[height<=720]", "-o", str(out_path), url], check=True)
    return out_path


def _draw_boxes(img_bgr: np.ndarray, dets, thickness: int = 1) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), thickness)
        label = f"{d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (int(d.x1), int(d.y1) - th - 4), (int(d.x1) + tw + 4, int(d.y1)), (0, 255, 0), -1)
        cv2.putText(out, label, (int(d.x1) + 2, int(d.y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def predict_video(
    video_path_or_url: str,
    model_config: str,
    ckpt: str | None = None,
    threshold: float = 0.3,
    device: str = "cpu",
    save_path: str | Path = "predict_out.mp4",
    stride: int = 4,
    max_frames: int | None = None,
    print_every: int = 30,
    temporal_n_frames: int = 8,
    temporal_stamp_threshold: float = 0.4,
    temporal_spawn_mask: np.ndarray | None = None,
    temporal_spawn_amplitude: float = 0.4,
) -> dict:
    """Run inference on a video (local path or URL). Writes an annotated mp4.
    Returns a stats dict (frame count, mean dets, det std, total time).
    """
    if _is_url(video_path_or_url):
        cache_dir = Path.cwd() / ".opndet_video_cache"
        video_path = _download_video(video_path_or_url, cache_dir)
    else:
        video_path = Path(video_path_or_url)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

    m = load_model(model_config, ckpt, device=device)
    in_ch, h, w = m.input_shape

    acc = None
    if in_ch == 4:
        from opndet.temporal import TailAccumulator
        acc = TailAccumulator((h // stride, w // stride),
                              n_frames=temporal_n_frames,
                              stamp_threshold=temporal_stamp_threshold,
                              spawn_mask=temporal_spawn_mask,
                              spawn_amplitude=temporal_spawn_amplitude,
                              stride=stride)
        print(f"  temporal mode: TailAccumulator(n_frames={temporal_n_frames}, "
              f"stamp_threshold={temporal_stamp_threshold})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # H.264 (avc1) plays in VSCode/browsers/Windows; cv2's mp4v (MPEG-4 Part 2)
    # is hit-and-miss. Try strategies in order:
    #   1. imageio-ffmpeg if installed (optional dep — ships its own static ffmpeg)
    #   2. system ffmpeg via shell pipe
    #   3. cv2 mp4v fallback (warn that container may not play everywhere)
    use_imageio = False
    try:
        import imageio_ffmpeg as _iff   # noqa: F401
        use_imageio = True
    except ImportError:
        pass
    has_sys_ffmpeg = shutil.which("ffmpeg") is not None

    writer = None
    used_codec = None
    if use_imageio:
        import imageio.v2 as iio
        writer = iio.get_writer(str(save_path), fps=fps, codec="libx264",
                                 quality=8, macro_block_size=1)
        used_codec = "imageio-ffmpeg/H.264"
    elif has_sys_ffmpeg:
        # Pipe BGR frames into ffmpeg via a subprocess for direct H.264 encoding.
        ff = subprocess.Popen([
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(save_path),
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        writer = ff
        used_codec = "system ffmpeg/H.264"
    else:
        cv_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not cv_writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {save_path}")
        writer = cv_writer
        used_codec = "cv2/mp4v"
        print(f"warning: H.264 unavailable. Wrote mp4v container (may not play in VSCode/browsers).")
        print(f"         install with: uv pip install imageio-ffmpeg  (or apt install ffmpeg)")

    n = 0
    det_counts = []
    t_start = time.perf_counter()
    target = max_frames if max_frames else total_frames or "?"
    print(f"processing {video_path.name}  ({total_frames or '?'} frames @ {fps:.1f} fps) -> {save_path}")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            prior_in = acc.acc if acc is not None else None
            t, info = preprocess(frame_bgr, h, w, in_ch=in_ch, prior=prior_in, stride=stride)
            with torch.no_grad():
                out = m(t.to(device))
            out_t = out["output"] if isinstance(out, dict) else out
            out_np = out_t.cpu().numpy()
            dets = decode(out_np[0], h, w, stride, threshold=threshold)
            if acc is not None:
                acc.update([((d.x1, d.y1, d.x2, d.y2), d.score) for d in dets])

            # render on the letterboxed frame so the output canvas matches model input dims
            canvas = np.full((h, w, 3), 114, dtype=np.uint8)
            sh, sw = frame_bgr.shape[:2]
            s = min(w / sw, h / sh)
            nw, nh = int(round(sw * s)), int(round(sh * s))
            ox, oy = (w - nw) // 2, (h - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame_bgr, (nw, nh))
            vis = _draw_boxes(canvas, dets)
            if used_codec.startswith("imageio"):
                # imageio expects RGB
                writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            elif used_codec.startswith("system ffmpeg"):
                writer.stdin.write(vis.tobytes())
            else:
                writer.write(vis)

            det_counts.append(len(dets))
            n += 1
            if n % print_every == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = n / max(elapsed, 1e-6)
                recent = det_counts[-print_every:]
                print(f"  frame {n}/{target}  proc {cur_fps:.1f} fps  "
                      f"dets last30: mean={np.mean(recent):.1f} std={np.std(recent):.2f}")
            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
        if used_codec.startswith("imageio"):
            writer.close()
        elif used_codec.startswith("system ffmpeg"):
            writer.stdin.close()
            writer.wait()
        else:
            writer.release()

    elapsed = time.perf_counter() - t_start
    counts = np.array(det_counts) if det_counts else np.zeros(0)
    stats = {
        "frames": n,
        "elapsed_s": elapsed,
        "process_fps": n / max(elapsed, 1e-6),
        "video_fps": fps,
        "det_mean": float(counts.mean()) if counts.size else 0.0,
        "det_std": float(counts.std()) if counts.size else 0.0,
        "det_min": int(counts.min()) if counts.size else 0,
        "det_max": int(counts.max()) if counts.size else 0,
        "out": str(save_path),
    }
    return stats


def predict_image(
    image_path: str | Path,
    model_config: str,
    ckpt: str | None = None,
    threshold: float = 0.3,
    device: str = "cpu",
    save_path: str | Path | None = None,
    stride: int = 4,
) -> list[dict]:
    m = load_model(model_config, ckpt, device=device)
    in_ch, h, w = m.input_shape
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    t, info = preprocess(img_bgr, h, w, in_ch=in_ch, prior=None, stride=stride)
    with torch.no_grad():
        out = m(t.to(device))
    out_t = out["output"] if isinstance(out, dict) else out
    out_np = out_t.cpu().numpy()
    dets = decode(out_np[0], h, w, stride, threshold=threshold)

    results = []
    for d in dets:
        x1, y1, x2, y2 = unletterbox_box(d.x1, d.y1, d.x2, d.y2, info)
        x1 = max(0.0, min(info["orig_w"] - 1, x1))
        y1 = max(0.0, min(info["orig_h"] - 1, y1))
        x2 = max(0.0, min(info["orig_w"] - 1, x2))
        y2 = max(0.0, min(info["orig_h"] - 1, y2))
        results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": d.score})

    if save_path is not None:
        vis = img_bgr.copy()
        for r in results:
            cv2.rectangle(vis, (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"])), (0, 255, 0), 2)
            cv2.putText(vis, f"{r['score']:.2f}", (int(r["x1"]), int(r["y1"]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(save_path), vis)
    return results
