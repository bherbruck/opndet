"""SAM2 → OBB preprocessing. ROADMAP §2.1 Phase 4a.

Reads a COCO json (AABB labels), runs SAM2 per image with the AABBs as box
prompts, fits an oriented rectangle to each mask, and writes YOLOv8-OBB
`<basename>.txt` files to an output dir. Idempotent (skips existing outputs).

Pure preprocessing — no model-side changes. SAM2 is an optional dep; importing
this module is cheap, but `run()` will raise a clear error if SAM2 is missing.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROUNDNESS_THRESHOLD = 1.15


def coco_bbox_to_xyxy(bbox) -> list[float]:
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def mask_to_obb_corners(mask: np.ndarray, fallback_bbox=None) -> np.ndarray | None:
    """Mask → 4 corners (float32, [4,2]) of the fitted oriented rectangle.
    Returns None if the mask is empty / contour too small.
    Round objects (aspect < ROUNDNESS_THRESHOLD) fall back to AABB corners.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    try:
        (cx, cy), (w_ell, h_ell), angle = cv2.fitEllipse(contour)
    except cv2.error:
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect).astype(np.float32)

    major = max(w_ell, h_ell)
    minor = min(w_ell, h_ell)
    if minor <= 0:
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect).astype(np.float32)

    if major / minor < ROUNDNESS_THRESHOLD:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    major_angle = angle if h_ell >= w_ell else angle + 90
    theta = np.deg2rad(major_angle - 90)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    half_maj, half_min = major / 2, minor / 2
    local = np.array([
        [-half_maj, -half_min],
        [ half_maj, -half_min],
        [ half_maj,  half_min],
        [-half_maj,  half_min],
    ], dtype=np.float32)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ R.T + np.array([cx, cy], dtype=np.float32)


def is_valid_rectangle(corners: np.ndarray | None, tolerance: float = 0.01) -> bool:
    if corners is None or len(corners) != 4:
        return False
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    if abs(sides[0] - sides[2]) > tolerance * max(sides[0], sides[2], 1e-6):
        return False
    if abs(sides[1] - sides[3]) > tolerance * max(sides[1], sides[3], 1e-6):
        return False
    v1 = corners[1] - corners[0]
    v2 = corners[2] - corners[1]
    norm_product = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_product < 1e-6:
        return False
    return abs(float(np.dot(v1, v2)) / norm_product) <= tolerance


def corners_to_yolo_obb_line(corners: np.ndarray, img_w: int, img_h: int, class_id: int = 0) -> str:
    """YOLOv8-OBB normalized format. Allows coords outside [0,1] for truncated objects."""
    normalized = corners.astype(np.float64).copy()
    normalized[:, 0] /= img_w
    normalized[:, 1] /= img_h
    coords = " ".join(f"{v:.6f}" for v in normalized.flatten())
    return f"{class_id} {coords}"


def _is_round_via_corners(corners: np.ndarray) -> bool:
    """Detect AABB-fallback corners (axis-aligned rectangle) for manifest stats."""
    return (abs(corners[0, 1] - corners[1, 1]) < 1e-3
            and abs(corners[2, 1] - corners[3, 1]) < 1e-3
            and abs(corners[0, 0] - corners[3, 0]) < 1e-3
            and abs(corners[1, 0] - corners[2, 0]) < 1e-3)


@dataclass
class ImageStats:
    n_objects: int = 0
    n_obb: int = 0
    n_round_fallback: int = 0
    n_invalid: int = 0


@dataclass
class RunStats:
    n_images_processed: int = 0
    n_images_skipped: int = 0
    n_objects_processed: int = 0
    n_obb_extracted: int = 0
    n_aabb_fallback: int = 0
    n_invalid_dropped: int = 0
    sam_model_used: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def process_image(img_path: Path, annotations: list[dict], predictor) -> tuple[list[str], ImageStats]:
    """Run SAM2 on one image; return (yolo_obb_lines, stats).
    `annotations` is the raw COCO ann list (each has a 'bbox' field [x,y,w,h]).
    `predictor` is a SAM2ImagePredictor (already moved to device).
    """
    import torch  # local import — module imports stay light

    stats = ImageStats()
    img = cv2.imread(str(img_path))
    if img is None:
        return [], stats
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    boxes_xyxy = np.array(
        [coco_bbox_to_xyxy(a["bbox"]) for a in annotations if a.get("bbox") is not None],
        dtype=np.float32,
    )
    if boxes_xyxy.size == 0:
        return [], stats
    stats.n_objects = len(boxes_xyxy)

    use_cuda = torch.cuda.is_available()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda
        else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
    )
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(img_rgb)
        masks, _scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )
    # SAM2 returns (N, 1, H, W) for batched boxes or (1, H, W) for a single one
    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim == 3 and len(boxes_xyxy) == 1:
        masks = masks[0:1]

    lines: list[str] = []
    for i, mask in enumerate(masks):
        corners = mask_to_obb_corners(mask, fallback_bbox=boxes_xyxy[i])
        if corners is None or not is_valid_rectangle(corners):
            stats.n_invalid += 1
            continue
        if _is_round_via_corners(corners):
            stats.n_round_fallback += 1
        stats.n_obb += 1
        lines.append(corners_to_yolo_obb_line(corners, w, h, class_id=0))
    return lines, stats


def _load_predictor(sam_model: str, device: str):
    """Lazy SAM2 import + predictor build. Raises a clear error if SAM2 isn't installed."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        # NOTE: the SAM2 repo's PyPI metadata name is `sam-2` (with hyphen), not
        # `sam2` — the `<name> @ git+URL` install form fails name-matching.
        # Use the bare git URL.
        raise RuntimeError(
            "SAM2 not installed. Install with one of:\n"
            "    pip install git+https://github.com/facebookresearch/sam2.git\n"
            "    uv pip install --system git+https://github.com/facebookresearch/sam2.git"
        ) from e

    hf_id_map = {
        "sam2_t": "facebook/sam2-hiera-tiny",
        "sam2_s": "facebook/sam2-hiera-small",
        "sam2_b": "facebook/sam2-hiera-base-plus",
        "sam2_l": "facebook/sam2-hiera-large",
        "sam2.1_t": "facebook/sam2.1-hiera-tiny",
        "sam2.1_s": "facebook/sam2.1-hiera-small",
        "sam2.1_b": "facebook/sam2.1-hiera-base-plus",
        "sam2.1_l": "facebook/sam2.1-hiera-large",
    }
    hf_id = hf_id_map.get(sam_model, sam_model)  # accept raw HF id too
    predictor = SAM2ImagePredictor.from_pretrained(hf_id, device=device)
    return predictor


def run(coco_json: str | Path, images_dir: str | Path, out_dir: str | Path,
        sam_model: str = "sam2_b", device: str = "cuda",
        max_images: int | None = None, progress_every: int = 25) -> RunStats:
    coco_json = Path(coco_json)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {im_id: [] for im_id in images_by_id}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    stats = RunStats(sam_model_used=sam_model,
                     timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()

    items = list(images_by_id.items())
    if max_images is not None:
        items = items[:max_images]

    predictor = None
    for idx, (im_id, im) in enumerate(items):
        img_path = images_dir / im["file_name"]
        out_path = out_dir / (Path(im["file_name"]).stem + ".txt")
        if not img_path.exists():
            stats.errors.append(f"missing image: {img_path}")
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            stats.n_images_skipped += 1
            continue
        anns = anns_by_image.get(im_id, [])
        if not anns:
            # write empty file to mark "processed, no objects" so we skip on rerun
            out_path.write_text("")
            stats.n_images_processed += 1
            continue

        if predictor is None:
            predictor = _load_predictor(sam_model, device)

        try:
            lines, im_stats = process_image(img_path, anns, predictor)
        except Exception as e:  # noqa: BLE001 — keep going on per-image faults
            stats.errors.append(f"{img_path.name}: {e}")
            continue
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats.n_images_processed += 1
        stats.n_objects_processed += im_stats.n_objects
        stats.n_obb_extracted += im_stats.n_obb
        stats.n_aabb_fallback += im_stats.n_round_fallback
        stats.n_invalid_dropped += im_stats.n_invalid

        if (idx + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (stats.n_images_processed) / max(elapsed, 1e-6)
            print(f"  [{idx + 1}/{len(items)}] processed={stats.n_images_processed} "
                  f"skipped={stats.n_images_skipped} obb={stats.n_obb_extracted} "
                  f"round={stats.n_aabb_fallback} drop={stats.n_invalid_dropped} "
                  f"({rate:.1f} img/s)")

    stats.duration_seconds = round(time.time() - t0, 2)
    manifest = {k: v for k, v in asdict(stats).items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return stats
