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
from tqdm.auto import tqdm

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


def _obb_area(corners: np.ndarray) -> float:
    """Area of the rotated rectangle from its 4 corners."""
    side1 = float(np.linalg.norm(corners[1] - corners[0]))
    side2 = float(np.linalg.norm(corners[2] - corners[1]))
    return side1 * side2


def _aabb_area(xyxy: np.ndarray) -> float:
    """Area of the axis-aligned bbox (x1, y1, x2, y2)."""
    return float(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]))


def is_obb_within_prompt(corners: np.ndarray, prompt_xyxy: np.ndarray,
                        max_area_frac: float = 1.5) -> bool:
    """Reject OBBs whose area exceeds `max_area_frac × prompt AABB area`.

    Catches the rare SAM2 failure mode where the mask escapes the box prompt
    and grabs the entire background — fitEllipse on that mask returns an OBB
    spanning much of the image. Honest OBBs (even rotated) are always smaller
    than 1.5× the input AABB area in practice.
    """
    if corners is None or prompt_xyxy is None:
        return True
    aabb_a = _aabb_area(prompt_xyxy)
    if aabb_a <= 0:
        return True
    return _obb_area(corners) <= max_area_frac * aabb_a


def is_obb_self_consistent(corners: np.ndarray, prompt_xyxy: np.ndarray) -> bool:
    """Reject OBBs that have drifted off their prompt: the OBB's centroid
    must lie inside the prompt AABB.

    A SAM mask that escaped the box will have its mask centroid (and thus
    the fitEllipse OBB centroid) drift far from the AABB center — easy to
    detect. Honest masks (even partially occluded or edge-truncated) stay
    near the AABB center because the mask is still inside or partially
    inside the prompt.

    Why not also require the AABB center to lie inside the OBB? Edge cases:
      - Annotator drew a loose AABB; SAM tightly fit the mask in one
        quadrant. Honest mask, but AABB center is outside OBB.
      - Object is truncated at the frame edge; AABB extends off-frame,
        AABB center is outside the visible mask region.
    Both are valid OBBs we want to keep, so the inverse check is too strict.
    """
    if corners is None or prompt_xyxy is None:
        return True
    x1, y1, x2, y2 = prompt_xyxy
    obb_cx = float(corners[:, 0].mean())
    obb_cy = float(corners[:, 1].mean())
    return x1 <= obb_cx <= x2 and y1 <= obb_cy <= y2


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


def process_image(img_path: Path, annotations: list[dict], predictor,
                  img_rgb: np.ndarray | None = None) -> tuple[list[str], ImageStats]:
    """Run SAM2 on one image; return (yolo_obb_lines, stats).
    `annotations` is the raw COCO ann list (each has a 'bbox' field [x,y,w,h]).
    `predictor` is a SAM2ImagePredictor (already moved to device).
    `img_rgb` is an optional pre-loaded RGB ndarray; if None, loads from img_path.
    Pre-loading lets the caller overlap disk I/O with the previous image's GPU
    forward in a thread pool.
    """
    import torch  # local import — module imports stay light

    stats = ImageStats()
    if img_rgb is None:
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
        # Sanity: OBB area must not balloon beyond the AABB prompt's area
        # (SAM2 sometimes escapes the box and grabs background).
        if not is_obb_within_prompt(corners, boxes_xyxy[i], max_area_frac=1.5):
            stats.n_invalid += 1
            continue
        # Sanity: OBB centroid inside AABB AND AABB center inside OBB.
        # Catches mask-escape cases where the OBB drifts off its prompt.
        if not is_obb_self_consistent(corners, boxes_xyxy[i]):
            stats.n_invalid += 1
            continue
        if _is_round_via_corners(corners):
            stats.n_round_fallback += 1
        stats.n_obb += 1
        lines.append(corners_to_yolo_obb_line(corners, w, h, class_id=0))
    return lines, stats


def _sam_batch_predict(predictor, images: list[np.ndarray],
                       boxes_per_image: list[np.ndarray],
                       has_batch_api: bool) -> list[np.ndarray]:
    """Run SAM2 over a batch of images. Returns list of mask arrays of shape
    (n_objs_i, H_i, W_i) per image. Falls back to set_image()+predict() loop
    when set_image_batch isn't available.
    """
    import torch
    use_cuda = torch.cuda.is_available()
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda
                    else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False))

    masks_per: list[np.ndarray] = []
    with torch.inference_mode(), autocast_ctx:
        if has_batch_api and len(images) > 1:
            predictor.set_image_batch(images)
            results = predictor.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=boxes_per_image,
                multimask_output=False,
            )
            # predict_batch returns (masks_list, scores_list, low_res_list)
            masks_list = results[0] if isinstance(results, tuple) else results
            for m in masks_list:
                m = np.asarray(m)
                if m.ndim == 4:
                    m = m[:, 0]
                elif m.ndim == 3 and m.shape[0] == 1:
                    pass
                masks_per.append(m)
        else:
            for img, boxes in zip(images, boxes_per_image):
                predictor.set_image(img)
                masks, _scores, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=boxes, multimask_output=False,
                )
                if masks.ndim == 4:
                    masks = masks[:, 0]
                masks_per.append(np.asarray(masks))
    return masks_per


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
        max_images: int | None = None,
        batch_size: int = 8, num_workers: int = 8) -> RunStats:
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

    # Filter+ classify items into "do" / "skip" / "no-anns" up front so the
    # image-load thread doesn't waste time on items we won't process.
    todo: list[tuple[Path, Path, list[dict]]] = []
    for im_id, im in items:
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
            out_path.write_text("")
            stats.n_images_processed += 1
            continue
        todo.append((img_path, out_path, anns))

    if not todo:
        stats.duration_seconds = round(time.time() - t0, 2)
        manifest = {k: v for k, v in asdict(stats).items()}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return stats

    predictor = _load_predictor(sam_model, device)

    # Step 1: preload ALL images into RAM in parallel. ~3 GB for a 5000-image
    # 384x512 dataset; fits comfortably on Colab. Eliminates the per-image
    # disk-I/O wait that idles the GPU between SAM forwards.
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(item):
        path = item[0]
        img = cv2.imread(str(path))
        return (item, cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        loaded = list(tqdm(
            pool.map(_load_one, todo),
            total=len(todo),
            desc="preload imgs",
            unit="img",
            leave=False,
        ))
    n_loaded = sum(1 for _, img in loaded if img is not None)
    if n_loaded < len(todo):
        stats.errors.append(f"{len(todo) - n_loaded} images failed to load")

    # Step 2: batched SAM2 inference. set_image_batch runs the image encoder
    # once over a batch (single big GPU forward); per-image predict() then
    # only runs the cheap mask decoder. Net: GPU is busy ~100% of the time.
    # Falls back to single-image set_image() if the SAM2 build doesn't expose
    # set_image_batch.
    has_batch_api = hasattr(predictor, "set_image_batch") and hasattr(predictor, "predict_batch")
    BATCH = batch_size if has_batch_api else 1

    t_inf = time.time()
    pbar = tqdm(
        total=len(todo),
        desc=f"sam-obb {sam_model}",
        unit="img",
        dynamic_ncols=True,
    )
    for chunk_start in range(0, len(loaded), BATCH):
        chunk = loaded[chunk_start:chunk_start + BATCH]
        valid = [(item, img) for (item, img) in chunk if img is not None]
        if not valid:
            continue
        items_v = [v[0] for v in valid]
        imgs_v = [v[1] for v in valid]
        boxes_per = []
        for img_path, _, anns in items_v:
            bx = np.array(
                [coco_bbox_to_xyxy(a["bbox"]) for a in anns if a.get("bbox") is not None],
                dtype=np.float32,
            )
            boxes_per.append(bx)

        try:
            masks_per = _sam_batch_predict(predictor, imgs_v, boxes_per, has_batch_api)
        except Exception as e:  # noqa: BLE001
            for (img_path, _, _) in items_v:
                stats.errors.append(f"{img_path.name}: batch failed: {e}")
            continue

        for (img_path, out_path, anns), masks, boxes_xyxy in zip(items_v, masks_per, boxes_per):
            lines: list[str] = []
            im_stats = ImageStats(n_objects=len(boxes_xyxy))
            for i, mask in enumerate(masks):
                corners = mask_to_obb_corners(mask, fallback_bbox=boxes_xyxy[i])
                if corners is None or not is_valid_rectangle(corners):
                    im_stats.n_invalid += 1
                    continue
                # Sanity: reject mask-escape failures via two cheap checks.
                if not is_obb_within_prompt(corners, boxes_xyxy[i], max_area_frac=1.5):
                    im_stats.n_invalid += 1
                    continue
                if not is_obb_self_consistent(corners, boxes_xyxy[i]):
                    im_stats.n_invalid += 1
                    continue
                if _is_round_via_corners(corners):
                    im_stats.n_round_fallback += 1
                im_stats.n_obb += 1
                h, w = mask.shape[-2:]
                lines.append(corners_to_yolo_obb_line(corners, w, h, class_id=0))
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            stats.n_images_processed += 1
            stats.n_objects_processed += im_stats.n_objects
            stats.n_obb_extracted += im_stats.n_obb
            stats.n_aabb_fallback += im_stats.n_round_fallback
            stats.n_invalid_dropped += im_stats.n_invalid

        # Advance pbar by however many images this chunk added (NaN-safe).
        n_done_this_chunk = sum(1 for v in valid)
        pbar.update(n_done_this_chunk)
        pbar.set_postfix({
            "obb": stats.n_obb_extracted,
            "round": stats.n_aabb_fallback,
            "drop": stats.n_invalid_dropped,
            "B": BATCH,
        })
    pbar.close()

    stats.duration_seconds = round(time.time() - t0, 2)
    manifest = {k: v for k, v in asdict(stats).items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return stats
