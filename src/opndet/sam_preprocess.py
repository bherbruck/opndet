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
    # Per-rule drop breakdown (sums to n_invalid)
    n_drop_geometry: int = 0      # is_valid_rectangle failed (cv2.fitEllipse junk)
    n_drop_area: int = 0          # OBB area > max_area_frac × AABB area
    n_drop_centroid: int = 0      # OBB centroid not inside AABB
    n_drop_no_corners: int = 0    # mask_to_obb_corners returned None (no contour)
    n_kept_edge: int = 0          # mask touched image edge — area+centroid checks skipped


@dataclass
class RunStats:
    n_images_processed: int = 0
    n_images_skipped: int = 0
    n_objects_processed: int = 0
    n_obb_extracted: int = 0
    n_aabb_fallback: int = 0
    n_invalid_dropped: int = 0
    # Per-rule drop breakdown (sums to n_invalid_dropped)
    n_drop_geometry: int = 0
    n_drop_area: int = 0
    n_drop_centroid: int = 0
    n_drop_no_corners: int = 0
    n_kept_edge: int = 0
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
        if corners is None:
            stats.n_drop_no_corners += 1
            stats.n_invalid += 1
            continue
        if not is_valid_rectangle(corners):
            stats.n_drop_geometry += 1
            stats.n_invalid += 1
            continue
        if not is_obb_within_prompt(corners, boxes_xyxy[i], max_area_frac=1.5):
            stats.n_drop_area += 1
            stats.n_invalid += 1
            continue
        if not is_obb_self_consistent(corners, boxes_xyxy[i]):
            stats.n_drop_centroid += 1
            stats.n_invalid += 1
            continue
        if _is_round_via_corners(corners):
            stats.n_round_fallback += 1
        stats.n_obb += 1
        lines.append(corners_to_yolo_obb_line(corners, w, h, class_id=0))
    return lines, stats


def _aabb_corners_from_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """4 corners of the axis-aligned bbox in (x1,y1) (x2,y1) (x2,y2) (x1,y2) order."""
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _min_area_rect_corners(mask: np.ndarray) -> np.ndarray | None:
    """Tightest rotated rectangle enclosing the visible mask pixels — used for
    truncated objects where fitEllipse would extrapolate past the image edge.
    Returns 4 corners in cv2.boxPoints order, or None if the mask is empty.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def _mask_touches_edge(mask: np.ndarray | None, margin: int = 2) -> bool:
    """True if the SAM mask reaches the image edge — indicates the object is
    truncated. fitEllipse on truncated masks extrapolates past the edge, so
    area + centroid sanity checks have to be relaxed for these.
    """
    if mask is None:
        return False
    m = np.asarray(mask)
    if m.size == 0:
        return False
    m = (m > 0)
    if not m.any():
        return False
    h, w = m.shape[-2:]
    return bool(
        m[..., :margin, :].any() or m[..., -margin:, :].any()
        or m[..., :, :margin].any() or m[..., :, -margin:].any()
    )


def _save_rejected_preview(img_rgb: np.ndarray, mask: np.ndarray | None,
                           prompt_xyxy: np.ndarray, corners: np.ndarray | None,
                           rule: str, out_path: Path) -> None:
    """Save a 3-panel diagnostic image: AABB | mask overlay | candidate OBB.
    Used to debug why an OBB was rejected.
    """
    h, w = img_rgb.shape[:2]
    panel1 = img_rgb.copy()
    if prompt_xyxy is not None:
        x1, y1, x2, y2 = prompt_xyxy.astype(int)
        cv2.rectangle(panel1, (int(x1), int(y1)), (int(x2), int(y2)), (200, 50, 220), 3)
    cv2.putText(panel1, "1. AABB prompt", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel1, "1. AABB prompt", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    panel2 = img_rgb.copy()
    if mask is not None and mask.size > 0:
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        red = np.zeros_like(panel2); red[:, :, 0] = 255
        panel2 = np.where(m[..., None] > 0,
                          cv2.addWeighted(panel2, 0.5, red, 0.5, 0),
                          panel2).astype(np.uint8)
    cv2.putText(panel2, "2. SAM mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel2, "2. SAM mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    panel3 = img_rgb.copy()
    if corners is not None:
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel3, [pts], isClosed=True, color=(50, 220, 50), thickness=3)
    cv2.putText(panel3, f"3. REJECTED: {rule}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel3, f"3. REJECTED: {rule}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (50, 50, 255), 1, cv2.LINE_AA)

    grid = np.concatenate([panel1, panel2, panel3], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


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
        batch_size: int = 8, num_workers: int = 8,
        save_rejected: int = 16,
        image_filter: str | Path | None = None) -> RunStats:
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
    if image_filter is not None:
        from opndet.dataset import _read_image_filter
        match, n_entries = _read_image_filter(image_filter)
        if not match:
            print(f"  image_filter {image_filter}: empty — processing all {len(items)} images")
        else:
            kept = [(iid, im) for (iid, im) in items
                    if Path(im["file_name"]).name in match or Path(im["file_name"]).stem in match]
            print(f"  image_filter {image_filter}: {n_entries} entries → SAM will run on {len(kept)}/{len(items)} images")
            if not kept:
                raise ValueError(f"image_filter {image_filter} matched 0 of this COCO's images — check the names")
            items = kept
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

    # Preview saves of REJECTED OBBs (3-panel: AABB | mask | candidate OBB)
    # Capped per-rule so we don't fill the disk on a bad run; lets the user
    # eyeball "what does the centroid-rejection failure mode look like".
    rejected_dir = out_dir / "_rejected"
    saved_per_rule: dict[str, int] = {}

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

        for (img_path, out_path, anns), masks, boxes_xyxy, img_rgb in zip(
                items_v, masks_per, boxes_per, imgs_v):
            lines: list[str] = []
            im_stats = ImageStats(n_objects=len(boxes_xyxy))

            def _maybe_save_reject(rule: str, mask, corners):
                if saved_per_rule.get(rule, 0) >= save_rejected:
                    return
                saved_per_rule[rule] = saved_per_rule.get(rule, 0) + 1
                fname = rejected_dir / f"{img_path.stem}_obj{i:02d}_{rule}.png"
                _save_rejected_preview(img_rgb, mask, boxes_xyxy[i], corners, rule, fname)

            for i, mask in enumerate(masks):
                # Truncated objects: SAM mask touches an image edge. fitEllipse
                # would extrapolate the ellipse off-frame, producing an OBB
                # that fails the area + centroid sanity checks. Use minAreaRect
                # instead — tightest rotated rectangle bounding the VISIBLE
                # mask pixels. Preserves rotation info, no extrapolation, fits
                # within the prompt AABB by construction.
                if _mask_touches_edge(mask):
                    corners = _min_area_rect_corners(mask)
                    if corners is None:
                        # mask had no contour after all — fall back to AABB
                        corners = _aabb_corners_from_xyxy(boxes_xyxy[i])
                    im_stats.n_kept_edge += 1
                else:
                    corners = mask_to_obb_corners(mask, fallback_bbox=boxes_xyxy[i])
                    if corners is None:
                        im_stats.n_drop_no_corners += 1
                        im_stats.n_invalid += 1
                        _maybe_save_reject("no_corners", mask, corners)
                        continue
                    if not is_valid_rectangle(corners):
                        im_stats.n_drop_geometry += 1
                        im_stats.n_invalid += 1
                        _maybe_save_reject("geometry", mask, corners)
                        continue
                    if not is_obb_within_prompt(corners, boxes_xyxy[i], max_area_frac=1.5):
                        im_stats.n_drop_area += 1
                        im_stats.n_invalid += 1
                        _maybe_save_reject("area", mask, corners)
                        continue
                    if not is_obb_self_consistent(corners, boxes_xyxy[i]):
                        im_stats.n_drop_centroid += 1
                        im_stats.n_invalid += 1
                        _maybe_save_reject("centroid", mask, corners)
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
            stats.n_drop_no_corners += im_stats.n_drop_no_corners
            stats.n_drop_geometry += im_stats.n_drop_geometry
            stats.n_drop_area += im_stats.n_drop_area
            stats.n_drop_centroid += im_stats.n_drop_centroid
            stats.n_kept_edge += im_stats.n_kept_edge

        # Advance pbar by however many images this chunk added (NaN-safe).
        n_done_this_chunk = sum(1 for v in valid)
        pbar.update(n_done_this_chunk)
        pbar.set_postfix({
            "obb": stats.n_obb_extracted,
            "round": stats.n_aabb_fallback,
            "edge": stats.n_kept_edge,
            "area": stats.n_drop_area,
            "ctr": stats.n_drop_centroid,
            "geo": stats.n_drop_geometry,
            "B": BATCH,
        })
    pbar.close()

    stats.duration_seconds = round(time.time() - t0, 2)
    manifest = {k: v for k, v in asdict(stats).items()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return stats


@dataclass
class SegRunStats:
    n_images_processed: int = 0
    n_images_skipped: int = 0
    n_objects_processed: int = 0
    n_empty_masks: int = 0          # SAM returned an all-zero mask for an object
    sam_model_used: str = ""
    clip_to_box: bool = True
    timestamp: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def _masks_to_instance_map(masks: np.ndarray, boxes_xyxy: np.ndarray, H: int, W: int,
                           clip_to_box: bool, box_pad: int = 1) -> tuple[np.ndarray, int]:
    """(n_obj,H,W) bool masks → a [H,W] uint16 instance-id label map (0=bg, k=obj k).

    `clip_to_box`: zero any mask pixel outside its prompt AABB (a few px of pad) — SAM2
    occasionally lets a mask escape the box and grab background; the AABB is GT, so cut it.
    Later objects overwrite earlier ones on overlap (fine for non-overlapping objects).
    """
    lbl = np.zeros((H, W), dtype=np.uint16)
    n_empty = 0
    for i, m in enumerate(masks):
        mm = (np.asarray(m) > 0)
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if clip_to_box and i < len(boxes_xyxy):
            x1, y1, x2, y2 = boxes_xyxy[i]
            keep = np.zeros((H, W), dtype=bool)
            xa, ya = max(0, int(np.floor(x1)) - box_pad), max(0, int(np.floor(y1)) - box_pad)
            xb, yb = min(W, int(np.ceil(x2)) + box_pad), min(H, int(np.ceil(y2)) + box_pad)
            keep[ya:yb, xa:xb] = True
            mm = mm & keep
        if not mm.any():
            n_empty += 1
            continue
        lbl[mm] = i + 1
    return lbl, n_empty


def run_seg(coco_json: str | Path, images_dir: str | Path, out_dir: str | Path,
            sam_model: str = "sam2_b", device: str = "cuda",
            max_images: int | None = None,
            batch_size: int = 8, num_workers: int = 8,
            clip_to_box: bool = True,
            image_filter: str | Path | None = None) -> SegRunStats:
    """Box-prompt SAM2 with the COCO GT AABBs and dump each image's per-instance masks as a
    16-bit `<stem>.png` instance-id label map (0=bg, k=instance k) to `out_dir`. Idempotent
    (skips existing non-empty outputs). This is `sam-obb` minus the OBB-fitting step — the
    seg head trains on these masks directly (true distance-transform dome). `clip_to_box`
    cuts any mask pixels that escaped the prompt AABB.
    """
    coco_json, images_dir, out_dir = Path(coco_json), Path(images_dir), Path(out_dir)
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

    stats = SegRunStats(sam_model_used=sam_model, clip_to_box=clip_to_box,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()

    items = list(images_by_id.items())
    if image_filter is not None:
        from opndet.dataset import _read_image_filter
        match, n_entries = _read_image_filter(image_filter)
        if not match:
            print(f"  image_filter {image_filter}: empty — processing all {len(items)} images")
        else:
            kept = [(iid, im) for (iid, im) in items
                    if Path(im["file_name"]).name in match or Path(im["file_name"]).stem in match]
            print(f"  image_filter {image_filter}: {n_entries} entries → SAM will run on {len(kept)}/{len(items)} images")
            if not kept:
                raise ValueError(f"image_filter {image_filter} matched 0 of this COCO's images — check the names")
            items = kept
    if max_images is not None:
        items = items[:max_images]

    todo: list[tuple[Path, Path, list[dict]]] = []
    for im_id, im in items:
        img_path = images_dir / im["file_name"]
        out_path = out_dir / (Path(im["file_name"]).stem + ".png")
        if not img_path.exists():
            stats.errors.append(f"missing image: {img_path}")
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            stats.n_images_skipped += 1
            continue
        anns = anns_by_image.get(im_id, [])
        if not anns:
            # no objects → an all-zero label map (so a re-run skips it; the dataset just gets an empty dome)
            cv2.imwrite(str(out_path), np.zeros((int(im["height"]), int(im["width"])), np.uint16))
            stats.n_images_processed += 1
            continue
        todo.append((img_path, out_path, anns))

    if not todo:
        stats.duration_seconds = round(time.time() - t0, 2)
        (out_dir / "manifest_seg.json").write_text(json.dumps(asdict(stats), indent=2))
        return stats

    predictor = _load_predictor(sam_model, device)
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(item):
        img = cv2.imread(str(item[0]))
        return (item, cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        loaded = list(tqdm(pool.map(_load_one, todo), total=len(todo), desc="preload imgs", unit="img", leave=False))

    has_batch_api = hasattr(predictor, "set_image_batch") and hasattr(predictor, "predict_batch")
    BATCH = batch_size if has_batch_api else 1
    pbar = tqdm(total=len(todo), desc=f"sam-seg {sam_model}", unit="img", dynamic_ncols=True)
    for chunk_start in range(0, len(loaded), BATCH):
        chunk = [(it, im) for (it, im) in loaded[chunk_start:chunk_start + BATCH] if im is not None]
        if not chunk:
            continue
        items_v = [c[0] for c in chunk]; imgs_v = [c[1] for c in chunk]
        boxes_per = [np.array([coco_bbox_to_xyxy(a["bbox"]) for a in anns if a.get("bbox") is not None],
                              dtype=np.float32) for (_, _, anns) in items_v]
        try:
            masks_per = _sam_batch_predict(predictor, imgs_v, boxes_per, has_batch_api)
        except Exception as e:  # noqa: BLE001
            for (img_path, _, _) in items_v:
                stats.errors.append(f"{img_path.name}: batch failed: {e}")
            continue
        for (img_path, out_path, anns), masks, boxes_xyxy, img_rgb in zip(items_v, masks_per, boxes_per, imgs_v):
            H, W = img_rgb.shape[:2]
            lbl, n_empty = _masks_to_instance_map(masks, boxes_xyxy, H, W, clip_to_box=clip_to_box)
            cv2.imwrite(str(out_path), lbl)   # 16-bit single-channel PNG
            stats.n_images_processed += 1
            stats.n_objects_processed += len(boxes_xyxy)
            stats.n_empty_masks += n_empty
        n_done = len(chunk)
        pbar.update(n_done)
        pbar.set_postfix({"objs": stats.n_objects_processed, "empty": stats.n_empty_masks, "B": BATCH})
    pbar.close()

    stats.duration_seconds = round(time.time() - t0, 2)
    (out_dir / "manifest_seg.json").write_text(json.dumps(asdict(stats), indent=2))
    return stats


# ── coco-segmentation → OBB sidecars (no SAM needed; masks are already the GT) ─────────────

def _seg_to_mask(seg, img_h: int, img_w: int) -> np.ndarray | None:
    """One COCO annotation's `segmentation` → [img_h, img_w] uint8 binary mask, or None.
    Handles polygons (cv2.fillPoly + a sanity clamp on wildly-out-of-bounds coords — a known
    Roboflow-export glitch) and RLE (the native `dataset._decode_coco_rle`, no pycocotools)."""
    if seg is None:
        return None
    if isinstance(seg, dict):
        from opndet.dataset import _decode_coco_rle
        m = _decode_coco_rle(seg)
        if m.shape != (img_h, img_w):
            m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return m.astype(np.uint8) if m.any() else None
    if isinstance(seg, list):
        mask = np.zeros((img_h, img_w), np.uint8)
        for ring in seg:
            try:
                pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
            except (ValueError, TypeError):
                continue
            if (pts.shape[0] < 3 or not np.isfinite(pts).all()
                    or (pts[:, 0] < -2).any() or (pts[:, 0] > img_w + 2).any()
                    or (pts[:, 1] < -2).any() or (pts[:, 1] > img_h + 2).any()):
                continue
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        return mask if mask.any() else None
    return None


@dataclass
class CocoToObbStats:
    n_images_processed: int = 0
    n_images_skipped: int = 0
    n_objects_processed: int = 0
    n_obb_extracted: int = 0
    n_aabb_fallback: int = 0
    n_invalid_dropped: int = 0
    n_drop_geometry: int = 0
    n_drop_area: int = 0
    n_drop_centroid: int = 0
    n_drop_no_seg: int = 0
    n_kept_edge: int = 0
    timestamp: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_coco_to_obb(coco_json: str | Path, out_dir: str | Path,
                    max_images: int | None = None,
                    image_filter: str | Path | None = None) -> CocoToObbStats:
    """Fit OBBs to a COCO json's `segmentation` polygons/RLE and dump YOLOv8-OBB `<stem>.txt`
    sidecars (the same format `opndet sam-obb` writes). No SAM, no image files needed — pure
    metadata transform. Reuses the OBB-fit gauntlet from `sam-obb` (fitEllipse with minAreaRect
    fallback for frame-clipped objects + area/centroid validity checks). Idempotent (skips
    existing non-empty outputs)."""
    coco_json = Path(coco_json); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_json) as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {im_id: [] for im_id in images_by_id}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann.get("bbox") is None:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    stats = CocoToObbStats(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()

    items = list(images_by_id.items())
    if image_filter is not None:
        from opndet.dataset import _read_image_filter
        match, n_entries = _read_image_filter(image_filter)
        if match:
            items = [(iid, im) for (iid, im) in items
                     if Path(im["file_name"]).name in match or Path(im["file_name"]).stem in match]
            print(f"  image_filter {image_filter}: {n_entries} entries → {len(items)} images")
    if max_images is not None:
        items = items[:max_images]

    pbar = tqdm(items, desc="coco-to-obb", unit="img", dynamic_ncols=True)
    for im_id, im in pbar:
        out_path = out_dir / (Path(im["file_name"]).stem + ".txt")
        if out_path.exists() and out_path.stat().st_size > 0:
            stats.n_images_skipped += 1
            continue
        H, W = int(im["height"]), int(im["width"])
        anns = anns_by_image.get(im_id, [])
        lines: list[str] = []
        for i, ann in enumerate(anns):
            stats.n_objects_processed += 1
            mask = _seg_to_mask(ann.get("segmentation"), H, W)
            if mask is None:
                stats.n_drop_no_seg += 1; stats.n_invalid_dropped += 1
                continue
            box_xyxy = np.array(coco_bbox_to_xyxy(ann["bbox"]), dtype=np.float32)
            # truncated → minAreaRect on the visible pixels (no off-frame fitEllipse extrapolation)
            if _mask_touches_edge(mask):
                corners = _min_area_rect_corners(mask)
                if corners is None:
                    corners = _aabb_corners_from_xyxy(box_xyxy)
                stats.n_kept_edge += 1
            else:
                corners = mask_to_obb_corners(mask, fallback_bbox=box_xyxy)
                if corners is None:
                    stats.n_drop_no_seg += 1; stats.n_invalid_dropped += 1; continue
                if not is_valid_rectangle(corners):
                    stats.n_drop_geometry += 1; stats.n_invalid_dropped += 1; continue
                if not is_obb_within_prompt(corners, box_xyxy, max_area_frac=1.5):
                    stats.n_drop_area += 1; stats.n_invalid_dropped += 1; continue
                if not is_obb_self_consistent(corners, box_xyxy):
                    stats.n_drop_centroid += 1; stats.n_invalid_dropped += 1; continue
            if _is_round_via_corners(corners):
                stats.n_aabb_fallback += 1
            stats.n_obb_extracted += 1
            lines.append(corners_to_yolo_obb_line(corners, W, H, class_id=0))
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats.n_images_processed += 1
        pbar.set_postfix({"obb": stats.n_obb_extracted, "edge": stats.n_kept_edge,
                          "drop": stats.n_invalid_dropped})

    stats.duration_seconds = round(time.time() - t0, 2)
    (out_dir / "manifest_coco_to_obb.json").write_text(json.dumps(asdict(stats), indent=2))
    return stats
