"""Tests for SAM → OBB preprocessing. ROADMAP §2.1 Phase 4a."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from opndet.sam_preprocess import (coco_bbox_to_xyxy, corners_to_yolo_obb_line,
                                   is_valid_rectangle, mask_to_obb_corners,
                                   process_image)


def _ellipse_mask(h: int, w: int, cx: float, cy: float, major: float, minor: float,
                  angle_deg: float = 0.0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (int(round(major / 2)), int(round(minor / 2))),
                angle_deg, 0, 360, 255, thickness=-1)
    return (mask > 0).astype(np.uint8)


def test_coco_bbox_to_xyxy_basic():
    assert coco_bbox_to_xyxy([1, 2, 10, 20]) == [1.0, 2.0, 11.0, 22.0]


def test_mask_to_obb_corners_axis_aligned_ellipse():
    mask = _ellipse_mask(200, 300, cx=150, cy=100, major=140, minor=40, angle_deg=0)
    corners = mask_to_obb_corners(mask)
    assert corners is not None and corners.shape == (4, 2)
    assert is_valid_rectangle(corners)
    # axis-aligned ⇒ x extents ≈ major, y extents ≈ minor
    xs, ys = corners[:, 0], corners[:, 1]
    assert abs((xs.max() - xs.min()) - 140) < 6
    assert abs((ys.max() - ys.min()) - 40) < 6


def test_mask_to_obb_corners_rotated_ellipse():
    mask = _ellipse_mask(300, 300, cx=150, cy=150, major=160, minor=40, angle_deg=45)
    corners = mask_to_obb_corners(mask)
    assert corners is not None
    assert is_valid_rectangle(corners)
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    long_side, short_side = max(sides), min(sides)
    assert abs(long_side - 160) < 10
    assert abs(short_side - 40) < 10


def test_mask_to_obb_corners_round_falls_back_to_aabb():
    # near-circular: aspect ratio ~1.0 < ROUNDNESS_THRESHOLD (1.15)
    mask = _ellipse_mask(200, 200, cx=100, cy=100, major=80, minor=78, angle_deg=0)
    corners = mask_to_obb_corners(mask)
    assert corners is not None
    # AABB fallback ⇒ axis-aligned (y of 0,1 equal; y of 2,3 equal)
    assert abs(corners[0, 1] - corners[1, 1]) < 1e-3
    assert abs(corners[2, 1] - corners[3, 1]) < 1e-3
    assert abs(corners[0, 0] - corners[3, 0]) < 1e-3
    assert abs(corners[1, 0] - corners[2, 0]) < 1e-3


def test_mask_to_obb_corners_empty_returns_none():
    assert mask_to_obb_corners(np.zeros((50, 50), dtype=np.uint8)) is None


def test_mask_to_obb_corners_tiny_blob_returns_none():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10, 10] = 1  # single pixel — contour <5 points
    assert mask_to_obb_corners(mask) is None


def test_is_valid_rectangle_true_for_real_rect():
    corners = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    assert is_valid_rectangle(corners)


def test_is_valid_rectangle_false_for_non_perpendicular():
    corners = np.array([[0, 0], [10, 0], [12, 5], [2, 5]], dtype=np.float32)
    assert not is_valid_rectangle(corners)


def test_is_valid_rectangle_false_for_unequal_opposite_sides():
    corners = np.array([[0, 0], [10, 0], [10, 5], [0, 8]], dtype=np.float32)
    assert not is_valid_rectangle(corners)


def test_is_valid_rectangle_false_for_none_or_wrong_shape():
    assert not is_valid_rectangle(None)
    assert not is_valid_rectangle(np.zeros((3, 2), dtype=np.float32))


def test_corners_to_yolo_obb_line_format_and_normalization():
    corners = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32)
    line = corners_to_yolo_obb_line(corners, img_w=400, img_h=200, class_id=0)
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 9
    floats = [float(x) for x in parts[1:]]
    # (0,0), (200/400=0.5, 0), (0.5, 100/200=0.5), (0, 0.5)
    assert floats == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5], abs=1e-6)


def test_corners_to_yolo_obb_line_allows_out_of_bounds():
    # truncated object at frame edge — corner extends beyond image
    corners = np.array([[-5, 0], [50, 0], [50, 30], [-5, 30]], dtype=np.float32)
    line = corners_to_yolo_obb_line(corners, img_w=100, img_h=100)
    floats = [float(x) for x in line.split()[1:]]
    assert floats[0] < 0  # negative coord preserved


class _FakeSAM2Predictor:
    """Stand-in for SAM2ImagePredictor: returns an elliptical mask per box prompt."""

    def __init__(self):
        self._h = 0
        self._w = 0

    def set_image(self, img_rgb: np.ndarray) -> None:
        self._h, self._w = img_rgb.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=False):
        boxes = np.atleast_2d(np.asarray(box, dtype=np.float32))
        masks = []
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # synthesize an elongated ellipse inside the box (long axis = box width)
            major = max(2.0, (x2 - x1) * 0.95)
            minor = max(2.0, (y2 - y1) * 0.4)
            masks.append(_ellipse_mask(self._h, self._w, cx, cy, major, minor, 0.0))
        masks_arr = np.stack(masks, axis=0)[:, None]  # (N, 1, H, W) — SAM2 batched shape
        scores = np.ones((len(boxes), 1), dtype=np.float32)
        return masks_arr, scores, None


def test_process_image_with_fake_predictor(tmp_path: Path):
    img = np.full((200, 300, 3), 30, dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    annotations = [
        {"bbox": [50, 80, 200, 40]},   # AABB: x,y,w,h → xyxy = (50,80,250,120)
        {"bbox": [10, 10, 30, 30]},    # near-square — predictor still elongates → fitEllipse path
    ]
    predictor = _FakeSAM2Predictor()
    lines, stats = process_image(img_path, annotations, predictor)
    assert stats.n_objects == 2
    assert len(lines) == stats.n_obb
    assert stats.n_obb >= 1  # at least the wide box should yield a valid OBB
    for ln in lines:
        parts = ln.split()
        assert parts[0] == "0"
        assert len(parts) == 9


def test_run_writes_manifest_and_skips_existing(tmp_path: Path, monkeypatch):
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    out_dir = tmp_path / "labels"
    img = np.full((100, 200, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "a.jpg"), img)
    cv2.imwrite(str(images_dir / "b.jpg"), img)

    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [20, 20, 100, 30], "iscrowd": 0},
        ],
        "categories": [{"id": 0, "name": "x"}],
    }
    coco_path = tmp_path / "ann.json"
    coco_path.write_text(json.dumps(coco))

    # patch loader to return the fake predictor (avoids SAM2 dep)
    import opndet.sam_preprocess as sp
    monkeypatch.setattr(sp, "_load_predictor", lambda *a, **kw: _FakeSAM2Predictor())

    stats = sp.run(coco_path, images_dir, out_dir, sam_model="sam2_b", device="cpu")
    assert (out_dir / "a.txt").exists()
    assert (out_dir / "b.txt").exists()  # empty (no anns)
    assert (out_dir / "b.txt").read_text() == ""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["sam_model_used"] == "sam2_b"
    assert manifest["n_images_processed"] == 2
    assert manifest["n_objects_processed"] == 1
    assert "timestamp" in manifest

    # Second run: all outputs already exist; image w/ content should be skipped,
    # the empty b.txt is treated as needing reprocess (size==0). a.txt is the
    # idempotency check.
    stats2 = sp.run(coco_path, images_dir, out_dir, sam_model="sam2_b", device="cpu")
    assert stats2.n_images_skipped >= 1


def test_load_predictor_missing_sam2_raises_clear_error(monkeypatch):
    import opndet.sam_preprocess as sp

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sam2"):
            raise ImportError("no sam2")
        return real_import(name, *a, **kw)

    # remove any cached sam2 module
    for k in list(sys.modules):
        if k.startswith("sam2"):
            del sys.modules[k]
    monkeypatch.setitem(sys.modules, "sam2", None)  # force ImportError on import sam2.*

    with pytest.raises(RuntimeError, match="SAM2 not installed"):
        sp._load_predictor("sam2_b", "cpu")


def test_run_coco_to_obb_polygons(tmp_path):
    """COCO with polygon segmentation → OBB sidecars (no SAM needed)."""
    import json
    import cv2
    from opndet.sam_preprocess import run_coco_to_obb
    # 200×150 canvas, one rotated ellipse (mask via polygon — sample 24 points around it)
    cx, cy, a, b, ang = 100.0, 75.0, 36.0, 14.0, 30.0
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cos_t, sin_t = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    xs = cx + a * np.cos(th) * cos_t - b * np.sin(th) * sin_t
    ys = cy + a * np.cos(th) * sin_t + b * np.sin(th) * cos_t
    ring = np.column_stack([xs, ys]).flatten().tolist()
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 150}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [60, 60, 80, 30], "iscrowd": 0,
                             "segmentation": [ring]}],
            "categories": [{"id": 0, "name": "obj"}]}
    cp = tmp_path / "ann.json"; cp.write_text(json.dumps(coco))
    out = tmp_path / "obb"
    stats = run_coco_to_obb(cp, out)
    assert stats.n_obb_extracted == 1 and stats.n_invalid_dropped == 0
    txt = (out / "a.txt").read_text().strip()
    parts = txt.split()
    assert len(parts) == 9 and parts[0] == "0"   # class_id + 4 (x,y) corners normalized
    # idempotent: re-run skips
    stats2 = run_coco_to_obb(cp, out)
    assert stats2.n_images_skipped == 1 and stats2.n_obb_extracted == 0


def test_run_coco_to_obb_rle(tmp_path):
    """COCO with RLE segmentation (uncompressed list-counts) → OBB sidecar."""
    import json
    from opndet.sam_preprocess import run_coco_to_obb
    # 12×8 image, RLE marks columns 4..7 (a 4-col strip, column-major counts)
    rle = {"size": [8, 12], "counts": [32, 32, 32]}   # 32 bg, 32 fg, 32 bg → cols 4..7 fg
    coco = {"images": [{"id": 1, "file_name": "b.png", "width": 12, "height": 8}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [4, 0, 4, 8], "iscrowd": 0,
                             "segmentation": rle}],
            "categories": [{"id": 0, "name": "obj"}]}
    cp = tmp_path / "ann.json"; cp.write_text(json.dumps(coco))
    out = tmp_path / "obb"
    stats = run_coco_to_obb(cp, out)
    # frame-clipped (the strip touches top+bottom edges) → minAreaRect path; still emits a line
    txt = (out / "b.txt").read_text().strip()
    assert txt and txt.startswith("0 ") and len(txt.split()) == 9
    assert stats.n_obb_extracted == 1
