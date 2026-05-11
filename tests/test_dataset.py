import json

import cv2
import numpy as np
import pytest


def _make_coco(tmp_path, names):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    images, anns = [], []
    for i, n in enumerate(names, start=1):
        cv2.imwrite(str(img_dir / n), img)
        images.append({"id": i, "file_name": n, "width": 64, "height": 64})
        anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": [8, 8, 16, 16], "iscrowd": 0})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "x"}]}
    coco_path = tmp_path / "ann.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, img_dir


def test_image_filter_none_keeps_all(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    samples = load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=None)
    assert len(samples) == 3


def test_image_filter_basenames(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    flt = tmp_path / "keep.txt"
    flt.write_text("a.jpg\nc.jpg\n")
    samples = load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=str(flt))
    assert sorted(s.image_path.name for s in samples) == ["a.jpg", "c.jpg"]


def test_image_filter_stems_and_comments(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    flt = tmp_path / "keep.txt"
    flt.write_text("# scenario 1\nb\n\n  c.jpg  \n")  # stem, blank, padded basename
    samples = load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=str(flt))
    assert sorted(s.image_path.name for s in samples) == ["b.jpg", "c.jpg"]


def test_image_filter_empty_file_keeps_all(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg", "b.jpg"])
    flt = tmp_path / "empty.txt"
    flt.write_text("# nothing here\n\n")
    samples = load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=str(flt))
    assert len(samples) == 2


def test_image_filter_missing_file_raises(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg"])
    with pytest.raises(FileNotFoundError):
        load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=str(tmp_path / "nope.txt"))


def test_image_filter_no_match_raises(tmp_path):
    from opndet.dataset import load_datasets
    coco, img_dir = _make_coco(tmp_path, ["a.jpg", "b.jpg"])
    flt = tmp_path / "keep.txt"
    flt.write_text("zzz.jpg\n")
    with pytest.raises(ValueError):
        load_datasets([{"coco": str(coco), "images": str(img_dir)}], image_filter=str(flt))
