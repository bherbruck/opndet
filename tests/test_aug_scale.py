"""scale_jitter / translate_frac augmentation (now wired into _geometric)."""
import numpy as np

from opndet.augment import AugConfig, _geometric


def _cfg(**kw):
    base = dict(brightness=0, contrast=0, gamma=(1.0, 1.0), hue=0, saturation=0,
                grayscale_prob=0, blur_prob=0, noise_sigma=0,
                hflip_prob=0, vflip_prob=0, rotate90_prob=0,
                scale_jitter=(1.0, 1.0), translate_frac=0.0,
                mosaic_prob=0, cutout_prob=0, min_visible_frac=0.5)
    base.update(kw)
    return AugConfig(**base)


def _img(h=128, w=160):
    return (np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3) % 255).astype(np.uint8)


def test_noop_when_default():
    img = _img()
    boxes = np.array([[20, 20, 60, 60], [100, 80, 130, 110]], np.float32)
    out_img, out_b, out_o = _geometric(img.copy(), boxes.copy(), _cfg(), np.random.default_rng(0), obbs=None)
    assert np.array_equal(out_img, img)
    assert np.array_equal(out_b, boxes)


def test_zoom_out_keeps_all_boxes_shrunk():
    img = _img()
    boxes = np.array([[20, 20, 60, 60], [100, 80, 130, 110]], np.float32)
    cfg = _cfg(scale_jitter=(0.5, 0.5))  # zoom out 2× → everything shrinks toward center
    _, out_b, _ = _geometric(img.copy(), boxes.copy(), cfg, np.random.default_rng(0), obbs=None)
    assert out_b.shape[0] == 2  # all kept (still fully in frame)
    # each box halved in size
    for orig, new in zip(boxes, out_b):
        ow = orig[2] - orig[0]; nw = new[2] - new[0]
        assert abs(nw - ow * 0.5) < 1.0


def test_zoom_in_drops_boxes_that_fall_out():
    img = _img(128, 160)
    # one box near center (survives a zoom-in), one near a corner (gets pushed out)
    boxes = np.array([[70, 56, 90, 72], [0, 0, 18, 18]], np.float32)
    cfg = _cfg(scale_jitter=(3.0, 3.0))  # zoom in 3× about the center
    _, out_b, _ = _geometric(img.copy(), boxes.copy(), cfg, np.random.default_rng(0), obbs=None)
    assert out_b.shape[0] == 1  # the corner box left the frame → dropped


def test_obb_theta_invariant_wh_scaled_and_kept_in_sync():
    img = _img()
    # boxes = AABB envelope of the OBBs (matches how OBB models feed _geometric)
    obbs = np.array([[80.0, 64.0, 30.0, 18.0, 0.7], [10.0, 10.0, 16.0, 16.0, 1.2]], np.float32)
    boxes = np.array([[65, 55, 95, 73], [2, 2, 18, 18]], np.float32)
    cfg = _cfg(scale_jitter=(3.0, 3.0))  # zoom in → drops the corner one
    _, out_b, out_o = _geometric(img.copy(), boxes.copy(), cfg, np.random.default_rng(0), obbs=obbs.copy())
    assert out_b.shape[0] == out_o.shape[0] == 1  # synced drop
    # θ unchanged; w,h scaled by 3
    assert abs(float(out_o[0, 4]) - 0.7) < 1e-4
    assert abs(float(out_o[0, 2]) - 30.0 * 3.0) < 1e-3
    assert abs(float(out_o[0, 3]) - 18.0 * 3.0) < 1e-3


def test_translate_only_shifts():
    img = _img(128, 160)
    boxes = np.array([[70, 56, 90, 72]], np.float32)  # near center so it stays in frame
    cfg = _cfg(translate_frac=0.1)
    _, out_b, _ = _geometric(img.copy(), boxes.copy(), cfg, np.random.default_rng(1), obbs=None)
    assert out_b.shape[0] == 1
    # size unchanged (translate only), center moved
    assert abs((out_b[0, 2] - out_b[0, 0]) - 20.0) < 1.0
    moved = abs((out_b[0, 0] + out_b[0, 2]) / 2 - 80.0) + abs((out_b[0, 1] + out_b[0, 3]) / 2 - 64.0)
    assert moved > 0.5
