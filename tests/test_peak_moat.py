"""hm_peak_moat: depress non-peak cells around each peak so PeakSuppress can't keep adjacent ties."""
import numpy as np

from opndet.encode import encode_targets, encode_targets_obb


class _Shim:
    img_h = 128
    img_w = 256
    stride = 4
    out_h = 32
    out_w = 64
    hm_blob_frac = 0.0
    hm_target = "ellipse"
    hm_ellipse_edge_margin = 0.1
    peak_kernel = 5


def _shim(**kw):
    s = _Shim()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_moat_off_by_default():
    box = np.array([[108, 34, 148, 94]], np.float32)  # center (128,64) → cell (16,32)
    hm = encode_targets(box, _shim())["hm"][0].numpy()
    assert hm[16, 32] > 0.99
    assert hm[16, 33] > 0.7   # ellipse dome value at d=1 cell — unmoated


def test_moat_depresses_neighbors():
    box = np.array([[108, 34, 148, 94]], np.float32)
    hm = encode_targets(box, _shim(hm_peak_moat=0.3))["hm"][0].numpy()
    assert hm[16, 32] > 0.99                       # peak untouched
    # every cell within L∞=2 of the peak, except the peak itself, ≤ 1.0 - 0.3
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            assert hm[16 + dy, 32 + dx] <= 0.7 + 1e-6, (dy, dx, hm[16 + dy, 32 + dx])


def test_adjacent_peaks_do_not_depress_each_other():
    # two box centers exactly 1 cell apart → cells (16,32) and (16,33)
    boxes = np.array([[108, 34, 148, 94], [112, 34, 152, 94]], np.float32)
    hm = encode_targets(boxes, _shim(hm_peak_moat=0.3))["hm"][0].numpy()
    assert hm[16, 32] > 0.99 and hm[16, 33] > 0.99   # both peaks survive the moat
    assert hm[16, 31] <= 0.7 + 1e-6                   # non-peak flanks still moated
    assert hm[16, 34] <= 0.7 + 1e-6


def test_moat_applies_to_obb_encoder():
    obbs = np.array([[128.0, 64.0, 40.0, 60.0, 0.0]], np.float32)  # cell (16,32)
    hm = encode_targets_obb(obbs, _shim(hm_peak_moat=0.3))["hm"][0].numpy()
    assert hm[16, 32] > 0.99
    assert hm[16, 33] <= 0.7 + 1e-6
