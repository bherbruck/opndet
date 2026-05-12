"""bbox-*-seg: full-res dense-dome segmentation head — build, export parity, encode, loss."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from opndet.presets import resolve
from opndet.yaml_build import build_model_from_yaml


def _export(m, x, path):
    torch.onnx.export(m, x, path, input_names=["input"], output_names=["output"],
                      opset_version=13, do_constant_folding=True, dynamo=False)


# ---- 1. preset builds, forwards [1,1,H,W], exports opset-13 edge-clean, ORT==PT ----
# f (smallest) + s (mid) cover both width extremes; p/n/m are mechanical clones of the same
# decoder graph — no need to ONNX-export all five every run (each export is ~2 s).
@pytest.mark.parametrize("preset", ["bbox-f-seg", "bbox-s-seg"])
def test_seg_preset_builds_and_exports_opset13(preset):
    m = build_model_from_yaml(resolve(preset)).eval()
    c, h, w = m.input_shape
    assert "dome" in m.aliases, "seg head must expose the 'dome' alias (used for head detection)"
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        out = m(x)
    ot = out["output"] if isinstance(out, dict) else out
    assert ot.shape == (1, 1, h, w), f"{preset}: expected full-res [1,1,{h},{w}], got {tuple(ot.shape)}"
    assert float(ot.min()) >= 0.0 and float(ot.max()) <= 1.0
    # raw (pre-sigmoid 1-ch logit) used by SegDomeLoss
    assert m.forward_with_alias(x, "raw").shape == (1, 1, h, w)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / f"{preset}.onnx")
        _export(m, x, p)
        from opndet.export import allowed_ops_for_tier
        om = onnx.load(p)
        allowed = allowed_ops_for_tier("edge")
        bad = sorted({n.op_type for n in om.graph.node} - allowed)
        assert not bad, f"{preset} (edge tier) exported forbidden ops: {bad}"
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        o2 = sess.run(None, {"input": x.numpy()})[0]
        assert float(np.abs(o2 - ot.numpy()).max()) < 1e-4


def test_seg_runs_at_non_default_size():
    # fully conv + nearest-resize → any size divisible by 32 works
    m = build_model_from_yaml(resolve("bbox-n-seg"), img_h=256, img_w=384).eval()
    out = m(torch.randn(1, 3, 256, 384))
    ot = out["output"] if isinstance(out, dict) else out
    assert ot.shape == (1, 1, 256, 384)


# ---- 2. encode_targets_seg: dense dome from OBBs / masks ----
class _Shim:
    img_h = 128
    img_w = 192
    stride = 4          # ignored by seg encode (it renders at seg_stride)
    seg_stride = 1
    hm_ellipse_edge_margin = 0.0


def test_encode_seg_dome_from_obb():
    from opndet.encode import encode_targets_seg
    cfg = _Shim()
    # one 40x24 box at the center, axis-aligned
    obbs = np.array([[96.0, 64.0, 40.0, 24.0, 0.0]], np.float32)
    t = encode_targets_seg(cfg, obbs=obbs)
    dome = t["dome"][0].numpy()
    assert dome.shape == (128, 192)
    assert dome[64, 96] > 0.99, "1.0 at the box center"
    assert dome[64, 96 + 22] == 0.0, "past the x semi-axis (20px) → 0"
    assert dome[64, 96 + 10] > 0.0, "inside → positive"
    # taller dimension is shorter here → x-spread > y-spread
    assert int((dome[64, :] > 0).sum()) > int((dome[:, 96] > 0).sum())


def test_encode_seg_dome_from_obb_rotated():
    from opndet.encode import encode_targets_seg
    obbs = np.array([[96.0, 64.0, 40.0, 16.0, np.pi / 2]], np.float32)  # long side now vertical
    dome = encode_targets_seg(_Shim(), obbs=obbs)["dome"][0].numpy()
    assert dome[64, 96] > 0.99
    assert int((dome[:, 96] > 0).sum()) > int((dome[64, :] > 0).sum())


def test_encode_seg_dome_from_masks_distance_transform():
    import cv2
    from opndet.encode import encode_targets_seg
    cfg = _Shim()
    # a 30px-radius disk → distance-transform dome: 1 at the center, ~linear to 0 at the rim
    mask = np.zeros((128, 192), np.uint8)
    cv2.circle(mask, (96, 64), 30, 1, -1)
    dome = encode_targets_seg(cfg, masks=[mask])["dome"][0].numpy()
    assert dome[64, 96] > 0.99
    assert 0.0 < dome[64, 96 + 20] < 0.6
    assert dome[64, 96 + 31] == 0.0  # outside the disk
    # two disks with a real gap between them, passed as two instance masks →
    # max-aggregated, so the gap column stays 0 (no bleed across it). Disk A
    # reaches x=83, disk B reaches x=97; x=90 is in the gap.
    da = np.zeros((128, 192), np.uint8); cv2.circle(da, (65, 64), 18, 1, -1)
    db = np.zeros((128, 192), np.uint8); cv2.circle(db, (115, 64), 18, 1, -1)
    d2 = encode_targets_seg(cfg, masks=[da, db])["dome"][0].numpy()
    assert d2[64, 65] > 0.99 and d2[64, 115] > 0.99
    assert d2[64, 90] == 0.0, "the gap column between the two disks is 0"


def test_encode_seg_dome_flat_top():
    import cv2
    from opndet.encode import encode_targets_seg
    class S(_Shim):
        seg_dome_ramp_px = 3
    # 30px-radius disk → flat-top plateau: 1.0 across the interior, dropoff only in the last ~3px.
    mask = np.zeros((128, 192), np.uint8); cv2.circle(mask, (96, 64), 30, 1, -1)
    dome = encode_targets_seg(S(), masks=[mask])["dome"][0].numpy()
    assert dome[64, 96] > 0.99                       # center: 1.0
    assert dome[64, 96 + 20] > 0.99                  # still deep inside → still 1.0 (NOT a proportional ramp)
    assert dome[64, 96 + 31] == 0.0                  # outside → 0
    assert 0.0 < dome[64, 96 + 28] < 1.0             # the ~3px edge ramp
    # OBB branch flat-tops too
    obbs = np.array([[96.0, 64.0, 40.0, 24.0, 0.0]], np.float32)
    d2 = encode_targets_seg(S(), obbs=obbs)["dome"][0].numpy()
    assert d2[64, 96] > 0.99 and d2[64, 96 + 14] > 0.99 and d2[64, 96 + 22] == 0.0
    # two disks with a gap → flat tops, still 0 in the gap (no bleed)
    da = np.zeros((128, 192), np.uint8); cv2.circle(da, (65, 64), 18, 1, -1)
    db = np.zeros((128, 192), np.uint8); cv2.circle(db, (115, 64), 18, 1, -1)
    d3 = encode_targets_seg(S(), masks=[da, db])["dome"][0].numpy()
    assert d3[64, 65] > 0.99 and d3[64, 115] > 0.99 and d3[64, 90] == 0.0


def test_encode_seg_instance_gap():
    import cv2
    from opndet.encode import encode_targets_seg
    # two disks ~tangent at x≈50 (cx 30 / 70, r=20)
    da = np.zeros((128, 192), np.uint8); cv2.circle(da, (30, 64), 20, 1, -1)
    db = np.zeros((128, 192), np.uint8); cv2.circle(db, (70, 64), 20, 1, -1)
    class S0(_Shim):
        seg_dome_ramp_px = 2; seg_instance_gap_px = 0
    class S6(_Shim):
        seg_dome_ramp_px = 2; seg_instance_gap_px = 6   # flat-top: fixed normalization → clean per-pixel compares
    zero_run = lambda row: int((row == 0).sum())
    d0 = encode_targets_seg(S0(), masks=[da, db])["dome"][0].numpy()
    d6 = encode_targets_seg(S6(), masks=[da, db])["dome"][0].numpy()
    # the inter-instance corridor is wider with the gap; interiors still saturated
    assert zero_run(d6[64, 42:58]) > zero_run(d0[64, 42:58])
    assert d6[64, 30] > 0.99 and d6[64, 70] > 0.99
    # contact-ONLY: a lone disk is byte-identical with gap 0 and gap 6 (no uniform erosion)
    solo0 = encode_targets_seg(S0(), masks=[da])["dome"][0].numpy()
    solo6 = encode_targets_seg(S6(), masks=[da])["dome"][0].numpy()
    assert np.array_equal(solo0, solo6)
    # ...and even when da is in a touching pair, its FAR rim (away from db, x≈10-15) is untouched
    assert np.array_equal(d6[:, :16], d0[:, :16])


def test_seg_break_mask_and_repulsion_loss():
    import cv2
    from opndet.encode import encode_targets_seg
    from opndet.loss import SegDomeLoss
    # two ~tangent disks → encode emits a `break` band between them; a lone disk → all-zero
    da = np.zeros((128, 192), np.uint8); cv2.circle(da, (40, 64), 22, 1, -1)
    db = np.zeros((128, 192), np.uint8); cv2.circle(db, (84, 64), 22, 1, -1)
    class S(_Shim):
        seg_dome_ramp_px = 2; seg_instance_gap_px = 2
    pair = encode_targets_seg(S(), masks=[da, db])
    assert "break" in pair and float(pair["break"].sum()) > 0
    bm = pair["break"][0].numpy(); ys, xs = np.nonzero(bm)
    assert 52 < xs.mean() < 72 and 48 < ys.mean() < 80     # the band straddles the contact line at x≈62
    solo = encode_targets_seg(S(), masks=[da])
    assert float(solo["break"].sum()) == 0.0               # one instance → no contact → empty
    empty = encode_targets_seg(_Shim(), obbs=np.zeros((0, 5), np.float32))
    assert "break" in empty and float(empty["break"].sum()) == 0.0
    # repulsion: w_break>0 penalises a prediction that bridges the break band; no-op when correct
    dome = pair["dome"]; brk = pair["break"]
    good = torch.logit(dome.clamp(1e-4, 1 - 1e-4))         # sigmoid(good) == dome (≈0 in the carved seam)
    bridge = good.clone(); bridge[brk > 0] = 0.0           # logit 0 → p=0.5 across the break band (a merge bridge)
    l0 = SegDomeLoss(w_break=0.0)
    l4 = SegDomeLoss(w_break=4.0)
    tgt = {"dome": dome, "break": brk}
    assert float(l4(bridge, tgt)["loss"]) > float(l0(bridge, tgt)["loss"])               # the break term punishes the bridge
    assert float(l4(bridge, tgt)["l_break"]) > float(l4(good, tgt)["l_break"]) >= 0.0     # bridging → larger l_break than the correct pred
    # w_break=0 → `break` mask ignored entirely (l_break stays 0)
    assert float(l0(bridge, tgt)["l_break"]) == 0.0


def test_encode_seg_clipped_edge_not_ramped():
    import cv2
    from opndet.encode import encode_targets_seg
    class S(_Shim):
        seg_dome_ramp_px = 3
    # a disk centered at (96,25) r=40 → extends to y=-15 → clipped at the frame top (rows <0 gone)
    m = np.zeros((128, 192), np.uint8); cv2.circle(m, (96, 25), 40, 1, -1)
    dome = encode_targets_seg(S(), masks=[m])["dome"][0].numpy()
    assert dome[0, 96] > 0.9, "the clipped frame-edge row stays saturated (object continues past the edge), not ramped"
    assert 0.0 < dome[64, 96] < 1.0, "the real curved edge (y≈65) still has the ~3px ramp"
    assert dome[70, 96] == 0.0, "outside the disk → 0"


def test_decode_coco_rle():
    from opndet.dataset import _decode_coco_rle
    exp = np.zeros((4, 4), np.uint8); exp[:, 2:] = 1   # left half bg, right half fg
    # uncompressed (list counts) and compressed (ASCII string) forms, column-major: [bg=8, fg=8]
    assert np.array_equal(_decode_coco_rle({"size": [4, 4], "counts": [8, 8]}), exp)
    assert np.array_equal(_decode_coco_rle({"size": [4, 4], "counts": "88"}), exp)
    # all-foreground 2x2 → counts [0, 4]
    assert np.array_equal(_decode_coco_rle({"size": [2, 2], "counts": [0, 4]}), np.ones((2, 2), np.uint8))
    # bytes counts (some loaders hand it back as bytes)
    assert np.array_equal(_decode_coco_rle({"size": [4, 4], "counts": b"88"}), exp)


def test_seg_coco_rle_gt_through_dataset(tmp_path):
    import json
    import cv2
    from opndet.dataset import OpndetDataset, load_coco_single_class
    from opndet.encode import encode_targets_seg
    img = (np.random.default_rng(1).random((40, 60, 3)) * 255).astype(np.uint8)
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # one instance = the left 30 columns (column-major counts: 30*40 fg, then 30*40 bg)
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 60, "height": 40}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 30, 40], "iscrowd": 0,
                             "segmentation": {"size": [40, 60], "counts": [0, 1200, 1200]}}],
            "categories": [{"id": 0, "name": "obj"}]}
    cp = tmp_path / "ann.json"; cp.write_text(json.dumps(coco))
    samps = load_coco_single_class(cp, img_dir)
    assert samps[0].coco_segs is not None and isinstance(samps[0].coco_segs[0], dict)
    class _Cfg:
        img_h = 40; img_w = 64; seg_stride = 1; seg_dome_ramp_px = 0; seg_instance_gap_px = 0
    def _enc(b, obbs=None, masks=None):
        return encode_targets_seg(_Cfg(), masks=masks)
    _enc._takes_masks = True
    ds = OpndetDataset(samps, 40, 64, augment_fn=None, encode_fn=_enc, mosaic_prob=0.0)
    assert ds._has_masks
    dome = ds[0][2]["dome"][0].numpy()
    assert dome.max() > 0.99 and int((dome > 0).sum()) > 200    # the left-half blob decoded + domed


def test_seg_coco_polygon_gt_through_dataset(tmp_path):
    import json
    import cv2
    from opndet.dataset import OpndetDataset, load_coco_single_class
    from opndet.encode import encode_targets_seg
    img = (np.random.default_rng(0).random((80, 100, 3)) * 255).astype(np.uint8)
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # two square instances via COCO polygon `segmentation`
    poly1 = [10, 20, 35, 20, 35, 55, 10, 55]   # x,y,x,y,... (a 25x35 box)
    poly2 = [60, 20, 88, 20, 88, 55, 60, 55]
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 80}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 25, 35], "segmentation": [poly1]},
                            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [60, 20, 28, 35], "segmentation": [poly2]}],
            "categories": [{"id": 0, "name": "obj"}]}
    cp = tmp_path / "ann.json"; cp.write_text(json.dumps(coco))
    samps = load_coco_single_class(cp, img_dir)
    assert len(samps) == 1 and samps[0].coco_segs is not None and len(samps[0].coco_segs) == 2
    class _Cfg:
        img_h = 64; img_w = 64; seg_stride = 1; seg_dome_ramp_px = 0; seg_instance_gap_px = 0
    def _enc(boxes, obbs=None, masks=None):
        return encode_targets_seg(_Cfg(), masks=masks)
    _enc._takes_masks = True
    ds = OpndetDataset(samps, 64, 64, augment_fn=None, encode_fn=_enc, mosaic_prob=0.0)
    assert ds._has_masks
    dome = ds[0][2]["dome"][0].numpy()
    assert dome.shape == (64, 64) and dome.max() > 0.99 and int((dome > 0).sum()) > 80


def test_seg_mask_gt_through_dataset(tmp_path):
    import cv2
    from opndet.dataset import OpndetDataset, Sample
    from opndet.encode import encode_targets_seg
    # a 3-channel image + an instance-id mask PNG (two disks: id 1 and id 2)
    img = (np.random.default_rng(0).random((80, 100, 3)) * 255).astype(np.uint8)
    ip = tmp_path / "x.png"; cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    lbl = np.zeros((80, 100), np.uint16)
    cv2.circle(lbl, (25, 40), 14, 1, -1); cv2.circle(lbl, (75, 40), 14, 2, -1)
    mp = tmp_path / "x.png_mask.png"; cv2.imwrite(str(mp), lbl)
    samp = Sample(image_path=ip, boxes=np.array([[11, 26, 39, 54], [61, 26, 89, 54]], np.float32),
                  img_w=100, img_h=80, mask_path=mp)
    class _Cfg:
        img_h = 64; img_w = 64; seg_stride = 1; seg_dome_ramp_px = 0; seg_instance_gap_px = 0
    def _enc(boxes, obbs=None, masks=None):
        return encode_targets_seg(_Cfg(), masks=masks)
    _enc._takes_masks = True
    ds = OpndetDataset([samp], 64, 64, augment_fn=None, encode_fn=_enc, mosaic_prob=0.0)
    img_t, boxes, targets = ds[0]
    dome = targets["dome"][0].numpy()
    assert dome.shape == (64, 64)
    assert dome.max() > 0.99                          # the disk interiors made it through letterbox + encode
    assert int((dome > 0).sum()) > 50                 # non-trivial footprint
    # mosaic auto-disabled when masks are present (so GT can't desync from the image)
    ds_m = OpndetDataset([samp], 64, 64, augment_fn=None, encode_fn=_enc, mosaic_prob=1.0)
    assert ds_m._has_masks and ds_m[0][2]["dome"][0].numpy().max() > 0.99


def test_encode_seg_empty():
    from opndet.encode import encode_targets_seg
    t = encode_targets_seg(_Shim(), obbs=np.zeros((0, 5), np.float32))
    assert float(t["dome"].abs().sum()) == 0.0
    assert t["dome"].shape == (1, 128, 192)


def test_encode_seg_stride_2():
    from opndet.encode import encode_targets_seg
    class S(_Shim):
        seg_stride = 2
    obbs = np.array([[96.0, 64.0, 40.0, 24.0, 0.0]], np.float32)
    dome = encode_targets_seg(S(), obbs=obbs)["dome"][0].numpy()
    assert dome.shape == (64, 96)
    assert dome[32, 48] > 0.99


# ---- 3. SegDomeLoss: QFL toward the dome + soft Dice on foreground ----
def test_predict_image_seg_path(tmp_path):
    import cv2

    from opndet.predict import predict_image
    img = (np.random.default_rng(0).random((150, 240, 3)) * 255).astype(np.uint8)
    ip = tmp_path / "x.jpg"; cv2.imwrite(str(ip), img)
    out = tmp_path / "vis.jpg"
    # untrained model + low threshold so the (≈0.1-everywhere) dome yields ≥1 blob
    res = predict_image(image_path=str(ip), model_config=resolve("bbox-n-seg"), ckpt=None,
                        threshold=0.05, device="cpu", save_path=str(out))
    assert out.exists()
    v = cv2.imread(str(out)); assert v.shape == img.shape   # vis is at original resolution
    # untrained model → dome ≈ 0.1 everywhere → no peak ≥ peak_thr → no instances ("no peak, no seg").
    # so res may be empty; if any blobs decoded, they must have the SegBlob keys.
    assert isinstance(res, list)
    assert not res or {"cx", "cy", "area_px", "peak", "x1", "y1", "x2", "y2"} == set(res[0])


def test_decode_seg_watershed_splits_touching():
    from opndet.decode import decode_seg
    # two cone-domes (1.0 at the apex, linear ramp to 0 at r=40) whose 0.3-contours overlap
    # (apexes 40px apart) → between them the dome DIPS to ~0.5, never to 0. Connected-components
    # at thr 0.3 merges them into one blob; the watershed-from-peaks split sees the two distinct
    # local maxima and splits the merged region at the ridge → two instances.
    yy, xx = np.mgrid[0:128, 0:256]
    da = np.clip(1.0 - np.hypot(xx - 90, yy - 64) / 40.0, 0.0, 1.0)
    db = np.clip(1.0 - np.hypot(xx - 130, yy - 64) / 40.0, 0.0, 1.0)
    dome = np.maximum(da, db).astype(np.float32)
    assert len(decode_seg(dome, threshold=0.3, min_area=4, mode="cc")) == 1          # naive: merged
    ws = decode_seg(dome, threshold=0.3, min_area=4, mode="watershed", peak_kernel=15, peak_thr=0.6)
    assert len(ws) == 2                                                              # split at the ridge
    xs = sorted(b.cx for b in ws)
    assert abs(xs[0] - 90) < 8 and abs(xs[1] - 130) < 8
    # an UNDER-confident dome (everything < peak_thr) → no peaks → no instances ("no peak, no seg")
    assert decode_seg(dome * 0.4, threshold=0.1, min_area=4, mode="watershed", peak_thr=0.6) == []


def test_decode_seg_blobs():
    import cv2
    from opndet.decode import decode_seg, decode_seg_batch
    dome = np.zeros((128, 192), np.float32)
    # two well-separated convex blobs (a disk and a small ellipse)
    cv2.circle(dome, (50, 64), 22, 1.0, -1)          # disk r=22 → area ≈ π·22² ≈ 1520
    cv2.ellipse(dome, (140, 64), (16, 10), 0, 0, 360, 0.9, -1)
    blobs = decode_seg(dome, threshold=0.5, min_area=4)
    assert len(blobs) == 2
    by_x = sorted(blobs, key=lambda b: b.cx)
    assert abs(by_x[0].cx - 50) < 2 and abs(by_x[0].cy - 64) < 2
    assert abs(by_x[1].cx - 140) < 2
    assert by_x[0].area_px > 1300                      # ≈ disk area in px
    assert by_x[0].peak >= 0.99 and by_x[1].peak >= 0.89
    # touching-but-distinct check: the dome between them is 0 → still 2 components
    assert blobs[0].peak >= blobs[1].peak              # sorted by descending peak
    # batch wrapper + min_area denoise
    dome2 = dome.copy(); dome2[0, 0] = 1.0             # 1-px speck
    out = np.stack([dome, dome2])[:, None]             # [2,1,H,W]
    bb = decode_seg_batch(out, threshold=0.5, min_area=4)
    assert len(bb) == 2 and len(bb[0]) == 2 and len(bb[1]) == 2  # the 1-px speck dropped


def test_seg_dome_loss_basic():
    from opndet.loss import SegDomeLoss
    B, H, W = 2, 32, 48
    dome = torch.zeros(B, 1, H, W)
    dome[:, 0, 12:20, 20:32] = torch.linspace(0.2, 1.0, 12).unsqueeze(0).repeat(8, 1)
    loss = SegDomeLoss(qfl_beta=2.0)
    good = torch.logit(dome.clamp(1e-4, 1 - 1e-4))
    bad = torch.zeros_like(dome)
    lo = loss(good, {"dome": dome})
    hi = loss(bad, {"dome": dome})
    assert float(lo["loss"]) < float(hi["loss"])
    assert "l_qfl" in lo and "l_dice" in lo
    bad.requires_grad_(True)
    out = loss(bad, {"dome": dome})
    out["loss"].backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all()


# ---- 4. train_seg: end-to-end (2 epochs on a tiny synthetic object dataset) ----
def _seg_fixture(tmp_path, n=14, sz=96):
    import cv2
    img_dir = tmp_path / "imgs"; obb_dir = tmp_path / "obb"
    img_dir.mkdir(); obb_dir.mkdir()
    images, anns, aid = [], [], 1
    rng = np.random.default_rng(0)
    for i in range(n):
        im = (rng.random((sz, sz, 3)) * 60 + 30).astype(np.uint8)
        lines = []
        for _ in range(2):
            cx, cy = int(rng.integers(20, sz - 20)), int(rng.integers(20, sz - 20))
            a, b = int(rng.integers(8, 14)), int(rng.integers(6, 11))
            cv2.ellipse(im, (cx, cy), (a, b), 0, 0, 360, (220, 210, 200), -1)
            x1, y1, x2, y2 = cx - a, cy - b, cx + a, cy + b
            anns.append({"id": aid, "image_id": i, "category_id": 1,
                         "bbox": [x1, y1, 2 * a, 2 * b], "area": 4 * a * b, "iscrowd": 0}); aid += 1
            lines.append("0 " + " ".join(f"{v:.6f}" for v in
                         [x1 / sz, y1 / sz, x2 / sz, y1 / sz, x2 / sz, y2 / sz, x1 / sz, y2 / sz]))
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), im)
        images.append({"id": i, "file_name": f"img{i}.jpg", "width": sz, "height": sz})
        (obb_dir / f"img{i}.txt").write_text("\n".join(lines) + "\n")
    import json
    coco = tmp_path / "ann.json"
    coco.write_text(json.dumps({"images": images, "annotations": anns,
                                "categories": [{"id": 1, "name": "object"}]}))
    return coco, img_dir, obb_dir


def test_train_seg_end_to_end(tmp_path):
    import yaml as _yaml

    from opndet.train import train  # exercises the seg auto-dispatch
    coco, img_dir, obb_dir = _seg_fixture(tmp_path)
    # image_filter: only the first 10 of the 14 fixture images → split 0.6/0.25/0.15 ⇒ val=2.
    # (without the filter being applied it'd be int(14*0.25)=3 — see the n_val assertion.)
    flt = tmp_path / "filter.txt"
    flt.write_text("\n".join(f"img{i}" for i in range(10)) + "\n")
    cfg = {
        "model_config": "bbox-n-seg", "model": {"img_h": 96, "img_w": 96},
        "device": "cpu", "amp": False, "seed": 0,
        "epochs": 1, "batch_size": 4, "lr": 1e-3, "warmup_steps": 2,   # 1 epoch — keep the test cheap
        "num_workers": 0, "ema_decay": 0.9, "ema_tau": 5,
        "vis_samples": 2, "metric_for_best": "dice", "auto_bundle": False,
        "data": {"sources": [{"coco": str(coco), "images": str(img_dir), "obb_dir": str(obb_dir)}],
                 "split_ratios": [0.6, 0.25, 0.15], "image_filter": str(flt)},
        "loss": {"qfl_beta": 2.0}, "augment": {"hflip_prob": 0.5},
        "runs_dir": str(tmp_path / "runs"), "name": "seg_smoke",
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(_yaml.safe_dump(cfg))
    train(str(cfg_path))
    ckpts = list((tmp_path / "runs").rglob("*_best.pt"))
    assert ckpts, "train_seg should have saved a best checkpoint"
    d = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert d["metric_for_best"] == "dice" and d["ema"] is not None
    assert {"dice", "fg_iou", "count_mae", "area_mape", "inst_iou_mean", "inst_iou_p10",
            "inst_recall", "inst_precision", "n_val"} <= set(d["metrics"])
    assert d["metrics"]["n_val"] == 2, "train_seg must honour data.image_filter (10 imgs × 0.25 = 2 val)"
    # vis only runs on a new-best (or first/last) epoch — here ep1 is first+best+last.
    # stable-once PNGs (deterministic on val): RGB + GT heatmap + GT decoded, one each per sample.
    rgb = list((tmp_path / "runs").rglob("vis/val_seg/sample_*_rgb.png"))
    gt_h = list((tmp_path / "runs").rglob("vis/val_seg/sample_*_gt_heat.png"))
    gt_s = list((tmp_path / "runs").rglob("vis/val_seg/sample_*_gt_seg.png"))
    assert len(rgb) == 2 and len(gt_h) == 2 and len(gt_s) == 2
    # per-epoch PNGs: predicted dome heatmap + predicted decoded instances, per sample × vis-epoch.
    ph = list((tmp_path / "runs").rglob("vis/val_seg/ep_*/sample_*_pred_heat.png"))
    ps = list((tmp_path / "runs").rglob("vis/val_seg/ep_*/sample_*_pred_seg.png"))
    assert len(ph) == 2 and len(ps) == 2  # 2 samples × 1 vis-epoch (ep1)
    # test vis (also gated on best/boundary) lands under vis/test_seg/
    assert list((tmp_path / "runs").rglob("vis/test_seg/sample_*_rgb.png"))
    # DuckDB store written (so the dashboard shows seg runs) with scalars + the val/seg vis
    rundir = ckpts[0].parent
    assert (rundir / "metrics.duckdb").exists()
    import duckdb
    con = duckdb.connect(str(rundir / "metrics.duckdb"), read_only=True)
    try:
        tags = {r[0] for r in con.execute("select distinct tag from scalars").fetchall()}
        assert {"train/loss", "lr", "val/dice"} <= tags
        assert "val/seg" in {r[0] for r in con.execute("select distinct tag from images").fetchall()}
        kinds = {r[0] for r in con.execute("select distinct kind from overlays").fetchall()}
        assert {"dome_pred", "dome_gt", "seg_pred", "seg_gt"} <= kinds   # heatmap + decoded, pred + gt
        # no box rows for a seg run — it's a dense-dome head, not a box model (an AABB fitted
        # around a dome blob is meaningless; per-blob extent is the `<N>px` label in the seg overlay)
        assert not con.execute("select count(*) from boxes").fetchone()[0]
    finally:
        con.close()
    # `opndet eval` routes a seg ckpt to the seg eval path
    from opndet.eval import run_eval
    r = run_eval(ckpt_path=str(ckpts[0]), config_path=None, split="test", out_dir=str(tmp_path / "evalout"))
    assert "seg" in r and {"dice", "fg_iou", "count_mae", "area_mape", "n_val"} <= set(r["seg"])
    assert (tmp_path / "evalout" / "seg_eval_test.md").exists()
