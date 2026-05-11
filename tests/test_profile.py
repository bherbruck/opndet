"""opndet profile — per-layer activity + ablation report."""
import numpy as np
import torch

from opndet.presets import resolve
from opndet.profile import profile
from opndet.yaml_build import build_model_from_yaml


def _make_ckpt(tmp_path, preset="bbox-n-obb"):
    m = build_model_from_yaml(resolve(preset))
    p = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "config": {"model_config": preset}, "epoch": 1, "step": 1}, p)
    return p


def _make_images(tmp_path, n=2, h=64, w=80):
    import cv2
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(d / f"img{i}.png"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return d


def test_profile_runs_and_reports(tmp_path):
    ck = _make_ckpt(tmp_path)
    imgs = _make_images(tmp_path, n=2)
    out = tmp_path / "report.html"
    res = profile(ckpt=ck, model="bbox-n-obb", images=imgs, out=out, n=2, device="cpu")
    assert out.exists() and out.stat().st_size > 1000
    assert res["total_params"] > 0
    assert res["n_images"] == 2
    layers = res["layers"]
    assert len(layers) > 10
    names = {l["name"] for l in layers}
    assert {"stem", "p1", "p3", "raw"} <= names
    for l in layers:
        assert {"name", "module", "channels", "params", "mean_abs", "live_frac", "ablation_delta"} <= set(l)
        assert 0.0 <= l["live_frac"] <= 1.0
        assert l["mean_abs"] >= 0.0
    # learned backbone/neck layers get an ablation-Δ; parameter-free decode ops don't.
    abl = {l["name"]: l["ablation_delta"] for l in layers}
    assert abl["p1"] is not None and abl["raw"] is None and abl["obj"] is None
    # Δ is normalized into [0, 1].
    for nm, v in abl.items():
        if v is not None:
            assert 0.0 <= v <= 1.0001, (nm, v)
    # the html embeds the per-layer activation thumbnails and the bar chart
    html = out.read_text()
    assert "data:image/png;base64," in html
    assert "ablation-Δ" in html


def test_profile_uses_ckpt_config_when_model_omitted(tmp_path):
    ck = _make_ckpt(tmp_path)
    imgs = _make_images(tmp_path, n=1)
    res = profile(ckpt=ck, model=None, images=imgs, out=tmp_path / "r.html", n=1, device="cpu")
    assert res["n_images"] == 1 and res["total_params"] > 0


def test_profile_single_image_path(tmp_path):
    ck = _make_ckpt(tmp_path)
    import cv2
    img = tmp_path / "one.png"
    cv2.imwrite(str(img), np.zeros((48, 48, 3), np.uint8))
    res = profile(ckpt=ck, model="bbox-n-obb", images=img, out=tmp_path / "r.html", n=8, device="cpu")
    assert res["n_images"] == 1
