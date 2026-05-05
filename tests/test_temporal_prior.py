from __future__ import annotations

import numpy as np
import pytest

from opndet.augment_temporal_prior import TemporalPriorSynth


def _box(cx: float, cy: float, w: float = 40.0, h: float = 40.0) -> np.ndarray:
    return np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2]], dtype=np.float32)


def test_zero_prior_prob_one_returns_zeros():
    synth = TemporalPriorSynth({"zero_prior_prob": 1.0}, seed=0)
    boxes = _box(256, 192)
    prior = synth(boxes, 384, 512)
    assert prior.shape == (96, 128)
    assert prior.max() == 0.0


def test_n_zero_no_history_no_spawn_zeros():
    synth = TemporalPriorSynth({
        "n_max": 0, "zero_prior_prob": 0.0, "spawn_zone_prob": 0.0,
    }, seed=0)
    boxes = _box(256, 192)
    prior = synth(boxes, 384, 512)
    assert prior.max() == 0.0


def test_output_shape_and_range_over_many_samples():
    synth = TemporalPriorSynth({}, seed=42)
    boxes = _box(256, 192)
    for _ in range(200):
        p = synth(boxes, 384, 512)
        assert p.shape == (96, 128)
        assert p.dtype == np.float32
        assert p.min() >= 0.0
        assert p.max() <= 1.0 + 1e-6


def test_stationary_motion_stamps_at_current_position():
    """motion=0 means object never moved — stamps land at current GT center.
    This is the only valid case where prior overlaps current GT."""
    synth = TemporalPriorSynth({
        "motion_axis_aligned_prob": 0.0,
        "motion_diagonal_prob": 0.0,
        "motion_zero_prob": 1.0,
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 4,
    }, seed=0)
    cx, cy = 256, 192
    boxes = _box(cx, cy)
    p = None
    for _ in range(20):
        p = synth(boxes, 384, 512, force_motion=(0.0, 0.0))
        if p.max() > 0:
            break
    assert p.max() > 0.0
    py, px = np.unravel_index(int(np.argmax(p)), p.shape)
    assert abs(px - cx / 4) <= 2
    assert abs(py - cy / 4) <= 2


def test_rightward_motion_trail_extends_left_of_current():
    """Object moving rightward (+dx) leaves trail behind (smaller x) since prior
    represents past frames. This is the most important correctness test —
    catches both current-frame leak and motion-direction inversion in one shot.
    """
    synth = TemporalPriorSynth({
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 6,
    }, seed=1)
    cx, cy = 320, 192
    boxes = _box(cx, cy, w=40, h=40)
    cur_cx_stride = cx / 4
    masses = []
    for _ in range(20):
        p = synth(boxes, 384, 512, force_motion=(8.0, 0.0))
        if p.sum() > 0:
            xs = np.arange(p.shape[1])[None, :]
            cog_x = float((p * xs).sum() / p.sum())
            masses.append(cog_x)
    assert len(masses) >= 5
    mean_cog = float(np.mean(masses))
    assert mean_cog < cur_cx_stride - 1.0, (
        f"trail center-of-gravity ({mean_cog:.2f}) must be left of current "
        f"position ({cur_cx_stride:.2f}) for rightward motion"
    )


def test_no_current_frame_leak_when_motion_nonzero():
    """Argmax of prior should never coincide exactly with current GT center
    when motion is nonzero — guards against accidentally stamping current GT."""
    synth = TemporalPriorSynth({
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 5,
    }, seed=2)
    cx, cy = 256, 192
    boxes = _box(cx, cy)
    cx_stride, cy_stride = cx / 4, cy / 4
    for _ in range(50):
        p = synth(boxes, 384, 512, force_motion=(10.0, 6.0))
        if p.max() > 0:
            py, px = np.unravel_index(int(np.argmax(p)), p.shape)
            d = ((px - cx_stride) ** 2 + (py - cy_stride) ** 2) ** 0.5
            assert d > 0.5, f"prior peak too close to current GT (d={d:.2f})"


def test_confidence_amplitude_bounded_by_range_max():
    """Stamp amplitudes can't exceed confidence_range upper bound."""
    synth = TemporalPriorSynth({
        "confidence_range": [0.5, 0.95],
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 1,
    }, seed=3)
    boxes = _box(256, 192)
    for _ in range(100):
        p = synth(boxes, 384, 512, force_motion=(8.0, 0.0))
        assert p.max() <= 0.95 + 1e-6


def test_output_in_range_zero_one_with_lots_of_overlap():
    """Even with overlapping stamps and FPs, prior must clip to [0, 1]."""
    synth = TemporalPriorSynth({
        "zero_prior_prob": 0.0,
        "false_positive_prob": 1.0,
        "spawn_zone_prob": 1.0,
        "n_max": 8,
        "confidence_range": [0.95, 0.95],
    }, seed=4)
    boxes = np.concatenate([_box(c, 192) for c in (100, 150, 200, 250, 300)], axis=0)
    for _ in range(20):
        p = synth(boxes, 384, 512, force_motion=(2.0, 0.0))
        assert p.min() >= 0.0
        assert p.max() <= 1.0 + 1e-6


def test_invalid_motion_probs_raises():
    with pytest.raises(AssertionError):
        TemporalPriorSynth({
            "motion_axis_aligned_prob": 0.5,
            "motion_diagonal_prob": 0.3,
            "motion_zero_prob": 0.3,
        })


def test_top_margin_excludes_box_intersecting_top_edge():
    """Margin test is intersection-based: any box touching the margin band
    is excluded. With margin_top=0.10 (38.4 px on H=384), a box with y1=10
    is excluded even though its center (cy=50) is below the margin band."""
    synth = TemporalPriorSynth({
        "margin_top": 0.10,
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 6,
    }, seed=10)
    H, W = 384, 512
    intersecting = _box(256, 50, w=40, h=80)  # y1=10, intersects margin band
    fully_inside = _box(256, 20)              # y1=0, fully in margin
    away = _box(256, 200)                      # well clear

    p_inter = sum((synth(intersecting, H, W, force_motion=(0.0, 8.0)).max() > 0)
                  for _ in range(20))
    p_fully = sum((synth(fully_inside, H, W, force_motion=(0.0, 8.0)).max() > 0)
                  for _ in range(20))
    p_away = sum((synth(away, H, W, force_motion=(0.0, 8.0)).max() > 0)
                 for _ in range(20))
    assert p_inter == 0, "expected no prior for box that touches top margin"
    assert p_fully == 0, "expected no prior for box fully in top margin"
    assert p_away >= 15


def test_margin_bottom_intersection_excludes_near_bottom():
    synth = TemporalPriorSynth({
        "margin_bottom": 0.10,             # 38.4 px band at bottom of H=384
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 6,
    }, seed=11)
    H, W = 384, 512
    touching = _box(256, 320, w=40, h=80)  # y2=360, 384-360=24 < 38.4 -> intersects
    nonzero = sum((synth(touching, H, W, force_motion=(0.0, -8.0)).max() > 0)
                  for _ in range(20))
    assert nonzero == 0


def test_object_skip_prob_one_excludes_all():
    """skip_prob=1.0 -> never any stamps from objects (only FP/spawn could fire)."""
    synth = TemporalPriorSynth({
        "object_skip_prob": 1.0,
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "n_max": 6,
    }, seed=12)
    boxes = _box(256, 192)
    for _ in range(20):
        p = synth(boxes, 384, 512, force_motion=(8.0, 0.0))
        assert p.max() == 0.0


def test_default_margins_off_preserves_old_behavior():
    """With all margins=0 and skip=0 (defaults), behavior unchanged from baseline."""
    synth = TemporalPriorSynth({
        "zero_prior_prob": 0.0,
        "false_positive_prob": 0.0,
        "spawn_zone_prob": 0.0,
        "object_drop_prob": 0.0,
        "n_max": 4,
    }, seed=13)
    near_top = _box(256, 20)
    nonzero = sum((synth(near_top, 384, 512, force_motion=(0.0, 8.0)).max() > 0)
                  for _ in range(20))
    assert nonzero >= 15
