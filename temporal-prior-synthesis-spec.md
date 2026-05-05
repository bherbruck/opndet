# Prior synthesis augmentation: implementation spec

## Context for Claude Code

opndet is adding temporal-mode model variants that take a 4th input channel (the "prior") alongside RGB. The prior at deployment is produced by a host-side accumulator (see `roadmap-section-7-revision.md`). For training, the prior must be synthesized per sample to match the deployment-time distribution.

This spec defines the synthesis augmentation module, its config surface, and how it integrates with the existing data pipeline.

## What to build

A new augmentation step `TemporalPriorSynth` that:

1. Runs **after** geometric augmentation (so the prior matches the augmented image's coordinate space)
2. Runs **before** photometric augmentation (so photometric ops can apply to RGB only — prior is not a photometric quantity)
3. Takes the (possibly augmented) GT boxes for the current sample
4. Produces a single-channel float32 prior heatmap at stride-4 resolution (96×128 for the standard 384×512 input)
5. Returns the image extended with the prior as a 4th channel

## File location

`src/opndet/augment_temporal_prior.py` — new module, separate from the main `augment.py` to keep concerns isolated. Imported and wired into the dataset pipeline only when the model has `in_ch: 4`.

## API

```python
class TemporalPriorSynth:
    def __init__(self, config: dict):
        """
        config keys (all optional, defaults sensible):
            n_max: int = 8
                Maximum tail length to synthesize. N is sampled uniform [0, n_max].
            stride: int = 4
                Output stride for the prior. Must match model's output stride.
            motion_axis_aligned_prob: float = 0.7
                Probability of axis-aligned motion vector.
            motion_diagonal_prob: float = 0.25
                Probability of diagonal (45/135/225/315 deg) motion.
            motion_zero_prob: float = 0.05
                Probability of zero (stationary) motion.
                The three above sum to 1.0.
            motion_speed_range: tuple[float, float] = (2.0, 15.0)
                Per-frame motion in pixels (input coords).
            motion_diagonal_jitter_deg: float = 10.0
                Std of angle jitter for diagonal motion.
            confidence_range: tuple[float, float] = (0.5, 0.95)
                Per-frame stamp confidence sampled uniform in this range.
            object_drop_prob: float = 0.05
                Probability of dropping an object's stamp from a given history frame
                (simulates detection miss).
            false_positive_prob: float = 0.10
                Probability that 1-3 false positive stamps are added.
            false_positive_count_range: tuple[int, int] = (1, 3)
            false_positive_amplitude_range: tuple[float, float] = (0.3, 0.5)
            spawn_zone_prob: float = 0.10
                Probability that a spawn-zone hot region is overlaid.
            spawn_zone_amplitude_range: tuple[float, float] = (0.3, 0.5)
            zero_prior_prob: float = 0.05
                Additional cold-start coverage beyond N=0 cases.
            gaussian_sigma_factor: float = 4.0
                bbox_edge / sigma_factor = stamp sigma. 4.0 puts bbox edge at ~2 sigma.
        """
        ...

    def __call__(self, image: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, list[Box]]:
        """
        image: (3, H, W) float32, post-geometric-aug, pre-photometric-aug
        boxes: current frame's GT boxes (post-geometric-aug)

        Returns:
            image_with_prior: (4, H, W) float32 — original image with prior concatenated
            boxes: unchanged (passed through for downstream use)
        """
        ...
```

## Synthesis algorithm (precise)

Pseudocode for `__call__`:

```python
def __call__(self, image, boxes):
    H, W = image.shape[1], image.shape[2]
    Hs, Ws = H // self.stride, W // self.stride  # 96, 128 for stride 4
    prior = np.zeros((Hs, Ws), dtype=np.float32)

    # Cold-start cases
    if rng.random() < self.zero_prior_prob:
        return self._concat(image, prior), boxes

    N = rng.integers(0, self.n_max + 1)  # uniform [0, n_max] inclusive
    if N == 0:
        # No history. Maybe overlay spawn zone, then return.
        if rng.random() < self.spawn_zone_prob:
            self._overlay_spawn_zone(prior, Hs, Ws)
        return self._concat(image, prior), boxes

    # Sample motion vector
    motion = self._sample_motion()  # (dx, dy) in pixels per frame, input coords

    # Stamp from oldest to newest
    fade_step = 1.0 / N
    for k in range(N, 0, -1):
        # Frame t-k: object positions offset by k * motion behind current
        for box in boxes:
            if rng.random() < self.object_drop_prob:
                continue  # simulate detection miss this frame

            # Offset box position by k * motion (objects came from offset-back)
            offset_box = self._shift_box(box, dx=-k * motion[0], dy=-k * motion[1])
            if not self._box_in_frame(offset_box, H, W):
                continue  # offset position is off-screen, skip

            # Sample per-frame confidence
            conf = rng.uniform(*self.confidence_range)
            # Apply fade for how many frames ago this stamp was
            faded_amplitude = max(conf - (k - 1) * fade_step, 0.0)
            if faded_amplitude <= 0:
                continue

            # Convert input-coord box to stride-coord, render Gaussian, max-merge
            stride_box = self._to_stride_coords(offset_box)
            self._stamp_gaussian(prior, stride_box, faded_amplitude)

    # False positives (rare, low amplitude)
    if rng.random() < self.false_positive_prob:
        n_fp = rng.integers(*self.false_positive_count_range)
        for _ in range(n_fp):
            fp_box = self._random_box(Ws, Hs, mean_size_from=boxes)
            fp_amp = rng.uniform(*self.false_positive_amplitude_range)
            self._stamp_gaussian(prior, fp_box, fp_amp)

    # Spawn zone overlay
    if rng.random() < self.spawn_zone_prob:
        self._overlay_spawn_zone(prior, Hs, Ws)

    return self._concat(image, prior), boxes


def _sample_motion(self):
    r = rng.random()
    speed = rng.uniform(*self.motion_speed_range)
    if r < self.motion_axis_aligned_prob:
        direction = rng.choice([(1,0), (-1,0), (0,1), (0,-1)])
        return (direction[0] * speed, direction[1] * speed)
    elif r < self.motion_axis_aligned_prob + self.motion_diagonal_prob:
        base_angle = rng.choice([45, 135, 225, 315])
        angle = base_angle + rng.normal(0, self.motion_diagonal_jitter_deg)
        return (math.cos(math.radians(angle)) * speed,
                math.sin(math.radians(angle)) * speed)
    else:
        return (0.0, 0.0)


def _stamp_gaussian(self, prior, box, amplitude):
    """
    Render anisotropic 2D Gaussian sized to the box, peak amplitude `amplitude`,
    max-merge into prior in-place. Box is in stride coordinates.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sigma_x = max(1.0, (x2 - x1) / self.gaussian_sigma_factor)
    sigma_y = max(1.0, (y2 - y1) / self.gaussian_sigma_factor)

    # Compute support region (3 sigma in each direction)
    rx = int(3 * sigma_x) + 1
    ry = int(3 * sigma_y) + 1
    x0 = max(0, int(cx - rx))
    x1_ = min(prior.shape[1], int(cx + rx) + 1)
    y0 = max(0, int(cy - ry))
    y1_ = min(prior.shape[0], int(cy + ry) + 1)
    if x1_ <= x0 or y1_ <= y0:
        return

    xx, yy = np.meshgrid(np.arange(x0, x1_), np.arange(y0, y1_))
    g = amplitude * np.exp(
        -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
    ).astype(np.float32)

    region = prior[y0:y1_, x0:x1_]
    np.maximum(region, g, out=region)


def _overlay_spawn_zone(self, prior, Hs, Ws):
    """
    Pick a random frame edge, render a moderate-amplitude band along it,
    decaying inward.
    """
    edge = rng.choice(['top', 'bottom', 'left', 'right'])
    band_depth = rng.integers(Hs // 6, Hs // 3)  # 16-32 cells deep at 96x128
    amplitude = rng.uniform(*self.spawn_zone_amplitude_range)

    if edge == 'top':
        for d in range(band_depth):
            row_amp = amplitude * (1 - d / band_depth)
            np.maximum(prior[d, :], row_amp, out=prior[d, :])
    elif edge == 'bottom':
        for d in range(band_depth):
            row_amp = amplitude * (1 - d / band_depth)
            np.maximum(prior[Hs - 1 - d, :], row_amp, out=prior[Hs - 1 - d, :])
    # similar for left/right ...
```

Box coord conversion utility (input coords → stride coords):

```python
def _to_stride_coords(self, box):
    return (box[0] / self.stride, box[1] / self.stride,
            box[2] / self.stride, box[3] / self.stride)
```

## Integration with existing dataset pipeline

In `src/opndet/dataset.py`, the `__getitem__` flow is roughly:

```
load image + boxes
apply geometric augmentation (flip, rotate, scale, translate, mosaic, cutout)
apply photometric augmentation (brightness, contrast, hue, etc.)
normalize
return (tensor, boxes)
```

Modify to:

```
load image + boxes
apply geometric augmentation
[NEW] if model.in_ch == 4: apply TemporalPriorSynth here
apply photometric augmentation (to RGB channels only — leave 4th channel alone)
normalize (separate normalization for RGB vs prior; prior is already in [0,1])
return (tensor, boxes)
```

Detection of whether to apply the synthesizer: check the model config's `in_ch` value. If 4, instantiate and apply. If 3, skip.

The simplest wiring is a flag in the dataset constructor:

```python
class CocoDataset:
    def __init__(self, ..., temporal_prior_config=None):
        ...
        self.prior_synth = (
            TemporalPriorSynth(temporal_prior_config)
            if temporal_prior_config is not None
            else None
        )

    def __getitem__(self, idx):
        ...
        if self.prior_synth is not None:
            image, boxes = self.prior_synth(image, boxes)
        ...
```

The training script reads model config, sees `in_ch: 4`, sets `temporal_prior_config={...}` from the train.yaml's `augment.temporal_prior` section.

## Config schema (added to train.yaml augment block)

```yaml
augment:
  # ... existing augment options ...

  temporal_prior:
    enabled: true                         # auto-true when model in_ch == 4
    n_max: 8
    motion_axis_aligned_prob: 0.7
    motion_diagonal_prob: 0.25
    motion_zero_prob: 0.05
    motion_speed_range: [2.0, 15.0]
    motion_diagonal_jitter_deg: 10.0
    confidence_range: [0.5, 0.95]
    object_drop_prob: 0.05
    false_positive_prob: 0.10
    false_positive_count_range: [1, 3]
    false_positive_amplitude_range: [0.3, 0.5]
    spawn_zone_prob: 0.10
    spawn_zone_amplitude_range: [0.3, 0.5]
    zero_prior_prob: 0.05
    gaussian_sigma_factor: 4.0
```

If the user doesn't specify the `temporal_prior` block, fall back to defaults (the values above).

## Critical correctness assertions

Bake these into the implementation:

1. **The prior must NOT include current-frame GT.** Stamps are at offset positions from current GT, simulating where objects were on previous frames. Add an assertion in test mode that verifies stamps are at offset positions, not at current GT centers (within some tolerance).

2. **The prior must be in [0, 1].** Add `assert prior.min() >= 0 and prior.max() <= 1` at the end of `__call__` in debug mode.

3. **The prior shape must match stride.** `assert prior.shape == (image.shape[1] // stride, image.shape[2] // stride)`.

4. **Photometric aug must not touch the prior channel.** Verify the photometric augmentation pipeline operates on channels [0:3] only when image is 4-channel. Probably needs a small change in `augment.py`.

## Visualization helper for QA

Provide a small utility that visualizes synthesized priors for a few sample images:

```python
# scripts/visualize_prior_synth.py
def visualize_samples(dataset, n=16, out_path="prior_samples.png"):
    """
    Sample n examples from dataset, render image + prior overlay side by side.
    Save as a grid PNG. Useful for visually QA'ing the synthesis is producing
    plausible priors.
    """
```

This catches synthesis bugs that might be hard to find via metrics alone — e.g., stamps in wrong coordinate frames, motion offsets pointing wrong direction, sigma scaling off, etc. Run it once after implementing and eyeball ~32 examples.

## Test cases

In `tests/test_temporal_prior.py`, verify:

1. **Zero history (N=0) produces zero prior** unless spawn zone fires.
2. **Stationary motion (motion=0,0) produces prior at exactly current GT position** (no offset).
3. **Axis-aligned motion produces trail offset in correct direction**: e.g., right-moving objects have trails extending leftward from current GT.
4. **Confidence-weighted amplitude.** Verify stamps at `confidence_range` upper bound peak near 0.95, not 1.0.
5. **Cold-start dropout works.** With `zero_prior_prob=1.0`, output prior is always all zeros.
6. **No leakage from current frame.** Verify (over many samples) that the prior's hot regions are NEVER exactly at current GT centers when motion is nonzero.
7. **Output shape.** `assert prior.shape == (96, 128)` for standard input.
8. **In-range values.** All prior values in [0, 1].

## Style notes

Match the existing opndet codebase style:

- Terse code, no docstrings on helpers unless WHY is non-obvious.
- Use numpy not torch in the synthesizer (runs on CPU in dataloader workers).
- Single-file module, no submodules.
- Imports at top: `numpy as np`, `math`, `dataclasses` if needed.
- Use `np.random.default_rng(seed)` for determinism, not `random` module.

## Definition of done

- `src/opndet/augment_temporal_prior.py` exists and implements `TemporalPriorSynth`.
- `dataset.py` is updated to invoke it when `in_ch == 4`.
- `augment.py` is updated to skip photometric ops on the 4th channel when present.
- `train.yaml` schema documents the `augment.temporal_prior` config block.
- `tests/test_temporal_prior.py` covers the eight test cases above.
- `scripts/visualize_prior_synth.py` produces a 16-sample grid PNG for visual QA.
- Default training config for `bbox-f-egg-tp` (a new preset to be created separately) wires this in.
