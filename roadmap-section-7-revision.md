# Roadmap revision: Section 7 — Temporal Prior Architecture

## Context for Claude Code

The existing Section 7 of `ROADMAP.md` describes a temporal prior input architecture, but the framing has evolved through design discussion. This spec replaces Section 7 with a corrected, simpler, more shippable version. The core architectural commitment is unchanged (4-channel input, prior is concatenated with RGB at the stem), but the *story* around it is much cleaner and several earlier design decisions have been revised.

## What to change

Replace the entire current Section 7 ("Temporal prior input — center stabilization via short tails") with the content below. Keep section numbering 7.1 through 7.9 but the content of each subsection changes substantially.

Do not modify Sections 1-6 or Section 8.

## Key conceptual shifts from the current Section 7

These are the points the new content reflects. Reference for review, not literal text to include:

1. **The prior is not just for "center stabilization."** It's a general "developer-controlled attention channel." Temporal accumulator is the most common use, but ROI masks, multi-camera fusion, operator heuristics, and spawn-zone hints are all valid uses of the same input.

2. **The model learns the combination, we don't engineer it.** Earlier drafts agonized over whether to use geometric mean, multiplicative boost, lock-on heuristics, etc. All of that is wrong. The model takes (RGB + prior) as input and learns the combination function during training. CNNs are good at this. Don't hand-engineer.

3. **K=1 (presence only) is the v1 commitment.** Previously the doc considered K=3 (presence + size memory) and K=5 (presence + size + rotation). Defer all of these. Center stability is the dominant deployment complaint; size memory is application-side averageable; rotation memory has no clear customer use case yet.

4. **Bbox-stamped Gaussians, not point stamps.** The accumulator stamps anisotropic Gaussians sized to the detection's bbox, with amplitude equal to the detection's confidence. Hard rectangles work but Gaussians give smoother gradients for the model to read. Stamp shape matches what the model produced (bbox), not arbitrary fixed-sigma points.

5. **Confidence-weighted accumulation, not binary.** Each detection contributes its confidence value to the accumulator (a 0.95 detection stamps at 0.95). This is the natural mechanism for flap suppression: confident persistent detections build hot trails; flappy weak detections never accumulate.

6. **Subtract first, then stamp.** Ordering matters. Fade the accumulator, then stamp current detections. Keeps current detections always at full amplitude even at stationary positions.

7. **The model handles motion via gradient-reading, not via explicit motion modeling.** For moving objects the accumulator naturally produces a trail offset behind the current position. The model learns to read the trail's gradient and place the detection just past the leading edge. This is well within CNN capabilities and doesn't require special architecture.

8. **Always-on conveyor scenario is the primary use case.** The system runs continuously. Cold start is one frame. Spawn zones at known entry edges help initial detection. Despawn happens via fade. Motion direction is fixed per deployment but varies across deployments — the model is trained on a distribution of directions.

9. **Snapshot mode is NOT a coequal mode.** Earlier drafts said "30-50% prior dropout for snapshot compatibility." Drop this. The temporal model is a separate product variant for video deployments; the snapshot model is for stills. Don't make one checkpoint serve both — it costs capacity and creates training/deployment distribution mismatch. The temporal model handles cold-start naturally because the accumulator simply starts at zero on frame 1, which the model sees as a normal "very faded prior" input — no special "snapshot compatibility" required.

10. **Equal-weight combining is the goal at deployment, but achieved by training, not by host-side math.** The model is trained on (RGB + prior) → GT and learns to weight them appropriately based on context. No host-side combining function (geometric mean, multiplicative boost, etc.) is needed. The deployment story is just "feed the prior in, get detections out."

11. **Spawn zones are operator-supplied static priors that get max-merged with the dynamic accumulator.** Operator provides a binary mask + amplitude (e.g., 0.4) for known entry points. The accumulator update applies the spawn mask at the end of each fade+stamp cycle so spawn zones don't fade away. This boosts initial detection at known spawn locations without retraining.

12. **The 7/10 → 10/10 detection consistency goal.** Current snapshot bbox-f-egg detects an object on roughly 7 of 10 consecutive frames; the temporal model targets 10/10. This is the headline success metric. Subordinate metrics (center stability, confidence consistency, bbox dimension stability) are downstream of this primary goal.

## Replacement content for Section 7

```markdown
## Part 7 — Temporal prior input (center stabilization and lock-on)

This section specifies opndet's temporal-mode architecture: a model variant that takes a 4th input channel (the "prior") alongside the standard 3-channel RGB input. The prior carries spatial information about where objects were recently detected, allowing the model to produce stable, locked-on detections frame-to-frame instead of the flappy, jittery output that pure-snapshot detectors produce on video streams.

The headline success metric: where snapshot bbox-f-egg detects a given object on roughly 7 of 10 consecutive frames, temporal-mode bbox-f-egg-tp targets 10/10. Subordinate metrics (center jitter, confidence flapping, bbox dimension stability) all improve as a consequence of the primary lock-on behavior.

### 7.1 Architecture

The temporal model is bbox-f-egg with `in_ch: 4`. The stem conv accepts 4 channels instead of 3. The prior heatmap is at stride 4 (matching output stride), 96×128 spatial resolution, single channel. Concatenated with RGB at the very front, processed by the stem conv alongside RGB.

Param count delta: ~108 weights added to the stem (12 stem output channels × 1 new input channel × 3×3 kernel). Negligible.

The rest of the network is identical to bbox-f-egg. Same backbone, same head, same peak suppression, same output format `[1, 5, H/4, W/4]`.

This is opt-in via a new YAML preset (`opndet-bbox-f-egg-tp.yaml`) and is *not* the default for any tier. Deployments must explicitly choose the temporal variant. The temporal variant is its own product, not a drop-in replacement for the snapshot model.

### 7.2 The accumulator (deployment-side)

The prior is produced at deployment by a host-side accumulator. Reference implementation:

```python
class TailAccumulator:
    def __init__(self, shape, n_frames=8, stamp_threshold=0.4,
                 spawn_mask=None, spawn_amplitude=0.4):
        self.acc = np.zeros(shape, dtype=np.float32)
        self.fade_step = 1.0 / n_frames
        self.stamp_threshold = stamp_threshold
        self.spawn_mask = spawn_mask
        self.spawn_amplitude = spawn_amplitude

    def update(self, detections):
        # 1. Fade accumulator
        self.acc = np.maximum(self.acc - self.fade_step, 0.0)

        # 2. Stamp current confident detections as Gaussian footprints
        for det in detections:
            if det.score >= self.stamp_threshold:
                self._stamp_gaussian(det.box, amplitude=det.score)

        # 3. Apply spawn-zone baseline (doesn't fade)
        if self.spawn_mask is not None:
            self.acc = np.maximum(self.acc, self.spawn_mask * self.spawn_amplitude)

        return self.acc

    def _stamp_gaussian(self, box, amplitude):
        cx, cy = box_center(box)
        sigma_x = (box[2] - box[0]) / 4   # bbox edge at ~2 sigma
        sigma_y = (box[3] - box[1]) / 4
        # Render anisotropic 2D Gaussian, max-merge into self.acc
        ...
```

Properties:

- **O(1) memory.** Single 96×128 fp32 buffer (~50 KB). No frame history maintained.
- **Confidence-weighted.** A detection's stamp amplitude equals its confidence. Confident detections build hot trails; flappy weak ones don't accumulate.
- **Bbox-shaped Gaussians.** Stamp shape matches the detection's bbox (anisotropic, sigma proportional to edge length). Mirrors what the model produces.
- **Bounded in [0, 1].** Max stamp amplitude is 1.0; fade ensures saturation doesn't exceed this.
- **Fade-then-stamp ordering.** Current detections always get full amplitude even at stationary positions.
- **Static spawn-zone overlay.** Operator-provided mask is max-merged after fade+stamp so it doesn't fade away.

### 7.3 Deployment knobs

| Parameter | Controls | Default | Tune for |
|---|---|---|---|
| `n_frames` (or `fade_step = 1/N`) | How long detection footprints persist | 8 | Higher (15-30) for slow scenes / detection dropouts; lower (3-5) for fast or chaotic scenes |
| `stamp_threshold` | Minimum confidence to enter accumulator | 0.4 | Higher rejects more noise; lower stamps more weak detections |
| `spawn_mask` | Static prior at known entry points | None | Set per deployment to operator-known spawn zones |
| `spawn_amplitude` | Strength of static spawn-zone hint | 0.4 | Higher = more help with initial detection; risk of false positives at edges |

All four are tunable at deployment without retraining.

### 7.4 Training procedure

The temporal model is trained on (RGB, synthesized prior, GT) triples. The synthesized prior must match the distribution the deployment accumulator produces from real frame history.

The synthesizer is a per-sample augmentation step that runs after geometric augmentation (so the prior matches the augmented image) and before normalization. It produces a 96×128 fp32 prior that is concatenated with the image as the 4th channel.

Synthesis algorithm:

```
For each training sample (current frame's GT boxes):
    1. Sample N uniform [0, 8]. N=0 means cold-start (zero prior).
    2. Sample motion vector for this scene:
       - 70%: axis-aligned (one of L/R/U/D) at uniform speed [2, 15] px/frame
       - 25%: diagonal (45/135/225/315 deg with small angle jitter) at same speed range
       - 5%: zero (stationary)
    3. For k in N..1 (oldest to newest):
       - Offset GT positions by k * motion_vector (objects came from offset-back position)
       - Sample per-frame confidence uniform [0.5, 0.95] for each object
       - Stamp Gaussian footprints at offset positions with that confidence amplitude
       - Apply fade: amplitude_after_fade = confidence - (k-1)/N
    4. With ~5% probability per object, drop the object's stamp entirely (simulate detection miss).
    5. With ~10% probability, add 1-3 false-positive stamps at random positions, low amplitude (0.3-0.5).
    6. With ~10% probability, add a spawn-zone hot region at a random frame edge.
    7. With ~5% probability, zero the entire prior (additional cold-start coverage beyond N=0 cases).
```

Critical correctness note: NEVER build the training prior from the current frame's GT. The prior must reflect what the *previous* frames' detections would have looked like, not what the current frame is. Building the prior from current GT teaches the model to copy the prior, which works perfectly in training and fails catastrophically at inference. Bake an assertion into the data pipeline: `assert prior_built_from_offset_history(t) and not prior_built_from(t)`.

### 7.5 What the model learns

Given (RGB + prior) → GT supervision across the synthesized distribution, the model learns to:

- **Use prior + RGB agreement as confidence boost.** Hot prior + matching RGB texture = confident detection at exactly that location.
- **Suppress phantoms.** Hot prior + RGB doesn't match (object left) = low confidence, no detection.
- **Detect new objects.** Cold prior + strong RGB = normal detection, slightly attenuated relative to prior-supported case.
- **Read motion gradients.** Trail in the prior with directional gradient → predict detection just past the leading edge, with RGB localizing the exact center.
- **Use spawn-zone hints.** Persistent moderate-amplitude region at frame edge → expect new objects to appear there, lift initial detections accordingly.

All of these emerge from training on the diverse synthesized distribution. No special architectural mechanisms required.

### 7.6 The first-frame question

When the system starts, the accumulator is zero. The model sees a zero prior and falls back to RGB-only detection. The synthesizer covers this case (N=0 samples + occasional zero-prior samples), so the model is trained for it.

Detection on frame 1 is approximately as good as snapshot mode would be. Frame 2 has a faint prior from frame 1's detections. By frame 5 or so, the prior is established and lock-on behavior kicks in.

For deployments that need stronger frame-1 detection, the operator-supplied spawn zone provides a static prior at known entry points, helping initial detection from the very first frame.

### 7.7 Use cases for the prior channel beyond temporal accumulation

The prior is a developer-controlled attention channel. The temporal accumulator is the most common source, but other valid sources include:

- **Static ROI mask.** Operator paints regions of interest; model focuses there.
- **Inactive-lane masking.** Multi-lane conveyor where some lanes are offline; mask out inactive lanes via dark prior values.
- **Multi-camera fusion.** Use detections from one camera (projected through calibration) as the prior for an adjacent camera.
- **Output of another model.** Depth maps, segmentation masks, motion masks from optical flow — anything spatial.
- **Engineered heuristics.** Operator-specific knowledge encoded as a static or slowly-updating prior.

The model is trained on a distribution of prior shapes (smooth fading accumulators, hard binary masks, sparse stamps, zero priors), so it generalizes to these alternative sources without retraining.

### 7.8 Tracker integration

Temporal mode produces stable per-frame detections but does not assign persistent identity. For applications that need ID-stable tracking (counting line crossings, etc.), pair temporal-mode opndet with ByteTrack, SORT, or similar.

ByteTrack on temporal-mode opndet detections should significantly outperform ByteTrack on snapshot-mode detections — trackers work much better when fed stable detections than jittery ones. The temporal prior addresses jitter at the source, before tracking.

### 7.9 Validation

Before promoting temporal mode beyond experimental status, verify the model uses the prior correctly.

Tests, all on synthesized or real video sequences:

1. **7-of-10 → 10-of-10 metric.** Primary success metric. On consecutive-frame video clips, measure the fraction of frames on which a given object is correctly detected. Temporal mode should hit ≥95% across representative clips; snapshot mode bbox-f-egg currently hits ~70-80%.

2. **Center stability.** For correctly-detected objects across consecutive frames, measure stddev of detected center position. Temporal mode should reduce this by ≥3× vs snapshot.

3. **Confidence stability.** Same setup, measure stddev of detection confidence. Temporal mode should produce smoother confidence trajectories (less flapping at borderline cases).

4. **Wrong prior doesn't break it.** Feed a prior with random offset from truth. Model degrades gracefully toward snapshot performance, doesn't produce phantom detections.

5. **Self-feedback stability.** Running the temporal model on a long video with its own predictions fed back as prior should converge to stable counts, not drift.

6. **N-invariance.** Same model at different fade rates (N=2, 5, 10, 30, 100) should all produce reasonable results. Validates that runtime N-tuning works.

7. **Highway preservation.** Synthetic dense-pack data should still produce per-object detections, not blob predictions.

8. **Tracker pairing.** ByteTrack on temporal-mode detections produces fewer ID switches per object lifetime than ByteTrack on snapshot-mode detections.

9. **Disappearance handling.** Object present for K frames, then disappears. Detection should drop within 1-2 frames (RGB absence overrides faded prior). No ghost detection persisting for the full fade duration.

10. **Spawn-zone bootstrap.** With a configured spawn zone, frame-1 detection at the spawn region should be measurably better than without.

Definition of done: temporal model variant ships as a separate preset (`bbox-f-egg-tp` or similar). Training synthesis is implemented and documented. Reference accumulator code is included in the SDK. Validation tests above all pass on representative test sequences. Documentation explains when to use temporal vs snapshot mode and how to configure the spawn zone for a deployment.
```

## Notes for Claude Code on style

- Match the existing roadmap's tone: terse, opinionated, engineering-focused, no buzzwords.
- Headers use `###` for subsections.
- Code blocks use triple backticks with no language tag (existing roadmap convention).
- Don't add hype language. The roadmap is intentionally sober.
- Preserve the existing definition-of-done pattern at the end of each subsection.

## What gets removed

The current Section 7 mentions the following ideas that are now removed entirely:

- "30-50% prior dropout for snapshot compatibility" (snapshot mode is a separate product, not a coequal mode in the same checkpoint)
- K=3 and K=5 prior variants (deferred until validated customer demand)
- Streaks/contrails framing (replaced with bbox-stamped Gaussians)
- Long-tail discussion (Default N is 8, range is small; long tails were over-thinking it)
- Lock-on as a separate engineered behavior (replaced with "the model learns it from training")
- Host-side boost combining functions (geometric mean, multiplicative boost, etc. — model learns the combination)

## What gets preserved from the existing Section 7

- The basic 4-channel architecture commitment
- The accumulator concept with subtract-first, then stamp
- The "snapshot model output stays [1, 5, H/4, W/4]" backward compat for the rest of the network
- The idea that the accumulator runs host-side and is fed in as model input
- The validation tests (refined and reordered)
- The pairing with ByteTrack for tracker integration
