# Engineering decisions and gotchas

Non-obvious choices and the reasoning behind them. Read before changing training defaults or "simplifying" code that looks redundant. Each entry has the commit hash for traceability.

This doc is for future LLMs / engineers who don't have session-context. The git log captures *what* changed; this captures *why we still have it that way*.

---

## Loss configuration

### `cls_loss: focal` is the cold-start safe default. VFL has a zero-IoU pos-gradient collapse mode.

**Symptom:** kitchen-sink configs with `cls_loss: vfl` would collapse to "predict nothing" between ep 2 and ep 5. Loss decreased monotonically, but only because positive predictions vanished entirely.

**Root cause:** VFL positive loss = `iou_target * BCE_pos`. At random init, predicted boxes are ~192×192 (sigmoid(0)=0.5 × img_size). For a 40×40 GT, IoU(192², 40²) ≈ 0.04 → `iou_target ≈ 0` at GT cells → positive gradient ≈ 0. Only the negative term has signal, which pulls all sigmoid output → 0. Model learns "predict nothing = safe."

**Fix:** Use `cls_loss: focal` for cold start. Plain focal has no IoU dependence; it always supervises positive cells regardless of box quality. Curr_2 (the working baseline) used focal by default.

**When VFL would be safe:** post-warmup, after boxes are roughly correctly sized (say after w_wh has ramped in). A future curriculum that switches focal → vfl after ep N would unlock VFL's ranking benefits without the cold-start failure mode. Not implemented yet; kitchen-sink stays focal throughout.

### `wh_loss: diou` is bf16-stable. CIoU + fp16 has gradient overflow risk.

**Symptom:** A run with `wh_loss: ciou` + `amp_dtype: fp16` had mAP@.5:.95 collapse from 0.852 → 0.495 around ep 30. Other metrics held but the box loss diverged.

**Root cause:** CIoU includes the aspect-ratio `v` term `(4/π² × (atan(w/h) − atan(w_gt/h_gt))²)`. At high aspect ratios this becomes large, and its gradient w.r.t. `w` and `h` involves sums of large products. fp16's 5-bit exponent saturates.

**Fix:** Either `amp_dtype: bf16` (8-bit exponent, no overflow) OR `wh_loss: diou` (CIoU minus the aspect-ratio term, just distance + IoU). bf16 is the cleaner architectural fix.

### Repulsion loss subtracts a baseline (commit 296acf4)

**Why:** RepGT-style penalty was `IoA(pred, neighbor_GT)`. For two GT boxes that already overlap (touching/stacked objects), even a perfect prediction (`pred ≡ GT_self`) had a non-zero phantom penalty — biasing the wh head toward shrunken boxes near any neighbor.

**Fix:** subtract `IoA(GT_self, GT_neighbor)` from the penalty. A perfect prediction now scores 0 regardless of GT overlap. Penalty fires only on EXCESS overlap beyond what the GTs already share.

**Future note:** OBB heads will mostly eliminate AABB GT overlap (rotated convex objects rarely have overlapping OBBs). Once OBB ships (§2.1), this baseline subtraction becomes mostly a no-op. Keep it for AABB compatibility.

### Assigner choice — TAL drives BOX REGRESSION only; cls stays Gaussian heatmap

**ROADMAP §1.8 Phase 3 (commit at-tip).** opndet's `-pro` variants default to `assigner: stal` (size-adaptive TAL). This is *unusual* compared to YOLO-family literature, where TAL drives BOTH cls and reg supervision via the alignment metric `t = s^α · IoU^β` as a soft cls target.

**What we do instead:** TAL/STAL pick the per-cell positive set for box regression only. The classification (objectness) head keeps its Gaussian heatmap target from `encode_targets_*`. The TAL alignment score `target_t` is computed and exposed for diagnostics but is NOT consumed by the loss.

**Why:** opndet's heatmap head produces *emergent segmentation* — the soft post-sigmoid map (`soft_obj`) reads as a high-quality blob mask after training, useful as a free byproduct for downstream consumers (count + rough mask). A peer-reviewed v8-style soft-label cls target would flatten the heatmap toward `t_normalized` instead of `1.0` at GT centers, weakening the centerness peak and degrading the segmentation byproduct. Keeping the Gaussian target preserves this property.

**Trade-off:** we lose the v8 "soft label" effect on cls, which on dense/occluded scenes can help cls calibration. Net: empirical evaluation on convex-object datasets shows the Gaussian-cls + TAL-reg hybrid produces stronger box regression (multiple supervised cells per GT) without sacrificing the centerness signature. Run `opndet analyze` after a TAL/STAL training run to verify the centerness map looks similar to the peak-baseline run; if not, tune `topk` lower (4 instead of 10) or revert to `assigner: peak`.

**OBB caveat.** TAL's IoU term ranks per-cell predictions against each GT. For OBB GTs we use the *enclosing AABB IoU* as the ranking signal, not true rotated IoU. Reasoning: rotated IoU requires either Sutherland-Hodgman polygon clipping (O(n) but no GPU kernel in PyTorch) or a Monte-Carlo approximation. Both are expensive for ranking; the enclosing-AABB IoU is monotonic with rotated IoU when objects are mostly axis-aligned (the common case), and remains a useful (if imperfect) ranking signal at high rotation. The angle channels are still supervised correctly via the per-cell `(sin2θ, cos2θ)` propagated from each GT's center cell to all cells assigned to that GT.

### Curriculum keys use long names; loss_fn uses short attrs (alias map, commit 0f90448)

**Why this is non-obvious:** The yaml curriculum block accepts keys like `repulsion_weight`, `count_weight`, `convexity_weight` (matching loss config keys). But `Loss.__init__` stores them into `self.rep_w`, `self.count_w`, `self.convex_w`. Without an alias map, `setattr(loss_fn, "repulsion_weight", v)` is a silent no-op (hasattr returns False), and the constructor-time weights remain live forever.

**The alias map** lives in `train.py::_curriculum_attr_aliases`. If you add a new loss component with both long and short names, ADD IT TO THIS MAP or curriculum will silently no-op for it. There's a one-time warn print at ep 1 if a key fails to resolve — watch for `WARN: curriculum key '...' has no loss_fn attribute`.

---

## Architecture / output

### `peak_kernel=7` for bbox-x specifically; other presets at k=5

**Why:** k=5 (radius 2, 8-px min spacing at stride=4) suppresses any non-tied pair within 2 cells. But two cells with sigmoid output that rounds to identical fp values (common at saturation) BOTH pass the `>= MaxPool` test since each IS the local max. Wider window (k=7, radius 3, 12-px min spacing) catches close-but-not-tied pairs out to 3 cells, killing the dominant case of adjacent duplicates.

**Why not all presets:** k=7 caps minimum detection density at 12 px center-to-center. For small/embedded presets (bbox-f/p/n), the user might want higher density. bbox-x is quality-first, objects are spaced ≥30 px in typical industrial framing, k=7 is safe.

**Tied-pair edge case:** If two cells have *exactly* identical fp values, no kernel size suppresses both. The architectural fix would be position-encoded tiebreak (add tiny position-dependent constant before MaxPool to break ties deterministically). Not implemented; if duplicates persist after k=7 + repulsion, hard-negative mining (§1.7) is the next attack surface.

### Output layout `[1, 5, H/4, W/4]` is the deployed contract — don't break it

The five channels (`obj`, `cx`, `cy`, `w`, `h`) and the post-suppression sparse `obj` channel are what client decoders expect. The `--diagnostic` ONNX export adds *additional* outputs but never modifies the production output. Variant heads (`bbox-x-hm2`, `bbox-x-flow`) have different output shapes and are documented as separate inference contracts in `docs/det-hm-variants.md`.

---

## Tier ops — server-only ops gated by `model.tier` field

opndet's preset YAMLs declare a deployment tier via `model.tier: edge | server` (default edge). The export-time op allowlist (`export.py::allowed_ops_for_tier`) is keyed on this tier; certain ops are accepted only at server tier even though they're opset-13 valid.

**`C2PSA` in -pro server presets is single-block multi-head spatial attention at p4 (post-SPPF) + matching neck stage (post-`b4`).** Multi-head qkv via 1×1 conv → softmax over flattened H×W → projection back, with residual. Adds ~5-15% mAP on small objects per YOLOv11 paper. Costs server-tier-only deployment: the emitted `MatMul` + `Softmax` ops compile cleanly under ONNX Runtime / OpenVINO 2022 CPU/GPU / Jetson but Myriad VPU breaks on dense softmax-over-spatial-tensors. Edge tier (`bbox-{f,p,n,s}-pro`) explicitly excludes attention to keep Myriad export viable.

**`SiLU` (`x * sigmoid(x)`) is registered as a server-tier-only activation.** Exports as `Mul` + `Sigmoid` — both ops are in the base ALLOWED_OPS, but the Mul+Sigmoid fusion pattern is reliably ORT/CPU/GPU only. Myriad VPU has SiLU compilation issues even when neither op is forbidden in isolation. `ConvBnAct` accepts `act: silu | relu6`; default `relu6` is the edge-safe path.

**`ResizeBilinear2xHalfPixel` is server-tier-only.** Bilinear-2x upsample with the `half_pixel` `coordinate_transformation_mode`. Asymmetric (the default for `ResizeNearest2x`) is the only Myriad-safe coord transform. The export check `check_resize_attrs(om, tier="edge")` raises `RuntimeError` if a Resize node has a non-asymmetric coord transform and the model is edge-tier.

**Tier defaults to `edge` if absent from YAML** — preserves backwards compatibility for any user preset authored before Phase 2.

**Param-count cost of attention.** C2PSA at both insertion sites adds: bbox-m-pro +0.45M (3.46M → 3.91M), bbox-l-pro +1.15M (8.53M → 9.69M), bbox-x-pro +1.90M (17.29M → 19.19M). Higher than the YOLOv11 spec's "~50-200K per insertion site" because we run full multi-head attention with a 3× channel qkv conv at each site rather than the spec's chunk-half optimization.

---

## Metrics

### Standard mAP@.5:.95 lies on small objects at stride=4. Shape-mAP is the honest companion.

**Math:** at stride=4 with ~1 px SGD residual, the architectural IoU ceiling for a 30-px object is ~0.94. mAP@.95 is *literally impossible* even on a perfect-modulo-noise model. mAP@.5:.95 averages 10 thresholds including unreachable ones, dragging the headline number down.

**Solution:** `map_50_95_shape` (commit 00365bb) — IoU computed after translating pred to share GT's center. Decouples mAP from sub-pixel center precision. Fair when stride limits center precision; converges with standard mAP as image resolution grows. Reported alongside, never replaces, standard mAP (we keep both for cross-paper comparability).

**Why not silently fudge mAP:** users compare opndet's mAP to YOLO/MMDet papers. Silently changing the definition would mislead. Two metrics, clearly named.

### `center_match` uses cell-window radius (commit 1ec5f2c, after reverting 6e38b1d's bbox-containment match)

**Why bbox-containment was reverted:** at random init, predicted bboxes default to ~192×192 (sigmoid(0)=0.5 × img_size). Every pred "contained" every GT center. F1_lenient read 0.80 at ep 2 of a collapsing run while real recall was random — the metric falsely indicated health.

**Current matching radius:** `max(cell_window/2 * stride, dist_frac * min(gw,gh), min_dist_px)`. cell_window=4 default → ±8 px floor (a 4×4-cell window centered on GT). Pure distance-based Hungarian; no bbox-size leniency. Ghost vs duplicate split: a non-matched pred is a *duplicate* if its center is within some other GT's radius, otherwise a *ghost* (phantom in empty space).

---

## Patience / early stopping

### Trajectory-patience > best-not-improved (commit 55bf3c4)

Single-metric patience kills runs when the watched metric saturates, even if other key metrics are still climbing. Multi-metric trajectory patience fits a slope per metric over a window; with the `any_climbing` rule, the run keeps going if ANY metric has positive normalized slope above `patience_min_slope`.

### Curriculum-aware floor (commit caadfc5)

Patience can't fire before `last_curriculum_end_epoch + window`. Without this, the run was stopping at ep 55 with curriculum-stage repulsion at 80% and count at 12.5% — the not-yet-ramped losses hadn't had time to move metrics, so existing slopes flattened on already-trained signals. Floor enforces "wait until all curriculum has finished + a window of post-curriculum measurement before considering trajectory-patience."

**Print line at training start** shows the resolved floor: `earliest-fire-epoch=N (curriculum_end=M+window)`. If the user sees this happening earlier than expected, a curriculum end_epoch may be misconfigured.

### `eval_threshold: auto` self-tunes via EMA (commit 1024c21)

Each epoch, evaluate computes `threshold_opt` (the threshold where F1 peaks for current model state). Setting `eval_threshold: auto` makes the next epoch's score_thresh = EMA of past `threshold_opt` values. Self-correcting: starts loose (bootstrap=0.10 to surface early predictions in viz), settles tight (~0.45-0.50) by ep 30. Removes the "guess the right operating threshold" config burden.

---

## Augmentation

### `min_visible_frac: 0.7` for kitchen-sink (vs default 0.5)

Cleaner training labels under cutout / scale aug. Boxes whose visible area falls below 70% are dropped from supervision. Prevents the model from being told to predict "this fragment of an object behind a cutout" — when it can't see most of the object, it shouldn't be supervised on its full bbox.

### `mosaic_prob: 0.0` for count-aware training

Mosaic mixes 4 images' worth of objects into one frame. The count loss expects `peak_count == n_gt`; mosaic stuffs 4× expected objects in one image, corrupting the count signal. Disable mosaic when `count_weight > 0`.

### `scale_jitter` disabled in current kitchen-sink

User explicitly set `scale_jitter: [1.0, 1.0]` (no jitter). May be re-enabled for scale-generalization runs. Architecture has limited scale invariance (single-scale stride=4 output); scale variation has to come from training data. Multi-scale output head (§3.3) is the architectural fix; scale_jitter is the data fix.

---

## Colab / deployment gotchas

### Always push commits

User pulls source via `pip install git+...@main` in Colab. Unpushed commits = stale install = redebugging the wrong code. Push immediately after every commit. Memory file `feedback_always_push.md` captures this as durable preference.

### In-process Colab training, not subprocess

`from opndet.train import train; train(cfg)` works for the auto-download-on-bundle. `!opndet train --config ...` does not — subprocess has no IPython kernel context, `files.download()` fails silently. Don't recommend the subprocess form for Colab.

### Colab numpy mismatches

Symptom: `_blas_supports_fpe`, `_center`, or tensorboard.compat.notf import errors. Cause: Colab silently ships incompatible numpy/scipy/tensorboard combinations. First-line fix: `Runtime > Change runtime type > Fallback runtime version`. Second-line: pin `numpy<2.3 scipy>=1.13 tensorboard>=2.16` at top of install cell + restart kernel. Memory file `feedback_colab_numpy_breakage.md` has the full debug path.

### Tensorboard optional (commit df11e83)

Training continues even if tensorboard import fails. Scalars still flow to DuckDB + dashboard. Add `tensorboard: false` to config to opt out explicitly.

---

## NO scipy. Use `lap` + `_optim.py` instead. — **HARD RULE**

opndet **does not depend on scipy**. Specifically: do NOT add `from scipy.optimize import ...` (or any other scipy import) to opndet code, even if it's "just one function" or "the standard way." Reasons, painful and concrete:

**Why scipy was removed (April 2026 incident):**
- scipy reaches deep into numpy's PRIVATE API (`numpy._core.umath._center`, `numpy.testing._private.utils.BLAS_SUPPORTS_FPE`, etc.)
- numpy bumps INTERNAL APIs without major-version-bumping (e.g. removed `_center` between 2.3 → 2.4)
- scipy then breaks. Their pyproject says `numpy>=2.0.0` (no upper bound), which is a lie about what their code actually requires
- Colab silently rolls numpy versions on a daily-ish cadence. We hit broken-scipy → broken-opndet ~weekly
- Forced explicit `numpy<2.3` pins in user notebooks AS A WORKAROUND. Each Colab session needed manual version-fighting.

**The fix:**
- `src/opndet/_optim.py` exposes `linear_sum_assignment` (wraps `lap.lapjv`) and `minimize_scalar_bounded` (pure-numpy golden-section search). scipy-compatible signatures so call sites are unchanged.
- `lap` (gatagat/lap) is a tiny C-compiled Hungarian solver. Talks to numpy via the public ndarray interface only — numpy private-API churn cannot reach it. Install footprint ~1 MB vs scipy's ~70 MB.
- `minimize_scalar_bounded` is ~30 LOC of pure Python. Zero external deps. Cannot break.

**Rule for future contributors / agents:**
1. Never add `import scipy` or `from scipy...` to opndet source. If a scipy function looks tempting, check if `_optim.py` already wraps an equivalent.
2. If you need a NEW optim/numerics primitive, prefer (in order):
   - Pure-numpy implementation in `_optim.py` (~30-150 LOC, zero risk)
   - A small specialized library with NARROW numpy surface (like `lap` for assignment)
   - NEVER scipy. Even for "just one call" — the dep cascades.
3. The replacements are scipy-compatible by signature so refactoring back/forth is cheap. Don't accept "I'll just use scipy for now and refactor later" — `_optim.py` already has the function you need.

**Same rule applies to other research-coded sprawling libs:**
- `pandas` — same offenders, large numpy private-API surface. Avoid for opndet.
- `xarray` — same.
- `statsmodels` — same.
- `sklearn` — bigger surface than scipy, but at least mostly stable. Hard ask before adding.

**OK to use:** torch, torchvision, numpy (public API), opencv-python-headless, onnx, onnxruntime, pyyaml, pillow, matplotlib, tqdm, tensorboard. All in pyproject.toml — that list is not aspirational, it's the entire dep set.

---

## Dep pinning (pyproject.toml has upper bounds)

opndet's `pyproject.toml` uses major-pin upper bounds (`<2.12`, `< 2.3`, `~=4.8`) on every dep, NOT the more common `>=` floors-only style. This is intentional.

**Why:** Colab and other shared-runtime environments roll their base-image package versions on no announced schedule. We had ~5 separate breakages in a single week from Colab silently bumping numpy 2.3→2.4, torch 2.10→2.11, tensorboard 2.19→2.20, protobuf 6→7, pillow 11→12, fsspec 2025→2026. Each cascade broke `pip install opndet` because our floors-only `>=` deps let pip resolve ANY new upstream version. Force-reinstall made it worse — even our pinned torch in install-cell-1 was overridden by opndet's transitive resolution in install-cell-2.

**The fix is upper bounds.** PEP 440 supports two styles:
- `~= X.Y` is "compatible release" (pip's closest analog to Cargo/Poetry `^`): `~=2.10` = `>=2.10, <3`. Locks at the next-major boundary.
- `>= X, < Y` is explicit. Used when we need a non-major-aligned ceiling (e.g. `numpy < 2.3` even though numpy 3 doesn't exist yet).

We use a mixed style: `~=` where the next-major boundary is the right block; explicit `>=, <` where we need to block a specific bad release.

**The trade-off:** users get bug-fix updates within the current major, but no silent breaking changes. When a new major comes out and we want to validate it, we expand the bound — we don't chase upstream daily.

**When to update the bounds:**
- An upstream dep ships a major version that we WANT to use (e.g. torch 2.12 ships a real perf win)
- Validate locally + update the upper bound in pyproject.toml + bump opndet version
- Don't preemptively widen "just to support latest" — there's no benefit if no opndet feature requires the newer dep

**Optional deps** (`dev`, `video`, `dashboard` in `[project.optional-dependencies]`) aren't bounded as strictly because they affect single subcommands; if `opndet dashboard` breaks because fastapi shipped a major, that's contained.

---

## Optimizer choice

### `optimizer: adamw` is the default. `optimizer: musgd` is opt-in for server-tier `-pro`.

AdamW with cosine LR + warmup is the default for every preset. MuSGD (Muon for hidden conv weights + AdamW for biases / BN / narrow heads, per YOLO26 / Jordan 2024) is opt-in via `optimizer: musgd` in the training yaml. It is intended only for server-tier `-pro` presets (`bbox-{m,l,x}-pro`) — train.py warns but does not block if you set it on edge-tier presets.

**Honest caveat.** Muon's empirical wins are most clearly demonstrated in transformer training. For convnet vision tasks the literature is thinner; the YOLO26 paper claims faster convergence but doesn't ship rigorous A/B numbers. For `bbox-x-pro` the partition lands ~99.8% of params in the Muon leg (17.2M of 17.3M; the remaining 0.2% is BN affine, biases, and the 1×{l,t,r,b,obj}-channel head convs). That is exactly the regime where the Newton-Schulz orthogonalization is supposed to matter most — uniform spectrum updates on every conv weight — but it is also the regime where convnet inductive bias may already be doing the work.

**Validate on your own dataset before promoting.** The flag exists so you can measure. If MuSGD doesn't move val mAP after a full schedule, stick with AdamW — there is no theoretical reason it must win for vision.

**Knobs.** `muon_momentum` (default 0.95, heavy-ball β), `muon_lr_scale` (default 1.0; multiplies the cosine-scheduled LR for the Muon leg only), `muon_ns_steps` (default 5; canonical Jordan setting). The cosine LR schedule applies to all param groups uniformly.

**State-dict compat.** MuSGD checkpoints carry a `_tag: "musgd-v1"` field; loading an AdamW checkpoint into a MuSGD optimizer (or vice versa) raises `ValueError` rather than silently corrupting. Switch optimizers on a fresh run, not mid-training.

---

## What NOT to remove "to clean up"

These look redundant or weird but are load-bearing:

- **`peak_eps = 5e-3`** in `_peak_mask`: fp16 tolerance margin for parity between PyTorch and ORT/Myriad. Without it, the actual max cell can be rounded just below pooled value and dropped. Don't lower without re-running parity tests.
- **`hm + eps` not `hm > MaxPool`**: arithmetic comparison instead of GreaterOrEqual op for OpenVINO 2022 Myriad which has a known `Logical_OR` lowering bug.
- **`ResizeNearest2x` not `Resize` with `coordinate_transformation_mode=half_pixel`**: opset-13 / Myriad require `asymmetric` mode.
- **`SplitChannels`, `Add`, `Sigmoid`, `Concat`** as separate named layers in YAML: lets `forward_with_alias` grab them for the loss / analyze. Removing the names breaks the layer slider in `opndet analyze`.
- **`focal_alpha=2.0, focal_beta=4.0`** as established defaults: 6.0 caused predict-nothing collapse during cold start in kitchen-sink experiments.
- **Repulsion baseline subtraction**: looks like extra math but corrects a real failure mode for overlapping AABB GTs.
