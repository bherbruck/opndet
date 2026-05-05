# DuckDB metrics + dashboard plan

Replace TensorBoard's append-only tfevents protobuf with a single DuckDB file
per run, plus a thin viewer over it. SQL replaces ~all of TB's bespoke
indexer/event-accumulator/smoothing/run-comparison code. Layered images
become first-class (per-pixel overlay layers + JSON box annotations) so the
viewer can do interactive opacity/threshold/filter without re-rendering.

The headline win for opndet specifically: queries like
`"show me the val frames where FN > 2 AND val/f1_opt_cal dropped this epoch"`
are a SQL JOIN — TB cannot do this. Image overlay toggling that TB lacks is
free in a custom viewer. And the duckdb file is one bundle-friendly artifact.

## Phased rollout

Build write side first. Validate with a temporal training run. Build the
viewer only after we have real data to point it at, so the viewer is shaped
by actual queries rather than guesses.

### Phase 1 — write side (~100 LOC)

**New module: `src/opndet/metrics_db.py`**

```python
class MetricsDB:
    def __init__(self, run_dir: Path):
        self.run = str(run_dir.name)
        self.con = duckdb.connect(str(run_dir / "metrics.duckdb"))
        self._init_schema()

    def add_scalar(self, ep: int, tag: str, value: float) -> None: ...
    def add_image(self, ep: int, tag: str, sample_idx: int, base_path: str) -> None: ...
    def add_overlay(self, ep: int, tag: str, sample_idx: int, kind: str, path: str) -> None: ...
    def add_boxes(self, ep: int, tag: str, sample_idx: int, kind: str,
                  boxes: np.ndarray, scores: np.ndarray | None = None,
                  meta: dict | None = None) -> None: ...
    def close(self) -> None: ...
```

**Schema:**

```sql
CREATE TABLE scalars (
    run TEXT, ep INT, tag TEXT, value DOUBLE, ts TIMESTAMP DEFAULT now()
);
CREATE INDEX scalars_idx ON scalars(run, tag, ep);

CREATE TABLE images (
    run TEXT, ep INT, tag TEXT, sample_idx INT, base_path TEXT
);
CREATE INDEX images_idx ON images(run, ep, tag);

CREATE TABLE overlays (
    run TEXT, ep INT, tag TEXT, sample_idx INT,
    kind TEXT,         -- 'prior_heat' | 'attention' | 'gt_mask' | ...
    path TEXT
);

CREATE TABLE boxes (
    run TEXT, ep INT, tag TEXT, sample_idx INT,
    kind TEXT,         -- 'pred' | 'gt' | 'tp' | 'fp' | 'fn'
    x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE,
    score DOUBLE,
    meta JSON
);
CREATE INDEX boxes_idx ON boxes(run, ep, tag, sample_idx, kind);

CREATE TABLE config (run TEXT, key TEXT, value TEXT);
```

**Wiring in `train.py`:**

- Init `db = MetricsDB(out_dir)` next to existing TB writer init
- Dual-write: every `writer.add_scalar(tag, value, ep)` also calls
  `db.add_scalar(ep, tag, value)`
- Replace `render_predictions` (which renders boxes onto a single PNG) with
  `save_predictions(model, vis_batch, ..., db, out_dir, ep)` that writes:
  - `runs/<exp>/vis/ep_<ep>/sample_<i>_rgb.png` — clean RGB
  - `runs/<exp>/vis/ep_<ep>/sample_<i>_prior.png` — JET-colormapped prior
    with per-pixel alpha = prior_value (only if input was 4-ch)
  - DB rows: one `images` row + one `overlays` row + one `boxes` row per
    pred/gt/tp/fp/fn
- Bundle picks up the duckdb file automatically (it's just one file in the
  run dir alongside last.pt)
- Behind config flags:
  - `metrics_db: true` (default true)
  - `keep_tb: true` (default true for one cycle, then default false)
- Performance: every write is `con.execute("INSERT INTO ...")`. For
  high-cardinality writes (boxes), batch with `executemany` per epoch.
  DuckDB's append-only mode is fast enough that we won't notice.

### Phase 2 — viewer (defer until after first temporal training run)

**Single file: `src/opndet/dashboard.py`**

Stack: Flask + chart.js + canvas. Target ~250 LOC. Single HTML template.

CLI: `opndet dashboard --run runs/exp1` opens `localhost:5000`. The dashboard
reads only from `runs/<exp>/metrics.duckdb` and image paths it references —
never modifies anything.

**Panels:**

- **Scalars**: line charts per tag, filterable by `LIKE 'val/%'` etc.
  Run comparison via SQL JOIN across multiple `--run` args. EMA smoothing is
  a SQL window function, not custom code.
- **Image viewer**: RGB base + overlay layers (opacity sliders), boxes drawn
  client-side from JSON via canvas (toggle per-kind, score-threshold slider,
  hover tooltip). Per-image controls:
  - **Layers**: RGB | prior_heat | attention | ... — checkbox + opacity slider per
  - **Boxes**: pred ☑ | gt ☑ | tp ☐ | fp ☐ | fn ☐ — per-kind checkbox
  - **Score threshold**: slider 0.0–1.0, re-filters pred boxes client-side
  - **Hover**: show score + meta JSON
- **Filter panel**: free-form SQL OR canned queries:
  - "show frames where FN > 2 AND val/f1_opt_cal dropped this epoch"
  - "show frames whose pred count differs from GT count by ≥ 2"
  - "show frames where any pred has score in [0.30, 0.40] (borderline)"

**HTML layout** (rough):
```
+-----------------------+ +-------------------------+
|  Scalars (chart.js)   | |  Image grid             |
|  smoothing slider     | |  layers/boxes controls  |
+-----------------------+ +-------------------------+
                          |  filter SQL textbox     |
                          +-------------------------+
```

### Phase 3 (maybe never) — drop TB writer

After validating the duckdb path works for a full training cycle and the
dashboard covers the things we actually use TB for, remove the dual-write.

## Storage shape

Per-run files:
```
runs/exp1/
  metrics.duckdb            # ~few MB after a 300-epoch run
  best.pt
  last.pt
  config.yaml
  tb/                       # legacy, deprecated after Phase 3
  vis/
    ep_001/
      sample_0_rgb.png
      sample_0_prior.png
      sample_1_rgb.png
      ...
    ep_010/
      ...
```

Bundle = zip the whole run dir. metrics.duckdb travels with the artifacts.

## Why this beats TB for opndet specifically

- **Layer toggling**: TB has none. Dashboard has it as a checkbox. Game over.
- **Score-threshold slider**: TB renders boxes once at fixed threshold; can't
  re-filter. Dashboard re-filters client-side from JSON.
- **FN/FP/TP filter**: TB doesn't know about box classification. Dashboard
  has them as separate `boxes` rows with `kind`.
- **Per-frame query**: "show frames where the model regressed" = SQL.
  Impossible in TB.
- **Notebook ad-hoc**: `pd.read_sql(...)` on the duckdb file. Free.
- **Bundle-friendly**: one file. tfevents shards are append-only protos
  nobody can query directly.

## Risks / known gotchas

- DuckDB write under multiprocessing. The training process is single-threaded
  for metrics writes (workers don't touch DB). Should be fine.
- DuckDB file size growth on long runs. Boxes table can get fat (vis_samples
  × n_dets × epochs). Mitigate: only write box rows for vis samples, not
  every val/test sample. Vis is typically 4–64 samples per epoch.
- Schema migration. Bake schema version into a `meta` table. Add migration
  helpers when the schema changes.
- Concurrent dashboard read while training writes. DuckDB supports concurrent
  read. Test before promising.

## Out of scope for v1

- Embeddings/projector (TB's tSNE thing). Nobody uses it for opndet.
- Histogram panel. Could add later via DuckDB `histogram()` function over
  scalars. Not needed for v1.
- Real-time streaming (auto-refresh during training). Manual refresh is fine.

## Sequencing with temporal model work

The temporal-mode (bbox-f-tp) variant needs the dashboard's image overlay
toggle most. That's the primary motivation. Suggested order:

1. Land Phase 1 (write side only) — small, low-risk, dual-writes alongside TB
2. Run the next temporal-mode training, collect data into duckdb
3. Build Phase 2 viewer, validating against the recorded data
4. After viewer covers everything we actually use, drop TB
