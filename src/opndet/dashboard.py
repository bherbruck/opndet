"""DuckDB-backed run dashboard.

Single-process FastAPI app reading metrics.duckdb in a run dir. Serves a
small SPA (chart.js + canvas) with:
- Scalar charts grouped by tag prefix
- Image viewer with togglable overlay layers (RGB base + JET prior heat +
  per-kind boxes drawn client-side from JSON)
- Score-threshold slider that re-filters pred boxes without re-rendering
- Free-form SQL query pane for ad-hoc lookups

Designed to run alongside training: another process writes to the duckdb
file, this process reads it. DuckDB supports concurrent reads.

Launch:
    opndet dashboard --run runs/exp1
or (from inside train.py with `dashboard: true`):
    auto-spawned subprocess that gets cleaned up at training exit
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def _open_db(run_dir: Path):
    import duckdb
    db_path = run_dir / "metrics.duckdb"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"metrics.duckdb not found in {run_dir}")
    return duckdb.connect(str(db_path), read_only=True)


def build_app(run_dir: Path) -> FastAPI:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    app = FastAPI(title=f"opndet dashboard · {run_dir.name}", version="0.1.0")

    # Static file serving for vis PNGs that boxes refer to
    app.mount("/files", StaticFiles(directory=str(run_dir)), name="files")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML.replace("__RUN_NAME__", run_dir.name)

    @app.get("/api/tags")
    def list_tags() -> dict[str, list[str]]:
        with _open_db(run_dir) as con:
            scalar_tags = [r[0] for r in con.execute("SELECT DISTINCT tag FROM scalars ORDER BY tag").fetchall()]
            image_tags = [r[0] for r in con.execute("SELECT DISTINCT tag FROM images ORDER BY tag").fetchall()]
        return {"scalars": scalar_tags, "images": image_tags}

    @app.get("/api/scalars")
    def get_scalars(tag: str = Query(...)) -> list[dict[str, Any]]:
        with _open_db(run_dir) as con:
            rows = con.execute(
                "SELECT ep, value FROM scalars WHERE tag = ? ORDER BY ep", [tag]
            ).fetchall()
        return [{"ep": r[0], "value": r[1]} for r in rows]

    @app.get("/api/scalars/multi")
    def get_scalars_multi(tags: str = Query(..., description="comma-separated tag list")) -> dict[str, list[dict[str, Any]]]:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        out: dict[str, list[dict[str, Any]]] = {}
        with _open_db(run_dir) as con:
            for t in tag_list:
                rows = con.execute(
                    "SELECT ep, value FROM scalars WHERE tag = ? ORDER BY ep", [t]
                ).fetchall()
                out[t] = [{"ep": r[0], "value": r[1]} for r in rows]
        return out

    @app.get("/api/epochs")
    def get_epochs(tag: str = Query(...)) -> list[int]:
        with _open_db(run_dir) as con:
            rows = con.execute(
                "SELECT DISTINCT ep FROM images WHERE tag = ? ORDER BY ep", [tag]
            ).fetchall()
        return [r[0] for r in rows]

    @app.get("/api/samples")
    def get_samples(tag: str = Query(...), ep: int = Query(...)) -> list[dict[str, Any]]:
        with _open_db(run_dir) as con:
            imgs = con.execute(
                "SELECT sample_idx, base_path FROM images WHERE tag = ? AND ep = ? ORDER BY sample_idx",
                [tag, ep],
            ).fetchall()
            overlays = con.execute(
                "SELECT sample_idx, kind, path FROM overlays WHERE tag = ? AND ep = ? ORDER BY sample_idx, kind",
                [tag, ep],
            ).fetchall()
            boxes = con.execute(
                "SELECT sample_idx, kind, x1, y1, x2, y2, score FROM boxes "
                "WHERE tag = ? AND ep = ? ORDER BY sample_idx, kind",
                [tag, ep],
            ).fetchall()
        ov_by_sample: dict[int, list[dict[str, str]]] = {}
        for s, kind, path in overlays:
            ov_by_sample.setdefault(s, []).append({"kind": kind, "url": f"/files/{path}"})
        bx_by_sample: dict[int, list[dict[str, Any]]] = {}
        for s, kind, x1, y1, x2, y2, score in boxes:
            bx_by_sample.setdefault(s, []).append({
                "kind": kind, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": score,
            })
        return [
            {
                "sample_idx": s,
                "rgb_url": f"/files/{path}",
                "overlays": ov_by_sample.get(s, []),
                "boxes": bx_by_sample.get(s, []),
            }
            for s, path in imgs
        ]

    @app.get("/api/config")
    def get_config() -> dict[str, str]:
        with _open_db(run_dir) as con:
            rows = con.execute("SELECT key, value FROM config ORDER BY key").fetchall()
        return {k: v for k, v in rows}

    @app.post("/api/sql")
    def run_sql(payload: dict) -> JSONResponse:
        q = payload.get("query", "").strip()
        if not q:
            raise HTTPException(400, "empty query")
        if any(kw in q.upper() for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH")):
            raise HTTPException(400, "read-only queries only")
        with _open_db(run_dir) as con:
            try:
                rows = con.execute(q).fetchall()
                cols = [d[0] for d in con.description] if con.description else []
            except Exception as e:
                raise HTTPException(400, f"SQL error: {e}")
        return JSONResponse({
            "columns": cols,
            "rows": [list(r) for r in rows[:1000]],
            "truncated": len(rows) > 1000,
        })

    return app


def serve(run_dir: str | Path, host: str = "127.0.0.1", port: int = 5000) -> None:
    import uvicorn
    app = build_app(Path(run_dir))
    print(f"opndet dashboard: http://{host}:{port}  (run: {Path(run_dir).resolve()})")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


def is_colab() -> bool:
    """Detect a Google Colab environment."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def embed_colab_iframe(port: int, height: int = 800) -> None:
    """Embed the dashboard as an iframe in the current Colab cell. Mirrors
    the same pattern that %tensorboard uses internally. Silent no-op outside
    Colab so call sites don't need a guard.
    """
    if not is_colab():
        return
    try:
        from google.colab import output
        output.serve_kernel_port_as_iframe(port, height=str(height))
    except Exception as e:  # best-effort
        print(f"opndet dashboard: iframe embed failed ({e}); open http://127.0.0.1:{port} manually")


def spawn_background(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 5000,
    embed_iframe: bool = True,
    wait_for_ready: float = 1.5,
):
    """Spawn the dashboard as a child process and (when in Colab and
    embed_iframe=True) auto-embed it as an iframe in the calling notebook
    cell. Returns the subprocess.Popen so callers can terminate() at exit.
    """
    import subprocess
    import sys
    import time
    cmd = [
        sys.executable, "-m", "opndet.cli", "dashboard",
        "--run", str(run_dir), "--host", host, "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if wait_for_ready > 0:
        time.sleep(wait_for_ready)
    if embed_iframe:
        embed_colab_iframe(port)
    return proc


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>opndet · __RUN_NAME__</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; background: #0e1116; color: #d6dee6; }
    header { background: #161b22; border-bottom: 1px solid #30363d; padding: 10px 16px; display: flex; align-items: center; gap: 16px; }
    header .title { font-weight: 600; color: #f0f6fc; }
    header .run { color: #7d8590; font-size: 13px; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 12px; padding: 12px; height: calc(100vh - 51px); }
    .pane { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; overflow: auto; }
    .pane h3 { margin: 0 0 8px; font-size: 13px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; }
    .tag-list label { display: block; padding: 4px 6px; cursor: pointer; border-radius: 3px; font-size: 13px; }
    .tag-list label:hover { background: #1f242c; }
    .tag-list input { margin-right: 6px; }
    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 10px; }
    .chart-card { background: #0e1116; border: 1px solid #30363d; border-radius: 4px; padding: 8px; }
    .chart-card .title { font-size: 12px; color: #c9d1d9; margin-bottom: 4px; }
    .chart-card canvas { width: 100% !important; height: 180px !important; }
    .image-controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; align-items: center; font-size: 12px; }
    .image-controls select, .image-controls input[type=range] { background: #0e1116; color: #d6dee6; border: 1px solid #30363d; border-radius: 3px; padding: 3px 6px; }
    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }
    .img-card { position: relative; background: #0e1116; border: 1px solid #30363d; border-radius: 4px; overflow: hidden; }
    .img-card .stage { position: relative; }
    .img-card .stage img { display: block; width: 100%; }
    .img-card canvas.boxes { position: absolute; inset: 0; pointer-events: auto; }
    .layer-toggles { padding: 6px 8px; font-size: 11px; color: #7d8590; display: flex; gap: 8px; flex-wrap: wrap; }
    .layer-toggles label { cursor: pointer; }
    button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 3px; padding: 4px 10px; cursor: pointer; font-family: inherit; font-size: 12px; }
    button:hover { background: #2d333b; }
    .sql-pane textarea { width: 100%; background: #0e1116; color: #d6dee6; border: 1px solid #30363d; border-radius: 3px; padding: 6px; font-family: inherit; font-size: 12px; min-height: 60px; }
    .sql-pane table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .sql-pane th, .sql-pane td { border: 1px solid #30363d; padding: 3px 6px; text-align: left; }
    .sql-pane th { background: #1f242c; }
    details { margin-top: 12px; }
    details summary { cursor: pointer; color: #7d8590; font-size: 12px; padding: 4px 0; }
  </style>
</head>
<body>
<header>
  <div class="title">opndet</div>
  <div class="run">run: __RUN_NAME__</div>
  <div style="flex:1"></div>
  <button onclick="refreshAll()">refresh</button>
</header>

<div class="grid">
  <div class="pane">
    <h3>scalars</h3>
    <div id="scalar-tags" class="tag-list"></div>
    <h3 style="margin-top:14px">image tags</h3>
    <div id="image-tags" class="tag-list"></div>

    <details>
      <summary>SQL</summary>
      <div class="sql-pane">
        <textarea id="sql-input" placeholder="SELECT ep, value FROM scalars WHERE tag LIKE 'val/%' ORDER BY ep DESC LIMIT 20"></textarea>
        <button onclick="runSQL()" style="margin-top:6px">run</button>
        <div id="sql-result" style="margin-top:8px"></div>
      </div>
    </details>
  </div>

  <div class="pane">
    <h3>charts</h3>
    <div id="charts" class="charts"></div>

    <h3 style="margin-top:18px">images</h3>
    <div class="image-controls">
      <label>tag <select id="img-tag"></select></label>
      <label>epoch <select id="img-ep"></select></label>
      <label>score ≥ <input id="score-thresh" type="range" min="0" max="1" step="0.01" value="0.2"> <span id="score-val">0.20</span></label>
      <label><input type="checkbox" id="show-pred" checked> pred</label>
      <label><input type="checkbox" id="show-gt" checked> gt</label>
      <label><input type="checkbox" id="show-tp"> tp</label>
      <label><input type="checkbox" id="show-fp"> fp</label>
      <label><input type="checkbox" id="show-fn"> fn</label>
      <label><input type="checkbox" id="show-prior" checked> prior heat</label>
      <label>α <input id="overlay-alpha" type="range" min="0" max="1" step="0.05" value="0.5"></label>
    </div>
    <div id="image-grid" class="image-grid"></div>
  </div>
</div>

<script>
const charts = {};
let scalarTags = [], imageTags = [];

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function refreshAll() {
  const tags = await api('/api/tags');
  scalarTags = tags.scalars;
  imageTags = tags.images;
  renderScalarTags();
  renderImageTags();
  // pre-select common interesting scalars
  const commonChecks = ['val/f1', 'val/f1_opt', 'val_cold/f1_opt', 'prior_lift/val/f1_opt', 'val_cal/f1', 'train/loss'];
  for (const t of commonChecks) {
    const el = document.querySelector(`input[data-scalar="${CSS.escape(t)}"]`);
    if (el) { el.checked = true; el.dispatchEvent(new Event('change')); }
  }
  // pick first image tag
  if (imageTags.length) {
    document.getElementById('img-tag').value = imageTags[0];
    await loadImageEpochs();
  }
}

function renderScalarTags() {
  const root = document.getElementById('scalar-tags');
  root.innerHTML = '';
  for (const tag of scalarTags) {
    const id = 'sc_' + tag.replace(/[^a-z0-9]/gi, '_');
    const lbl = document.createElement('label');
    lbl.innerHTML = `<input type="checkbox" data-scalar="${tag}" id="${id}"> ${tag}`;
    root.appendChild(lbl);
  }
  root.addEventListener('change', e => {
    if (e.target.matches('input[data-scalar]')) {
      const tag = e.target.dataset.scalar;
      if (e.target.checked) addChart(tag); else removeChart(tag);
    }
  });
}

function renderImageTags() {
  const root = document.getElementById('image-tags');
  const sel = document.getElementById('img-tag');
  root.innerHTML = '';
  sel.innerHTML = '';
  for (const tag of imageTags) {
    const lbl = document.createElement('label');
    lbl.textContent = tag;
    root.appendChild(lbl);
    const opt = document.createElement('option');
    opt.value = opt.textContent = tag;
    sel.appendChild(opt);
  }
}

async function addChart(tag) {
  if (charts[tag]) return;
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.id = 'card_' + tag.replace(/[^a-z0-9]/gi, '_');
  card.innerHTML = `<div class="title">${tag}</div><canvas></canvas>`;
  document.getElementById('charts').appendChild(card);
  const data = await api('/api/scalars?tag=' + encodeURIComponent(tag));
  const ctx = card.querySelector('canvas').getContext('2d');
  charts[tag] = new Chart(ctx, {
    type: 'line',
    data: { labels: data.map(d => d.ep), datasets: [{ label: tag, data: data.map(d => d.value), borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', tension: 0.2, pointRadius: 1 }] },
    options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#7d8590' }, grid: { color: '#21262d' } }, y: { ticks: { color: '#7d8590' }, grid: { color: '#21262d' } } } }
  });
}

function removeChart(tag) {
  if (!charts[tag]) return;
  charts[tag].destroy();
  delete charts[tag];
  const card = document.getElementById('card_' + tag.replace(/[^a-z0-9]/gi, '_'));
  card?.remove();
}

async function loadImageEpochs() {
  const tag = document.getElementById('img-tag').value;
  if (!tag) return;
  const eps = await api('/api/epochs?tag=' + encodeURIComponent(tag));
  const sel = document.getElementById('img-ep');
  sel.innerHTML = '';
  for (const ep of eps) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = ep;
    sel.appendChild(opt);
  }
  if (eps.length) { sel.value = eps[eps.length - 1]; await loadImages(); }
}

async function loadImages() {
  const tag = document.getElementById('img-tag').value;
  const ep = document.getElementById('img-ep').value;
  if (!tag || !ep) return;
  const samples = await api(`/api/samples?tag=${encodeURIComponent(tag)}&ep=${ep}`);
  const grid = document.getElementById('image-grid');
  grid.innerHTML = '';
  for (const s of samples) renderSample(grid, s);
}

function renderSample(grid, s) {
  const card = document.createElement('div');
  card.className = 'img-card';
  const stage = document.createElement('div');
  stage.className = 'stage';
  card.appendChild(stage);

  const baseImg = document.createElement('img');
  baseImg.src = s.rgb_url;
  stage.appendChild(baseImg);

  for (const ov of s.overlays) {
    const img = document.createElement('img');
    img.src = ov.url;
    img.className = 'overlay';
    img.dataset.kind = ov.kind;
    img.style.position = 'absolute';
    img.style.inset = '0';
    img.style.opacity = (document.getElementById('show-prior').checked ? document.getElementById('overlay-alpha').value : 0);
    img.style.mixBlendMode = 'normal';
    stage.appendChild(img);
  }

  const cv = document.createElement('canvas');
  cv.className = 'boxes';
  stage.appendChild(cv);

  const layerInfo = document.createElement('div');
  layerInfo.className = 'layer-toggles';
  layerInfo.textContent = `boxes: ${s.boxes.length}  overlays: ${s.overlays.map(o => o.kind).join(', ') || 'none'}`;
  card.appendChild(layerInfo);

  grid.appendChild(card);

  baseImg.onload = () => {
    cv.width = baseImg.naturalWidth;
    cv.height = baseImg.naturalHeight;
    drawBoxes(cv, s.boxes);
  };
  card._sample = s;
}

function drawBoxes(canvas, boxes) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const thresh = parseFloat(document.getElementById('score-thresh').value);
  const showByKind = {
    pred: document.getElementById('show-pred').checked,
    gt:   document.getElementById('show-gt').checked,
    tp:   document.getElementById('show-tp').checked,
    fp:   document.getElementById('show-fp').checked,
    fn:   document.getElementById('show-fn').checked,
  };
  const colorByKind = {
    pred: '#39c860', gt: '#ff5edb', tp: '#39c860', fp: '#ff6b35', fn: '#3aa6ff',
  };
  for (const b of boxes) {
    if (!showByKind[b.kind]) continue;
    if (b.kind === 'pred' && b.score != null && b.score < thresh) continue;
    ctx.strokeStyle = colorByKind[b.kind] || '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    if (b.score != null) {
      ctx.fillStyle = colorByKind[b.kind];
      ctx.font = '12px monospace';
      ctx.fillText(b.score.toFixed(2), b.x1 + 2, b.y1 + 12);
    }
  }
}

function rerenderBoxes() {
  document.querySelectorAll('.img-card').forEach(card => {
    const cv = card.querySelector('canvas.boxes');
    drawBoxes(cv, card._sample.boxes);
  });
}

function rerenderOverlays() {
  const showPrior = document.getElementById('show-prior').checked;
  const a = document.getElementById('overlay-alpha').value;
  document.querySelectorAll('.img-card .overlay').forEach(img => {
    img.style.opacity = showPrior ? a : 0;
  });
}

async function runSQL() {
  const q = document.getElementById('sql-input').value;
  try {
    const r = await api('/api/sql', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) });
    const root = document.getElementById('sql-result');
    let html = `<div style="color:#7d8590;font-size:11px;margin-bottom:4px">${r.rows.length} rows${r.truncated ? ' (truncated to 1000)' : ''}</div>`;
    html += '<table><thead><tr>' + r.columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    for (const row of r.rows) html += '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
    html += '</tbody></table>';
    root.innerHTML = html;
  } catch (e) {
    document.getElementById('sql-result').textContent = 'error: ' + e.message;
  }
}

document.getElementById('img-tag').addEventListener('change', loadImageEpochs);
document.getElementById('img-ep').addEventListener('change', loadImages);
document.getElementById('score-thresh').addEventListener('input', e => {
  document.getElementById('score-val').textContent = parseFloat(e.target.value).toFixed(2);
  rerenderBoxes();
});
['show-pred','show-gt','show-tp','show-fp','show-fn'].forEach(id => {
  document.getElementById(id).addEventListener('change', rerenderBoxes);
});
document.getElementById('show-prior').addEventListener('change', rerenderOverlays);
document.getElementById('overlay-alpha').addEventListener('input', rerenderOverlays);

refreshAll();
setInterval(refreshAll, 30000);  // auto-refresh every 30s while training
</script>
</body>
</html>
"""
