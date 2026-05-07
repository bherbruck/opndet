"""DuckDB-backed run dashboard with auto-discovery.

Point at a single run dir OR a runs parent dir — server scans for
**/metrics.duckdb files and surfaces them as selectable runs in the UI.
Mirrors TB's --logdir behavior. New runs that appear during training are
picked up by the next /api/runs poll (UI auto-refreshes every 30s).

Endpoints (all read-only, all accept optional ?run=<name>):
    /api/runs                   — discovered runs with mtime + path
    /api/tags?run=NAME          — distinct scalar + image tags for that run
    /api/scalars?tag=T&run=N    — (ep, value) series
    /api/scalars/multi?tags=T1,T2&run=N
    /api/epochs?tag=T&run=N
    /api/samples?tag=T&ep=E&run=N
    /api/config?run=N
    /api/sql                    — raw SQL across attached runs (advanced)

Static files: each run's vis assets served under /files/<run_name>/...
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import traceback


def _discover_runs(root: Path) -> dict[str, Path]:
    """Return {run_name: run_dir} for every dir under root that contains
    metrics.duckdb. If root itself contains metrics.duckdb, return just that.
    Sorted by mtime descending (newest first). Returns {} (not raise) if
    root doesn't exist yet or has no runs — the dashboard polls every 30s
    so newly-created runs get picked up automatically."""
    if not root.exists():
        return {}
    try:
        if (root / "metrics.duckdb").exists():
            return {root.name: root}
        out: list[tuple[Path, float]] = []
        for p in root.iterdir():
            if p.is_dir() and (p / "metrics.duckdb").exists():
                try:
                    out.append((p, (p / "metrics.duckdb").stat().st_mtime))
                except OSError:
                    continue
        out.sort(key=lambda x: -x[1])
        return {p.name: p for p, _ in out}
    except (OSError, PermissionError):
        return {}


def _open_db(run_dir: Path):
    """Open the run's metrics.duckdb for read.

    DuckDB takes a process-level file lock even in read_only=True mode, and
    the writer (training process) holds it for the entire run. So we read
    from a shadow copy in /tmp keyed by mtime — refreshed lazily when the
    source file changes. Tradeoff: dashboard sees a snapshot from up to
    one request ago, never blocks on the writer.
    """
    import duckdb
    import shutil
    src = run_dir / "metrics.duckdb"
    if not src.exists():
        raise HTTPException(404, f"metrics.duckdb missing in {run_dir}")
    shadow_root = Path("/tmp") / "opndet_dash_shadow"
    shadow_root.mkdir(parents=True, exist_ok=True)
    # one shadow file per run dir; encode the resolved path so different
    # runs (or the same name in different roots) don't collide.
    import hashlib
    key = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:16]
    shadow = shadow_root / f"{run_dir.name}_{key}.duckdb"
    # Always copy: DuckDB uses a WAL whose updates don't always bump the
    # main file's mtime, so an mtime-based cache misses recent commits.
    # Files are small (KB–few MB); copy is sub-ms.
    try:
        shutil.copy2(src, shadow)
        wal = src.with_suffix(".duckdb.wal")
        wal_shadow = shadow.with_suffix(".duckdb.wal")
        if wal.exists():
            shutil.copy2(wal, wal_shadow)
        elif wal_shadow.exists():
            wal_shadow.unlink()  # stale WAL after writer checkpointed
    except Exception as e:
        if not shadow.exists():
            raise HTTPException(503, f"metrics.duckdb temporarily unavailable: {e}")
    return duckdb.connect(str(shadow), read_only=True)


def build_app(root_dir: Path) -> FastAPI:
    root_dir = Path(root_dir).resolve()
    # Tolerate missing root — dashboard launches before training has had a
    # chance to create anything. Empty discovery returns [] from /api/runs;
    # the frontend polls every 30s and picks up new runs automatically.
    app = FastAPI(title=f"opndet · {root_dir.name}", version="0.2.0")

    @app.exception_handler(Exception)
    async def _all_errors(request: Request, exc: Exception):
        """Return the full traceback in the response. The dashboard isn't
        public-facing — verbose errors > silent 500s."""
        tb = traceback.format_exc()
        # Also print to server stderr so the cell shows it
        print(f"\n[dashboard 500] {request.method} {request.url.path}\n{tb}", flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__, "traceback": tb,
                     "path": str(request.url.path)},
        )

    @app.get("/api/health")
    def api_health() -> dict:
        return {
            "ok": True,
            "root": str(root_dir),
            "exists": root_dir.exists(),
            "n_runs": len(_discover_runs(root_dir)),
        }

    def _resolve_run(name: str | None) -> Path | None:
        """Returns None if no runs exist yet — endpoints handle that as
        empty data, not as an error."""
        runs = _discover_runs(root_dir)
        if not runs:
            return None
        if name is None:
            return next(iter(runs.values()))
        if name not in runs:
            return None
        return runs[name]

    @app.get("/files/{run_name}/{path:path}")
    def serve_file(run_name: str, path: str):
        runs = _discover_runs(root_dir)
        if run_name not in runs:
            raise HTTPException(404)
        full = (runs[run_name] / path).resolve()
        if not str(full).startswith(str(runs[run_name].resolve())):
            raise HTTPException(403)
        if not full.exists():
            raise HTTPException(404)
        return FileResponse(full)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML.replace("__ROOT_NAME__", root_dir.name)

    @app.get("/api/runs")
    def api_runs() -> list[dict[str, Any]]:
        runs = _discover_runs(root_dir)
        return [
            {
                "name": name,
                "path": str(p),
                "mtime": (p / "metrics.duckdb").stat().st_mtime,
            }
            for name, p in runs.items()
        ]

    @app.get("/api/tags")
    def api_tags(run: str | None = Query(None)) -> dict[str, list[str]]:
        run_dir = _resolve_run(run)
        if run_dir is None:
            return {"scalars": [], "images": []}
        with _open_db(run_dir) as con:
            scalar_tags = [r[0] for r in con.execute("SELECT DISTINCT tag FROM scalars ORDER BY tag").fetchall()]
            image_tags = [r[0] for r in con.execute("SELECT DISTINCT tag FROM images ORDER BY tag").fetchall()]
        return {"scalars": scalar_tags, "images": image_tags}

    @app.get("/api/scalars")
    def api_scalars(tag: str = Query(...), run: str | None = Query(None)) -> list[dict[str, Any]]:
        run_dir = _resolve_run(run)
        if run_dir is None:
            return []
        with _open_db(run_dir) as con:
            rows = con.execute("SELECT ep, value FROM scalars WHERE tag = ? ORDER BY ep", [tag]).fetchall()
        return [{"ep": r[0], "value": r[1]} for r in rows]

    @app.get("/api/scalars/multi")
    def api_scalars_multi(
        tags: str = Query(..., description="comma-separated tag list"),
        run: str | None = Query(None),
    ) -> dict[str, list[dict[str, Any]]]:
        run_dir = _resolve_run(run)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        out: dict[str, list[dict[str, Any]]] = {t: [] for t in tag_list}
        if run_dir is None:
            return out
        with _open_db(run_dir) as con:
            for t in tag_list:
                rows = con.execute("SELECT ep, value FROM scalars WHERE tag = ? ORDER BY ep", [t]).fetchall()
                out[t] = [{"ep": r[0], "value": r[1]} for r in rows]
        return out

    @app.get("/api/epochs")
    def api_epochs(tag: str = Query(...), run: str | None = Query(None)) -> list[int]:
        run_dir = _resolve_run(run)
        if run_dir is None:
            return []
        with _open_db(run_dir) as con:
            rows = con.execute("SELECT DISTINCT ep FROM images WHERE tag = ? ORDER BY ep", [tag]).fetchall()
        return [r[0] for r in rows]

    @app.get("/api/samples")
    def api_samples(
        tag: str = Query(...), ep: int = Query(...), run: str | None = Query(None),
    ) -> list[dict[str, Any]]:
        run_dir = _resolve_run(run)
        if run_dir is None:
            return []
        run_name = run_dir.name
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
            ov_by_sample.setdefault(s, []).append({"kind": kind, "url": f"/files/{run_name}/{path}"})
        bx_by_sample: dict[int, list[dict[str, Any]]] = {}
        for s, kind, x1, y1, x2, y2, score in boxes:
            bx_by_sample.setdefault(s, []).append({
                "kind": kind, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score,
            })
        return [
            {
                "sample_idx": s,
                "rgb_url": f"/files/{run_name}/{path}",
                "overlays": ov_by_sample.get(s, []),
                "boxes": bx_by_sample.get(s, []),
            }
            for s, path in imgs
        ]

    @app.get("/api/config")
    def api_config(run: str | None = Query(None)) -> dict[str, str]:
        run_dir = _resolve_run(run)
        if run_dir is None:
            return {}
        with _open_db(run_dir) as con:
            rows = con.execute("SELECT key, value FROM config ORDER BY key").fetchall()
        return {k: v for k, v in rows}

    @app.post("/api/sql")
    def api_sql(payload: dict, run: str | None = Query(None)) -> JSONResponse:
        q = (payload.get("query") or "").strip()
        if not q:
            return JSONResponse({"columns": [], "rows": [], "truncated": False, "error": "empty query"})
        if any(kw in q.upper() for kw in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE")):
            return JSONResponse({"columns": [], "rows": [], "truncated": False, "error": "read-only queries only"})
        run_dir = _resolve_run(run)
        if run_dir is None:
            return JSONResponse({"columns": [], "rows": [], "truncated": False, "error": "no runs available yet"})
        with _open_db(run_dir) as con:
            try:
                # Allow ATTACH for cross-run queries — but only paths under root_dir.
                if "ATTACH" in q.upper():
                    import re
                    paths = re.findall(r"ATTACH\s+'([^']+)'", q, flags=re.IGNORECASE)
                    for p in paths:
                        rp = Path(p).resolve()
                        if not str(rp).startswith(str(root_dir)):
                            return JSONResponse({"columns": [], "rows": [], "truncated": False,
                                                 "error": f"ATTACH outside root not allowed: {p}"})
                rows = con.execute(q).fetchall()
                cols = [d[0] for d in con.description] if con.description else []
            except Exception as e:
                return JSONResponse({"columns": [], "rows": [], "truncated": False, "error": f"SQL error: {e}"})

        def _coerce(v):
            # JSON can't carry datetime/Decimal/bytes/etc — stringify those.
            if v is None or isinstance(v, (str, int, float, bool, list, dict)):
                return v
            try:
                import datetime as _dt
                if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
                    return v.isoformat()
            except Exception:
                pass
            return str(v)

        return JSONResponse({
            "columns": cols,
            "rows": [[_coerce(v) for v in r] for r in rows[:1000]],
            "truncated": len(rows) > 1000,
        })

    @app.get("/api/scalars/runs")
    def api_scalars_runs(
        tag: str = Query(...),
        runs: str = Query(..., description="comma-separated run names"),
    ) -> dict[str, list[dict[str, Any]]]:
        """Return per-run scalar series for a single tag — one entry per
        requested run, empty list when that run has no data for the tag.
        Used by the UI to plot N lines on the same chart for cross-run
        comparison."""
        all_runs = _discover_runs(root_dir)
        names = [r.strip() for r in runs.split(",") if r.strip()]
        out: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
        for n in names:
            if n not in all_runs:
                continue
            try:
                with _open_db(all_runs[n]) as con:
                    rows = con.execute(
                        "SELECT ep, value FROM scalars WHERE tag = ? ORDER BY ep", [tag]
                    ).fetchall()
                    out[n] = [{"ep": r[0], "value": r[1]} for r in rows]
            except Exception:
                continue
        return out

    return app


def serve(root_dir: str | Path, host: str = "127.0.0.1", port: int = 5000) -> None:
    import uvicorn
    app = build_app(Path(root_dir))
    print(f"opndet dashboard: http://{host}:{port}  (root: {Path(root_dir).resolve()})")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


def spawn_background(
    root_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 5000,
    wait_for_ready: float = 1.5,
    quiet: bool = False,
):
    """Spawn the dashboard as a child process. Prints localhost:port to
    stdout so the URL is visible in any cell/log. Returns subprocess.Popen.

    quiet=False (default): subprocess stderr inherits the parent so errors
    are visible. Pass quiet=True to silence completely.
    """
    import subprocess
    import sys
    import time
    cmd = [
        sys.executable, "-m", "opndet.cli", "dashboard",
        "--root", str(root_dir), "--host", host, "--port", str(port),
    ]
    if quiet:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # stdout silenced (uvicorn startup banner is noisy), stderr inherits
        # so tracebacks reach the cell — most common debug path.
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    print(f"opndet dashboard: http://localhost:{port}", flush=True)
    if wait_for_ready > 0:
        time.sleep(wait_for_ready)
    return proc


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>opndet · __ROOT_NAME__</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; background: #0e1116; color: #d6dee6; }
    header { background: #161b22; border-bottom: 1px solid #30363d; padding: 10px 16px; display: flex; align-items: center; gap: 16px; }
    header .title { font-weight: 600; color: #f0f6fc; }
    header select { background: #0e1116; color: #d6dee6; border: 1px solid #30363d; border-radius: 3px; padding: 4px 8px; font-family: inherit; }
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
  <div class="title">opndet · __ROOT_NAME__</div>
  <div style="flex:1"></div>
  <span id="empty-banner" style="color:#ff6b35;font-size:12px;display:none">no runs yet — waiting…</span>
  <button onclick="refreshAll()">refresh</button>
</header>

<div class="grid">
  <div class="pane">
    <h3 style="display:flex;justify-content:space-between;align-items:center">
      <span>runs</span>
      <span id="runs-count" style="color:#39c860;font-size:11px">0 selected</span>
    </h3>
    <div id="run-list" class="tag-list" style="max-height:220px;overflow:auto;border:1px solid #21262d;border-radius:3px;padding:4px;margin-bottom:6px"></div>
    <div style="display:flex;gap:6px;margin-bottom:14px">
      <button onclick="toggleAllRuns(true)" style="flex:1">select all</button>
      <button onclick="toggleAllRuns(false)" style="flex:1">deselect all</button>
    </div>

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
    <h3 style="display:flex;justify-content:space-between;align-items:center">
      <span>charts</span>
      <span style="font-size:11px;color:#7d8590;text-transform:none;letter-spacing:0">
        smoothing <input id="smooth-slider" type="range" min="0" max="0.99" step="0.01" value="0" style="vertical-align:middle">
        <span id="smooth-val">0.00</span>
      </span>
    </h3>
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
      <label><input type="checkbox" id="show-trail" checked> trail</label>
      <label><input type="checkbox" id="show-prior" checked> prior heat</label>
      <label>α <input id="overlay-alpha" type="range" min="0" max="1" step="0.05" value="0.5"></label>
    </div>
    <div id="image-grid" class="image-grid"></div>
  </div>
</div>

<script>
const charts = {};
let selectedRuns = [], scalarTags = [], imageTags = [];
const RUN_COLORS = ['#58a6ff', '#39c860', '#ff6b35', '#ffb86c', '#bd93f9', '#ff79c6', '#8be9fd', '#f1fa8c'];

// Persist selection across reloads. URL hash is primary (works inside
// Colab's sandboxed iframe where localStorage is blocked, and makes the
// view shareable). localStorage is a fallback for fresh URL visits.
const LS_RUNS = 'opndet:selectedRuns';
const LS_KNOWN = 'opndet:knownRuns';
const LS_SCALARS = 'opndet:selectedScalars';

function lsGet(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

function hashGet() {
  const h = (location.hash || '').replace(/^#/, '');
  const params = new URLSearchParams(h);
  const r = (params.get('runs')    || '').split(',').filter(Boolean);
  const s = (params.get('scalars') || '').split(',').filter(Boolean);
  const k = (params.get('known')   || '').split(',').filter(Boolean);
  return {runs: r, scalars: s, known: k};
}
function hashSet(runs, scalars, known) {
  const params = new URLSearchParams();
  if (runs.length)    params.set('runs', runs.join(','));
  if (scalars.length) params.set('scalars', scalars.join(','));
  if (known.length)   params.set('known', known.join(','));
  const newHash = '#' + params.toString();
  if (location.hash !== newHash) {
    history.replaceState(null, '', location.pathname + location.search + newHash);
  }
}
function persistedGet(kind) {
  const h = hashGet();
  if (h[kind].length) return h[kind];
  if (kind === 'runs')    return lsGet(LS_RUNS, []);
  if (kind === 'scalars') return lsGet(LS_SCALARS, []);
  if (kind === 'known')   return lsGet(LS_KNOWN, []);
  return [];
}
function persistedSet(runs, scalars, known) {
  hashSet(runs, scalars, known);
  lsSet(LS_RUNS, runs);
  lsSet(LS_SCALARS, scalars);
  lsSet(LS_KNOWN, known);
}

const primaryRun = () => selectedRuns[0] || null;
const qrun = () => primaryRun() ? '&run=' + encodeURIComponent(primaryRun()) : '';

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    // soft-fail — show empty rather than crash the UI
    console.warn(`api ${path} -> ${r.status}`);
    return null;
  }
  return r.json();
}

async function refreshRuns() {
  const runs = await api('/api/runs') || [];
  const list = document.getElementById('run-list');
  list.innerHTML = '';

  // Load prior selection from URL hash + localStorage on first call.
  if (selectedRuns.length === 0 && list.dataset.bootstrapped !== '1') {
    selectedRuns = persistedGet('runs');
    list.dataset.bootstrapped = '1';
  }
  const known = new Set(persistedGet('known'));
  const currentNames = new Set(runs.map(r => r.name));

  // Auto-select runs that have appeared since our last visit. A run the
  // user explicitly unchecked WHILE it was visible stays in known, so it
  // won't auto-reselect.
  let autoAdded = 0;
  for (const name of currentNames) {
    if (!known.has(name) && !selectedRuns.includes(name)) {
      selectedRuns.push(name);
      autoAdded++;
    }
  }
  // Drop selections that no longer exist (run dir deleted)
  selectedRuns = selectedRuns.filter(n => currentNames.has(n));

  // First-ever load with no known runs: select the most recent.
  if (known.size === 0 && selectedRuns.length === 0 && runs.length > 0) {
    selectedRuns.push(runs[0].name);
  }

  persistedSet(selectedRuns, Object.keys(charts), [...currentNames]);

  const sel = new Set(selectedRuns);
  const fmt = new Intl.DateTimeFormat(undefined, {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
  });
  for (const r of runs) {
    const dt = fmt.format(new Date(r.mtime * 1000));
    const id = 'run_' + r.name.replace(/[^a-z0-9]/gi, '_');
    const lbl = document.createElement('label');
    lbl.title = `${r.path}\n${new Date(r.mtime * 1000).toString()}`;
    lbl.innerHTML = `<input type="checkbox" data-run="${r.name}" id="${id}"${sel.has(r.name) ? ' checked' : ''}> <span style="font-weight:500">${r.name}</span> <span style="color:#7d8590;font-size:11px">${dt}</span>`;
    list.appendChild(lbl);
  }
  if (runs.length === 0) {
    document.getElementById('empty-banner').style.display = 'inline';
    selectedRuns = [];
    document.getElementById('runs-count').textContent = '0 selected';
    return autoAdded;
  }
  document.getElementById('empty-banner').style.display = 'none';
  syncSelectedRuns();
  return autoAdded;
}

function syncSelectedRuns() {
  selectedRuns = [...document.querySelectorAll('#run-list input[data-run]:checked')].map(el => el.dataset.run);
  document.getElementById('runs-count').textContent = `${selectedRuns.length} selected`;
  persistedSet(selectedRuns, Object.keys(charts), persistedGet('known'));
}

function toggleAllRuns(on) {
  document.querySelectorAll('#run-list input[data-run]').forEach(el => el.checked = on);
  syncSelectedRuns();
  refreshAll();
}

async function refreshAll() {
  await refreshRuns();
  if (!primaryRun()) {
    // no runs yet — wipe lists, leave UI quiet
    document.getElementById('scalar-tags').innerHTML = '';
    document.getElementById('image-tags').innerHTML = '';
    document.getElementById('img-tag').innerHTML = '';
    document.getElementById('img-ep').innerHTML = '';
    document.getElementById('image-grid').innerHTML = '';
    return;
  }
  const tags = await api('/api/tags?run=' + encodeURIComponent(primaryRun())) || {scalars: [], images: []};
  scalarTags = tags.scalars; imageTags = tags.images;
  renderScalarTags();
  renderImageTags();
  // pre-check from persisted state (URL hash > localStorage) if present,
  // else common defaults. Call addChart directly because dispatching a
  // synthetic 'change' event doesn't bubble to the parent listener by
  // default.
  if (Object.keys(charts).length === 0) {
    let toCheck = persistedGet('scalars');
    if (!Array.isArray(toCheck) || toCheck.length === 0) {
      toCheck = ['val/f1', 'val/f1_opt', 'val_cold/f1_opt', 'prior_lift/val/f1_opt', 'val_cal/f1', 'train/loss'];
    }
    for (const t of toCheck) {
      const el = document.querySelector(`input[data-scalar="${CSS.escape(t)}"]`);
      if (el) {
        el.checked = true;
        addChart(t);
      }
    }
    persistedSet(selectedRuns, Object.keys(charts), persistedGet('known'));
  } else {
    // re-fetch existing charts with the new run selection
    for (const tag of Object.keys(charts)) {
      removeChart(tag);
      addChart(tag);
    }
  }
  if (imageTags.length && !document.getElementById('img-tag').value) {
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
  root.onchange = e => {
    if (e.target.matches('input[data-scalar]')) {
      const tag = e.target.dataset.scalar;
      if (e.target.checked) addChart(tag); else removeChart(tag);
      persistedSet(selectedRuns, Object.keys(charts), persistedGet('known'));
    }
  };
}

function renderImageTags() {
  const root = document.getElementById('image-tags');
  const sel = document.getElementById('img-tag');
  root.innerHTML = ''; sel.innerHTML = '';
  for (const tag of imageTags) {
    const lbl = document.createElement('label');
    lbl.textContent = tag;
    root.appendChild(lbl);
    const opt = document.createElement('option');
    opt.value = opt.textContent = tag;
    sel.appendChild(opt);
  }
}

// EMA smoothing — alpha closer to 1 = heavier smoothing. Returns a new
// array of {x, y} points with debiased EMA (TB-style).
function emaSmooth(points, alpha) {
  if (alpha <= 0 || points.length === 0) return points;
  let last = 0, debias = 0;
  return points.map((p, i) => {
    last = last * alpha + (1 - alpha) * p.y;
    debias = debias * alpha + (1 - alpha) * 1;
    return { x: p.x, y: last / Math.max(debias, 1e-9) };
  });
}

function currentSmoothing() {
  return parseFloat(document.getElementById('smooth-slider').value);
}

async function addChart(tag) {
  if (charts[tag]) return;
  if (selectedRuns.length === 0) return;
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.id = 'card_' + tag.replace(/[^a-z0-9]/gi, '_');
  card.innerHTML = `<div class="title">${tag}</div><canvas></canvas>`;
  document.getElementById('charts').appendChild(card);

  const perRun = await api(`/api/scalars/runs?tag=${encodeURIComponent(tag)}&runs=${selectedRuns.map(encodeURIComponent).join(',')}`) || {};
  const alpha = currentSmoothing();
  const datasets = [];
  selectedRuns.forEach((run, i) => {
    const series = (perRun[run] || []).map(d => ({ x: d.ep, y: d.value }));
    const color = RUN_COLORS[i % RUN_COLORS.length];
    if (alpha > 0) {
      // raw line, faint dashed (kept for reference)
      datasets.push({
        label: `${run} (raw)`,
        data: series,
        borderColor: color + '66',
        borderDash: [3, 3],
        borderWidth: 1,
        pointRadius: 0,
        tension: 0,
      });
    }
    datasets.push({
      label: run,
      data: alpha > 0 ? emaSmooth(series, alpha) : series,
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 2,
      tension: 0.2,
      pointRadius: 1,
    });
  });

  const ctx = card.querySelector('canvas').getContext('2d');
  charts[tag] = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      animation: false,
      parsing: false,
      // index mode = vertical crosshair, all runs' values at the hovered
      // epoch shown together. intersect:false so you don't have to land
      // on a point.
      interaction: { mode: 'index', intersect: false, axis: 'x' },
      plugins: {
        legend: {
          display: selectedRuns.length > 1,
          labels: {
            color: '#c9d1d9', boxWidth: 12,
            // hide the "(raw)" entries from the legend
            filter: (item) => !item.text.endsWith('(raw)'),
          },
        },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0e1116', borderColor: '#30363d', borderWidth: 1,
          titleColor: '#f0f6fc', bodyColor: '#c9d1d9',
          callbacks: {
            // skip "(raw)" entries in the tooltip
            beforeBody: () => null,
            label: (ctx) => {
              if (ctx.dataset.label.endsWith('(raw)')) return null;
              return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}  @ep ${ctx.parsed.x}`;
            },
          },
        },
      },
      scales: {
        x: { type: 'linear', ticks: { color: '#7d8590' }, grid: { color: '#21262d' }, title: { display: false } },
        y: { ticks: { color: '#7d8590' }, grid: { color: '#21262d' } },
      },
    },
  });
}

function reapplySmoothing() {
  const alpha = currentSmoothing();
  document.getElementById('smooth-val').textContent = alpha.toFixed(2);
  // Re-fetch each chart so the raw + smoothed datasets are rebuilt.
  // Cheap because data is small and the API serves directly from shadow db.
  for (const tag of Object.keys(charts)) { removeChart(tag); addChart(tag); }
}

function removeChart(tag) {
  if (!charts[tag]) return;
  charts[tag].destroy();
  delete charts[tag];
  document.getElementById('card_' + tag.replace(/[^a-z0-9]/gi, '_'))?.remove();
}

async function loadImageEpochs() {
  const tag = document.getElementById('img-tag').value;
  if (!tag) return;
  const eps = await api('/api/epochs?tag=' + encodeURIComponent(tag) + qrun());
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
  const samples = await api(`/api/samples?tag=${encodeURIComponent(tag)}&ep=${ep}` + qrun());
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
    img.src = ov.url; img.className = 'overlay'; img.dataset.kind = ov.kind;
    img.style.position = 'absolute'; img.style.inset = '0';
    img.style.opacity = (document.getElementById('show-prior').checked ? document.getElementById('overlay-alpha').value : 0);
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
    trail: document.getElementById('show-trail')?.checked ?? true,
  };
  const colorByKind = { pred: '#39c860', gt: '#ff5edb', tp: '#39c860', fp: '#ff6b35', fn: '#3aa6ff', trail: '#ffffff' };
  for (const b of boxes) {
    if (b.kind === 'trail') {
      if (!showByKind.trail) continue;
      ctx.strokeStyle = colorByKind.trail; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2); ctx.stroke();
      ctx.fillStyle = colorByKind.trail;
      ctx.beginPath(); ctx.arc(b.x1, b.y1, 1.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(b.x2, b.y2, 2.5, 0, Math.PI * 2); ctx.fill();
      continue;
    }
    if (!showByKind[b.kind]) continue;
    if (b.kind === 'pred' && b.score != null && b.score < thresh) continue;
    ctx.strokeStyle = colorByKind[b.kind] || '#fff';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    if (b.kind === 'pred' && b.score != null) {
      ctx.fillStyle = colorByKind[b.kind];
      ctx.font = '12px monospace';
      ctx.fillText(b.score.toFixed(2), b.x1 + 2, b.y1 + 12);
    }
  }
}

function rerenderBoxes() {
  document.querySelectorAll('.img-card').forEach(card => {
    drawBoxes(card.querySelector('canvas.boxes'), card._sample.boxes);
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
  const root = document.getElementById('sql-result');
  try {
    const run = primaryRun();
    const url = '/api/sql' + (run ? '?run=' + encodeURIComponent(run) : '');
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    const r = await resp.json().catch(() => ({error: 'bad response'}));
    if (r.error) {
      root.innerHTML = `<div style="color:#ff6b35;font-size:12px">error: ${r.error}</div>`;
      return;
    }
    let html = `<div style="color:#7d8590;font-size:11px;margin-bottom:4px">${r.rows.length} rows${r.truncated ? ' (truncated to 1000)' : ''}</div>`;
    html += '<table><thead><tr>' + (r.columns || []).map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    for (const row of (r.rows || [])) html += '<tr>' + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
    html += '</tbody></table>';
    root.innerHTML = html;
  } catch (e) {
    root.textContent = 'error: ' + (e.message || e);
  }
}

document.getElementById('run-list').addEventListener('change', async e => {
  if (!e.target.matches('input[data-run]')) return;
  syncSelectedRuns();
  await refreshAll();
});
document.getElementById('img-tag').addEventListener('change', loadImageEpochs);
document.getElementById('img-ep').addEventListener('change', loadImages);
document.getElementById('score-thresh').addEventListener('input', e => {
  document.getElementById('score-val').textContent = parseFloat(e.target.value).toFixed(2);
  rerenderBoxes();
});
['show-pred','show-gt','show-tp','show-fp','show-fn','show-trail'].forEach(id => {
  document.getElementById(id).addEventListener('change', rerenderBoxes);
});
document.getElementById('show-prior').addEventListener('change', rerenderOverlays);
document.getElementById('overlay-alpha').addEventListener('input', rerenderOverlays);
document.getElementById('smooth-slider').addEventListener('input', () => {
  document.getElementById('smooth-val').textContent = currentSmoothing().toFixed(2);
});
document.getElementById('smooth-slider').addEventListener('change', reapplySmoothing);

refreshAll();
setInterval(async () => {
  const autoAdded = await refreshRuns();
  // re-fetch charts only when something actually changed
  if (autoAdded > 0) {
    for (const tag of Object.keys(charts)) { removeChart(tag); addChart(tag); }
  }
}, 30000);
</script>
</body>
</html>
"""
