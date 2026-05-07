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

    @app.get("/api/export/scalars.csv")
    def api_export_scalars(
        runs: str | None = Query(None, description="comma-separated; omit for all selected runs implicit"),
        tags: str | None = Query(None, description="comma-separated tag filter; omit for all"),
    ) -> "Response":
        """Stream a long-format CSV of scalars across the requested runs.

        Columns: run,ep,tag,value
        Filter optionally by ?runs=...&tags=... (each comma-separated).
        Suitable for pandas.read_csv directly.
        """
        from fastapi.responses import StreamingResponse
        all_runs = _discover_runs(root_dir)
        run_names = [r.strip() for r in (runs or "").split(",") if r.strip()] or list(all_runs.keys())
        tag_filter = [t.strip() for t in (tags or "").split(",") if t.strip()]

        def _gen():
            yield "run,ep,tag,value\n"
            for n in run_names:
                if n not in all_runs:
                    continue
                try:
                    with _open_db(all_runs[n]) as con:
                        if tag_filter:
                            placeholders = ",".join(["?"] * len(tag_filter))
                            q = f"SELECT ep, tag, value FROM scalars WHERE tag IN ({placeholders}) ORDER BY tag, ep"
                            rows = con.execute(q, tag_filter).fetchall()
                        else:
                            rows = con.execute(
                                "SELECT ep, tag, value FROM scalars ORDER BY tag, ep"
                            ).fetchall()
                        for ep, tag, value in rows:
                            # CSV-quote the run/tag if they contain commas/quotes/newlines
                            t_safe = tag.replace('"', '""')
                            n_safe = n.replace('"', '""')
                            yield f'"{n_safe}",{ep},"{t_safe}",{value}\n'
                except Exception as e:
                    # surface failures inline rather than aborting the stream
                    yield f'"{n}",,,error: {str(e).replace(",", " ")}\n'

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"opndet_scalars_{ts}.csv"
        return StreamingResponse(
            _gen(), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

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
    /* explicit fixed-height canvas wrapper so Chart.js can't stretch the
       canvas to fill arbitrary parent height (was causing image-tall charts) */
    .chart-card .canvas-wrap { position: relative; width: 100%; height: 180px; }
    .chart-card canvas { position: absolute !important; inset: 0; width: 100% !important; height: 100% !important; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
    .chart-group { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 12px; }
    .chart-group > summary { cursor: pointer; padding: 8px 12px; font-size: 13px; color: #f0f6fc; font-weight: 600; user-select: none; list-style: none; display: flex; align-items: center; gap: 8px; }
    .chart-group > summary::before { content: "▸"; transition: transform 0.15s; color: #7d8590; font-weight: normal; }
    .chart-group[open] > summary::before { transform: rotate(90deg); }
    .chart-group > summary .count { color: #7d8590; font-size: 11px; font-weight: normal; margin-left: auto; }
    .chart-group > .charts { padding: 0 12px 12px; }
    .tabs { display: flex; gap: 0; border-bottom: 1px solid #30363d; margin-bottom: 12px; }
    .tab { padding: 8px 16px; cursor: pointer; color: #7d8590; border-bottom: 2px solid transparent; user-select: none; font-size: 13px; }
    .tab:hover { color: #c9d1d9; }
    .tab.active { color: #f0f6fc; border-bottom-color: #58a6ff; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
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
  <label style="font-size:11px;color:#7d8590;display:flex;gap:6px;align-items:center">
    <input type="checkbox" id="autorefresh-toggle" checked> auto
    <select id="autorefresh-interval" style="background:#0e1116;color:#d6dee6;border:1px solid #30363d;border-radius:3px;padding:2px 6px;font-family:inherit;font-size:11px">
      <option value="5000">5s</option>
      <option value="10000" selected>10s</option>
      <option value="30000">30s</option>
      <option value="60000">60s</option>
      <option value="300000">5m</option>
    </select>
  </label>
  <button onclick="pollUpdate(true)">refresh</button>
</header>

<div class="grid">
  <div class="pane">
    <h3 style="display:flex;justify-content:space-between;align-items:center">
      <span>runs</span>
      <span id="runs-count" style="color:#39c860;font-size:11px">0 selected</span>
    </h3>
    <div id="run-list" class="tag-list" style="max-height:60vh;overflow:auto;border:1px solid #21262d;border-radius:3px;padding:4px;margin-bottom:6px"></div>
    <div style="display:flex;gap:6px;margin-bottom:14px">
      <button onclick="toggleAllRuns(true)" style="flex:1">select all</button>
      <button onclick="toggleAllRuns(false)" style="flex:1">deselect all</button>
    </div>

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
    <div class="tabs">
      <div class="tab active" data-tab="charts">charts</div>
      <div class="tab" data-tab="images">images</div>
    </div>

    <div class="tab-panel active" data-panel="charts">
      <div style="display:flex;justify-content:flex-end;align-items:center;gap:14px;font-size:11px;color:#7d8590;margin-bottom:8px">
        <button onclick="exportCSV()" style="font-size:11px;padding:3px 8px">⬇ csv</button>
        <label>smoothing
          <input id="smooth-slider" type="range" min="0" max="0.99" step="0.01" value="0" style="vertical-align:middle;margin-left:6px">
          <span id="smooth-val" style="margin-left:4px">0.00</span>
        </label>
      </div>
      <div id="chart-groups"></div>
    </div>

    <div class="tab-panel" data-panel="images">
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
</div>

<script>
const charts = {};
let selectedRuns = [], scalarTags = [], imageTags = [];
// Perceptually-distinct hue-cycled palette. Replaces the old palette where
// orange + amber + yellow were too close, and blue + cyan blurred together.
// 10 hues evenly spaced around the wheel at high saturation; readable on
// dark bg.
const RUN_COLORS = [
  '#ef4444',  // red
  '#f97316',  // orange
  '#eab308',  // yellow
  '#84cc16',  // lime
  '#10b981',  // emerald
  '#06b6d4',  // cyan
  '#3b82f6',  // blue
  '#8b5cf6',  // violet
  '#d946ef',  // fuchsia
  '#ec4899',  // pink
];

// Deterministic color per run name — same run always renders in the same
// color regardless of selection order. djb2-ish string hash → palette idx.
function colorFor(run) {
  let h = 5381;
  for (let i = 0; i < run.length; i++) h = (((h << 5) + h) + run.charCodeAt(i)) >>> 0;
  return RUN_COLORS[h % RUN_COLORS.length];
}

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
  const r = (params.get('runs')        || '').split(',').filter(Boolean);
  const s = (params.get('scalars')     || '').split(',').filter(Boolean);
  const k = (params.get('known')       || '').split(',').filter(Boolean);
  const o = (params.get('openGroups')  || '').split(',').filter(Boolean);
  const t = params.get('tab') || 'charts';
  return {runs: r, scalars: s, known: k, openGroups: o, tab: t};
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
  if (kind === 'tab')         return h.tab || 'charts';
  if (kind === 'openGroups')  return h.openGroups.length ? h.openGroups : (lsGet('opndet:openGroups', []) || []);
  if (h[kind] && h[kind].length) return h[kind];
  if (kind === 'runs')        return lsGet(LS_RUNS, []);
  if (kind === 'scalars')     return lsGet(LS_SCALARS, []);
  if (kind === 'known')       return lsGet(LS_KNOWN, []);
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
    // Deterministic color per run name. Selected runs show full color;
    // unselected dim the swatch via opacity so the assignment is still
    // visible but not distracting.
    const color = colorFor(r.name);
    const checked = sel.has(r.name);
    lbl.innerHTML = `<input type="checkbox" data-run="${r.name}" id="${id}"${checked ? ' checked' : ''}><span class="swatch" style="background:${color};opacity:${checked ? 1 : 0.3}"></span><span style="font-weight:500">${r.name}</span> <span style="color:#7d8590;font-size:11px">${dt}</span>`;
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
    // no runs yet — clear what's currently rendered, leave UI quiet
    document.getElementById('chart-groups').innerHTML = '';
    document.getElementById('img-tag').innerHTML = '';
    document.getElementById('img-ep').innerHTML = '';
    document.getElementById('image-grid').innerHTML = '';
    return;
  }
  // Tags = UNION across all selected runs. The most-recent run might not
  // have any scalars yet (e.g. just started training), so fetching tags
  // only from primaryRun would leave the page blank for a few epochs.
  const allScalars = new Set(), allImages = new Set();
  for (const run of selectedRuns) {
    const t = await api('/api/tags?run=' + encodeURIComponent(run));
    if (!t) continue;
    (t.scalars || []).forEach(x => allScalars.add(x));
    (t.images  || []).forEach(x => allImages.add(x));
  }
  scalarTags = [...allScalars].sort();
  imageTags  = [...allImages].sort();
  await renderAllCharts();
  renderImageTagDropdown();
  if (imageTags.length && !document.getElementById('img-tag').value) {
    document.getElementById('img-tag').value = imageTags[0];
    await loadImageEpochs();
  }
}

// Group key = first slash component of the tag (e.g. "val/f1" → "val",
// "prior_lift/val/f1_opt" → "prior_lift"). Default-open whitelist for
// the most-watched groups.
const GROUP_OPEN_DEFAULT = new Set(['val', 'val_cal', 'test', 'prior_lift', 'train']);
const GROUP_ORDER = ['train', 'val', 'val_cal', 'val_cold', 'test', 'test_cold',
                      'prior_lift', 'eval', 'time', 'lr', 'misc'];

function groupKey(tag) {
  const slash = tag.indexOf('/');
  return slash === -1 ? 'misc' : tag.slice(0, slash);
}
function groupSort(a, b) {
  const ai = GROUP_ORDER.indexOf(a), bi = GROUP_ORDER.indexOf(b);
  if (ai >= 0 && bi >= 0) return ai - bi;
  if (ai >= 0) return -1;
  if (bi >= 0) return 1;
  return a.localeCompare(b);
}

async function renderAllCharts() {
  // tear down any existing charts; we rebuild from the current scalarTags
  for (const tag of Object.keys(charts)) removeChart(tag);
  const root = document.getElementById('chart-groups');
  root.innerHTML = '';

  if (scalarTags.length === 0 || selectedRuns.length === 0) return;

  // bucket
  const groups = {};
  for (const tag of scalarTags) {
    const g = groupKey(tag);
    (groups[g] ||= []).push(tag);
  }
  const sortedGroups = Object.keys(groups).sort(groupSort);

  // remember which groups the user collapsed/expanded across this session
  const openSet = new Set(persistedGet('openGroups').length ? persistedGet('openGroups') : sortedGroups.filter(g => GROUP_OPEN_DEFAULT.has(g)));

  for (const g of sortedGroups) {
    const det = document.createElement('details');
    det.className = 'chart-group';
    det.dataset.group = g;
    if (openSet.has(g)) det.open = true;
    const sum = document.createElement('summary');
    sum.innerHTML = `<span>${g}</span><span class="count">${groups[g].length}</span>`;
    det.appendChild(sum);
    const charts_div = document.createElement('div');
    charts_div.className = 'charts';
    det.appendChild(charts_div);
    root.appendChild(det);
    // only render charts when the group is open (saves bandwidth/rerender)
    if (det.open) {
      for (const tag of groups[g]) await addChart(tag, charts_div);
    }
    // lazy-load on first expand
    det.addEventListener('toggle', async () => {
      saveOpenGroups();
      if (det.open && charts_div.childElementCount === 0) {
        for (const tag of groups[g]) await addChart(tag, charts_div);
      }
    });
  }
}

function saveOpenGroups() {
  const open = [...document.querySelectorAll('.chart-group[open]')].map(d => d.dataset.group);
  // store via persistedSet — extends the existing hash with an openGroups field
  const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  if (open.length) params.set('openGroups', open.join(','));
  else params.delete('openGroups');
  history.replaceState(null, '', location.pathname + location.search + '#' + params.toString());
  try { localStorage.setItem('opndet:openGroups', JSON.stringify(open)); } catch {}
}

function renderImageTagDropdown() {
  const sel = document.getElementById('img-tag');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const tag of imageTags) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = tag;
    sel.appendChild(opt);
  }
  if (prev && imageTags.includes(prev)) sel.value = prev;
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

async function addChart(tag, parent) {
  if (charts[tag]) return;
  if (selectedRuns.length === 0) return;
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.id = 'card_' + tag.replace(/[^a-z0-9]/gi, '_');
  card.innerHTML = `<div class="title">${tag}</div><div class="canvas-wrap"><canvas></canvas></div>`;
  (parent || document.querySelector('#chart-groups .chart-group[open] .charts') || document.getElementById('chart-groups')).appendChild(card);

  const perRun = await api(`/api/scalars/runs?tag=${encodeURIComponent(tag)}&runs=${selectedRuns.map(encodeURIComponent).join(',')}`) || {};
  const alpha = currentSmoothing();
  const datasets = [];
  selectedRuns.forEach((run) => {
    const series = (perRun[run] || []).map(d => ({ x: d.ep, y: d.value }));
    const color = colorFor(run);
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
      responsive: true,
      maintainAspectRatio: false,
      // index mode = vertical crosshair, all runs' values at the hovered
      // epoch shown together. intersect:false so you don't have to land
      // on a point.
      interaction: { mode: 'index', intersect: false, axis: 'x' },
      plugins: {
        // Per-chart legend off — run color is shown beside each run in the
        // left sidebar, so the same legend on every chart is just noise.
        legend: { display: false },
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

async function reapplySmoothing() {
  const alpha = currentSmoothing();
  document.getElementById('smooth-val').textContent = alpha.toFixed(2);
  // Re-render the entire group structure so chart cards land back in
  // their correct group accordion.
  await renderAllCharts();
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
  // Union of epochs across all selected runs.
  const allEps = new Set();
  for (const run of selectedRuns) {
    const eps = await api(`/api/epochs?tag=${encodeURIComponent(tag)}&run=${encodeURIComponent(run)}`) || [];
    eps.forEach(e => allEps.add(e));
  }
  const eps = [...allEps].sort((a, b) => a - b);
  const sel = document.getElementById('img-ep');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const ep of eps) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = ep;
    sel.appendChild(opt);
  }
  if (eps.length) {
    sel.value = (prev && eps.includes(parseInt(prev))) ? prev : eps[eps.length - 1];
    await loadImages();
  }
}

async function loadImages() {
  const tag = document.getElementById('img-tag').value;
  const ep = document.getElementById('img-ep').value;
  const root = document.getElementById('image-grid');
  root.innerHTML = '';
  if (!tag || !ep) {
    root.innerHTML = `<div style="color:#7d8590;font-size:12px;padding:8px">no tag/epoch selected</div>`;
    return;
  }
  if (selectedRuns.length === 0) {
    root.innerHTML = `<div style="color:#7d8590;font-size:12px;padding:8px">no runs selected</div>`;
    return;
  }
  // One section per selected run, color-tagged. Skip runs that don't have
  // a sample at this (tag, ep) — they may not have hit vis_every yet.
  let total = 0;
  for (const run of selectedRuns) {
    const samples = await api(`/api/samples?tag=${encodeURIComponent(tag)}&ep=${ep}&run=${encodeURIComponent(run)}`) || [];
    if (samples.length === 0) continue;
    const sec = document.createElement('div');
    const color = colorFor(run);
    sec.innerHTML = `<div style="margin:14px 0 6px;font-size:13px;color:#c9d1d9;font-weight:600;display:flex;align-items:center;gap:6px"><span class="swatch" style="background:${color}"></span>${run} <span style="color:#7d8590;font-size:11px;font-weight:400">${samples.length} samples</span></div>`;
    const sub = document.createElement('div');
    sub.className = 'image-grid';
    sec.appendChild(sub);
    root.appendChild(sec);
    for (const s of samples) renderSample(sub, s);
    total += samples.length;
  }
  if (total === 0) {
    root.innerHTML = `<div style="color:#7d8590;font-size:12px;padding:8px">no selected run has samples for ${tag} @ epoch ${ep}</div>`;
  }
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

function exportCSV() {
  if (selectedRuns.length === 0) {
    alert('Select at least one run to export.');
    return;
  }
  const params = new URLSearchParams();
  params.set('runs', selectedRuns.join(','));
  // optional tag filter — only export tags currently rendered as charts
  // if any are open. Otherwise dump everything.
  if (Object.keys(charts).length > 0) {
    params.set('tags', Object.keys(charts).join(','));
  }
  // Trigger download via a temporary anchor so the file pops up named.
  const a = document.createElement('a');
  a.href = '/api/export/scalars.csv?' + params.toString();
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// Tabs
async function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.dataset.panel === name));
  const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  if (name === 'charts') params.delete('tab'); else params.set('tab', name);
  history.replaceState(null, '', location.pathname + location.search + '#' + params.toString());
  // When switching INTO images, re-fetch tag list and (re)load images.
  // This handles the case where images appeared mid-run after the initial
  // page load — without this you'd have to manually click refresh.
  if (name === 'images') {
    if (selectedRuns.length === 0) return;
    const allImages = new Set();
    for (const run of selectedRuns) {
      const t = await api('/api/tags?run=' + encodeURIComponent(run));
      if (!t) continue;
      (t.images || []).forEach(x => allImages.add(x));
    }
    imageTags = [...allImages].sort();
    renderImageTagDropdown();
    if (imageTags.length) {
      const sel = document.getElementById('img-tag');
      if (!sel.value || !imageTags.includes(sel.value)) sel.value = imageTags[0];
      await loadImageEpochs();
    }
  }
}
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => activateTab(t.dataset.tab));
});
activateTab(persistedGet('tab') || 'charts');

refreshAll();

// Background poll: incrementally update existing charts (in-place data
// swap, no flicker) + redraw run list. If a new run appeared OR new scalar
// tags appeared since last render, do a full re-render so new lines/groups
// show up. If `force` is true (manual refresh), always rebuild tag union
// + chart groups but keep existing chart data slot until renderAllCharts
// fills them so the area never goes empty.
async function pollUpdate(force = false) {
  const autoAdded = await refreshRuns();

  // Re-fetch the tag union — handles new scalar tags emerging mid-training
  // (e.g. test/* and val_cal/* don't appear until first calibrate fires).
  const prevScalarKey = scalarTags.join('|');
  if (selectedRuns.length > 0) {
    const allScalars = new Set(), allImages = new Set();
    for (const run of selectedRuns) {
      const t = await api('/api/tags?run=' + encodeURIComponent(run));
      if (!t) continue;
      (t.scalars || []).forEach(x => allScalars.add(x));
      (t.images  || []).forEach(x => allImages.add(x));
    }
    scalarTags = [...allScalars].sort();
    imageTags  = [...allImages].sort();
    renderImageTagDropdown();
  }
  const tagsChanged = scalarTags.join('|') !== prevScalarKey;

  if (autoAdded > 0 || tagsChanged || (force && Object.keys(charts).length === 0)) {
    await renderAllCharts();
    return;
  }
  if (Object.keys(charts).length === 0 || selectedRuns.length === 0) return;

  // Existing charts: fetch latest series, swap dataset.data in place.
  const tags = Object.keys(charts);
  const runs = selectedRuns;
  for (const tag of tags) {
    const chart = charts[tag];
    const perRun = await api(`/api/scalars/runs?tag=${encodeURIComponent(tag)}&runs=${runs.map(encodeURIComponent).join(',')}`) || {};
    const alpha = currentSmoothing();
    runs.forEach((run, i) => {
      const seriesRaw = (perRun[run] || []).map(d => ({ x: d.ep, y: d.value }));
      const smoothedDS = chart.data.datasets.find(d => d.label === run);
      const rawDS = chart.data.datasets.find(d => d.label === `${run} (raw)`);
      const smoothed = alpha > 0 ? emaSmooth(seriesRaw, alpha) : seriesRaw;
      if (!smoothedDS) return;
      smoothedDS.data = smoothed;
      if (rawDS) rawDS.data = seriesRaw;
    });
    chart.update('none');
  }
}
// Auto-refresh: configurable interval + toggle, persisted in hash/localStorage.
let _pollTimer = null;
function applyAutorefresh() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  const on = document.getElementById('autorefresh-toggle').checked;
  const ms = parseInt(document.getElementById('autorefresh-interval').value, 10);
  // persist
  const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  params.set('autorefresh', on ? '1' : '0');
  params.set('refreshMs', String(ms));
  history.replaceState(null, '', location.pathname + location.search + '#' + params.toString());
  try {
    localStorage.setItem('opndet:autorefresh', on ? '1' : '0');
    localStorage.setItem('opndet:refreshMs', String(ms));
  } catch {}
  if (on) _pollTimer = setInterval(() => pollUpdate(false), ms);
}
// Restore prior settings
{
  const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  let on = params.get('autorefresh');
  let ms = params.get('refreshMs');
  if (on === null) { try { on = localStorage.getItem('opndet:autorefresh'); } catch {} }
  if (ms === null) { try { ms = localStorage.getItem('opndet:refreshMs'); } catch {} }
  if (on !== null) document.getElementById('autorefresh-toggle').checked = (on === '1');
  if (ms !== null) document.getElementById('autorefresh-interval').value = ms;
}
document.getElementById('autorefresh-toggle').addEventListener('change', applyAutorefresh);
document.getElementById('autorefresh-interval').addEventListener('change', applyAutorefresh);
applyAutorefresh();
</script>
</body>
</html>
"""
