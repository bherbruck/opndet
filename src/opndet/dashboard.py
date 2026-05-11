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

import hashlib
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse


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


_DB_STAT_THROTTLE_S = 2.0
_db_cache: dict[str, dict] = {}        # run_dir(resolved) -> {conn, lock, sig, checked, shadow}
_db_cache_lock = threading.Lock()


def _stat_sig(p: Path):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


class _PooledConn:
    """Context manager handed back by _open_db: yields a cached, shared read-only
    DuckDB connection under a per-run lock. __exit__ releases the lock; it never
    closes the connection (the cache owns it). Keeps the existing
    `with _open_db(run_dir) as con:` call sites working unchanged."""
    __slots__ = ("_conn", "_lock")

    def __init__(self, conn, lock):
        self._conn, self._lock = conn, lock

    def __enter__(self):
        self._lock.acquire()
        return self._conn

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _open_db(run_dir: Path):
    """Return a context manager yielding a read-only DuckDB connection to a /tmp
    shadow copy of the run's metrics.duckdb.

    DuckDB takes a process-level file lock even in read_only mode and the writer
    (training process) holds it for the whole run, so we read from a shadow copy.
    The copy AND the connection are CACHED per run and refreshed only when the
    source db or its WAL changes (re-stat'd at most every few seconds). Re-copying
    the file and re-opening DuckDB on every request was the dashboard's main
    latency sink — a hot request now just hands back the pooled connection.
    """
    import duckdb
    import shutil
    import hashlib

    src = run_dir / "metrics.duckdb"
    if not src.exists():
        raise HTTPException(404, f"metrics.duckdb missing in {run_dir}")
    key = str(run_dir.resolve())
    wal = src.with_suffix(".duckdb.wal")
    now = time.monotonic()

    with _db_cache_lock:
        entry = _db_cache.get(key)
        # Hot path: source checked recently — reuse the pooled connection as-is.
        if entry is not None and (now - entry["checked"]) < _DB_STAT_THROTTLE_S:
            return _PooledConn(entry["conn"], entry["lock"])
        sig = (_stat_sig(src), _stat_sig(wal))
        if entry is not None and sig == entry["sig"]:
            entry["checked"] = now
            return _PooledConn(entry["conn"], entry["lock"])

        # First open, or the db/WAL changed since last refresh: copy to a fresh
        # shadow file (fresh name so any in-flight reader keeps its own inode)
        # and open a new connection. The old cache entry's connection is released
        # when its last in-flight _PooledConn exits (CPython refcount).
        shadow_root = Path("/tmp") / "opndet_dash_shadow"
        shadow_root.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        shadow = shadow_root / f"{run_dir.name}_{h}_{int(now * 1000)}.duckdb"
        try:
            shutil.copy2(src, shadow)
            if wal.exists():
                shutil.copy2(wal, shadow.with_suffix(".duckdb.wal"))
        except Exception as e:
            if entry is not None:                       # serve stale rather than 503
                entry["checked"] = now
                return _PooledConn(entry["conn"], entry["lock"])
            raise HTTPException(503, f"metrics.duckdb temporarily unavailable: {e}")

        conn = duckdb.connect(str(shadow), read_only=True)
        _db_cache[key] = {"conn": conn, "lock": threading.Lock(),
                          "sig": sig, "checked": now, "shadow": shadow}
        # Best-effort cleanup of this run's older shadow files (and WALs).
        for old in shadow_root.glob(f"{run_dir.name}_{h}_*.duckdb*"):
            if not str(old).startswith(str(shadow)):
                try:
                    old.unlink()
                except OSError:
                    pass
        return _PooledConn(conn, _db_cache[key]["lock"])


# ---------------------------------------------------------------------------
# Content-addressed vis assets. visualize.py re-writes the clean base RGB into
# a fresh ep<NNN>/ dir every viz epoch — byte-identical content, and often the
# same val image across runs — so /files/<run>/<path> URLs never dedupe. Hash
# the bytes and serve at /blob/<hash> instead: the browser then downloads each
# distinct image at most once, ever, across all runs and epochs.
_blob_lock = threading.Lock()
_blob_hash_by_path: dict[str, tuple[str, float, int]] = {}   # resolved path -> (hash, mtime, size)
_blob_path_by_hash: dict[str, Path] = {}                     # hash -> resolved path


def _blob_ref(run_dir: Path, rel_path: str) -> str:
    """Return a /blob/<hash> URL for a vis file under a run dir. On any error
    (missing file, escape attempt) fall back to the plain /files/ URL so a
    broken asset degrades to a 404 rather than a 500."""
    fallback = f"/files/{run_dir.name}/{rel_path}"
    try:
        full = (run_dir / rel_path).resolve()
        if not str(full).startswith(str(run_dir.resolve())):
            return fallback
        st = full.stat()
        key = str(full)
        with _blob_lock:
            cached = _blob_hash_by_path.get(key)
            if cached and cached[1] == st.st_mtime and cached[2] == st.st_size:
                h = cached[0]
            else:
                h = hashlib.blake2b(full.read_bytes(), digest_size=12).hexdigest()
                _blob_hash_by_path[key] = (h, st.st_mtime, st.st_size)
            _blob_path_by_hash[h] = full
        return f"/blob/{h}"
    except OSError:
        return fallback


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
        # Per-epoch vis assets live at distinct paths and never change once
        # written — let the browser cache them so a poll doesn't re-pull MB of PNGs.
        return FileResponse(full, headers={"Cache-Control": "public, max-age=86400, immutable"})

    @app.get("/blob/{h}")
    def serve_blob(h: str):
        with _blob_lock:
            p = _blob_path_by_hash.get(h)
        if p is None or not p.exists():
            # Unknown hash (e.g. server restarted, stale URL in a cached page).
            # The client re-fetches /api/samples, which repopulates the map and
            # hands back fresh /blob URLs — so this self-heals.
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "public, max-age=31536000, immutable"})

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
                "SELECT sample_idx, kind, x1, y1, x2, y2, score, meta FROM boxes "
                "WHERE tag = ? AND ep = ? ORDER BY sample_idx, kind",
                [tag, ep],
            ).fetchall()
        ov_by_sample: dict[int, list[dict[str, str]]] = {}
        for s, kind, path in overlays:
            ov_by_sample.setdefault(s, []).append({"kind": kind, "url": _blob_ref(run_dir, path)})
        bx_by_sample: dict[int, list[dict[str, Any]]] = {}
        for s, kind, x1, y1, x2, y2, score, meta in boxes:
            entry = {"kind": kind, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}
            # OBB corners + θ live in meta. Pass through so the JS renderer can
            # draw rotated polylines instead of axis-aligned rectangles.
            if meta:
                m = meta if isinstance(meta, dict) else None
                if m is None:
                    try:
                        import json as _json
                        m = _json.loads(meta)
                    except Exception:
                        m = None
                if isinstance(m, dict):
                    if "corners" in m:
                        entry["corners"] = m["corners"]
                    if "theta" in m:
                        entry["theta"] = m["theta"]
                    if "points" in m:
                        entry["points"] = m["points"]
            bx_by_sample.setdefault(s, []).append(entry)
        return [
            {
                "sample_idx": s,
                "rgb_url": _blob_ref(run_dir, path),
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

    def _name_list(payload: dict, key: str) -> list[str]:
        """Pull a run/tag name list from a JSON body — accepts a list or a
        comma-separated string under `key`."""
        v = payload.get(key, [])
        if isinstance(v, str):
            v = v.split(",")
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    @app.post("/api/tags/bulk")
    def api_tags_bulk(payload: dict) -> dict[str, Any]:
        """One round-trip: union scalar+image tags across all requested runs,
        plus per-run breakdown so the frontend can tell which run owns
        which tag without re-fetching. Body: {"runs": [...]}."""
        names = _name_list(payload, "runs")
        all_runs = _discover_runs(root_dir)
        per_run: dict[str, dict[str, list[str]]] = {}
        scalar_set, image_set = set(), set()
        for n in names:
            if n not in all_runs:
                per_run[n] = {"scalars": [], "images": []}
                continue
            try:
                with _open_db(all_runs[n]) as con:
                    s = [r[0] for r in con.execute("SELECT DISTINCT tag FROM scalars ORDER BY tag").fetchall()]
                    i = [r[0] for r in con.execute("SELECT DISTINCT tag FROM images ORDER BY tag").fetchall()]
                    per_run[n] = {"scalars": s, "images": i}
                    scalar_set.update(s); image_set.update(i)
            except Exception:
                per_run[n] = {"scalars": [], "images": []}
        return {
            "scalars": sorted(scalar_set),
            "images": sorted(image_set),
            "per_run": per_run,
        }

    @app.post("/api/scalars/bulk")
    def api_scalars_bulk(payload: dict) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """One request returns the full {tag: {run: [{ep,value}]}} matrix.
        Body: {"runs": [...], "tags": [...]?} — omit/empty tags for all tags.
        POST (not GET) so large run/tag lists don't blow the URL-length limit."""
        run_names = _name_list(payload, "runs")
        tag_filter = _name_list(payload, "tags")
        all_runs = _discover_runs(root_dir)
        # collect: dict[tag][run] -> list[(ep,value)]
        out: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for n in run_names:
            if n not in all_runs:
                continue
            try:
                with _open_db(all_runs[n]) as con:
                    if tag_filter:
                        placeholders = ",".join(["?"] * len(tag_filter))
                        q = f"SELECT tag, ep, value FROM scalars WHERE tag IN ({placeholders}) ORDER BY tag, ep"
                        rows = con.execute(q, tag_filter).fetchall()
                    else:
                        rows = con.execute("SELECT tag, ep, value FROM scalars ORDER BY tag, ep").fetchall()
                    for tag, ep, value in rows:
                        out.setdefault(tag, {}).setdefault(n, []).append({"ep": ep, "value": value})
            except Exception:
                continue
        # ensure every requested run is keyed under each tag (empty if no data)
        for tag in (tag_filter or list(out.keys())):
            d = out.setdefault(tag, {})
            for n in run_names:
                d.setdefault(n, [])
        return out

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

    # SPA: serve the built Vite bundle at "/". Registered last so every /api/*
    # and /files/* route above takes precedence. html=True makes unknown paths
    # fall back to index.html (client-side routing). If the bundle hasn't been
    # built (fresh source checkout), fall back to a tiny placeholder page.
    static_dir = Path(__file__).parent / "dashboard_static"
    if (static_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")
    else:
        @app.get("/", response_class=HTMLResponse)
        def _no_bundle() -> str:
            return (
                "<h2>opndet dashboard</h2><p>frontend bundle not built. "
                "Run <code>scripts/build_dashboard.sh</code> (needs bun/node) "
                "or <code>cd frontend &amp;&amp; bun install &amp;&amp; bun run build</code>.</p>"
            )

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

