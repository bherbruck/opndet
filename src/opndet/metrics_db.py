"""Per-run DuckDB metrics store. Replaces TB tfevents append-only protobuf
with a single queryable file: SQL handles smoothing, run-comparison, and
the queries TB structurally cannot do (e.g. "show frames where FN>2 AND
val/f1_opt_cal dropped this epoch").

One file per run dir: <run_dir>/metrics.duckdb. Image paths are relative to
the run dir so the whole bundle (db + vis PNGs + ckpts) is portable.

Layered image model: images table stores the RGB base path, overlays stores
0..N kind-tagged overlay PNGs (prior_heat, attention, etc.), boxes stores
JSON-serializable per-detection rows with kind=pred|gt|tp|fp|fn so the
viewer can do client-side score-threshold filtering and per-kind toggle
without re-rendering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scalars (
    run TEXT, ep INTEGER, tag TEXT, value DOUBLE, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS scalars_idx ON scalars(run, tag, ep);

CREATE TABLE IF NOT EXISTS images (
    run TEXT, ep INTEGER, tag TEXT, sample_idx INTEGER, base_path TEXT
);
CREATE INDEX IF NOT EXISTS images_idx ON images(run, ep, tag);

CREATE TABLE IF NOT EXISTS overlays (
    run TEXT, ep INTEGER, tag TEXT, sample_idx INTEGER,
    kind TEXT, path TEXT
);
CREATE INDEX IF NOT EXISTS overlays_idx ON overlays(run, ep, tag, sample_idx);

CREATE TABLE IF NOT EXISTS boxes (
    run TEXT, ep INTEGER, tag TEXT, sample_idx INTEGER,
    kind TEXT,
    x1 DOUBLE, y1 DOUBLE, x2 DOUBLE, y2 DOUBLE,
    score DOUBLE,
    meta JSON
);
CREATE INDEX IF NOT EXISTS boxes_idx ON boxes(run, ep, tag, sample_idx, kind);

CREATE TABLE IF NOT EXISTS config (run TEXT, key TEXT, value TEXT);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class MetricsDB:
    def __init__(self, run_dir: str | Path):
        import duckdb
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.run = run_dir.name
        self.path = run_dir / "metrics.duckdb"
        self.con = duckdb.connect(str(self.path))
        self.con.execute(_SCHEMA)
        self.con.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value=excluded.value",
            ["schema_version", "1"],
        )
        self._scalar_buf: list[tuple[str, int, str, float]] = []

    def add_scalar(self, ep: int, tag: str, value: float) -> None:
        self._scalar_buf.append((self.run, int(ep), str(tag), float(value)))
        if len(self._scalar_buf) >= 64:
            self.flush_scalars()

    def flush_scalars(self) -> None:
        if not self._scalar_buf:
            return
        self.con.executemany(
            "INSERT INTO scalars (run, ep, tag, value) VALUES (?, ?, ?, ?)",
            self._scalar_buf,
        )
        self._scalar_buf.clear()

    def add_image(self, ep: int, tag: str, sample_idx: int, base_path: str | Path) -> None:
        bp = self._rel(base_path)
        self.con.execute(
            "INSERT INTO images (run, ep, tag, sample_idx, base_path) VALUES (?, ?, ?, ?, ?)",
            [self.run, int(ep), str(tag), int(sample_idx), bp],
        )

    def add_overlay(self, ep: int, tag: str, sample_idx: int, kind: str, path: str | Path) -> None:
        p = self._rel(path)
        self.con.execute(
            "INSERT INTO overlays (run, ep, tag, sample_idx, kind, path) VALUES (?, ?, ?, ?, ?, ?)",
            [self.run, int(ep), str(tag), int(sample_idx), str(kind), p],
        )

    def add_boxes(
        self,
        ep: int,
        tag: str,
        sample_idx: int,
        kind: str,
        boxes_xyxy: np.ndarray | Iterable,
        scores: np.ndarray | Iterable | None = None,
        meta: dict[int, dict] | None = None,
    ) -> None:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        if scores is not None:
            sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        else:
            sc = np.full(boxes.shape[0], np.nan, dtype=np.float64)
        rows = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = None if np.isnan(sc[i]) else float(sc[i])
            row_meta = (meta.get(i) if meta else None) or {}
            rows.append((
                self.run, int(ep), str(tag), int(sample_idx), str(kind),
                float(x1), float(y1), float(x2), float(y2),
                score, json.dumps(row_meta) if row_meta else "{}",
            ))
        if rows:
            self.con.executemany(
                "INSERT INTO boxes (run, ep, tag, sample_idx, kind, x1, y1, x2, y2, score, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def set_config(self, cfg: dict[str, Any]) -> None:
        rows = [(self.run, k, json.dumps(v) if not isinstance(v, str) else v) for k, v in _flatten(cfg).items()]
        if rows:
            self.con.execute("DELETE FROM config WHERE run = ?", [self.run])
            self.con.executemany("INSERT INTO config (run, key, value) VALUES (?, ?, ?)", rows)

    def close(self) -> None:
        self.flush_scalars()
        self.con.close()

    def _rel(self, p: str | Path) -> str:
        p = Path(p)
        try:
            return str(p.relative_to(self.run_dir))
        except ValueError:
            return str(p)


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
