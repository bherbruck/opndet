// Typed client for the opndet dashboard FastAPI backend.

export interface RunInfo {
  name: string;
  path: string;
  mtime: number; // unix seconds
}

export interface Point {
  ep: number;
  value: number;
}

export interface TagsBulk {
  scalars: string[];
  images: string[];
  per_run: Record<string, { scalars: string[]; images: string[] }>;
}

export type ScalarsBulk = Record<string, Record<string, Point[]>>; // tag -> run -> series

export interface Box {
  kind: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number | null;
  corners?: number[][]; // [[x,y]*4] image px — OBB
  theta?: number;
  points?: number[][];
}

export interface Sample {
  sample_idx: number;
  rgb_url: string;
  overlays: { kind: string; url: string }[];
  boxes: Box[];
}

export interface SqlResult {
  columns: string[];
  rows: unknown[][];
  truncated: boolean;
  error?: string;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return (await r.json()) as T;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return (await r.json()) as T;
}

const qs = (o: Record<string, string | undefined>) =>
  Object.entries(o)
    .filter(([, v]) => v != null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v as string)}`)
    .join("&");

export const api = {
  runs: () => getJSON<RunInfo[]>("/api/runs"),
  health: () => getJSON<{ ok: boolean; root: string; n_runs: number }>("/api/health"),
  // bulk reads are POST: run/tag lists can get long, and these are live-polled
  // training metrics — caching them would be wrong anyway.
  tagsBulk: (runs: string[]) => postJSON<TagsBulk>("/api/tags/bulk", { runs }),
  scalarsBulk: (runs: string[], tags?: string[]) =>
    postJSON<ScalarsBulk>("/api/scalars/bulk", { runs, tags: tags ?? [] }),
  epochs: (run: string, tag: string) => getJSON<number[]>(`/api/epochs?${qs({ run, tag })}`),
  samples: (run: string, tag: string, ep: number) =>
    getJSON<Sample[]>(`/api/samples?${qs({ run, tag, ep: String(ep) })}`),
  config: (run: string) => getJSON<Record<string, string>>(`/api/config?${qs({ run })}`),
  sql: (run: string, query: string) =>
    postJSON<SqlResult>(`/api/sql?${qs({ run })}`, { query }).catch(
      (e): SqlResult => ({ columns: [], rows: [], truncated: false, error: String(e) }),
    ),
  scalarsCsvUrl: (runs: string[], tags?: string[]) =>
    `/api/export/scalars.csv?${qs({ runs: runs.join(","), tags: tags?.join(",") })}`,
};
