import type { Point } from "./api";

// Distinct, reasonably colourblind-friendly palette for run lines.
const PALETTE = [
  "#4f9dff", "#ff8c42", "#5fd35f", "#e15c5c", "#b07be0",
  "#56c6c6", "#d6c14a", "#e87ab0", "#9aa7b5", "#7fd07f",
  "#c97f4a", "#7a9de0", "#d05fc6", "#5fae7a", "#d0a05f",
];

const _runColors = new Map<string, string>();
let _next = 0;
export function runColor(name: string): string {
  let c = _runColors.get(name);
  if (!c) {
    c = PALETTE[_next % PALETTE.length];
    _next++;
    _runColors.set(name, c);
  }
  return c;
}

export function timeAgo(unixSec: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - unixSec));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// Group tags by the prefix before the first "/". Tags without a slash go in "".
export function groupTags(tags: string[]): { group: string; tags: string[] }[] {
  const m = new Map<string, string[]>();
  for (const t of tags) {
    const i = t.indexOf("/");
    const g = i >= 0 ? t.slice(0, i) : "";
    if (!m.has(g)) m.set(g, []);
    m.get(g)!.push(t);
  }
  return [...m.entries()]
    .sort(([a], [b]) => (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b)))
    .map(([group, tags]) => ({ group, tags: tags.sort() }));
}

// EMA smoothing à la TensorBoard's slider. weight in [0,1); 0 = no smoothing.
export function emaSmooth(series: Point[], weight: number): Point[] {
  if (weight <= 0 || series.length === 0) return series;
  const out: Point[] = [];
  let last = series[0].value;
  let debias = 0;
  for (const p of series) {
    if (Number.isFinite(p.value)) {
      last = last * weight + (1 - weight) * p.value;
      debias = debias * weight + (1 - weight);
      out.push({ ep: p.ep, value: last / (debias || 1) });
    } else {
      out.push(p);
    }
  }
  return out;
}

