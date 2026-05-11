import { useEffect, useRef } from "react";
import uPlot from "uplot";
import type { Point } from "../api";
import { emaSmooth, runColor } from "../util";

export interface ChartSeries {
  run: string;
  points: Point[];
}

interface Props {
  title: string;
  series: ChartSeries[];
  smoothing: number; // 0..1
  logY: boolean;
  height?: number;
}

const fmt = (v: number) =>
  Math.abs(v) >= 1e6 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(3) : `${Number(v.toPrecision(5))}`;

// Build uPlot's column-oriented data: shared x = sorted union of epochs,
// one y array per run aligned to that x (null where the run has no point).
function buildData(series: ChartSeries[], smoothing: number): uPlot.AlignedData {
  const epochSet = new Set<number>();
  for (const s of series) for (const p of s.points) epochSet.add(p.ep);
  const xs = [...epochSet].sort((a, b) => a - b);
  const xi = new Map(xs.map((e, i) => [e, i]));
  const ys: (number | null)[][] = series.map((s) => {
    const arr: (number | null)[] = new Array(xs.length).fill(null);
    for (const p of emaSmooth(s.points, smoothing)) {
      const i = xi.get(p.ep);
      if (i != null) arr[i] = Number.isFinite(p.value) ? p.value : null;
    }
    return arr;
  });
  return [xs, ...ys] as uPlot.AlignedData;
}

// Floating hover tooltip — replaces uPlot's legend so the plot gets the space.
function tooltipPlugin(): uPlot.Plugin {
  let el: HTMLDivElement | null = null;
  return {
    hooks: {
      init: (u) => {
        el = document.createElement("div");
        el.style.cssText =
          "position:absolute;pointer-events:none;z-index:10;display:none;white-space:nowrap;" +
          "background:rgba(11,14,19,0.94);border:1px solid #2e3a47;border-radius:4px;padding:4px 7px;" +
          "font:11px ui-monospace,monospace;color:#d6dee6;line-height:1.35";
        u.over.appendChild(el);
      },
      setCursor: (u) => {
        const { idx, left, top } = u.cursor;
        if (!el) return;
        if (idx == null || left == null || top == null || left < 0) {
          el.style.display = "none";
          return;
        }
        const xs = u.data[0] as number[];
        let html = `<b>ep ${xs[idx]}</b>`;
        for (let s = 1; s < u.series.length; s++) {
          const v = u.data[s]?.[idx] as number | null | undefined;
          if (v == null) continue;
          const ser = u.series[s];
          const c = typeof ser.stroke === "function" ? ser.stroke(u, s) : (ser.stroke as string);
          html += `<br><span style="color:${c}">●</span> ${ser.label}: ${fmt(v)}`;
        }
        el.innerHTML = html;
        el.style.display = "block";
        const ow = u.over.clientWidth, oh = u.over.clientHeight;
        const w = el.offsetWidth, h = el.offsetHeight;
        el.style.left = `${left + 10 + w > ow ? left - 10 - w : left + 10}px`;
        el.style.top = `${top + 10 + h > oh ? Math.max(0, top - 10 - h) : top + 10}px`;
      },
      destroy: () => {
        el?.remove();
        el = null;
      },
    },
  };
}

export function Chart({ title, series, smoothing, logY, height = 220 }: Props) {
  const elRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  // uPlot is imperative — keep the live data in a ref so the "create plot"
  // effect reads it without listing it as a dep (it should only re-run on a
  // *structural* change: title, run set, y-scale).
  const liveRef = useRef({ series, smoothing });
  liveRef.current = { series, smoothing };
  const runKey = series.map((s) => s.run).join("|");

  useEffect(() => {
    const host = elRef.current;
    if (!host) return;
    const { series: ser, smoothing: sm } = liveRef.current;
    const opts: uPlot.Options = {
      title,
      width: host.clientWidth || 360,
      height,
      legend: { show: false },
      // drag a box (x+y) to zoom into a region; double-click resets. points:false
      // kills uPlot's per-series focus dot (it parks one at the origin until you
      // hover) — the tooltip plugin is the hover readout.
      cursor: { focus: { prox: 30 }, points: { show: false }, drag: { x: true, y: true, uni: 12 } },
      // x is epoch numbers, not timestamps — without time:false uPlot renders
      // them as dates ("12/31/69 7:00pm" = unix 0).
      scales: { x: { time: false }, y: { distr: logY ? 3 : 1 } },
      plugins: [tooltipPlugin()],
      axes: [
        { stroke: "#8b97a3", grid: { stroke: "#222a33" }, ticks: { stroke: "#2a333d" } },
        { stroke: "#8b97a3", grid: { stroke: "#222a33" }, ticks: { stroke: "#2a333d" }, size: 52 },
      ],
      series: [
        { label: "ep" },
        ...ser.map((s) => ({
          label: s.run,
          stroke: runColor(s.run),
          width: 1.75,
          points: { show: false },
          spanGaps: false,
        })),
      ],
    };
    const u = new uPlot(opts, buildData(ser, sm), host);
    plotRef.current = u;
    const ro = new ResizeObserver(() => u.setSize({ width: host.clientWidth, height }));
    ro.observe(host);
    return () => {
      ro.disconnect();
      u.destroy();
      plotRef.current = null;
    };
  }, [title, logY, height, runKey]);

  useEffect(() => {
    plotRef.current?.setData(buildData(series, smoothing));
  }, [series, smoothing]);

  // Reserve the height even while uPlot is (re)creating — otherwise the grid
  // momentarily collapses on a run-set change and the page scrolls to the top.
  return <div className="w-full" ref={elRef} style={{ minHeight: height + 30 }} />;
}
