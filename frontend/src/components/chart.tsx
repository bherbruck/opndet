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

export function Chart({ title, series, smoothing, logY, height = 220 }: Props) {
  const elRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  // uPlot is imperative — we keep the live `series`/`smoothing` in a ref so the
  // "create plot" effect can read them without listing them as deps (it should
  // only re-run when the chart's *structure* changes: title, run set, y-scale).
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
      cursor: { focus: { prox: 24 } },
      legend: { live: true },
      scales: { y: { distr: logY ? 3 : 1 } },
      axes: [
        { stroke: "#8b97a3", grid: { stroke: "#222a33" }, ticks: { stroke: "#2a333d" } },
        { stroke: "#8b97a3", grid: { stroke: "#222a33" }, ticks: { stroke: "#2a333d" }, size: 56 },
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

  // Data / smoothing changes → update in place (no flicker, keeps zoom/pan).
  useEffect(() => {
    plotRef.current?.setData(buildData(series, smoothing));
  }, [series, smoothing]);

  return <div className="w-full" ref={elRef} />;
}
