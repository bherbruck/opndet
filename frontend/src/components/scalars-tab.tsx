import { useMemo, useState } from "react";
import { useLocalStorage } from "usehooks-ts";
import type { ScalarsBulk, TagsBulk } from "../api";
import { groupTags } from "../util";
import { Chart, type ChartSeries } from "./chart";

interface Props {
  selected: string[];
  tags: TagsBulk | null;
  scalars: ScalarsBulk | null;
}

export function ScalarsTab({ selected, tags, scalars }: Props) {
  const [smoothing, setSmoothing] = useLocalStorage("opndet.smoothing", 0);
  const [logY, setLogY] = useLocalStorage("opndet.logY", false);
  const [minEp, setMinEp] = useLocalStorage("opndet.minEp", 0);
  const [collapsed, setCollapsed] = useLocalStorage<Record<string, boolean>>("opndet.groupsCollapsed", {});
  const [filter, setFilter] = useState("");

  const toggleGroup = (g: string) => setCollapsed((c) => ({ ...c, [g]: !c[g] }));

  const maxEp = useMemo(() => {
    let m = 0;
    if (scalars) for (const byRun of Object.values(scalars)) for (const pts of Object.values(byRun)) {
      const last = pts[pts.length - 1]?.ep;
      if (last != null && last > m) m = last;
    }
    return m;
  }, [scalars]);
  const effMinEp = Math.min(minEp, maxEp);

  const groups = useMemo(() => {
    const all = tags?.scalars ?? [];
    const f = filter.trim().toLowerCase();
    return groupTags(f ? all.filter((t) => t.toLowerCase().includes(f)) : all);
  }, [tags, filter]);

  if (selected.length === 0)
    return <div className="p-3.5 text-fgdim">select one or more runs in the sidebar.</div>;
  if (!tags || !scalars) return <div className="p-3.5 text-fgdim">loading…</div>;
  if ((tags.scalars ?? []).length === 0)
    return <div className="p-3.5 text-fgdim">no scalars logged yet for these runs.</div>;

  return (
    <div>
      <div className="sticky top-0 z-10 flex flex-wrap items-center gap-3.5 border-b border-line bg-bg px-3.5 py-2.5">
        <label className="flex items-center gap-1.5 text-fgdim">
          smoothing
          <input type="range" min={0} max={0.99} step={0.01} value={smoothing} onChange={(e) => setSmoothing(Number(e.target.value))} />
          <span className="w-9 text-right text-fg">{smoothing.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-1.5 text-fgdim">
          <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log-y
        </label>
        {maxEp > 0 && (
          <label className="flex items-center gap-1.5 text-fgdim">
            ep ≥
            <input type="range" min={0} max={maxEp} step={1} value={effMinEp} onChange={(e) => setMinEp(Number(e.target.value))} />
            <span className="w-12 text-right text-fg">{effMinEp}{effMinEp > 0 && <button type="button" className="ml-1 text-fgdim hover:text-fg" onClick={() => setMinEp(0)} title="reset">✕</button>}</span>
          </label>
        )}
        <input className="field min-w-[140px] flex-1" placeholder="filter tags…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>

      <div className="p-3.5">
        {groups.map(({ group, tags: gtags }) => {
          const isCol = !!collapsed[group];
          return (
            <section key={group || "_"} className="mb-3.5">
              <div
                className="cursor-pointer select-none border-b border-line px-0.5 py-1 text-[12px] tracking-[0.06em] text-fgdim uppercase"
                onClick={() => toggleGroup(group)}
              >
                {isCol ? "▸" : "▾"} {group || "ungrouped"} <span className="text-line2">({gtags.length})</span>
              </div>
              {!isCol && (
                <div className="mt-2.5 grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(360px,1fr))]">
                  {gtags.map((tag) => {
                    const perRun = scalars[tag] ?? {};
                    const series: ChartSeries[] = selected
                      .map((r) => ({ run: r, points: (perRun[r] ?? []).filter((p) => p.ep >= effMinEp) }))
                      .filter((s) => s.points.length > 0);
                    return (
                      <div key={tag} className="rounded-md border border-line bg-bg1 px-2 pt-1.5 pb-2">
                        {series.length === 0 ? (
                          <div className="p-4 text-fgdim">
                            <div className="mb-1 font-semibold text-fg">{tag}</div>no data{effMinEp > 0 ? ` ≥ ep ${effMinEp}` : ""}
                          </div>
                        ) : (
                          <Chart title={tag} series={series} smoothing={smoothing} logY={logY} />
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
