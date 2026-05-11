import { useEffect, useMemo, useRef } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocalStorage } from "usehooks-ts";
import { api } from "./api";
import { timeAgo } from "./util";
import { RunSidebar } from "./components/run-sidebar";
import { ScalarsTab } from "./components/scalars-tab";
import { ImagesTab } from "./components/images-tab";
import { ConfigTab } from "./components/config-tab";
import { SqlTab } from "./components/sql-tab";

type Tab = "scalars" | "images" | "config" | "sql";
const TABS: { id: Tab; label: string }[] = [
  { id: "scalars", label: "scalars" },
  { id: "images", label: "images" },
  { id: "config", label: "config" },
  { id: "sql", label: "sql" },
];
const REFRESH_OPTS = [
  { ms: 0, label: "off" },
  { ms: 5000, label: "5s" },
  { ms: 15000, label: "15s" },
  { ms: 30000, label: "30s" },
  { ms: 60000, label: "60s" },
];

export function App() {
  const [selected, setSelected] = useLocalStorage<string[]>("opndet.selected", []);
  const [tab, setTab] = useLocalStorage<Tab>("opndet.tab", "scalars");
  const [refreshMs, setRefreshMs] = useLocalStorage<number>("opndet.refreshMs", 30000);
  const [known, setKnown] = useLocalStorage<string[]>("opndet.known", []);
  const refetchInterval = refreshMs > 0 ? refreshMs : false;

  const runsQ = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval,
    select: (rs) => [...rs].sort((a, b) => b.mtime - a.mtime),
  });
  const runs = useMemo(() => runsQ.data ?? [], [runsQ.data]);
  const runNames = useMemo(() => runs.map((r) => r.name), [runs]);

  // Auto-select runs that appeared since we last looked; drop ones that vanished.
  // Runs the user explicitly unchecked stay in `known`, so they don't pop back.
  const knownRef = useRef(known);
  knownRef.current = known;
  useEffect(() => {
    if (runNames.length === 0) return;
    const fresh = runNames.filter((n) => !knownRef.current.includes(n));
    setKnown(runNames);
    setSelected((prev) => {
      const kept = prev.filter((n) => runNames.includes(n));
      return fresh.length ? [...new Set([...kept, ...fresh])] : kept;
    });
  }, [runNames.join("|")]);

  const sel = useMemo(() => selected.filter((n) => runNames.includes(n)), [selected, runNames]);

  const tagsQ = useQuery({
    queryKey: ["tagsBulk", sel],
    queryFn: () => api.tagsBulk(sel),
    enabled: sel.length > 0,
    placeholderData: keepPreviousData,
    refetchInterval,
  });
  const scalarsQ = useQuery({
    queryKey: ["scalarsBulk", sel],
    queryFn: () => api.scalarsBulk(sel),
    enabled: sel.length > 0,
    placeholderData: keepPreviousData,
    refetchInterval,
  });

  const csvUrl = useMemo(() => api.scalarsCsvUrl(sel), [sel]);
  const lastSync = Math.max(runsQ.dataUpdatedAt, tagsQ.dataUpdatedAt, scalarsQ.dataUpdatedAt);
  const fetching = runsQ.isFetching || tagsQ.isFetching || scalarsQ.isFetching;
  const err = (runsQ.error ?? tagsQ.error ?? scalarsQ.error)?.toString();
  const rootName = document.title.includes("·") ? document.title.split("·")[1].trim() : "runs";

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-none items-center gap-3.5 border-b border-line bg-bg1 px-3.5 py-2">
        <span className="font-bold text-white">
          opndet <span className="font-normal text-fgdim">· {rootName}</span>
        </span>
        <span className="text-fgdim">
          {runs.length} run{runs.length === 1 ? "" : "s"} · {sel.length} shown
        </span>
        {err && <span className="text-warn">⚠ {err}</span>}
        <span className="flex-1" />
        <span className="text-fgdim">
          {fetching ? "syncing…" : lastSync ? `synced ${timeAgo(lastSync / 1000)}` : "…"}
        </span>
        <label className="flex items-center gap-1.5 text-fgdim">
          auto
          <select className="field" value={refreshMs} onChange={(e) => setRefreshMs(Number(e.target.value))}>
            {REFRESH_OPTS.map((o) => (
              <option key={o.ms} value={o.ms}>{o.label}</option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={() => { runsQ.refetch(); tagsQ.refetch(); scalarsQ.refetch(); }}>
          refresh
        </button>
        <a className="btn no-underline" href={csvUrl}>csv</a>
      </header>

      <nav className="flex flex-none gap-1 border-b border-line bg-bg1 px-3.5 pt-1.5">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab-btn ${tab === t.id ? "tab-btn-on" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="flex min-h-0 flex-1">
        <aside className="w-[250px] flex-none overflow-y-auto border-r border-line bg-bg1 p-2">
          <RunSidebar runs={runs} selected={selected} onChange={setSelected} />
        </aside>
        <main className="min-w-0 flex-1 overflow-y-auto p-3">
          {tab === "scalars" && <ScalarsTab selected={sel} tags={tagsQ.data ?? null} scalars={scalarsQ.data ?? null} />}
          {tab === "images" && <ImagesTab runs={runs} refetchInterval={refetchInterval} />}
          {tab === "config" && <ConfigTab selected={sel} />}
          {tab === "sql" && <SqlTab runs={runs} selected={sel} />}
        </main>
      </div>
    </div>
  );
}
