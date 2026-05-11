import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type RunInfo, type ScalarsBulk, type TagsBulk } from "./api";
import { loadLS, saveLS, timeAgo } from "./util";
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
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selected, setSelected] = useState<string[]>(() => loadLS<string[]>("opndet.selected", []));
  const [tab, setTab] = useState<Tab>(() => loadLS<Tab>("opndet.tab", "scalars"));
  const [refreshMs, setRefreshMs] = useState<number>(() => loadLS<number>("opndet.refreshMs", 30000));
  const [tags, setTags] = useState<TagsBulk | null>(null);
  const [scalars, setScalars] = useState<ScalarsBulk | null>(null);
  const [lastSync, setLastSync] = useState<number>(0);
  const [err, setErr] = useState<string | null>(null);
  const knownRef = useRef<string[]>(loadLS<string[]>("opndet.known", []));

  useEffect(() => saveLS("opndet.selected", selected), [selected]);
  useEffect(() => saveLS("opndet.tab", tab), [tab]);
  useEffect(() => saveLS("opndet.refreshMs", refreshMs), [refreshMs]);

  const refreshRuns = useCallback(async () => {
    const list = await api.runs();
    list.sort((a, b) => b.mtime - a.mtime);
    setRuns(list);
    // Auto-select runs that appeared since we last looked. Runs the user
    // explicitly unchecked stay in `known`, so they don't pop back.
    const names = list.map((r) => r.name);
    const fresh = names.filter((n) => !knownRef.current.includes(n));
    knownRef.current = names;
    saveLS("opndet.known", names);
    if (fresh.length) {
      setSelected((prev) => {
        const next = [...new Set([...prev.filter((n) => names.includes(n)), ...fresh])];
        return next;
      });
    } else {
      // Drop selections for runs that disappeared.
      setSelected((prev) => prev.filter((n) => names.includes(n)));
    }
    return names;
  }, []);

  const refreshData = useCallback(async (sel: string[]) => {
    if (sel.length === 0) {
      setTags(null);
      setScalars(null);
      return;
    }
    const [t, s] = await Promise.all([api.tagsBulk(sel), api.scalarsBulk(sel)]);
    setTags(t);
    setScalars(s);
  }, []);

  const syncAll = useCallback(async () => {
    try {
      const names = await refreshRuns();
      // Any newly auto-selected run lands in `selected` state and the
      // selection effect re-fetches; here just refresh what's already shown.
      await refreshData(selected.filter((n) => names.includes(n)));
      setLastSync(Date.now());
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, [refreshRuns, refreshData, selected]);

  // Initial load.
  useEffect(() => {
    syncAll();
  }, []);

  // Re-fetch data when the selection changes (debounced a touch).
  useEffect(() => {
    const id = setTimeout(() => {
      refreshData(selected).then(() => setLastSync(Date.now())).catch((e) => setErr(String(e)));
    }, 60);
    return () => clearTimeout(id);
  }, [selected, refreshData]);

  // Autorefresh loop — paused when the tab is hidden.
  useEffect(() => {
    if (refreshMs <= 0) return;
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      if (document.visibilityState === "visible") {
        try {
          const names = await refreshRuns();
          await refreshData(selected.filter((n) => names.includes(n)));
          setLastSync(Date.now());
          setErr(null);
        } catch (e) {
          setErr(String(e));
        }
      }
    };
    const id = setInterval(tick, refreshMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [refreshMs, refreshRuns, refreshData, selected]);

  const csvUrl = useMemo(() => api.scalarsCsvUrl(selected), [selected]);
  const rootName = document.title.includes("·") ? document.title.split("·")[1].trim() : "runs";

  return (
    <div className="flex h-full flex-col">
      {/* header */}
      <header className="flex flex-none items-center gap-3.5 border-b border-line bg-bg1 px-3.5 py-2">
        <span className="font-bold text-white">
          opndet <span className="font-normal text-fgdim">· {rootName}</span>
        </span>
        <span className="text-fgdim">
          {runs.length} run{runs.length === 1 ? "" : "s"} · {selected.length} shown
        </span>
        {err && <span className="text-warn">⚠ {err}</span>}
        <span className="flex-1" />
        <span className="text-fgdim">{lastSync ? `synced ${timeAgo(lastSync / 1000)}` : "…"}</span>
        <label className="flex items-center gap-1.5 text-fgdim">
          auto
          <select
            className="field"
            value={refreshMs}
            onChange={(e) => setRefreshMs(Number(e.target.value))}
          >
            {REFRESH_OPTS.map((o) => (
              <option key={o.ms} value={o.ms}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={syncAll}>
          refresh
        </button>
        <a className="btn no-underline" href={csvUrl}>
          csv
        </a>
      </header>

      {/* tabs */}
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

      {/* body */}
      <div className="flex min-h-0 flex-1">
        <aside className="w-[250px] flex-none overflow-y-auto border-r border-line bg-bg1 p-2">
          <RunSidebar runs={runs} selected={selected} onChange={setSelected} />
        </aside>
        <main className="min-w-0 flex-1 overflow-y-auto p-3">
          {tab === "scalars" && <ScalarsTab selected={selected} tags={tags} scalars={scalars} />}
          {tab === "images" && <ImagesTab selected={selected} tags={tags} />}
          {tab === "config" && <ConfigTab selected={selected} />}
          {tab === "sql" && <SqlTab runs={runs} selected={selected} />}
        </main>
      </div>
    </div>
  );
}
