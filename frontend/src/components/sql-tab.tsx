import { useState } from "react";
import { useLocalStorage } from "usehooks-ts";
import { api, type RunInfo, type SqlResult } from "../api";

interface Props {
  runs: RunInfo[];
  selected: string[];
}

const DEFAULT_Q = "SELECT tag, count(*) AS n, min(ep) AS first_ep, max(ep) AS last_ep\nFROM scalars GROUP BY tag ORDER BY tag";

export function SqlTab({ runs, selected }: Props) {
  const [query, setQuery] = useLocalStorage("opndet.sqlQuery", DEFAULT_Q);
  const [target, setTarget] = useState<string>(selected[0] ?? runs[0]?.name ?? "");
  const [res, setRes] = useState<SqlResult | null>(null);
  const [running, setRunning] = useState(false);

  const run = async () => {
    if (!target) return;
    setRunning(true);
    try {
      setRes(await api.sql(target, query));
    } finally {
      setRunning(false);
    }
  };

  const names = runs.map((r) => r.name);

  return (
    <div>
      <textarea
        className="h-28 w-full resize-y rounded border border-line2 bg-bg2 p-2 font-mono text-[12px] text-fg"
        spellCheck={false}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); }}
      />
      <div className="my-2 flex items-center gap-2.5">
        <label className="text-fgdim">
          run db&nbsp;
          <select className="field" value={target} onChange={(e) => setTarget(e.target.value)}>
            {names.length === 0 && <option value="">(none)</option>}
            {names.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={run} disabled={running || !target}>
          {running ? "running…" : "run (⌘/ctrl+enter)"}
        </button>
        <span className="text-fgdim">read-only · ATTACH allowed under the runs root · 1000-row cap</span>
      </div>
      {res?.error && <div className="px-0.5 py-1.5 whitespace-pre-wrap text-warn">{res.error}</div>}
      {res && !res.error && (
        <div className="max-h-[60vh] overflow-auto rounded border border-line">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                {res.columns.map((c) => (
                  <th key={c} className="sticky top-0 border border-line bg-bg2 px-2 py-0.5 text-left whitespace-nowrap">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {res.rows.map((row, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: result rows have no stable id
                <tr key={i}>
                  {row.map((v, j) => (
                    // biome-ignore lint/suspicious/noArrayIndexKey: columns are positional
                    <td key={j} className="border border-line px-2 py-0.5 whitespace-nowrap">{v == null ? "" : String(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {res.rows.length === 0 && <div className="p-2 text-fgdim">0 rows</div>}
          {res.truncated && <div className="p-2 text-fgdim">… truncated at 1000 rows</div>}
        </div>
      )}
    </div>
  );
}
