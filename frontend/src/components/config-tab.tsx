import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { runColor } from "../util";

interface Props {
  selected: string[];
}

export function ConfigTab({ selected }: Props) {
  const [diffOnly, setDiffOnly] = useState(true);

  const cfgQ = useQuery({
    queryKey: ["config", selected],
    enabled: selected.length > 0,
    placeholderData: keepPreviousData,
    queryFn: async () => {
      const pairs = await Promise.all(
        selected.map((r) => api.config(r).then((c) => [r, c] as const).catch(() => [r, {}] as const)),
      );
      return Object.fromEntries(pairs) as Record<string, Record<string, string>>;
    },
  });

  if (selected.length === 0)
    return <div className="p-3.5 text-fgdim">select one or more runs in the sidebar.</div>;
  if (!cfgQ.data) return <div className="p-3.5 text-fgdim">loading…</div>;
  const cfgs = cfgQ.data;

  const keys = [...new Set(selected.flatMap((r) => Object.keys(cfgs[r] ?? {})))].sort();
  const differs = (k: string) => new Set(selected.map((r) => cfgs[r]?.[k] ?? " ")).size > 1;
  const rows = diffOnly && selected.length > 1 ? keys.filter(differs) : keys;

  return (
    <div className="p-3.5">
      <label className="mb-2 flex items-center gap-1.5 text-fgdim">
        <input type="checkbox" checked={diffOnly} onChange={(e) => setDiffOnly(e.target.checked)} disabled={selected.length < 2} />
        show only differing keys
      </label>
      {rows.length === 0 ? (
        <div className="text-fgdim">{keys.length ? "all shown configs agree on every key." : "no config rows."}</div>
      ) : (
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="sticky top-0 border border-line bg-bg2 px-2 py-[3px] text-left">key</th>
              {selected.map((r) => (
                <th key={r} className="sticky top-0 border border-line bg-bg2 px-2 py-[3px] text-left">
                  <span className="inline-block h-2.5 w-2.5 rounded-[2px] align-middle" style={{ background: runColor(r) }} /> {r}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((k) => {
              const d = selected.length > 1 && differs(k);
              return (
                <tr key={k} className={d ? "bg-[rgba(225,92,92,0.08)]" : ""}>
                  <td className={`border border-line px-2 py-[3px] whitespace-nowrap ${d ? "text-white" : "text-fgdim"}`}>{k}</td>
                  {selected.map((r) => (
                    <td key={r} className="border border-line px-2 py-[3px] break-words whitespace-pre-wrap">
                      {cfgs[r]?.[k] ?? <span className="text-fgdim">—</span>}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
