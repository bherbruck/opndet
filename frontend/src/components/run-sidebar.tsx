import type { RunInfo } from "../api";
import { runColor, timeAgo } from "../util";

interface Props {
  runs: RunInfo[];
  selected: string[];
  onChange: (next: string[]) => void;
}

export function RunSidebar({ runs, selected, onChange }: Props) {
  const sel = new Set(selected);
  const toggle = (name: string) => {
    const next = new Set(sel);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    onChange([...next]);
  };

  return (
    <div>
      <h3 className="mx-0.5 mt-1 mb-2 text-[12px] tracking-[0.06em] text-fgdim uppercase">runs</h3>
      <div className="mb-2 flex gap-1.5">
        <button type="button" className="btn flex-1 px-1 py-0.5 text-[11px]" onClick={() => onChange(runs.map((r) => r.name))}>
          all
        </button>
        <button type="button" className="btn flex-1 px-1 py-0.5 text-[11px]" onClick={() => onChange([])}>
          none
        </button>
      </div>
      {runs.length === 0 && <div className="px-1 text-fgdim">no runs found yet…</div>}
      {runs.map((r) => {
        const on = sel.has(r.name);
        return (
          <div
            key={r.name}
            className={`flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 hover:bg-bg2 ${on ? "" : "opacity-60"}`}
            onClick={() => toggle(r.name)}
            title={r.path}
          >
            <input type="checkbox" checked={on} readOnly className="pointer-events-none" />
            <span className="h-2.5 w-2.5 flex-none rounded-[2px]" style={{ background: runColor(r.name) }} />
            <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{r.name}</span>
            <span className="flex-none text-[11px] text-fgdim">{timeAgo(r.mtime)}</span>
          </div>
        );
      })}
    </div>
  );
}
