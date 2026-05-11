import type { RunInfo } from "../api";
import { runColor, timeAgo } from "../util";

interface Props {
  runs: RunInfo[]; // already ordered oldest → newest by the caller
  mode: "multi" | "single";
  selected: string[]; // multi: the checked set · single: a 1-element array
  onChange: (next: string[]) => void;
}

export function RunSidebar({ runs, mode, selected, onChange }: Props) {
  const sel = new Set(selected);
  const pick = (name: string) => {
    if (mode === "single") {
      onChange([name]);
      return;
    }
    const next = new Set(sel);
    next.has(name) ? next.delete(name) : next.add(name);
    onChange([...next]);
  };

  return (
    <div>
      <h3 className="mx-0.5 mt-1 mb-2 text-[12px] tracking-[0.06em] text-fgdim uppercase">
        {mode === "single" ? "run · images" : "runs"}
      </h3>
      {mode === "multi" && (
        <div className="mb-2 flex gap-1.5">
          <button type="button" className="btn flex-1 px-1 py-0.5 text-[11px]" onClick={() => onChange(runs.map((r) => r.name))}>
            all
          </button>
          <button type="button" className="btn flex-1 px-1 py-0.5 text-[11px]" onClick={() => onChange([])}>
            none
          </button>
        </div>
      )}
      {runs.length === 0 && <div className="px-1 text-fgdim">no runs found yet…</div>}
      {runs.map((r) => {
        const on = sel.has(r.name);
        return (
          <div
            key={r.name}
            className={`flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 hover:bg-bg2 ${on ? "" : "opacity-60"}`}
            onClick={() => pick(r.name)}
            title={r.path}
          >
            <input
              type={mode === "single" ? "radio" : "checkbox"}
              name={mode === "single" ? "img-run" : undefined}
              checked={on}
              readOnly
              className="pointer-events-none"
            />
            <span className="h-2.5 w-2.5 flex-none rounded-[2px]" style={{ background: runColor(r.name) }} />
            <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{r.name}</span>
            <span className="flex-none text-[11px] text-fgdim">{timeAgo(r.mtime)}</span>
          </div>
        );
      })}
    </div>
  );
}
