import { useCallback, useEffect, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocalStorage } from "usehooks-ts";
import { api, type RunInfo } from "../api";
import { runColor } from "../util";
import { Lightbox } from "./lightbox";
import { SampleView } from "./sample-view";

interface Props {
  runs: RunInfo[];
  imgRun: string; // owned by App (the sidebar single-select drives it)
  refetchInterval: number | false;
}

const isTyping = (el: EventTarget | null) =>
  el instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName);

/** Images view. Run picker is the sidebar (single-select); this tab owns the
 * tag, epoch, and lightbox. Always defaults to the latest epoch; ↑/→ and ↓/←
 * step through epochs (on the grid and inside the lightbox). */
export function ImagesTab({ runs, imgRun, refetchInterval }: Props) {
  const [imgTag, setImgTag] = useLocalStorage("opndet.imgTag", "");
  const [ep, setEp] = useState<number | null>(null);
  const [lbIdx, setLbIdx] = useState<number | null>(null);

  const tagsQ = useQuery({
    queryKey: ["imgTags", imgRun],
    enabled: !!imgRun,
    placeholderData: keepPreviousData,
    refetchInterval,
    queryFn: () => api.tagsBulk([imgRun]),
  });
  const imageTags = tagsQ.data?.images ?? [];

  useEffect(() => {
    if (imageTags.length === 0) return;
    if (!imgTag || !imageTags.includes(imgTag)) setImgTag(imageTags[0]);
  }, [imageTags, imgTag, setImgTag]);

  const hasTag = !!imgRun && imageTags.includes(imgTag);

  const epochsQ = useQuery({
    queryKey: ["imgEpochs", imgRun, imgTag],
    enabled: hasTag,
    placeholderData: keepPreviousData,
    refetchInterval,
    queryFn: () => api.epochs(imgRun, imgTag).then((e) => [...e].sort((a, b) => a - b)),
  });
  const epochs = epochsQ.data ?? [];

  // Default to the LATEST epoch; only move if the current pick disappears.
  useEffect(() => {
    if (epochs.length === 0) { setEp(null); return; }
    setEp((cur) => (cur != null && epochs.includes(cur) ? cur : epochs[epochs.length - 1]));
  }, [epochs]);

  const samplesQ = useQuery({
    queryKey: ["imgSamples", imgRun, imgTag, ep],
    enabled: hasTag && ep != null,
    placeholderData: keepPreviousData,
    refetchInterval,
    queryFn: () => api.samples(imgRun, imgTag, ep as number),
  });
  const samples = samplesQ.data ?? [];
  const epIdx = ep == null ? -1 : epochs.indexOf(ep);

  const stepEpoch = useCallback(
    (delta: number) => {
      setEp((cur) => {
        if (epochs.length === 0) return cur;
        const i = cur == null ? epochs.length - 1 : epochs.indexOf(cur);
        const j = Math.max(0, Math.min(epochs.length - 1, (i < 0 ? epochs.length - 1 : i) + delta));
        return epochs[j];
      });
    },
    [epochs],
  );

  // ↑/→ next epoch, ↓/← prev — but not while typing, and not while the lightbox
  // is open (it forwards arrow keys itself so it doesn't double-step).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (lbIdx != null || isTyping(e.target)) return;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); stepEpoch(+1); }
      else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); stepEpoch(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lbIdx, stepEpoch]);

  if (runs.length === 0) return <div className="px-1 py-8 text-fgdim">no runs found yet…</div>;

  const busy = tagsQ.isFetching || epochsQ.isFetching || samplesQ.isFetching;
  const lbSample = lbIdx == null ? undefined : samples.find((s) => s.sample_idx === lbIdx);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3.5">
        <span className="flex items-center gap-1.5 font-semibold">
          <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: runColor(imgRun) }} />
          {imgRun || "(no run)"}
        </span>
        {imageTags.length > 0 && (
          <label className="flex items-center gap-1.5 text-fgdim">
            tag
            <select className="field" value={imgTag} onChange={(e) => setImgTag(e.target.value)}>
              {imageTags.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        )}
        {epochs.length > 0 && ep != null && (
          <label className="flex min-w-[240px] flex-1 items-center gap-2 text-fgdim">
            epoch
            <button type="button" className="btn px-1.5 py-0" onClick={() => stepEpoch(-1)} disabled={epIdx <= 0} title="prev (↓/←)">◀</button>
            <input
              type="range" min={0} max={epochs.length - 1} step={1} value={Math.max(0, epIdx)} className="flex-1"
              onChange={(e) => setEp(epochs[Number(e.target.value)])}
            />
            <button type="button" className="btn px-1.5 py-0" onClick={() => stepEpoch(+1)} disabled={epIdx >= epochs.length - 1} title="next (↑/→)">▶</button>
            <button type="button" className="btn px-1.5 py-0 text-[11px]" onClick={() => setEp(epochs[epochs.length - 1])} title="latest">⏭</button>
            <span className="w-20 text-right text-fg">ep {ep} <span className="text-fgdim">({epIdx + 1}/{epochs.length})</span></span>
          </label>
        )}
        {busy && <span className="text-fgdim">…</span>}
      </div>

      {!hasTag ? (
        <div className="text-fgdim">{imageTags.length === 0 ? `no image samples logged for "${imgRun}".` : "pick a tag."}</div>
      ) : ep == null ? (
        <div className="text-fgdim">no epochs with samples.</div>
      ) : samples.length === 0 ? (
        <div className="text-fgdim">no samples at ep {ep}.</div>
      ) : (
        <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(240px,1fr))]">
          {samples.map((s) => (
            <button
              key={s.sample_idx}
              type="button"
              className="relative block cursor-zoom-in overflow-hidden rounded border border-line2 hover:border-accent"
              onClick={() => setLbIdx(s.sample_idx)}
            >
              <SampleView src={s.rgb_url} boxes={s.boxes} />
              <span className="absolute top-0.5 left-1 text-[10px] text-[#cfe] [text-shadow:0_0_3px_#000]">#{s.sample_idx}</span>
            </button>
          ))}
        </div>
      )}

      {lbSample && (
        <Lightbox
          sample={lbSample}
          caption={`${imgRun} · ${imgTag} · ep ${ep} · #${lbIdx} · (${epIdx + 1}/${epochs.length})`}
          onClose={() => setLbIdx(null)}
          onStepEpoch={stepEpoch}
        />
      )}
    </div>
  );
}
