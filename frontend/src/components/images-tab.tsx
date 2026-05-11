import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocalStorage } from "usehooks-ts";
import { api, type RunInfo, type Sample } from "../api";
import { runColor, timeAgo } from "../util";
import { Lightbox } from "./lightbox";
import { SampleView } from "./sample-view";

interface Props {
  runs: RunInfo[]; // most-recent first
  refetchInterval: number | false;
}

/** Independent of the scalars run-selection on purpose: you're usually watching
 * one (the currently-running) run's samples and don't want a scalars tweak to
 * yank it. Single-run picker, defaults to the most recently active run. */
export function ImagesTab({ runs, refetchInterval }: Props) {
  const runNames = useMemo(() => runs.map((r) => r.name), [runs]);
  const [imgRun, setImgRun] = useLocalStorage("opndet.imgRun", "");
  const [imgTag, setImgTag] = useLocalStorage("opndet.imgTag", "");
  const [ep, setEp] = useState<number | null>(null);
  const [lb, setLb] = useState<{ sample: Sample; caption: string } | null>(null);

  // Default / keep imgRun valid → most recently active run.
  useEffect(() => {
    if (runNames.length === 0) return;
    if (!imgRun || !runNames.includes(imgRun)) setImgRun(runNames[0]);
  }, [runNames, imgRun, setImgRun]);

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

  // Default to the LAST epoch; only move if the current pick disappears.
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

  if (runs.length === 0) return <div className="px-1 py-8 text-fgdim">no runs found yet…</div>;

  const epIdx = ep == null ? 0 : epochs.indexOf(ep);
  const busy = tagsQ.isFetching || epochsQ.isFetching || samplesQ.isFetching;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3.5">
        <label className="flex items-center gap-1.5 text-fgdim">
          run
          <select className="field" value={imgRun} onChange={(e) => setImgRun(e.target.value)}>
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name} · {timeAgo(r.mtime)}</option>
            ))}
          </select>
        </label>
        <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: runColor(imgRun) }} />
        {imageTags.length > 0 && (
          <label className="flex items-center gap-1.5 text-fgdim">
            tag
            <select className="field" value={imgTag} onChange={(e) => setImgTag(e.target.value)}>
              {imageTags.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        )}
        {epochs.length > 0 && ep != null && (
          <label className="flex min-w-[220px] flex-1 items-center gap-2 text-fgdim">
            epoch
            <input
              type="range" min={0} max={epochs.length - 1} step={1} value={epIdx} className="flex-1"
              onChange={(e) => setEp(epochs[Number(e.target.value)])}
            />
            <button type="button" className="btn px-1.5 py-0 text-[11px]" onClick={() => setEp(epochs[epochs.length - 1])} title="jump to latest">⏭</button>
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
              onClick={() => setLb({ sample: s, caption: `${imgRun} · ${imgTag} · ep ${ep} · #${s.sample_idx}` })}
            >
              <SampleView src={s.rgb_url} boxes={s.boxes} />
              <span className="absolute top-0.5 left-1 text-[10px] text-[#cfe] [text-shadow:0_0_3px_#000]">#{s.sample_idx}</span>
            </button>
          ))}
        </div>
      )}

      {lb && <Lightbox sample={lb.sample} caption={lb.caption} onClose={() => setLb(null)} />}
    </div>
  );
}
