import { useCallback, useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocalStorage } from "usehooks-ts";
import { api, type RunInfo } from "../api";
import { runColor } from "../util";
import { DetailView } from "./detail-view";
import { SampleView, type Layers } from "./sample-view";

interface Props {
  runs: RunInfo[];
  imgRun: string; // owned by App (the sidebar single-select drives it)
  refetchInterval: number | false;
}

const isTyping = (el: EventTarget | null) =>
  el instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName);

export function ImagesTab({ runs, imgRun, refetchInterval }: Props) {
  const [imgTag, setImgTag] = useLocalStorage("opndet.imgTag", "");
  const [ep, setEp] = useState<number | null>(null);
  const [detailIdx, setDetailIdx] = useState<number | null>(null);

  // Annotation layers — persisted, shared by the grid thumbnails and detail view.
  const [overlaysOn, setOverlaysOn] = useLocalStorage<Record<string, boolean>>("opndet.layerOverlays", {});
  const [overlayOpacity, setOverlayOpacity] = useLocalStorage("opndet.layerOpacity", 0.55);
  const [showGt, setShowGt] = useLocalStorage("opndet.layerGt", true);
  const [showPred, setShowPred] = useLocalStorage("opndet.layerPred", true);
  const [showConf, setShowConf] = useLocalStorage("opndet.layerConf", true);
  const [confMin, setConfMin] = useLocalStorage("opndet.layerConfMin", 0);
  const layers: Layers = useMemo(
    () => ({ overlaysOn, overlayOpacity, gt: showGt, pred: showPred, conf: showConf, confMin }),
    [overlaysOn, overlayOpacity, showGt, showPred, showConf, confMin],
  );

  const tagsQ = useQuery({
    queryKey: ["imgTags", imgRun],
    enabled: !!imgRun,
    refetchInterval,
    queryFn: () => api.tagsBulk([imgRun]),
  });
  const imageTags = tagsQ.data?.images ?? [];

  useEffect(() => {
    if (imageTags.length === 0) return;
    if (!imgTag || !imageTags.includes(imgTag)) setImgTag(imageTags[0]);
  }, [imageTags, imgTag, setImgTag]);

  const hasTag = !!imgRun && imageTags.includes(imgTag);

  useEffect(() => { setEp(null); }, [imgRun, imgTag]);
  useEffect(() => { setDetailIdx(null); }, [imgRun, imgTag]);

  const epochsQ = useQuery({
    queryKey: ["imgEpochs", imgRun, imgTag],
    enabled: hasTag,
    refetchInterval,
    queryFn: () => api.epochs(imgRun, imgTag).then((e) => [...e].sort((a, b) => a - b)),
  });
  const epochs = epochsQ.data ?? [];

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
  const overlayKinds = useMemo(
    () => [...new Set(samples.flatMap((s) => s.overlays.map((o) => o.kind)))].sort(),
    [samples],
  );
  const anyOverlayOn = overlayKinds.some((k) => overlaysOn[k]);

  const stepEpoch = useCallback((delta: number) => {
    setEp((cur) => {
      if (epochs.length === 0) return cur;
      const i = cur == null ? epochs.length - 1 : epochs.indexOf(cur);
      const j = Math.max(0, Math.min(epochs.length - 1, (i < 0 ? epochs.length - 1 : i) + delta));
      return epochs[j];
    });
  }, [epochs]);
  const stepSample = useCallback((delta: number) => {
    setDetailIdx((cur) => {
      if (samples.length === 0 || cur == null) return cur;
      const i = samples.findIndex((s) => s.sample_idx === cur);
      const j = Math.max(0, Math.min(samples.length - 1, (i < 0 ? 0 : i) + delta));
      return samples[j].sample_idx;
    });
  }, [samples]);

  // Grid view: ↑/↓ step the epoch (no "current image" here; the detail view
  // handles its own keys — ←/→ image, ↑/↓ epoch).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (detailIdx != null || isTyping(e.target)) return;
      if (e.key === "ArrowUp") { e.preventDefault(); stepEpoch(+1); }
      else if (e.key === "ArrowDown") { e.preventDefault(); stepEpoch(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detailIdx, stepEpoch]);

  if (runs.length === 0) return <div className="p-3.5 text-fgdim">no runs found yet…</div>;

  const busy = tagsQ.isFetching || epochsQ.isFetching || samplesQ.isFetching;
  const detailSample = detailIdx == null ? undefined : samples.find((s) => s.sample_idx === detailIdx);

  return (
    <div className="flex h-full flex-col">
      {/* header (run / tag / epoch + layer toggles) — stays put; content scrolls */}
      <div className="shrink-0 border-b border-line bg-bg px-3.5 pt-2 pb-1.5">
        <div className="flex flex-wrap items-center gap-3.5">
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
              <input type="range" min={0} max={epochs.length - 1} step={1} value={Math.max(0, epIdx)} className="flex-1" onChange={(e) => setEp(epochs[Number(e.target.value)])} />
              <button type="button" className="btn px-1.5 py-0" onClick={() => stepEpoch(+1)} disabled={epIdx >= epochs.length - 1} title="next (↑/→)">▶</button>
              <button type="button" className="btn px-1.5 py-0 text-[11px]" onClick={() => setEp(epochs[epochs.length - 1])} title="latest">⏭</button>
              <span className="w-20 text-right text-fg">ep {ep} <span className="text-fgdim">({epIdx + 1}/{epochs.length})</span></span>
            </label>
          )}
          {busy && <span className="text-fgdim">…</span>}
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-3.5 text-fgdim">
          <span className="uppercase tracking-[0.06em] text-[11px]">layers</span>
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} /> <span style={{ color: "#39c860" }}>pred</span></label>
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} /> <span style={{ color: "#ff5edb" }}>gt</span></label>
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={showConf} onChange={(e) => setShowConf(e.target.checked)} /> conf</label>
          <label className="flex items-center gap-1.5" title="hide pred boxes below this score (visual only — doesn't change metrics)">
            conf ≥
            <input type="range" min={0} max={0.95} step={0.05} value={confMin} onChange={(e) => setConfMin(Number(e.target.value))} />
            <span className="w-9 text-right text-fg">{confMin.toFixed(2)}</span>
          </label>
          {overlayKinds.map((k) => (
            <label key={k} className="flex items-center gap-1.5">
              <input type="checkbox" checked={!!overlaysOn[k]} onChange={(e) => setOverlaysOn((o) => ({ ...o, [k]: e.target.checked }))} /> {k}
            </label>
          ))}
          {overlayKinds.length > 0 && (
            <label className={`flex items-center gap-1.5 ${anyOverlayOn ? "" : "opacity-40"}`}>
              opacity
              <input type="range" min={0.05} max={1} step={0.05} value={overlayOpacity} onChange={(e) => setOverlayOpacity(Number(e.target.value))} disabled={!anyOverlayOn} />
              <span className="w-8 text-right text-fg">{overlayOpacity.toFixed(2)}</span>
            </label>
          )}
        </div>
      </div>

      {/* content: grid or in-place detail view */}
      {detailSample ? (
        <DetailView
          sample={detailSample}
          caption={`${imgTag} · ep ${ep} (${epIdx + 1}/${epochs.length}) · #${detailIdx}`}
          layers={layers}
          onBack={() => setDetailIdx(null)}
          onStepEpoch={stepEpoch}
          onStepSample={samples.length > 1 ? stepSample : undefined}
        />
      ) : (
        <div className="flex-1 overflow-y-auto p-3.5">
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
                  onClick={() => setDetailIdx(s.sample_idx)}
                >
                  <SampleView src={s.rgb_url} boxes={s.boxes} overlays={s.overlays} layers={layers} />
                  <span className="absolute top-0.5 left-1 text-[10px] text-[#cfe] [text-shadow:0_0_3px_#000]">#{s.sample_idx}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
