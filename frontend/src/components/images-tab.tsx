import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useLocalStorage } from "usehooks-ts";
import { api, type Sample, type TagsBulk } from "../api";
import { runColor } from "../util";
import { Lightbox } from "./lightbox";

interface Props {
  selected: string[];
  tags: TagsBulk | null;
  refetchInterval: number | false;
}

export function ImagesTab({ selected, tags, refetchInterval }: Props) {
  const imageTags = tags?.images ?? [];
  const [imgTag, setImgTag] = useLocalStorage("opndet.imgTag", "");
  const [ep, setEp] = useState<number | null>(null);
  const [lb, setLb] = useState<{ sample: Sample; caption: string } | null>(null);

  // Keep imgTag valid as the available tags change.
  useEffect(() => {
    if (imageTags.length === 0) return;
    if (!imgTag || !imageTags.includes(imgTag)) setImgTag(imageTags[0]);
  }, [imageTags, imgTag, setImgTag]);

  const runsWithTag = useMemo(
    () => selected.filter((r) => (tags?.per_run?.[r]?.images ?? []).includes(imgTag)),
    [selected, tags, imgTag],
  );

  const epochsQ = useQuery({
    queryKey: ["epochs", imgTag, runsWithTag],
    enabled: !!imgTag && runsWithTag.length > 0,
    placeholderData: keepPreviousData,
    refetchInterval,
    queryFn: async () => {
      const pairs = await Promise.all(
        runsWithTag.map((r) => api.epochs(r, imgTag).then((e) => [r, e] as const).catch(() => [r, []] as const)),
      );
      return Object.fromEntries(pairs) as Record<string, number[]>;
    },
  });
  const epochsByRun = epochsQ.data ?? {};
  const allEpochs = useMemo(
    () => [...new Set(Object.values(epochsByRun).flat())].sort((a, b) => a - b),
    [epochsByRun],
  );

  // Snap `ep` to the latest available whenever the epoch set changes.
  useEffect(() => {
    if (allEpochs.length === 0) { setEp(null); return; }
    setEp((cur) => (cur != null && allEpochs.includes(cur) ? cur : allEpochs[allEpochs.length - 1]));
  }, [allEpochs]);

  const samplesQ = useQuery({
    queryKey: ["samples", imgTag, ep, runsWithTag],
    enabled: !!imgTag && ep != null && runsWithTag.length > 0,
    placeholderData: keepPreviousData,
    queryFn: async () => {
      const e = ep as number;
      const pairs = await Promise.all(
        runsWithTag.map((r) =>
          ((epochsByRun[r] ?? []).includes(e) ? api.samples(r, imgTag, e) : Promise.resolve([]))
            .then((s) => [r, s] as const)
            .catch(() => [r, []] as const),
        ),
      );
      return Object.fromEntries(pairs) as Record<string, Sample[]>;
    },
  });
  const samplesByRun = samplesQ.data ?? {};

  if (selected.length === 0)
    return <div className="px-1 py-8 text-fgdim">select one or more runs in the sidebar.</div>;
  if (!tags) return <div className="px-1 py-8 text-fgdim">loading…</div>;
  if (imageTags.length === 0)
    return <div className="px-1 py-8 text-fgdim">no image samples logged for these runs.</div>;

  const epIdx = ep == null ? 0 : allEpochs.indexOf(ep);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3.5">
        <label className="flex items-center gap-1.5 text-fgdim">
          tag
          <select className="field" value={imgTag} onChange={(e) => setImgTag(e.target.value)}>
            {imageTags.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        {allEpochs.length > 0 && ep != null && (
          <label className="flex flex-1 items-center gap-2 text-fgdim">
            epoch
            <input
              type="range" min={0} max={allEpochs.length - 1} step={1} value={epIdx}
              className="flex-1"
              onChange={(e) => setEp(allEpochs[Number(e.target.value)])}
            />
            <span className="w-16 text-right text-fg">{ep} <span className="text-fgdim">({epIdx + 1}/{allEpochs.length})</span></span>
          </label>
        )}
        {(epochsQ.isFetching || samplesQ.isFetching) && <span className="text-fgdim">loading…</span>}
      </div>

      {runsWithTag.length === 0 ? (
        <div className="text-fgdim">none of the selected runs have samples for "{imgTag}".</div>
      ) : (
        <div className="flex flex-col gap-3.5">
          {runsWithTag.map((r) => {
            const samples = samplesByRun[r] ?? [];
            const hasEp = ep != null && (epochsByRun[r] ?? []).includes(ep);
            return (
              <div key={r} className="rounded-md border border-line bg-bg1">
                <div className="flex items-center gap-2 border-b border-line px-2 py-1.5">
                  <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: runColor(r) }} />
                  <span className="font-semibold">{r}</span>
                  <span className="text-fgdim">
                    {hasEp ? `${samples.length} sample${samples.length === 1 ? "" : "s"}` : `no ep ${ep}`}
                  </span>
                </div>
                <div className="flex gap-2 overflow-x-auto p-2">
                  {samples.map((s) => (
                    <button
                      key={s.sample_idx}
                      type="button"
                      className="relative flex-none cursor-zoom-in overflow-hidden rounded border border-line2"
                      onClick={() => setLb({ sample: s, caption: `${r} · ${imgTag} · ep ${ep} · #${s.sample_idx}` })}
                    >
                      <img className="block h-[150px] w-auto" src={s.rgb_url} alt={`#${s.sample_idx}`} loading="lazy" />
                      <span className="absolute top-0.5 left-1 text-[10px] text-[#cfe] [text-shadow:0_0_3px_#000]">#{s.sample_idx}</span>
                    </button>
                  ))}
                  {hasEp && samples.length === 0 && <span className="p-2 text-fgdim">no samples</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {lb && <Lightbox sample={lb.sample} caption={lb.caption} onClose={() => setLb(null)} />}
    </div>
  );
}
