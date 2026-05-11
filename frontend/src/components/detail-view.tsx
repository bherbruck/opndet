import { useEffect, useRef, useState } from "react";
import type { Sample } from "../api";
import { SampleView, type Layers } from "./sample-view";

interface Props {
  sample: Sample;
  caption: string;
  layers: Layers;
  onBack: () => void;
  onStepEpoch?: (delta: number) => void;
  onStepSample?: (delta: number) => void;
}

/** In-place detail view — fills the images-tab content area (not a modal). The
 * image is scaled to fill the area. ↑/↓ flips through samples, ←/→ through
 * epochs, esc returns to the grid. Annotation layers come from the shared
 * toggles in the images-tab header. */
export function DetailView({ sample, caption, layers, onBack, onStepEpoch, onStepSample }: Props) {
  const areaRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onBack();
      else if (e.key === "ArrowRight") { e.preventDefault(); onStepSample?.(+1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); onStepSample?.(-1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); onStepEpoch?.(+1); }
      else if (e.key === "ArrowDown") { e.preventDefault(); onStepEpoch?.(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    const apply = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Largest width that keeps the image inside the available area.
  const fitW =
    nat && box.w > 0 && box.h > 0 ? Math.round(Math.min(box.w, box.h * (nat.w / nat.h))) : undefined;

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-line px-1 py-1.5">
        <button type="button" className="btn px-2 py-0" onClick={onBack}>← grid</button>
        {onStepSample && (
          <span className="inline-flex items-center gap-1 text-fgdim">
            <button type="button" className="btn px-2 py-0" onClick={() => onStepSample(-1)} title="prev image (←)">◀ img</button>
            <button type="button" className="btn px-2 py-0" onClick={() => onStepSample(+1)} title="next image (→)">img ▶</button>
          </span>
        )}
        {onStepEpoch && (
          <span className="inline-flex items-center gap-1 text-fgdim">
            <button type="button" className="btn px-2 py-0" onClick={() => onStepEpoch(-1)} title="prev epoch (↓)">◀ ep</button>
            <button type="button" className="btn px-2 py-0" onClick={() => onStepEpoch(+1)} title="next epoch (↑)">ep ▶</button>
          </span>
        )}
        <span className="font-semibold">{caption}</span>
        <span className="ml-auto text-fgdim">←/→ image · ↑/↓ epoch · esc back</span>
      </div>
      <div ref={areaRef} className="flex flex-1 items-center justify-center overflow-auto p-2">
        <SampleView
          src={sample.rgb_url}
          boxes={sample.boxes}
          overlays={sample.overlays}
          layers={layers}
          widthPx={fitW}
          onLoadNatural={(w, h) => setNat({ w, h })}
        />
      </div>
    </div>
  );
}
