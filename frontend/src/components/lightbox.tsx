import { useEffect, useState } from "react";
import type { Sample } from "../api";
import { SampleView } from "./sample-view";

interface Props {
  sample: Sample;
  caption: string;
  onClose: () => void;
  onStepEpoch?: (delta: number) => void;
  onStepSample?: (delta: number) => void;
}

/** Full-bleed image viewer: the image is always scaled to fill the viewport
 * (no zoom controls — pointless). ↑/↓ flips through samples, ←/→ through
 * epochs. Overlay PNGs + boxes toggle on top. */
export function Lightbox({ sample, caption, onClose, onStepEpoch, onStepSample }: Props) {
  const overlayKinds = sample.overlays.map((o) => o.kind);
  const [shownOverlays, setShownOverlays] = useState<Set<string>>(new Set());
  const [showBoxes, setShowBoxes] = useState(true);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [vp, setVp] = useState({ w: window.innerWidth, h: window.innerHeight });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowUp") { e.preventDefault(); onStepSample?.(+1); }
      else if (e.key === "ArrowDown") { e.preventDefault(); onStepSample?.(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); onStepEpoch?.(+1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); onStepEpoch?.(-1); }
    };
    const onResize = () => setVp({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("resize", onResize); };
  });

  // Fit: largest width that keeps the image inside ~92vw / ~80vh.
  const fitW = nat ? Math.round(Math.min(vp.w * 0.94, vp.h * 0.82 * (nat.w / nat.h))) : undefined;
  const toggleOverlay = (k: string) =>
    setShownOverlays((s) => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/85" onClick={onClose}>
      <div className="flex max-h-[97vh] max-w-[98vw] flex-col gap-2 rounded-lg border border-line2 bg-bg1 p-2.5" onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold">{caption}</span>
          {onStepSample && (
            <span className="inline-flex items-center gap-1 text-fgdim">
              <button type="button" className="btn px-2 py-0" onClick={() => onStepSample(-1)} title="prev image (↓)">◀ img</button>
              <button type="button" className="btn px-2 py-0" onClick={() => onStepSample(+1)} title="next image (↑)">img ▶</button>
            </span>
          )}
          {onStepEpoch && (
            <span className="inline-flex items-center gap-1 text-fgdim">
              <button type="button" className="btn px-2 py-0" onClick={() => onStepEpoch(-1)} title="prev epoch (←)">◀ ep</button>
              <button type="button" className="btn px-2 py-0" onClick={() => onStepEpoch(+1)} title="next epoch (→)">ep ▶</button>
            </span>
          )}
          {overlayKinds.map((k) => (
            <label key={k} className="inline-flex items-center gap-1.5 text-fgdim">
              <input type="checkbox" checked={shownOverlays.has(k)} onChange={() => toggleOverlay(k)} /> {k}
            </label>
          ))}
          {sample.boxes.length > 0 && (
            <label className="inline-flex items-center gap-1.5 text-fgdim">
              <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> boxes ({sample.boxes.length})
            </label>
          )}
          <button type="button" className="btn ml-auto" onClick={onClose}>close ✕</button>
        </div>
        <div className="flex items-center justify-center overflow-auto">
          <SampleView
            src={sample.rgb_url}
            boxes={sample.boxes}
            overlays={sample.overlays}
            shownOverlays={shownOverlays}
            showBoxes={showBoxes}
            widthPx={fitW}
            onLoadNatural={(w, h) => setNat({ w, h })}
          />
        </div>
        <div className="text-fgdim">↑/↓ image · ←/→ epoch · esc close</div>
      </div>
    </div>
  );
}
