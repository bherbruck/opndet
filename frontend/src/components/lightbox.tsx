import { useEffect, useState } from "react";
import type { Sample } from "../api";
import { SampleView } from "./sample-view";

interface Props {
  sample: Sample;
  caption: string;
  onClose: () => void;
}

export function Lightbox({ sample, caption, onClose }: Props) {
  const overlayKinds = sample.overlays.map((o) => o.kind);
  const [shownOverlays, setShownOverlays] = useState<Set<string>>(new Set());
  const [showBoxes, setShowBoxes] = useState(true);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState<number | null>(null); // null → "fit to viewport"

  const fitZoom = nat ? Math.min(1, (window.innerWidth * 0.88) / nat.w, (window.innerHeight * 0.78) / nat.h) : 1;
  const effZoom = zoom ?? fitZoom;
  const bump = (f: number) => setZoom((z) => Math.max(0.05, Math.min(8, (z ?? fitZoom) * f)));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "+" || e.key === "=") bump(1.25);
      else if (e.key === "-" || e.key === "_") bump(1 / 1.25);
      else if (e.key === "0") setZoom(fitZoom);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const toggleOverlay = (k: string) =>
    setShownOverlays((s) => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85" onClick={onClose}>
      <div
        className="flex max-h-[94vh] max-w-[96vw] flex-col gap-2 rounded-lg border border-line2 bg-bg1 p-2.5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold">{caption}</span>
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
          <span className="ml-auto inline-flex items-center gap-1.5 text-fgdim">
            <button type="button" className="btn px-2 py-0" onClick={() => bump(1 / 1.25)}>−</button>
            <button type="button" className="btn px-2 py-0" onClick={() => setZoom(fitZoom)}>fit</button>
            <button type="button" className="btn px-2 py-0" onClick={() => setZoom(1)}>1×</button>
            <button type="button" className="btn px-2 py-0" onClick={() => bump(1.25)}>+</button>
            <span className="w-12 text-right text-fg">{Math.round(effZoom * 100)}%</span>
          </span>
          <button type="button" className="btn" onClick={onClose}>close ✕</button>
        </div>
        <div className="overflow-auto" style={{ maxHeight: "84vh", maxWidth: "92vw" }}>
          <SampleView
            src={sample.rgb_url}
            boxes={sample.boxes}
            overlays={sample.overlays}
            shownOverlays={shownOverlays}
            showBoxes={showBoxes}
            widthPx={nat ? Math.max(32, Math.round(nat.w * effZoom)) : undefined}
            onLoadNatural={(w, h) => setNat({ w, h })}
          />
        </div>
        <div className="text-fgdim">drag scrollbars / scroll to pan · +/− to zoom · 0 = fit · esc = close</div>
      </div>
    </div>
  );
}
