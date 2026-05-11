import { useEffect, useRef, useState } from "react";
import type { Box, Sample } from "../api";

interface Props {
  sample: Sample;
  caption: string;
  onClose: () => void;
}

function drawBoxes(canvas: HTMLCanvasElement, w: number, h: number, boxes: Box[]) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(1.5, Math.round(Math.min(w, h) / 320));
  ctx.font = `${Math.max(11, Math.round(Math.min(w, h) / 60))}px ui-monospace, monospace`;
  const colorFor = (kind: string) => (kind.startsWith("gt") ? "#5fd35f" : kind.startsWith("pred") ? "#4f9dff" : "#e8c14a");
  for (const b of boxes) {
    const c = colorFor(b.kind);
    ctx.strokeStyle = c;
    ctx.fillStyle = c;
    if (b.corners && b.corners.length >= 4) {
      ctx.beginPath();
      ctx.moveTo(b.corners[0][0], b.corners[0][1]);
      for (let i = 1; i < b.corners.length; i++) ctx.lineTo(b.corners[i][0], b.corners[i][1]);
      ctx.closePath();
      ctx.stroke();
    } else if (b.points && b.points.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(b.points[0][0], b.points[0][1]);
      for (let i = 1; i < b.points.length; i++) ctx.lineTo(b.points[i][0], b.points[i][1]);
      ctx.closePath();
      ctx.stroke();
    } else {
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }
    if (b.score != null) ctx.fillText(b.score.toFixed(2), b.x1 + 2, Math.max(10, b.y1 - 3));
  }
}

export function Lightbox({ sample, caption, onClose }: Props) {
  const overlayKinds = sample.overlays.map((o) => o.kind);
  const [shownOverlays, setShownOverlays] = useState<Set<string>>(new Set());
  const [showBoxes, setShowBoxes] = useState(true);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (canvasRef.current && nat) drawBoxes(canvasRef.current, nat.w, nat.h, showBoxes ? sample.boxes : []);
  }, [nat, showBoxes, sample.boxes]);

  const toggleOverlay = (k: string) =>
    setShownOverlays((s) => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85" onClick={onClose}>
      <div
        className="flex max-h-[94vh] max-w-[94vw] flex-col gap-2 rounded-lg border border-line2 bg-bg1 p-2.5"
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
          <button type="button" className="btn ml-auto" onClick={onClose}>close ✕</button>
        </div>
        <div className="relative overflow-auto">
          <img
            className="block max-h-[80vh] max-w-[88vw]"
            src={sample.rgb_url}
            alt={caption}
            onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          />
          {sample.overlays
            .filter((o) => shownOverlays.has(o.kind))
            .map((o) => (
              <img key={o.kind} className="pointer-events-none absolute inset-0 h-full w-full" src={o.url} alt={o.kind} />
            ))}
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
        </div>
      </div>
    </div>
  );
}
