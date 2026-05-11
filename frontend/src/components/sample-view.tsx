import { useEffect, useRef, useState } from "react";
import type { Box } from "../api";

interface Props {
  src: string;
  boxes: Box[];
  overlays?: { kind: string; url: string }[];
  shownOverlays?: Set<string>;
  showBoxes?: boolean;
  /** CSS width in px for the image (used for lightbox zoom). Omit → fill container. */
  widthPx?: number;
  onLoadNatural?: (w: number, h: number) => void;
}

const boxColor = (kind: string) =>
  kind.startsWith("gt") ? "#5fd35f" : kind.startsWith("pred") ? "#4f9dff" : "#e8c14a";

function paint(canvas: HTMLCanvasElement, w: number, h: number, boxes: Box[]) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(1.5, Math.round(Math.min(w, h) / 300));
  ctx.font = `${Math.max(10, Math.round(Math.min(w, h) / 55))}px ui-monospace, monospace`;
  ctx.textBaseline = "bottom";
  for (const b of boxes) {
    const c = boxColor(b.kind);
    ctx.strokeStyle = c;
    ctx.fillStyle = c;
    const poly = b.corners && b.corners.length >= 3 ? b.corners : b.points && b.points.length >= 2 ? b.points : null;
    if (poly) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.stroke();
    } else {
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }
    if (b.score != null) ctx.fillText(b.score.toFixed(2), b.x1 + 2, Math.max(11, b.y1 - 2));
  }
}

/** Image + overlay PNGs + a box/OBB-polyline canvas, all co-registered: the
 * canvas internal resolution is the image's natural size and it's CSS-stretched
 * to whatever the <img> renders at, so boxes line up at any zoom. The <img> is
 * w-full, so the rendered width is controlled by `widthPx` (lightbox) or the
 * parent grid cell (thumbnail). */
export function SampleView({
  src, boxes, overlays = [], shownOverlays, showBoxes = true, widthPx, onLoadNatural,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (canvasRef.current && nat) paint(canvasRef.current, nat.w, nat.h, showBoxes ? boxes : []);
  }, [nat, showBoxes, boxes]);

  return (
    <span className="relative block leading-none" style={widthPx ? { width: widthPx } : undefined}>
      <img
        className="block h-auto w-full"
        src={src}
        alt=""
        loading="lazy"
        onLoad={(e) => {
          const w = e.currentTarget.naturalWidth, h = e.currentTarget.naturalHeight;
          setNat({ w, h });
          onLoadNatural?.(w, h);
        }}
      />
      {overlays
        .filter((o) => shownOverlays?.has(o.kind))
        .map((o) => (
          <img key={o.kind} className="pointer-events-none absolute inset-0 h-full w-full" src={o.url} alt={o.kind} />
        ))}
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
    </span>
  );
}
