import { useEffect, useRef, useState } from "react";
import type { Box } from "../api";

interface Props {
  src: string;
  boxes: Box[];
  overlays?: { kind: string; url: string }[];
  shownOverlays?: Set<string>;
  showBoxes?: boolean;
  /** CSS width in px for the image (lightbox zoom). Omit → fill the container. */
  widthPx?: number;
  onLoadNatural?: (w: number, h: number) => void;
}

// Original dashboard palette — keep these.
const COLOR_BY_KIND: Record<string, string> = {
  pred: "#39c860", gt: "#ff5edb", tp: "#39c860", fp: "#ff6b35", fn: "#3aa6ff", trail: "#ffffff",
};

/** corners may arrive as [[x,y]*4] or as a flat [x0,y0,x1,y1,x2,y2,x3,y3]. */
function cornerPairs(b: Box): number[][] | null {
  const c = b.corners as unknown;
  if (Array.isArray(c) && c.length >= 3 && Array.isArray(c[0])) return c as number[][];
  if (Array.isArray(c) && c.length === 8 && typeof c[0] === "number") {
    const f = c as number[];
    return [[f[0], f[1]], [f[2], f[3]], [f[4], f[5]], [f[6], f[7]]];
  }
  if (b.points && b.points.length >= 2) return b.points;
  return null;
}

function paint(canvas: HTMLCanvasElement, natW: number, natH: number, boxes: Box[]) {
  // Bitmap = rendered CSS size; draw in natural coords via ctx.scale.
  const rw = Math.max(1, Math.round(canvas.clientWidth));
  const rh = Math.max(1, Math.round(canvas.clientHeight || (rw * natH) / natW));
  if (canvas.width !== rw) canvas.width = rw;
  if (canvas.height !== rh) canvas.height = rh;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, rw, rh);
  ctx.scale(rw / natW, rh / natH);

  // zoom = rendered px per natural px. Line/text widths are specified in
  // *rendered* px and converted back to natural units (÷ zoom) so they scale
  // up when the image is blown up (fit > 1×) but never go thinner than the
  // floor on thumbnails (zoom < 1).
  const zoom = rw / natW;
  const px = (renderedPx: number, floorPx = renderedPx) => Math.max(floorPx, renderedPx * zoom) / zoom;
  const lw = px(2);          // box stroke ≈ 2px, scales with zoom
  ctx.lineJoin = "round";
  ctx.textBaseline = "top";
  ctx.font = `${px(13)}px ui-monospace, monospace`;

  for (const b of boxes) {
    const color = COLOR_BY_KIND[b.kind] ?? "#ffffff";
    if (b.kind === "trail") {
      ctx.strokeStyle = color;
      ctx.lineWidth = px(1);
      ctx.beginPath(); ctx.moveTo(b.x1, b.y1); ctx.lineTo(b.x2, b.y2); ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(b.x1, b.y1, px(1.5), 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(b.x2, b.y2, px(2.5), 0, Math.PI * 2); ctx.fill();
      continue;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    const poly = cornerPairs(b);
    if (poly) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.stroke();
    } else {
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }
    if (b.score != null) {
      ctx.fillStyle = color;
      const lx = poly ? Math.min(...poly.map((p) => p[0])) : b.x1;
      const ly = poly ? Math.min(...poly.map((p) => p[1])) : b.y1;
      // top-left, just inside the box border
      ctx.fillText(b.score.toFixed(2), lx + lw + px(1), ly + lw + px(1));
    }
  }
}

/** Image + overlay PNGs + a box/OBB-polyline canvas, all co-registered: the
 * canvas bitmap tracks the image's *rendered* size, so boxes align and line
 * widths stay constant at any zoom or thumbnail size. */
export function SampleView({
  src, boxes, overlays = [], shownOverlays, showBoxes = true, widthPx, onLoadNatural,
}: Props) {
  const wrapRef = useRef<HTMLSpanElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);

  const repaint = () => {
    if (canvasRef.current && nat) paint(canvasRef.current, nat.w, nat.h, showBoxes ? boxes : []);
  };
  // Repaint on data change.
  useEffect(repaint, [nat, showBoxes, boxes]);
  // Repaint on rendered-size change (zoom, grid reflow, window resize).
  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(repaint);
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [nat, showBoxes, boxes]);

  return (
    <span ref={wrapRef} className="relative block leading-none" style={widthPx ? { width: widthPx } : undefined}>
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
