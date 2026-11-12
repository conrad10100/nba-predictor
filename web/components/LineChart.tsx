"use client";

import { useMemo, useRef, useState } from "react";

export type Series = {
  key: string;
  label: string;
  color: string; // CSS var, e.g. "var(--series-1)"
  values: (number | null)[];
  dashed?: boolean;
  markers?: boolean;
};

type Props = {
  title: string;
  xLabels: string[];
  series: Series[];
  band?: { lo: (number | null)[]; hi: (number | null)[]; label: string };
  format?: (v: number) => string;
  height?: number;
  yMin?: number;
};

const PAD = { top: 12, right: 64, bottom: 26, left: 36 };
const WIDTH = 640;

export function LineChart({ title, xLabels, series, band, format = (v) => v.toFixed(1), height = 200, yMin }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const n = xLabels.length;

  const { y, ticks } = useMemo(() => {
    const all = [...series.flatMap((s) => s.values), ...(band ? [...band.lo, ...band.hi] : [])].filter((v): v is number => v != null);
    const lo = yMin ?? Math.min(0, ...all);
    const hi = Math.max(1, ...all);
    const step = niceStep((hi - lo) / 4);
    const top = Math.ceil(hi / step) * step;
    const bottom = Math.floor(lo / step) * step;
    const scale = (v: number) => PAD.top + (1 - (v - bottom) / (top - bottom || 1)) * (height - PAD.top - PAD.bottom);
    const t: number[] = [];
    for (let v = bottom; v <= top + 1e-9; v += step) t.push(v);
    return { y: scale, ticks: t };
  }, [series, band, height, yMin]);

  const x = (i: number) => PAD.left + (n <= 1 ? 0.5 : i / (n - 1)) * (WIDTH - PAD.left - PAD.right);

  const path = (values: (number | null)[]) => {
    let d = "";
    let pen = false;
    values.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const bandPath = useMemo(() => {
    if (!band) return "";
    const idx = band.lo.map((v, i) => (v != null && band.hi[i] != null ? i : -1)).filter((i) => i >= 0);
    // Draw one polygon per contiguous run of points that have a range.
    const runs: number[][] = [];
    idx.forEach((i) => (runs.length && runs.at(-1)!.at(-1) === i - 1 ? runs.at(-1)!.push(i) : runs.push([i])));
    return runs
      .map((r) => {
        if (r.length === 1) {
          const i = r[0];
          return `M${x(i) - 3},${y(band.hi[i]!)}H${x(i) + 3}V${y(band.lo[i]!)}H${x(i) - 3}Z`;
        }
        const top = r.map((i) => `${x(i)},${y(band.hi[i]!)}`).join("L");
        const bottom = [...r].reverse().map((i) => `${x(i)},${y(band.lo[i]!)}`).join("L");
        return `M${top}L${bottom}Z`;
      })
      .join("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [band, y, n]);

  const onMove = (e: React.PointerEvent) => {
    const rect = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const i = Math.round(((px - PAD.left) / (WIDTH - PAD.left - PAD.right)) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  const labelEvery = Math.max(1, Math.ceil(n / 8));
  const lastIdx = (values: (number | null)[]) => values.findLastIndex((v) => v != null);

  return (
    <figure className="rounded-lg border border-border bg-surface p-3">
      <figcaption className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        {(series.length > 1 || band) && (
          <span className="flex flex-wrap gap-3 text-xs text-text-2">
            {series.map((s) => (
              <span key={s.key} className="flex items-center gap-1.5">
                <svg width="16" height="8" aria-hidden>
                  <line x1="0" y1="4" x2="16" y2="4" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} />
                </svg>
                {s.label}
              </span>
            ))}
            {band && (
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: "var(--band)" }} />
                {band.label}
              </span>
            )}
          </span>
        )}
      </figcaption>
      <div className="relative">
        <svg
          ref={ref}
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="w-full touch-none select-none"
          role="img"
          aria-label={title}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth="1" />
              <text x={PAD.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize="10" fill="var(--muted)" className="num">
                {format(t)}
              </text>
            </g>
          ))}
          {xLabels.map((l, i) =>
            i % labelEvery === 0 || i === n - 1 ? (
              <text key={i} x={x(i)} y={height - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">
                {l}
              </text>
            ) : null,
          )}
          {band && <path d={bandPath} fill="var(--band)" />}
          {series.map((s) => (
            <g key={s.key}>
              <path d={path(s.values)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeDasharray={s.dashed ? "5 4" : undefined} />
              {s.markers &&
                s.values.map((v, i) =>
                  v == null ? null : <circle key={i} cx={x(i)} cy={y(v)} r="3.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />,
                )}
              {lastIdx(s.values) >= 0 && (
                <text
                  x={x(lastIdx(s.values)) + 8}
                  y={y(s.values[lastIdx(s.values)]!)}
                  dy="0.32em"
                  fontSize="10"
                  fill="var(--text-2)"
                  className="num"
                >
                  {s.label} {format(s.values[lastIdx(s.values)]!)}
                </text>
              )}
            </g>
          ))}
          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={height - PAD.bottom} stroke="var(--muted)" strokeWidth="1" />
              {series.map((s) =>
                s.values[hover] == null ? null : (
                  <circle key={s.key} cx={x(hover)} cy={y(s.values[hover]!)} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
                ),
              )}
            </g>
          )}
        </svg>
        {hover != null && (
          <div
            className="pointer-events-none absolute top-1 z-10 min-w-36 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs shadow-sm"
            style={{
              left: `${(x(hover) / WIDTH) * 100}%`,
              transform: x(hover) > WIDTH * 0.6 ? "translateX(calc(-100% - 10px))" : "translateX(10px)",
            }}
          >
            <div className="mb-1 font-medium">{xLabels[hover]}</div>
            {series.map((s) => (
              <div key={s.key} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-text-2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                  {s.label}
                </span>
                <span className="num font-medium">{s.values[hover] == null ? "–" : format(s.values[hover]!)}</span>
              </div>
            ))}
            {band && band.lo[hover] != null && (
              <div className="flex justify-between gap-3 text-text-2">
                <span>{band.label}</span>
                <span className="num">{format(band.lo[hover]!)}–{format(band.hi[hover]!)}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </figure>
  );
}

function niceStep(raw: number) {
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const norm = raw / mag;
  return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
}
