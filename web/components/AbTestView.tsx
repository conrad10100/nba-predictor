"use client";

import { useEffect, useState } from "react";

import { LineChart } from "@/components/LineChart";
import { api, fmt, STAT_LABEL, TARGETS, type AbResult, type AbStat, type AbTest } from "@/lib/api";

const SWITCH_LABEL: Record<string, [string, string]> = {
  defender_features: ["Without defender ratings", "With defender ratings"],
};

function setupLabel(params: Record<string, number>) {
  const entries = Object.entries(params);
  if (!entries.length) return "–";
  return entries.map(([k, v]) => SWITCH_LABEL[k]?.[v ? 1 : 0] ?? `${k}=${v}`).join(", ");
}

function daysBetween(a: string, b: string) {
  return Math.round((new Date(b).getTime() - new Date(a).getTime()) / 86_400_000);
}

export function AbTestView() {
  const [data, setData] = useState<AbTest | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.abTest().then(setData).catch((e) => setError(String(e.message ?? e))); }, []);

  if (error) return <p className="text-critical">{error}</p>;
  if (!data) return <p className="text-text-2">Loading A/B test…</p>;

  const verdicts = TARGETS.flatMap((t) => (data.stats[t]?.verdicts ?? []).map((v) => ({ ...v, target: t })))
    .sort((a, b) => b.decided_on.localeCompare(a.decided_on) || a.target.localeCompare(b.target));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">A/B test</h1>
        <p className="max-w-3xl text-sm text-text-2">
          The live model (A) makes your picks. A shadow model (B), the same model with one feature set flipped, predicts every
          slate alongside it without affecting picks. After {data.week_days} days both are compared on the same real games: B takes
          over only if its over/under hit rate against the DraftKings line is at least {(data.min_hit_rate_gain * 100).toFixed(0)} point
          higher, over at least {data.min_line_predictions} lines (pushes and predictions exactly on the line don&apos;t count). Then a new
          week starts.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {TARGETS.map((t) => {
          const s = data.stats[t];
          return s ? <StatCard key={t} target={t} s={s} minGain={data.min_hit_rate_gain} minLines={data.min_line_predictions} weekDays={data.week_days} /> : null;
        })}
      </div>

      <section className="grid gap-4 lg:grid-cols-3">
        {TARGETS.map((t) => {
          const days = data.stats[t]?.week?.days ?? [];
          if (days.length < 2 || days[0].live_graded == null) return null;
          const running = (who: "live" | "shadow") => {
            let w = 0, g = 0;
            return days.map((d) => {
              w += d[`${who}_wins`] ?? 0;
              g += d[`${who}_graded`] ?? 0;
              return g ? (w / g) * 100 : null;
            });
          };
          return (
            <LineChart
              key={t}
              title={`${STAT_LABEL[t]}: over/under hit rate so far this week`}
              xLabels={days.map((d) => d.date.slice(5).replace("-", "/"))}
              series={[
                { key: "a", label: "A", color: "var(--series-1)", values: running("live"), markers: true },
                { key: "b", label: "B", color: "var(--series-2)", values: running("shadow"), markers: true },
              ]}
              format={(v) => `${v.toFixed(0)}%`}
            />
          );
        })}
      </section>

      <section className="overflow-x-auto rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-2.5 text-sm font-medium">Weekly results</div>
        {verdicts.length === 0 ? (
          <p className="px-4 py-3 text-sm text-text-2">No week has finished yet.</p>
        ) : (
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left text-xs text-text-2">
              <tr className="border-b border-border">
                <th className="px-4 py-2 font-medium">Games</th>
                <th className="px-3 py-2 font-medium">Stat</th>
                <th className="px-3 py-2 font-medium">B (shadow)</th>
                <th className="px-3 py-2 text-right font-medium">Lines</th>
                <th className="px-3 py-2 text-right font-medium">Over/under hit A / B</th>
                <th className="px-3 py-2 text-right font-medium">Avg miss A / B</th>
                <th className="px-4 py-2 text-right font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {verdicts.map((v) => (
                <tr key={`${v.target}-${v.decided_on}`} className="border-b border-border/60 last:border-0">
                  <td className="num px-4 py-1.5">{v.start.slice(5)} – {v.end.slice(5)}</td>
                  <td className="px-3 py-1.5">{STAT_LABEL[v.target]}</td>
                  <td className="px-3 py-1.5 text-text-2">{setupLabel(v.shadow_params)}</td>
                  <td className="num px-3 py-1.5 text-right">{v.n_lines}</td>
                  <td className="num px-3 py-1.5 text-right">{pct1(v.live_hit_rate)} / {pct1(v.shadow_hit_rate)}</td>
                  <td className="num px-3 py-1.5 text-right text-text-2">±{fmt(v.live_mae_lines, 2)} / ±{fmt(v.shadow_mae_lines, 2)}</td>
                  <td className={`px-4 py-1.5 text-right text-xs ${v.switched ? "text-good" : "text-text-2"}`}>
                    {v.switched ? "B won, switched" : "A kept"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

function StatCard({ target, s, minGain, minLines, weekDays }: {
  target: (typeof TARGETS)[number]; s: AbStat; minGain: number; minLines: number; weekDays: number;
}) {
  const w = s.week;
  const today = new Date().toISOString().slice(0, 10);
  const day = s.since ? Math.min(weekDays, daysBetween(s.since, today) + 1) : 0;
  const a = w?.live_hit_rate;
  const b = w?.shadow_hit_rate;
  const diff = a != null && b != null ? b - a : null; // > 0: B wins more often
  const pts = (v: number) => `${(Math.abs(v) * 100).toFixed(1)} pt${Math.abs(v) * 100 === 1 ? "" : "s"}`;
  let status = "Waiting for graded games";
  let tone = "text-text-2";
  if (s.locked.length) {
    status = "Paused: setting forced by hand";
  } else if (!s.since) {
    status = "Starts with the next prediction run";
  } else if (diff != null) {
    const enough = (w?.n_lines ?? 0) >= minLines;
    if (diff >= minGain) {
      status = `B ahead by ${pts(diff)}${enough ? "" : ` (needs ${minLines}+ lines)`}`;
      tone = "text-good";
    } else if (diff > 0) {
      status = `B ahead by ${pts(diff)}, under the ${(minGain * 100).toFixed(0)}-point bar`;
    } else if (diff < 0) {
      status = `A ahead by ${pts(diff)}`;
    } else {
      status = "Tied";
    }
  }

  return (
    <section className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="font-medium">{STAT_LABEL[target]}</h2>
        {s.since && <span className="text-xs text-text-2">day {day} of {weekDays} · decides {s.decides_on?.slice(5)}</span>}
      </div>
      <p className={`mt-1 text-sm ${tone}`}>{status}</p>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <Side label="A · live" setup={setupLabel(s.live_params)} r={w} who="live" color="var(--series-1)" />
        <Side label="B · shadow" setup={setupLabel(s.shadow_params)} r={w} who="shadow" color="var(--series-2)" />
      </div>
      <p className="mt-3 text-xs text-text-2">
        {w ? `${w.n_lines} graded predictions with DraftKings lines (${w.n} total) since ${w.start.slice(5)}.` : "No graded games yet."}
      </p>
    </section>
  );
}

function Side({ label, setup, r, who, color }: { label: string; setup: string; r: AbResult | null; who: "live" | "shadow"; color: string }) {
  const mae = r?.[`${who}_mae_lines`];
  const hit = r?.[`${who}_hit_rate`];
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
        {label}
      </div>
      <div className="mt-0.5 text-xs text-text-2">{setup}</div>
      <div className="num mt-2 text-2xl font-semibold">{pct1(hit)}</div>
      <div className="text-xs text-text-2">over/under hit rate</div>
      <div className="num mt-1 text-sm">{mae == null ? "–" : `±${mae.toFixed(2)}`} <span className="text-xs text-text-2">avg miss</span></div>
    </div>
  );
}

function pct1(v: number | null | undefined) {
  return v == null ? "–" : `${(v * 100).toFixed(1)}%`;
}
