"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { LineChart } from "@/components/LineChart";
import { api, exportUrl, fmt, pct, STAT_LABEL, TARGETS, type DayEval, type EvalTotals, type GradedPlayer, type Target } from "@/lib/api";

const SEGMENT_LABEL: Record<string, string> = {
  b2b: "Back-to-back", rested: "Rested", same_conference: "Same conference", cross_conference: "East vs West",
  home: "Home", away: "Away", starters: "Starters (28+ min)", bench: "Bench", teammate_out: "Key teammate out", injury_listed: "On injury report",
};

export function AccuracyView() {
  const [days, setDays] = useState<DayEval[] | null>(null);
  const [totals, setTotals] = useState<Partial<EvalTotals>>({});
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    api.evaluation()
      .then((r) => { setDays(r.days); setTotals(r.totals); setSelected(r.days.at(-1)?.date ?? null); })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const dates = useMemo(() => [...new Set(days?.map((d) => d.date))].sort(), [days]);
  const byTarget = (t: Target) => dates.map((d) => days!.find((e) => e.date === d && e.target === t) ?? null);

  if (error) return <p className="text-critical">{error}</p>;
  if (!days) return <p className="text-text-2">Loading accuracy…</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Prediction accuracy</h1>
          <p className="text-sm text-text-2">
            Every morning, yesterday&apos;s pre-game predictions are graded against the box scores. {dates.length} days tracked.
          </p>
        </div>
        <a href={exportUrl} className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm hover:bg-surface-2">
          Export all predictions (CSV)
        </a>
      </div>

      {dates.length === 0 && (
        <p className="rounded-lg border border-border bg-surface px-4 py-3 text-text-2">
          No graded days yet. The daily job stores each morning&apos;s predictions and grades them the next morning once the games are final.
        </p>
      )}

      {dates.length > 0 && (
        <>
          <section className="grid gap-3 md:grid-cols-3">
            {TARGETS.map((t) => totals[t] && <TotalsCard key={t} target={t} totals={totals[t]!} />)}
          </section>

          <section className="grid gap-4 lg:grid-cols-3">
            {TARGETS.map((t) => (
              <LineChart
                key={t}
                title={`${STAT_LABEL[t]}: average miss per player`}
                xLabels={dates.map((d) => d.slice(5).replace("-", "/"))}
                series={[
                  { key: "model", label: "Model", color: "var(--series-1)", values: byTarget(t).map((e) => e?.mae ?? null), markers: dates.length < 30 },
                  { key: "base", label: "L10 avg", color: "var(--series-2)", dashed: true, values: byTarget(t).map((e) => e?.baseline_mae ?? null) },
                ]}
                format={(v) => v.toFixed(1)}
              />
            ))}
          </section>

          <LineChart
            title="Share of players predicted within ±3 pts / ±2 reb / ±2 ast"
            xLabels={dates.map((d) => d.slice(5).replace("-", "/"))}
            series={TARGETS.map((t, i) => ({
              key: t, label: STAT_LABEL[t], color: `var(--series-${i + 1})`, values: byTarget(t).map((e) => (e ? e.within_close_pct * 100 : null)),
            }))}
            format={(v) => `${v.toFixed(0)}%`}
            yMin={0}
          />

          <SegmentTable days={days} />
          <DayDetail dates={dates} days={days} selected={selected} onSelect={setSelected} />
        </>
      )}
    </div>
  );
}

function TotalsCard({ target, totals }: { target: Target; totals: EvalTotals[Target] }) {
  const better = totals.baseline_mae != null && totals.mae < totals.baseline_mae;
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-sm text-text-2">{STAT_LABEL[target]} · season to date</div>
      <div className="num mt-1 text-3xl font-semibold">±{fmt(totals.mae, 2)}</div>
      <div className="text-xs text-text-2">
        average miss per player · L10 average misses by ±{fmt(totals.baseline_mae, 2)}{" "}
        <span className={better ? "text-good" : "text-critical"}>{better ? "✓ beating it" : "✕ behind it"}</span>
      </div>
      <dl className="num mt-3 grid grid-cols-2 gap-y-1 text-sm">
        <dt className="text-text-2">Within ±{totals.close_threshold}</dt><dd className="text-right">{pct(totals.within_close_pct)}</dd>
        <dt className="text-text-2">Closer than L10</dt><dd className="text-right">{pct(totals.beat_baseline_pct)}</dd>
        <dt className="text-text-2">Inside range (goal 80%)</dt><dd className="text-right">{pct(totals.interval_coverage)}</dd>
        <dt className="text-text-2">Bias</dt><dd className="text-right">{totals.bias > 0 ? "+" : ""}{fmt(totals.bias, 2)} {totals.bias > 0 ? "(too high)" : "(too low)"}</dd>
        <dt className="text-text-2">Graded</dt><dd className="text-right">{totals.n} over {totals.days} days</dd>
      </dl>
    </div>
  );
}

function SegmentTable({ days }: { days: DayEval[] }) {
  const rows = useMemo(() => {
    const acc: Record<string, Partial<Record<Target, { n: number; err: number; bias: number }>>> = {};
    for (const d of days) {
      for (const [seg, v] of Object.entries(d.segments)) {
        if (v.mae == null) continue;
        const cell = ((acc[seg] ??= {})[d.target] ??= { n: 0, err: 0, bias: 0 });
        cell.n += v.n;
        cell.err += v.mae * v.n;
        cell.bias += (v.bias ?? 0) * v.n;
      }
    }
    return Object.keys(SEGMENT_LABEL).filter((s) => acc[s]).map((s) => ({ seg: s, cells: acc[s] }));
  }, [days]);

  return (
    <section className="overflow-x-auto rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2.5 text-sm font-medium">Accuracy by situation (average miss, bias)</div>
      <table className="w-full min-w-[560px] text-sm">
        <thead className="text-left text-xs text-text-2">
          <tr className="border-b border-border">
            <th className="px-4 py-2 font-medium">Situation</th>
            {TARGETS.map((t) => <th key={t} className="px-3 py-2 text-right font-medium">{STAT_LABEL[t]}</th>)}
            <th className="px-4 py-2 text-right font-medium">Players</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ seg, cells }) => (
            <tr key={seg} className="border-b border-border/60 last:border-0">
              <td className="px-4 py-1.5">{SEGMENT_LABEL[seg]}</td>
              {TARGETS.map((t) => {
                const c = cells[t];
                return (
                  <td key={t} className="num px-3 py-1.5 text-right">
                    {c ? <>±{(c.err / c.n).toFixed(2)} <span className="text-xs text-muted">({c.bias >= 0 ? "+" : ""}{(c.bias / c.n).toFixed(2)})</span></> : "–"}
                  </td>
                );
              })}
              <td className="num px-4 py-1.5 text-right text-text-2">{cells.pts?.n ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function DayDetail({ dates, days, selected, onSelect }: { dates: string[]; days: DayEval[]; selected: string | null; onSelect: (d: string) => void }) {
  const [loaded, setLoaded] = useState<{ date: string; players: GradedPlayer[] } | null>(null);
  useEffect(() => {
    if (!selected) return;
    api.evaluationDay(selected).then((players) => setLoaded({ date: selected, players }));
  }, [selected]);
  const players = loaded?.date === selected ? loaded.players : null;

  const summary = days.filter((d) => d.date === selected);

  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2.5">
        <span className="text-sm font-medium">Day detail</span>
        <select value={selected ?? ""} onChange={(e) => onSelect(e.target.value)} className="rounded-md border border-border bg-bg px-2 py-1 text-sm">
          {[...dates].reverse().map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <span className="text-sm text-text-2">
          {summary.map((s) => `${STAT_LABEL[s.target]}: ${Math.round(s.within_close_pct * s.n)} of ${s.n} within ±${s.close_threshold}`).join(" · ")}
        </span>
      </div>
      <div className="max-h-[480px] overflow-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="sticky top-0 bg-surface text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Player</th>
              {TARGETS.map((t) => <th key={t} className="px-3 py-2 text-right font-medium">{t.toUpperCase()} pred → actual</th>)}
            </tr>
          </thead>
          <tbody>
            {players?.map((p) => (
              <tr key={p.player_id} className="border-b border-border/60 last:border-0">
                <td className="px-4 py-1.5">
                  <Link href={`/player/${p.player_id}`} className="hover:underline">{p.name}</Link>
                  <span className="ml-1.5 text-xs text-muted">{p.team}</span>
                </td>
                {TARGETS.map((t) => {
                  const s = p[t];
                  const close = s && Math.abs(s.error) <= (t === "pts" ? 3 : 2);
                  return (
                    <td key={t} className="num px-3 py-1.5 text-right">
                      {s ? (
                        <>
                          {fmt(s.predicted)} → <span className="font-medium">{s.actual}</span>{" "}
                          <span className={`text-xs ${close ? "text-good" : "text-text-2"}`}>{close ? "✓" : ""} {s.error > 0 ? "+" : ""}{fmt(s.error)}</span>
                        </>
                      ) : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
            {players == null && <tr><td className="px-4 py-3 text-text-2" colSpan={4}>Loading…</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
