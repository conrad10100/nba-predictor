"use client";

import { useEffect, useMemo, useState } from "react";

import { LineChart } from "@/components/LineChart";
import { api, fmt, pct, STAT_LABEL, TARGETS, type Kind, type Target, type VsLine, type VsLineReport } from "@/lib/api";

/** Model accuracy on the players DraftKings posts a line for, next to the line's own accuracy. */
export function VsLineSection() {
  const [report, setReport] = useState<VsLineReport | null>(null);
  useEffect(() => { api.accuracyVsLine().then(setReport).catch(() => setReport(null)); }, []);

  const dates = useMemo(() => [...new Set(report?.days.map((d) => d.date))].sort(), [report]);
  if (!report || dates.length === 0) return null;
  const at = (t: Target) => dates.map((d) => report.days.find((x) => x.date === d && x.stat === t) ?? null);

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Players with DraftKings lines: model vs. the line</h2>
        <p className="max-w-3xl text-sm text-text-2">
          The numbers above average over every player who played, including bench players who are easy to predict. This compares
          the model only on players DraftKings posted a line for, against the line itself as a prediction. To beat the book over
          time, the model&apos;s average miss has to be smaller than the line&apos;s.
        </p>
      </div>
      {(["live", "backtest"] as Kind[]).map((kind) => {
        const totals = report.totals[kind];
        if (!totals || Object.keys(totals).length === 0) return null;
        return (
          <div key={kind}>
            <h3 className="mb-2 text-sm font-medium text-text-2">{kind === "live" ? "Live predictions" : "Backtest"}</h3>
            <div className="grid gap-3 md:grid-cols-3">
              {TARGETS.map((t) => totals[t] && <VsLineCard key={t} target={t} v={totals[t]!} />)}
            </div>
          </div>
        );
      })}
      <div className="grid gap-4 lg:grid-cols-3">
        {TARGETS.map((t) => (
          <LineChart
            key={t}
            title={`${STAT_LABEL[t]}: average miss, model vs. DraftKings line`}
            xLabels={dates.map((d) => d.slice(5).replace("-", "/"))}
            series={[
              { key: "model", label: "Model", color: "var(--series-1)", values: at(t).map((e) => e?.model_mae ?? null), markers: dates.length < 30 },
              { key: "line", label: "DK line", color: "var(--series-2)", dashed: true, values: at(t).map((e) => e?.line_mae ?? null) },
            ]}
            format={(v) => v.toFixed(1)}
          />
        ))}
      </div>
    </section>
  );
}

function VsLineCard({ target, v }: { target: Target; v: VsLine }) {
  const ahead = v.model_mae < v.line_mae;
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-sm text-text-2">{STAT_LABEL[target]} · {v.n} props</div>
      <div className="num mt-1 flex items-baseline gap-2">
        <span className="text-3xl font-semibold">±{fmt(v.model_mae, 2)}</span>
        <span className="text-sm text-text-2">vs DraftKings ±{fmt(v.line_mae, 2)}</span>
      </div>
      <div className={`text-xs ${ahead ? "text-good" : "text-critical"}`}>
        {ahead ? "✓ model is more accurate than the line" : "✕ the line is more accurate than the model"}
      </div>
      <dl className="num mt-3 grid grid-cols-2 gap-y-1 text-sm">
        <dt className="text-text-2">Model closer</dt><dd className="text-right">{pct(v.model_closer)}</dd>
        <dt className="text-text-2">Line closer</dt><dd className="text-right">{pct(v.line_closer)}</dd>
        <dt className="text-text-2">Model bias</dt><dd className="text-right">{v.model_bias > 0 ? "+" : ""}{fmt(v.model_bias, 2)}</dd>
        <dt className="text-text-2">Line bias</dt><dd className="text-right">{v.line_bias > 0 ? "+" : ""}{fmt(v.line_bias, 2)}</dd>
      </dl>
    </div>
  );
}
