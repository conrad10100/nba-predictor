"use client";

import { useEffect, useState } from "react";

import { api, fmt, pct, STAT_LABEL, type ModelMetrics } from "@/lib/api";

export function ModelView() {
  const [m, setM] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setError(String(e.message ?? e))); }, []);

  if (error) return <p className="text-critical">{error}</p>;
  if (!m) return <p className="text-text-2">Loading picks…</p>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Picks</h1>
        <p className="text-sm text-text-2">
          Retrained every morning. Accuracy below is on the most recent 3 weeks of games, which the model did not train on.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {m.latest.map((r) => {
          const cfg = m.config.find((c) => c.target === r.target);
          const top = Object.entries(r.top_features).slice(0, 10);
          const max = Math.max(...top.map(([, v]) => v), 1e-9);
          return (
            <section key={r.target} className="rounded-lg border border-border bg-surface p-4">
              <div className="flex items-baseline justify-between">
                <h2 className="font-medium">{STAT_LABEL[r.target]}</h2>
                <span className={`text-xs ${r.promoted ? "text-good" : "text-critical"}`}>{r.promoted ? "✓ live" : "✕ not promoted"}</span>
              </div>
              <div className="num mt-2 text-3xl font-semibold">±{fmt(r.mae, 2)}</div>
              <div className="text-xs text-text-2">average miss · last-10 average misses by ±{fmt(r.baseline_mae, 2)}</div>
              <dl className="num mt-3 grid grid-cols-2 gap-y-1 text-sm">
                <dt className="text-text-2">Inside 80% range</dt><dd className="text-right">{pct(r.interval_coverage)}</dd>
                <dt className="text-text-2">Algorithm</dt><dd className="text-right">{cfg?.params.model === "xgboost" ? "XGBoost" : "HistGradientBoosting"}</dd>
                <dt className="text-text-2">Recency half-life</dt><dd className="text-right">{cfg?.params.half_life_days ?? "–"} days</dd>
                <dt className="text-text-2">Bias correction</dt><dd className="text-right">{cfg ? `${cfg.bias_correction >= 0 ? "+" : ""}${cfg.bias_correction.toFixed(2)}` : "–"}</dd>
                <dt className="text-text-2">Training rows</dt><dd className="text-right">{r.n_train.toLocaleString()}</dd>
              </dl>
              {cfg?.note && <p className="mt-2 text-xs text-text-2">Tuning: {cfg.note}</p>}

              <h3 className="mt-4 mb-2 text-xs font-medium text-text-2">What moves the prediction most</h3>
              <ul className="space-y-1">
                {top.map(([name, v]) => (
                  <li key={name} className="grid grid-cols-[1fr_auto] items-center gap-2 text-xs" title={`Removing it increases average miss by ${v.toFixed(3)}`}>
                    <div>
                      <div className="truncate text-text-2">{name}</div>
                      <div className="mt-0.5 h-1.5 rounded-r-sm" style={{ width: `${Math.max(2, (v / max) * 100)}%`, background: "var(--series-1)" }} />
                    </div>
                    <span className="num">{v.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>

      <section className="overflow-x-auto rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-2.5 text-sm font-medium">Training history</div>
        <table className="w-full min-w-[560px] text-sm">
          <thead className="text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Trained</th>
              <th className="px-3 py-2 font-medium">Stat</th>
              <th className="px-3 py-2 text-right font-medium">Avg miss</th>
              <th className="px-3 py-2 text-right font-medium">L10 avg miss</th>
              <th className="px-4 py-2 text-right font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {[...m.history].reverse().map((h) => (
              <tr key={`${h.version}-${h.target}`} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5">{h.trained_at.slice(0, 16).replace("T", " ")}</td>
                <td className="px-3 py-1.5">{STAT_LABEL[h.target]}</td>
                <td className="num px-3 py-1.5 text-right">±{fmt(h.mae, 2)}</td>
                <td className="num px-3 py-1.5 text-right text-text-2">±{fmt(h.baseline_mae, 2)}</td>
                <td className="px-4 py-1.5 text-right text-xs">{h.promoted ? "✓ live" : "✕ not promoted"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
