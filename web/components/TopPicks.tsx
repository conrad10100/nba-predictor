"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { api, fmt, STAT_LABEL, TARGETS, type Pick, type Target } from "@/lib/api";

const oddsText = (o: number | null) => (o == null ? "" : o > 0 ? `+${o}` : `${o}`);

/** The night's top 5 over/under picks per stat: biggest gap between the model and the DraftKings line. */
export function TopPicks({ date }: { date: string }) {
  const [loaded, setLoaded] = useState<{ date: string; picks: Record<Target, Pick[]> } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    api.picks(date)
      .then((r) => { if (live) { setLoaded(r); setFailed(false); } })
      .catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  }, [date]);

  if (failed || !loaded || loaded.date !== date) return null;
  if (TARGETS.every((t) => loaded.picks[t].length === 0)) {
    return (
      <p className="rounded-lg border border-border bg-surface px-4 py-3 text-sm text-text-2">
        No DraftKings player lines for this date yet. They&apos;re stored on the morning and 5pm runs.
      </p>
    );
  }

  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="text-sm font-medium">Top picks vs. DraftKings</h2>
        <p className="text-xs text-text-2">
          Biggest gap between the model and the line · <Link href="/performance" className="underline">track record</Link>
        </p>
      </div>
      <div className="grid divide-y divide-border md:grid-cols-3 md:divide-x md:divide-y-0">
        {TARGETS.map((t) => (
          <div key={t} className="px-4 py-3">
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-text-2">{STAT_LABEL[t]}</h3>
            {loaded.picks[t].length === 0 ? (
              <p className="text-sm text-muted">No lines yet</p>
            ) : (
              <ol className="space-y-2">
                {loaded.picks[t].map((p, i) => <PickRow key={`${p.game_id}-${p.player_id}`} p={p} rank={i + 1} />)}
              </ol>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function PickRow({ p, rank }: { p: Pick; rank: number }) {
  const graded = p.result === "win" || p.result === "loss" || p.result === "push";
  const badge = p.result === "win" ? "bg-good/15 text-good" : p.result === "loss" ? "bg-critical/15 text-critical" : "bg-surface-2 text-text-2";
  return (
    <li className="flex items-start gap-2 text-sm">
      <span className="num w-4 shrink-0 text-text-2">{rank}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <Link href={`/player/${p.player_id}`} className="truncate font-medium hover:underline">{p.player ?? p.player_id}</Link>
          <span className="num shrink-0 font-semibold">
            {p.pick === "over" ? "Over" : "Under"} {fmt(p.line)}
            {graded && <span className={`ml-1.5 rounded px-1 py-0.5 text-xs font-medium ${badge}`}>{p.result.toUpperCase()}</span>}
          </span>
        </div>
        <div className="num flex justify-between gap-2 text-xs text-text-2">
          <span className="truncate">{p.team}{p.opp ? ` vs ${p.opp}` : ""}</span>
          <span className="shrink-0">
            model {fmt(p.predicted)} ({p.edge > 0 ? "+" : ""}{fmt(p.edge)}) · {p.prob == null ? "" : `${Math.round(p.prob * 100)}%`} · {oddsText(p.odds_assumed ? null : p.odds)}
            {graded && p.actual != null && <> · actual {p.actual}</>}
          </span>
        </div>
      </div>
    </li>
  );
}
