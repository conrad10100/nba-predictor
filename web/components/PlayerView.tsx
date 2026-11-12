"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { LineChart } from "@/components/LineChart";
import { api, fmt, STAT_LABEL, TARGETS, type PlayerDetail } from "@/lib/api";

export function PlayerView() {
  const { id } = useParams<{ id: string }>();
  const [player, setPlayer] = useState<PlayerDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.player(id).then(setPlayer).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  if (error) return <p className="text-critical">{error}</p>;
  if (!player) return <p className="text-text-2">Loading player…</p>;

  const labels = player.games.map((g) => `${g.date.slice(5).replace("-", "/")} ${g.opp ?? ""}`);
  const graded = player.games.filter((g) => g.predicted).length;

  return (
    <div className="space-y-5">
      <div>
        <Link href="/" className="text-sm text-text-2 hover:underline">← Slate</Link>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{player.name}</h1>
        <p className="text-sm text-text-2">
          {player.team ?? "Free agent"} · {player.position ?? "–"} · last {player.games.length} games
          {graded > 0 ? ` · ${graded} with a pre-game prediction` : " · no tracked predictions yet"}
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {TARGETS.map((t) => (
          <LineChart
            key={t}
            title={STAT_LABEL[t]}
            xLabels={labels}
            series={[
              { key: "actual", label: "Actual", color: "var(--series-1)", values: player.games.map((g) => g[t]), markers: true },
              { key: "pred", label: "Predicted", color: "var(--series-2)", dashed: true, values: player.games.map((g) => g.predicted?.[t]?.predicted ?? null) },
            ]}
            band={{ label: "Predicted range", lo: player.games.map((g) => g.predicted?.[t]?.p10 ?? null), hi: player.games.map((g) => g.predicted?.[t]?.p90 ?? null) }}
            format={(v) => v.toFixed(0)}
          />
        ))}
      </div>

      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Date</th>
              <th className="px-2 py-2 font-medium">Opp</th>
              <th className="px-2 py-2 text-right font-medium">MIN</th>
              {TARGETS.map((t) => (
                <th key={t} className="px-2 py-2 text-right font-medium">{t.toUpperCase()} (pred)</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...player.games].reverse().map((g) => (
              <tr key={g.game_id} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5">{g.date}</td>
                <td className="px-2 py-1.5">{g.opp}</td>
                <td className="num px-2 py-1.5 text-right text-text-2">{fmt(g.min, 0)}</td>
                {TARGETS.map((t) => (
                  <td key={t} className="num px-2 py-1.5 text-right">
                    <span className="font-medium">{g[t]}</span>
                    <span className="ml-1.5 text-xs text-muted">{g.predicted?.[t] ? `(${fmt(g.predicted[t]!.predicted)})` : ""}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
