"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { LineChart } from "@/components/LineChart";
import {
  api, fmt, performanceExportUrl, STAT_LABEL, TARGETS,
  type Kind, type Performance, type Pick as PickRow, type PickResult, type Record_, type StatKey, type Target,
} from "@/lib/api";

// At standard -110 juice you need to win 110/210 of decided picks to break even.
const BREAK_EVEN = 110 / 210;
const PAGE = 200;

const STAT_NAME: Record<StatKey, string> = { all: "All picks", ...STAT_LABEL };
const STATS: StatKey[] = ["all", ...TARGETS];

const short = (d: string) => d.slice(5).replace("-", "/");
const recordText = (r: Pick<Record_, "wins" | "losses" | "pushes">) => `${r.wins}–${r.losses}${r.pushes ? `–${r.pushes}` : ""}`;

export function PerformanceView() {
  const [kind, setKind] = useState<Kind | null>(null);
  const [data, setData] = useState<Partial<Record<Kind, Performance>>>({});
  const [error, setError] = useState<string | null>(null);

  // Open on live picks once there are any; until then the backtest is the only history.
  useEffect(() => {
    api.performance("live")
      .then((live) => {
        setData((d) => ({ ...d, live }));
        setKind(live.available.live > 0 || live.available.backtest === 0 ? "live" : "backtest");
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (kind && !data[kind]) {
      api.performance(kind).then((p) => setData((d) => ({ ...d, [kind]: p }))).catch((e) => setError(String(e.message ?? e)));
    }
  }, [kind, data]);

  const perf = kind ? data[kind] : undefined;
  if (error) return <p className="text-critical">{error}</p>;
  if (!kind || !perf) return <p className="text-text-2">Loading performance…</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-3xl">
          <h1 className="text-2xl font-semibold tracking-tight">Performance vs. DraftKings lines</h1>
          <p className="text-sm text-text-2">
            For every player with a DraftKings points, rebounds or assists line, the model&apos;s pick is <strong>over</strong> when
            it predicts more than the line and <strong>under</strong> when it predicts less. The pick wins when the player finishes on
            that side of the line; finishing exactly on it is a push. Lines are the last pre-game line.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Toggle
            value={kind}
            onChange={setKind}
            options={[
              { value: "live", label: `Live (${perf.available.live})` },
              { value: "backtest", label: `Backtest (${perf.available.backtest})` },
            ]}
          />
          <a href={performanceExportUrl} className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm hover:bg-surface-2">
            Export CSV
          </a>
        </div>
      </div>

      {kind === "backtest" && (
        <p className="max-w-3xl text-xs text-text-2">
          <strong>Backtest:</strong> each week was predicted by a model trained only on games before it, and graded against that
          night&apos;s closing DraftKings line. The backtest knows who actually played, which live pre-game predictions don&apos;t
          always know, so expect live results to run a little worse.
        </p>
      )}

      {perf.picks.length === 0 ? (
        <p className="rounded-lg border border-border bg-surface px-4 py-3 text-text-2">
          {kind === "live"
            ? "No live picks graded yet. Tonight's predictions are graded tomorrow morning once the box scores are in."
            : "No backtest picks with DraftKings lines yet."}
        </p>
      ) : (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {STATS.map((s) => perf.totals[s] && <RecordCard key={s} stat={s} record={perf.totals[s]!} />)}
          </section>
          <RunningTally perf={perf} />
          <EdgeTable perf={perf} />
          <TopPerformers perf={perf} />
          <PicksTable picks={perf.picks} />
        </>
      )}
    </div>
  );
}

function Toggle<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-md border border-border bg-surface p-0.5 text-sm">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded px-2.5 py-1 ${value === o.value ? "bg-surface-2 font-medium text-text" : "text-text-2 hover:text-text"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function WinPct({ value, className = "" }: { value: number | null; className?: string }) {
  if (value == null) return <span className={className}>–</span>;
  return <span className={`${value >= BREAK_EVEN ? "text-good" : "text-critical"} ${className}`}>{(value * 100).toFixed(1)}%</span>;
}

function RecordCard({ stat, record }: { stat: StatKey; record: Record_ }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-sm text-text-2">{STAT_NAME[stat]}</div>
      <div className="num mt-1 text-3xl font-semibold">{recordText(record)}</div>
      <div className="num text-sm">
        <WinPct value={record.win_pct} className="font-medium" /> <span className="text-text-2">win rate</span>
      </div>
      <div className="mt-1 text-xs text-muted">
        W–L{record.pushes ? "–push" : ""} · break-even at -110 is {(BREAK_EVEN * 100).toFixed(1)}%
      </div>
    </div>
  );
}

function RunningTally({ perf }: { perf: Performance }) {
  const dates = useMemo(() => [...new Set(perf.running.map((r) => r.date))].sort(), [perf]);
  const at = (stat: StatKey) => dates.map((d) => perf.running.find((r) => r.date === d && r.stat === stat) ?? null);
  const all = at("all");

  return (
    <section className="space-y-4">
      {dates.length >= 2 && (
        <LineChart
          title="Running win rate, all picks (cumulative)"
          xLabels={dates.map(short)}
          series={[
            { key: "all", label: "Model", color: "var(--series-1)", values: all.map((r) => (r?.win_pct == null ? null : r.win_pct * 100)),
              markers: dates.length < 30 },
            { key: "be", label: "Break-even", color: "var(--muted)", dashed: true, values: dates.map(() => BREAK_EVEN * 100) },
          ]}
          format={(v) => `${v.toFixed(0)}%`}
        />
      )}
      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-2.5 text-sm font-medium">Running tally by night</div>
        <div className="max-h-[360px] overflow-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="sticky top-0 bg-surface text-left text-xs text-text-2">
              <tr className="border-b border-border">
                <th className="px-4 py-2 font-medium">Night</th>
                <th className="px-3 py-2 text-right font-medium">That night</th>
                {TARGETS.map((t) => <th key={t} className="px-3 py-2 text-right font-medium">{STAT_LABEL[t]} (night)</th>)}
                <th className="px-3 py-2 text-right font-medium">Season record</th>
                <th className="px-4 py-2 text-right font-medium">Season win %</th>
              </tr>
            </thead>
            <tbody>
              {[...dates].reverse().map((d) => {
                const r = all[dates.indexOf(d)]!;
                return (
                  <tr key={d} className="border-b border-border/60 last:border-0">
                    <td className="px-4 py-1.5">{d}</td>
                    <td className="num px-3 py-1.5 text-right">{recordText({ wins: r.day_wins, losses: r.day_losses, pushes: r.day_pushes })}</td>
                    {TARGETS.map((t) => {
                      const c = perf.running.find((x) => x.date === d && x.stat === t);
                      return (
                        <td key={t} className="num px-3 py-1.5 text-right text-text-2">
                          {c ? recordText({ wins: c.day_wins, losses: c.day_losses, pushes: c.day_pushes }) : "–"}
                        </td>
                      );
                    })}
                    <td className="num px-3 py-1.5 text-right font-medium">{recordText(r)}</td>
                    <td className="num px-4 py-1.5 text-right"><WinPct value={r.win_pct} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function EdgeTable({ perf }: { perf: Performance }) {
  return (
    <section className="overflow-x-auto rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2.5 text-sm">
        <span className="font-medium">Win rate by edge</span>
        <span className="ml-2 text-text-2">how far the prediction was from the line, in points / rebounds / assists</span>
      </div>
      <table className="w-full min-w-[420px] text-sm">
        <thead className="text-left text-xs text-text-2">
          <tr className="border-b border-border">
            <th className="px-4 py-2 font-medium">Edge</th>
            <th className="px-3 py-2 text-right font-medium">Record</th>
            <th className="px-4 py-2 text-right font-medium">Win %</th>
          </tr>
        </thead>
        <tbody>
          {perf.by_edge.map((b) => (
            <tr key={b.label} className="border-b border-border/60 last:border-0">
              <td className="px-4 py-1.5">{b.label}</td>
              <td className="num px-3 py-1.5 text-right">{b.picks ? recordText(b) : "–"}</td>
              <td className="num px-4 py-1.5 text-right"><WinPct value={b.win_pct} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {perf.no_pick > 0 && (
        <p className="border-t border-border px-4 py-2 text-xs text-text-2">
          {perf.no_pick} prediction{perf.no_pick === 1 ? "" : "s"} landed exactly on the line and made no pick.
        </p>
      )}
    </section>
  );
}

function ResultBadge({ result }: { result: PickResult | null }) {
  if (!result) return <span className="text-muted">–</span>;
  const cls = result === "win" ? "bg-good/15 text-good" : result === "loss" ? "bg-critical/15 text-critical" : "bg-surface-2 text-text-2";
  return <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${cls}`}>{result === "no pick" ? "no pick" : result.toUpperCase()}</span>;
}

function PickText({ pick, line }: { pick: "over" | "under" | null; line: number | null }) {
  if (!pick || line == null) return <span className="text-muted">–</span>;
  return <>{pick === "over" ? "Over" : "Under"} {fmt(line)}</>;
}

function TopPerformers({ perf }: { perf: Performance }) {
  const dates = useMemo(() => [...new Set(perf.top_performers.map((r) => r.date))].sort().reverse(), [perf]);
  const [date, setDate] = useState<string | null>(null);
  const [stat, setStat] = useState<Target>("pts");
  const shown = date && dates.includes(date) ? date : dates[0];
  const rows = perf.top_performers.filter((r) => r.date === shown && r.stat === stat);
  const inTop5 = rows.filter((r) => r.model_rank != null && r.model_rank <= 5).length;
  const night = rows.reduce((acc, r) => ({ ...acc, [r.result ?? "none"]: (acc[r.result ?? "none"] ?? 0) + 1 }), {} as Record<string, number>);
  const summary = perf.top_summary;

  if (!dates.length) return null;
  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2.5">
        <span className="text-sm font-medium">Top 5 performers of the night</span>
        <select value={shown} onChange={(e) => setDate(e.target.value)} className="rounded-md border border-border bg-bg px-2 py-1 text-sm">
          {dates.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <Toggle value={stat} onChange={setStat} options={TARGETS.map((t) => ({ value: t, label: STAT_LABEL[t] }))} />
      </div>
      <p className="px-4 pt-3 text-sm text-text-2">
        The night&apos;s actual top 5 in {STAT_LABEL[stat].toLowerCase()}. The model had <strong className="text-text">{inTop5} of 5</strong> in
        its own projected top 5; against the line it went{" "}
        <strong className="text-text">{night.win ?? 0}–{night.loss ?? 0}{night.push ? `–${night.push}` : ""}</strong>
        {rows.length - (night.win ?? 0) - (night.loss ?? 0) - (night.push ?? 0) > 0 && " (players without a line aren't counted)"}.
      </p>
      {summary && (
        <p className="px-4 pt-1 text-xs text-text-2">
          Across {summary.nights} night{summary.nights === 1 ? "" : "s"}: on top-5 performers the model is{" "}
          {STATS.filter((s) => summary.record[s]?.picks).map((s, i) => (
            <span key={s}>
              {i > 0 && " · "}
              {STAT_NAME[s].toLowerCase()} {recordText(summary.record[s]!)} (<WinPct value={summary.record[s]!.win_pct} />)
            </span>
          ))}
          . On average it also had{" "}
          {TARGETS.filter((t) => summary.in_model_top5[t] != null).map((t) => `${summary.in_model_top5[t]!.toFixed(1)} (${t.toUpperCase()})`).join(", ")}{" "}
          of the 5 in its own top 5.
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[640px] text-sm">
          <thead className="text-left text-xs text-text-2">
            <tr className="border-y border-border">
              <th className="px-4 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Player</th>
              <th className="px-3 py-2 text-right font-medium">Actual</th>
              <th className="px-3 py-2 text-right font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">Model rank</th>
              <th className="px-3 py-2 text-right font-medium">Line</th>
              <th className="px-3 py-2 font-medium">Model pick</th>
              <th className="px-4 py-2 font-medium">Result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.player_id} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5 text-text-2">{r.rank}</td>
                <td className="px-3 py-1.5">
                  <Link href={`/player/${r.player_id}`} className="hover:underline">{r.player ?? r.player_id}</Link>
                  <span className="ml-1.5 text-xs text-muted">{r.team}</span>
                </td>
                <td className="num px-3 py-1.5 text-right font-medium">{r.actual}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(r.predicted)}</td>
                <td className="num px-3 py-1.5 text-right">{r.model_rank == null ? "–" : `#${r.model_rank}`}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(r.line)}</td>
                <td className="num px-3 py-1.5"><PickText pick={r.pick} line={r.line} /></td>
                <td className="px-4 py-1.5"><ResultBadge result={r.result} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PicksTable({ picks }: { picks: PickRow[] }) {
  const dates = useMemo(() => [...new Set(picks.map((p) => p.date))].sort().reverse(), [picks]);
  const [date, setDate] = useState("all");
  const [stat, setStat] = useState<StatKey>("all");
  const [result, setResult] = useState<"all" | PickResult>("all");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(PAGE);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return picks
      .filter((p) => (date === "all" || p.date === date) && (stat === "all" || p.stat === stat) && (result === "all" || p.result === result)
        && (!q || (p.player ?? "").toLowerCase().includes(q) || (p.team ?? "").toLowerCase() === q))
      .sort((a, b) => b.date.localeCompare(a.date) || (a.player ?? "").localeCompare(b.player ?? "") || a.stat.localeCompare(b.stat));
  }, [picks, date, stat, result, query]);
  const rec = filtered.reduce((r, p) => ({ ...r, [p.result]: (r[p.result] ?? 0) + 1 }), {} as Partial<Record<PickResult, number>>);
  const decided = (rec.win ?? 0) + (rec.loss ?? 0);

  const select = "rounded-md border border-border bg-bg px-2 py-1 text-sm";
  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        <span className="mr-1 text-sm font-medium">Every pick</span>
        <select value={date} onChange={(e) => { setDate(e.target.value); setLimit(PAGE); }} className={select} aria-label="Night">
          <option value="all">All nights</option>
          {dates.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <select value={stat} onChange={(e) => { setStat(e.target.value as StatKey); setLimit(PAGE); }} className={select} aria-label="Stat">
          {STATS.map((s) => <option key={s} value={s}>{s === "all" ? "All stats" : STAT_LABEL[s]}</option>)}
        </select>
        <select value={result} onChange={(e) => { setResult(e.target.value as "all" | PickResult); setLimit(PAGE); }} className={select} aria-label="Result">
          <option value="all">All results</option>
          <option value="win">Wins</option>
          <option value="loss">Losses</option>
          <option value="push">Pushes</option>
          <option value="no pick">No pick</option>
        </select>
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setLimit(PAGE); }}
          placeholder="Player or team"
          className={`${select} w-40`}
          aria-label="Search player or team"
        />
        <span className="num ml-auto text-sm text-text-2">
          {filtered.length} picks · {rec.win ?? 0}–{rec.loss ?? 0}{rec.push ? `–${rec.push}` : ""}
          {decided > 0 && <> · <WinPct value={(rec.win ?? 0) / decided} /></>}
        </span>
      </div>
      <div className="max-h-[640px] overflow-auto">
        <table className="w-full min-w-[820px] text-sm">
          <thead className="sticky top-0 bg-surface text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Night</th>
              <th className="px-3 py-2 font-medium">Player</th>
              <th className="px-3 py-2 font-medium">Stat</th>
              <th className="px-3 py-2 text-right font-medium">DK line</th>
              <th className="px-3 py-2 text-right font-medium">Opened</th>
              <th className="px-3 py-2 text-right font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">Edge</th>
              <th className="px-3 py-2 font-medium">Pick</th>
              <th className="px-3 py-2 text-right font-medium">Actual</th>
              <th className="px-4 py-2 font-medium">Result</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, limit).map((p) => (
              <tr key={`${p.game_id}-${p.player_id}-${p.stat}`} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5 text-text-2">{short(p.date)}</td>
                <td className="px-3 py-1.5">
                  <Link href={`/player/${p.player_id}`} className="hover:underline">{p.player ?? p.player_id}</Link>
                  <span className="ml-1.5 text-xs text-muted">{p.team}{p.opp ? ` vs ${p.opp}` : ""}</span>
                </td>
                <td className="px-3 py-1.5">{STAT_LABEL[p.stat]}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(p.line)}</td>
                <td className="num px-3 py-1.5 text-right text-text-2">{fmt(p.open_line)}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(p.predicted)}</td>
                <td className="num px-3 py-1.5 text-right text-text-2">{p.edge > 0 ? "+" : ""}{fmt(p.edge)}</td>
                <td className="num px-3 py-1.5"><PickText pick={p.pick} line={p.line} /></td>
                <td className="num px-3 py-1.5 text-right font-medium">{p.actual}</td>
                <td className="px-4 py-1.5"><ResultBadge result={p.result} /></td>
              </tr>
            ))}
            {filtered.length === 0 && <tr><td colSpan={10} className="px-4 py-3 text-text-2">No picks match these filters.</td></tr>}
          </tbody>
        </table>
      </div>
      {filtered.length > limit && (
        <button type="button" onClick={() => setLimit((l) => l + PAGE)} className="w-full border-t border-border px-4 py-2 text-sm text-accent hover:bg-surface-2">
          Show {Math.min(PAGE, filtered.length - limit)} more of {filtered.length - limit} remaining
        </button>
      )}
    </section>
  );
}

