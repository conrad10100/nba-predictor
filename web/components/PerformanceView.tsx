"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useMemo, useState } from "react";

import { LineChart } from "@/components/LineChart";
import {
  api, fmt, performanceExportUrl, STAT_LABEL, TARGETS,
  type Kind, type Performance, type Pick as PickRow, type PickResult, type Phase, type Record_, type RunningRow, type StatKey, type Target, type Top5Method,
} from "@/lib/api";

// At standard -110 juice you need to win 110/210 of decided picks to break even.
const BREAK_EVEN = 110 / 210;
const PAGE = 200;

const STAT_NAME: Record<StatKey, string> = { all: "All props", ...STAT_LABEL };
const STATS: StatKey[] = ["all", ...TARGETS];

const short = (d: string) => d.slice(5).replace("-", "/");
const recordText = (r: { wins: number; losses: number; pushes: number }) => `${r.wins}–${r.losses}${r.pushes ? `–${r.pushes}` : ""}`;
const oddsText = (o: number | null | undefined) => (o == null ? "–" : o > 0 ? `+${o}` : `${o}`);
const pctText = (v: number | null | undefined, digits = 0) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);

function Units({ value, className = "" }: { value: number | null | undefined; className?: string }) {
  if (value == null) return <span className={`text-muted ${className}`}>–</span>;
  const v = Math.abs(value) < 0.005 ? 0 : value;
  const cls = v > 0 ? "text-good" : v < 0 ? "text-critical" : "text-text-2";
  return <span className={`num ${cls} ${className}`}>{v > 0 ? "+" : ""}{v.toFixed(2)}u</span>;
}

/** Closing line value: did the line move toward the side the model took against the opening line? */
function ClvText({ r }: { r: Pick<Record_, "clv_n" | "clv_avg" | "clv_toward" | "clv_away"> }) {
  if (!r.clv_n || r.clv_avg == null) return <span className="text-muted">–</span>;
  const good = (r.clv_toward ?? 0) > (r.clv_away ?? 0);
  return (
    <span className="num">
      <span className={good ? "text-good" : "text-critical"}>{pctText(r.clv_toward)} toward</span>
      <span className="text-text-2"> / {pctText(r.clv_away)} away · avg {r.clv_avg >= 0 ? "+" : ""}{r.clv_avg.toFixed(2)}</span>
    </span>
  );
}

function WinPct({ value, className = "" }: { value: number | null; className?: string }) {
  if (value == null) return <span className={className}>–</span>;
  return <span className={`${value >= BREAK_EVEN ? "text-good" : "text-critical"} ${className}`}>{(value * 100).toFixed(1)}%</span>;
}

const PHASE: Record<Phase, { title: string; about: string; other: { href: string; label: string }; provisional: string }> = {
  morning: {
    title: "Performance vs. DraftKings: morning picks",
    about: "Picks as they stood after the morning run (about 5am ET), before the evening injury update.",
    other: { href: "/performance/after-injuries", label: "Compare with picks after the 6pm ET injury update →" },
    provisional: "Locked at the morning run. Once a game tips off its picks can't change, and lines DraftKings posts later never count.",
  },
  final: {
    title: "Performance vs. DraftKings: after the injury update",
    about: "Picks as they stood after the 6pm ET injury update, the last update before most tip-offs (earlier games keep their morning picks).",
    other: { href: "/performance", label: "← Compare with the morning picks" },
    provisional: "Provisional until the 6pm ET injury update locks it. Once a game tips off its picks can't change, and lines DraftKings posts later never count.",
  },
};

export function PerformanceView({ phase = "morning" }: { phase?: Phase }) {
  const [kind, setKind] = useState<Kind | null>(null);
  const [data, setData] = useState<Partial<Record<Kind, Performance>>>({});
  const [error, setError] = useState<string | null>(null);

  // Open on live picks once there are any (including tonight's pending top 5); until then the backtest is the only history.
  useEffect(() => {
    api.performance("live", phase)
      .then((live) => {
        setData((d) => ({ ...d, live }));
        const hasLive = live.available.live > 0 || live.top5.gap.picks.length > 0;
        setKind(hasLive || live.available.backtest === 0 ? "live" : "backtest");
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [phase]);

  useEffect(() => {
    if (kind && !data[kind]) {
      api.performance(kind, phase).then((p) => setData((d) => ({ ...d, [kind]: p }))).catch((e) => setError(String(e.message ?? e)));
    }
  }, [kind, data, phase]);

  const perf = kind ? data[kind] : undefined;
  if (error) return <p className="text-critical">{error}</p>;
  if (!kind || !perf) return <p className="text-text-2">Loading performance…</p>;
  const empty = perf.picks.length === 0 && perf.top5.gap.picks.length === 0;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-3xl">
          <h1 className="text-2xl font-semibold tracking-tight">{PHASE[phase].title}</h1>
          <p className="text-sm text-text-2">
            {PHASE[phase].about}{" "}
            <Link href={PHASE[phase].other.href} className="text-accent underline">{PHASE[phase].other.label}</Link>
          </p>
          <p className="mt-1 text-sm text-text-2">
            Every player prop line on DraftKings becomes a pick: <strong>over</strong> when the model predicts more than the line,
            <strong> under</strong> when it predicts less. Each pick risks 1 unit at DraftKings&apos; price for that side; a push returns
            the stake and a player who doesn&apos;t play is void, as the book would settle it.
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
          night&apos;s closing DraftKings line and price. The backtest knows who actually played, which live pre-game predictions
          don&apos;t always know, so expect live results to run a little worse.
        </p>
      )}

      {empty ? (
        <p className="rounded-lg border border-border bg-surface px-4 py-3 text-text-2">
          {kind === "live"
            ? "No live picks yet. Tonight's lines are stored before tip-off and the picks are graded the next morning."
            : "No backtest picks with DraftKings lines yet."}
        </p>
      ) : (
        <>
          <ConfidentSection perf={perf} provisional={PHASE[phase].provisional} />
          {perf.picks.length > 0 && (
            <>
              <section className="space-y-3">
                <h2 className="text-lg font-semibold tracking-tight">Every prop pick</h2>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {STATS.map((s) => perf.totals[s] && <RecordCard key={s} label={STAT_NAME[s]} record={perf.totals[s]!} />)}
                </div>
              </section>
              <RunningTally rows={perf.running} title="Running tally, every prop pick" byStat />
              <div className="grid gap-4 lg:grid-cols-2">
                <BucketTable
                  title="By model confidence"
                  note="does a more confident pick win more often?"
                  first="Confidence"
                  rows={perf.by_confidence}
                  showAvgProb
                />
                <BucketTable title="By edge" note="distance between prediction and line" first="Edge" rows={perf.by_edge} />
              </div>
              <TopPerformers perf={perf} />
              <PicksTable picks={perf.picks} noPick={perf.no_pick} />
            </>
          )}
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

function Tile({ label, children, sub }: { label: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-sm text-text-2">{label}</div>
      <div className="num mt-1 text-3xl font-semibold">{children}</div>
      {sub && <div className="mt-1 text-xs text-text-2">{sub}</div>}
    </div>
  );
}

function RecordCard({ label, record }: { label: string; record: Record_ }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-sm text-text-2">{label}</div>
      <div className="num mt-1 text-3xl font-semibold">{recordText(record)}</div>
      <dl className="num mt-2 grid grid-cols-2 gap-y-0.5 text-sm">
        <dt className="text-text-2">Win rate</dt><dd className="text-right"><WinPct value={record.win_pct} /></dd>
        <dt className="text-text-2">Units</dt><dd className="text-right"><Units value={record.units} /></dd>
        <dt className="text-text-2">ROI</dt><dd className="text-right">{pctText(record.roi, 1)}</dd>
        <dt className="text-text-2">Expected (EV)</dt><dd className="text-right"><Units value={record.expected_units} /></dd>
        <dt className="col-span-2 mt-1 text-xs text-text-2">Line move after open (CLV)</dt>
        <dd className="col-span-2 text-xs"><ClvText r={record} /></dd>
      </dl>
    </div>
  );
}

const METHOD: Record<Top5Method, { label: string; title: string; about: string }> = {
  gap: {
    label: "Biggest gap vs line",
    title: "Top 5 player-points picks: biggest gap from the line",
    about: "Each night, the five player-points props where the model's prediction is furthest from the DraftKings line (e.g. it predicts 30 on a line of 20.5), picking over or under accordingly.",
  },
  confidence: {
    label: "Highest confidence",
    title: "Top 5 player-points picks: highest confidence",
    about: "Each night, the five player-points props where the model gives its pick the best chance to win. Like the gap, but a gap counts for less on a player whose scoring swings a lot.",
  },
};

function ConfidentSection({ perf, provisional }: { perf: Performance; provisional: string }) {
  const [method, setMethod] = useState<Top5Method>("gap");
  const c = perf.top5[method];
  const t = c.totals;
  const nights = useMemo(() => [...new Set(c.picks.map((p) => p.date))].sort().reverse(), [c.picks]);
  const [night, setNight] = useState<string | null>(null);
  const shown = night && nights.includes(night) ? night : nights[0];
  const rows = c.picks.filter((p) => p.date === shown).sort((a, b) => (a.rank ?? 9) - (b.rank ?? 9));
  const dates = c.running.map((r) => r.date);
  const other: Top5Method = method === "gap" ? "confidence" : "gap";
  const o = perf.top5[other].totals;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{METHOD[method].title}</h2>
          <p className="max-w-3xl text-sm text-text-2">
            {METHOD[method].about} Confidence is the model&apos;s chance the pick wins, from its prediction and calibrated 80% range,
            using only what was known before tip-off. EV is its expected profit at the DraftKings price.
          </p>
        </div>
        <Toggle value={method} onChange={setMethod} options={(["gap", "confidence"] as const).map((m) => ({ value: m, label: METHOD[m].label }))} />
      </div>
      {o.picks > 0 && (
        <p className="text-xs text-text-2">
          For comparison, {METHOD[other].label.toLowerCase()}: {recordText(o)} (<WinPct value={o.win_pct} />), <Units value={o.units} />.
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Tile label="Record" sub={t.picks ? `${t.picks} graded picks over ${dates.length} night${dates.length === 1 ? "" : "s"}` : "no graded picks yet"}>
          {recordText(t)}
        </Tile>
        <Tile label="Win rate" sub={`break-even at -110 is ${(BREAK_EVEN * 100).toFixed(1)}%; avg confidence ${pctText(t.avg_prob)}`}>
          <WinPct value={t.win_pct} />
        </Tile>
        <Tile label="Units won" sub={`ROI ${pctText(t.roi, 1)} · 1 unit per pick at DraftKings' price`}>
          <Units value={t.units} />
        </Tile>
        <Tile label="Expected units (EV)" sub={`model's EV per pick: ${t.picks ? `${t.expected_units / t.picks >= 0 ? "+" : ""}${(t.expected_units / t.picks).toFixed(3)}u` : "–"}`}>
          <Units value={t.expected_units} />
        </Tile>
      </div>
      <p className="text-sm text-text-2">
        <span className="font-medium text-text">Closing line value:</span> after the line opened, it moved toward the model&apos;s side vs
        away from it: <ClvText r={t} />. Lines that keep moving toward the model&apos;s side are the best early sign of a real edge,
        long before the win-loss record settles.
      </p>

      {dates.length >= 2 && (
        <LineChart
          title="Cumulative units, top 5 picks"
          xLabels={dates.map(short)}
          series={[
            { key: "units", label: "Actual", color: "var(--series-1)", values: c.running.map((r) => r.units), markers: dates.length < 30 },
            { key: "ev", label: "Expected (EV)", color: "var(--series-2)", dashed: true, values: c.running.map((r) => r.expected_units) },
          ]}
          format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}u`}
        />
      )}

      {nights.length > 0 && (
        <div className="rounded-lg border border-border bg-surface">
          <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2.5">
            <span className="text-sm font-medium">Picks for</span>
            <select value={shown} onChange={(e) => setNight(e.target.value)} className="rounded-md border border-border bg-bg px-2 py-1 text-sm" aria-label="Night">
              {nights.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
            <NightSummary rows={rows} />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] whitespace-nowrap text-sm">
              <thead className="text-left text-xs text-text-2">
                <tr className="border-b border-border">
                  <th className="px-4 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Player</th>
                  <th className="px-3 py-2 text-right font-medium">Model</th>
                  <th className="px-3 py-2 text-right font-medium">Gap</th>
                  <th className="px-3 py-2 font-medium">Pick</th>
                  <th className="px-3 py-2 text-right font-medium">DK price</th>
                  <th className="px-3 py-2 text-right font-medium">Confidence</th>
                  <th className="px-3 py-2 text-right font-medium">DK implied</th>
                  <th className="px-3 py-2 text-right font-medium">EV</th>
                  <th className="px-3 py-2 text-right font-medium">Actual</th>
                  <th className="px-3 py-2 font-medium">Result</th>
                  <th className="px-4 py-2 text-right font-medium">Units</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={`${p.game_id}-${p.player_id}`} className="border-b border-border/60 last:border-0">
                    <td className="num px-4 py-1.5 text-text-2">{p.rank}</td>
                    <td className="px-3 py-1.5"><PlayerCell p={p} /></td>
                    <td className="num px-3 py-1.5 text-right">{fmt(p.predicted)}</td>
                    <td className="num px-3 py-1.5 text-right text-text-2">{p.edge > 0 ? "+" : ""}{fmt(p.edge)}</td>
                    <td className="num px-3 py-1.5 font-medium"><PickText pick={p.pick} line={p.line} /></td>
                    <td className="num px-3 py-1.5 text-right">{oddsText(p.odds)}{p.odds_assumed && <span className="text-muted">*</span>}</td>
                    <td className="num px-3 py-1.5 text-right">{pctText(p.prob)}</td>
                    <td className="num px-3 py-1.5 text-right text-text-2">{pctText(p.market_prob)}</td>
                    <td className="num px-3 py-1.5 text-right"><Units value={p.ev} /></td>
                    <td className="num px-3 py-1.5 text-right font-medium">{p.actual ?? "–"}</td>
                    <td className="px-3 py-1.5"><ResultBadge result={p.result} /></td>
                    <td className="num px-4 py-1.5 text-right"><Units value={p.units} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows.some((p) => p.result === "pending") && (
            <p className="border-t border-border px-4 py-2 text-xs text-text-2">
              {provisional}
            </p>
          )}
          {rows.some((p) => p.odds_assumed) && <p className="border-t border-border px-4 py-2 text-xs text-text-2">* price not posted; graded at -110.</p>}
        </div>
      )}

      {c.running.length > 0 && <RunningTally rows={c.running} title={`Night by night, top 5 by ${METHOD[method].label.toLowerCase()}`} />}
    </section>
  );
}

function NightSummary({ rows }: { rows: PickRow[] }) {
  const n = (r: PickResult) => rows.filter((p) => p.result === r).length;
  const units = rows.reduce((s, p) => s + (p.units ?? 0), 0);
  const pending = n("pending");
  return (
    <span className="num text-sm text-text-2">
      {n("win")}–{n("loss")}{n("push") ? `–${n("push")}` : ""} · <Units value={units} />
      {pending > 0 && ` · ${pending} pending`}
      {n("void") > 0 && ` · ${n("void")} void (didn't play)`}
    </span>
  );
}

function RunningTally({ rows, title, byStat = false }: { rows: RunningRow[]; title: string; byStat?: boolean }) {
  const all = rows.filter((r) => r.stat === "all");
  const dates = all.map((r) => r.date);
  return (
    <section className="space-y-4">
      {byStat && dates.length >= 2 && (
        <LineChart
          title="Running win rate, every prop pick"
          xLabels={dates.map(short)}
          series={[
            { key: "all", label: "Model", color: "var(--series-1)", values: all.map((r) => (r.win_pct == null ? null : r.win_pct * 100)), markers: dates.length < 30 },
            { key: "be", label: "B/E", color: "var(--muted)", dashed: true, values: dates.map(() => BREAK_EVEN * 100) },
          ]}
          format={(v) => `${v.toFixed(0)}%`}
        />
      )}
      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-2.5 text-sm font-medium">{title}</div>
        <div className="max-h-[360px] overflow-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="sticky top-0 bg-surface text-left text-xs text-text-2">
              <tr className="border-b border-border">
                <th className="px-4 py-2 font-medium">Night</th>
                <th className="px-3 py-2 text-right font-medium">That night</th>
                {byStat && TARGETS.map((t) => <th key={t} className="px-3 py-2 text-right font-medium">{STAT_LABEL[t]}</th>)}
                <th className="px-3 py-2 text-right font-medium">Night units</th>
                <th className="px-3 py-2 text-right font-medium">Running record</th>
                <th className="px-3 py-2 text-right font-medium">Win %</th>
                <th className="px-3 py-2 text-right font-medium">Running units</th>
                <th className="px-4 py-2 text-right font-medium">Expected</th>
              </tr>
            </thead>
            <tbody>
              {[...all].reverse().map((r) => (
                <tr key={r.date} className="border-b border-border/60 last:border-0">
                  <td className="px-4 py-1.5">{r.date}</td>
                  <td className="num px-3 py-1.5 text-right">{recordText({ wins: r.day_wins, losses: r.day_losses, pushes: r.day_pushes })}</td>
                  {byStat && TARGETS.map((t) => {
                    const c = rows.find((x) => x.date === r.date && x.stat === t);
                    return (
                      <td key={t} className="num px-3 py-1.5 text-right text-text-2">
                        {c ? recordText({ wins: c.day_wins, losses: c.day_losses, pushes: c.day_pushes }) : "–"}
                      </td>
                    );
                  })}
                  <td className="px-3 py-1.5 text-right"><Units value={r.day_units} /></td>
                  <td className="num px-3 py-1.5 text-right font-medium">{recordText(r)}</td>
                  <td className="num px-3 py-1.5 text-right"><WinPct value={r.win_pct} /></td>
                  <td className="px-3 py-1.5 text-right font-medium"><Units value={r.units} /></td>
                  <td className="px-4 py-1.5 text-right"><Units value={r.expected_units} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function BucketTable({ title, note, first, rows, showAvgProb = false }: {
  title: string; note: string; first: string; rows: ({ label: string } & Record_)[]; showAvgProb?: boolean;
}) {
  return (
    <section className="overflow-x-auto rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2.5 text-sm">
        <span className="font-medium">{title}</span>
        <span className="ml-2 text-text-2">{note}</span>
      </div>
      <table className="w-full min-w-[420px] text-sm">
        <thead className="text-left text-xs text-text-2">
          <tr className="border-b border-border">
            <th className="px-4 py-2 font-medium">{first}</th>
            <th className="px-3 py-2 text-right font-medium">Record</th>
            <th className="px-3 py-2 text-right font-medium">Win %</th>
            {showAvgProb && <th className="px-3 py-2 text-right font-medium">Avg confidence</th>}
            <th className="px-4 py-2 text-right font-medium">Units</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => (
            <tr key={b.label} className="border-b border-border/60 last:border-0">
              <td className="px-4 py-1.5">{b.label}</td>
              <td className="num px-3 py-1.5 text-right">{b.picks ? recordText(b) : "–"}</td>
              <td className="num px-3 py-1.5 text-right"><WinPct value={b.win_pct} /></td>
              {showAvgProb && <td className="num px-3 py-1.5 text-right text-text-2">{pctText(b.avg_prob)}</td>}
              <td className="px-4 py-1.5 text-right">{b.picks ? <Units value={b.units} /> : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ResultBadge({ result }: { result: PickResult | null }) {
  if (!result) return <span className="text-muted">–</span>;
  const cls = result === "win" ? "bg-good/15 text-good" : result === "loss" ? "bg-critical/15 text-critical" : "bg-surface-2 text-text-2";
  const label = result === "win" || result === "loss" || result === "push" ? result.toUpperCase() : result;
  return <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${cls}`}>{label}</span>;
}

function PickText({ pick, line }: { pick: "over" | "under" | null; line: number | null }) {
  if (!pick || line == null) return <span className="text-muted">–</span>;
  return <>{pick === "over" ? "Over" : "Under"} {fmt(line)}</>;
}

function PlayerCell({ p }: { p: { player_id: number; player: string | null; team: string | null; opp: string | null } }) {
  return (
    <>
      <Link href={`/player/${p.player_id}`} className="hover:underline">{p.player ?? p.player_id}</Link>
      <span className="ml-1.5 text-xs text-muted">{p.team}{p.opp ? ` vs ${p.opp}` : ""}</span>
    </>
  );
}

function TopPerformers({ perf }: { perf: Performance }) {
  const dates = useMemo(() => [...new Set(perf.top_performers.map((r) => r.date))].sort().reverse(), [perf]);
  const [date, setDate] = useState<string | null>(null);
  const [stat, setStat] = useState<Target>("pts");
  const shown = date && dates.includes(date) ? date : dates[0];
  const rows = perf.top_performers.filter((r) => r.date === shown && r.stat === stat);
  const inTop5 = rows.filter((r) => r.model_rank != null && r.model_rank <= 5).length;
  const count = (r: PickResult) => rows.filter((x) => x.result === r).length;

  if (!dates.length) return null;
  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-2.5">
        <span className="text-sm font-medium">Top 5 performers of the night</span>
        <select value={shown} onChange={(e) => setDate(e.target.value)} className="rounded-md border border-border bg-bg px-2 py-1 text-sm" aria-label="Night">
          {dates.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <Toggle value={stat} onChange={setStat} options={TARGETS.map((t) => ({ value: t, label: STAT_LABEL[t] }))} />
      </div>
      <p className="px-4 pt-3 text-sm text-text-2">
        The night&apos;s actual top 5 in {STAT_LABEL[stat].toLowerCase()}. The model had <strong className="text-text">{inTop5} of 5</strong> in
        its own projected top 5; against the line it went{" "}
        <strong className="text-text">{count("win")}–{count("loss")}{count("push") ? `–${count("push")}` : ""}</strong>.
      </p>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[640px] text-sm">
          <thead className="text-left text-xs text-text-2">
            <tr className="border-y border-border">
              <th className="px-4 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Player</th>
              <th className="px-3 py-2 text-right font-medium">Actual</th>
              <th className="px-3 py-2 text-right font-medium">Model</th>
              <th className="px-3 py-2 text-right font-medium">Model rank</th>
              <th className="px-3 py-2 font-medium">Model pick</th>
              <th className="px-4 py-2 font-medium">Result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.player_id} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5 text-text-2">{r.rank}</td>
                <td className="px-3 py-1.5"><PlayerCell p={{ ...r, opp: null }} /></td>
                <td className="num px-3 py-1.5 text-right font-medium">{r.actual}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(r.predicted)}</td>
                <td className="num px-3 py-1.5 text-right">{r.model_rank == null ? "–" : `#${r.model_rank}`}</td>
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

function PicksTable({ picks, noPick }: { picks: PickRow[]; noPick: number }) {
  const dates = useMemo(() => [...new Set(picks.map((p) => p.date))].sort().reverse(), [picks]);
  const [date, setDate] = useState("all");
  const [stat, setStat] = useState<StatKey>("all");
  const [result, setResult] = useState<"all" | PickResult>("all");
  const [top, setTop] = useState<"all" | Top5Method>("all");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(PAGE);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return picks
      .filter((p) => (date === "all" || p.date === date) && (stat === "all" || p.stat === stat) && (result === "all" || p.result === result)
        && (top === "all" || (top === "gap" ? p.top5_gap : p.top5)) && (!q || (p.player ?? "").toLowerCase().includes(q) || (p.team ?? "").toLowerCase() === q))
      .sort((a, b) => b.date.localeCompare(a.date) || (a.player ?? "").localeCompare(b.player ?? "") || a.stat.localeCompare(b.stat));
  }, [picks, date, stat, result, top, query]);
  const count = (r: PickResult) => filtered.filter((p) => p.result === r).length;
  const units = filtered.reduce((s, p) => s + (p.units ?? 0), 0);
  const decided = count("win") + count("loss");
  const reset = () => setLimit(PAGE);

  const select = "rounded-md border border-border bg-bg px-2 py-1 text-sm";
  return (
    <section className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        <span className="mr-1 text-sm font-medium">Every pick</span>
        <select value={date} onChange={(e) => { setDate(e.target.value); reset(); }} className={select} aria-label="Filter night">
          <option value="all">All nights</option>
          {dates.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <select value={stat} onChange={(e) => { setStat(e.target.value as StatKey); reset(); }} className={select} aria-label="Stat">
          {STATS.map((s) => <option key={s} value={s}>{s === "all" ? "All stats" : STAT_LABEL[s]}</option>)}
        </select>
        <select value={result} onChange={(e) => { setResult(e.target.value as "all" | PickResult); reset(); }} className={select} aria-label="Result">
          <option value="all">All results</option>
          <option value="win">Wins</option>
          <option value="loss">Losses</option>
          <option value="push">Pushes</option>
          <option value="void">Void</option>
          <option value="no pick">No pick</option>
        </select>
        <select value={top} onChange={(e) => { setTop(e.target.value as "all" | Top5Method); reset(); }} className={select} aria-label="Top 5">
          <option value="all">All picks</option>
          <option value="gap">Top 5: biggest gap</option>
          <option value="confidence">Top 5: highest confidence</option>
        </select>
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); reset(); }}
          placeholder="Player or team"
          className={`${select} w-40`}
          aria-label="Search player or team"
        />
        <span className="num ml-auto text-sm text-text-2">
          {filtered.length} picks · {count("win")}–{count("loss")}{count("push") ? `–${count("push")}` : ""}
          {decided > 0 && <> · <WinPct value={count("win") / decided} /></>} · <Units value={units} />
        </span>
      </div>
      <div className="max-h-[640px] overflow-auto">
        <table className="w-full min-w-[1120px] whitespace-nowrap text-sm">
          <thead className="sticky top-0 bg-surface text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Night</th>
              <th className="px-3 py-2 font-medium">Player</th>
              <th className="px-3 py-2 font-medium">Stat</th>
              <th className="px-3 py-2 text-right font-medium">DK line</th>
              <th className="px-3 py-2 text-right font-medium" title="How far the line moved after opening, toward (+) or away from (−) the side the model took against the opening line">Line move</th>
              <th className="px-3 py-2 text-right font-medium">Over / under</th>
              <th className="px-3 py-2 text-right font-medium">Model</th>
              <th className="px-3 py-2 font-medium">Pick</th>
              <th className="px-3 py-2 text-right font-medium">Confidence</th>
              <th className="px-3 py-2 text-right font-medium">EV</th>
              <th className="px-3 py-2 text-right font-medium">Actual</th>
              <th className="px-3 py-2 font-medium">Result</th>
              <th className="px-4 py-2 text-right font-medium">Units</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, limit).map((p) => (
              <tr key={`${p.game_id}-${p.player_id}-${p.stat}`} className="border-b border-border/60 last:border-0">
                <td className="num px-4 py-1.5 text-text-2">{short(p.date)}</td>
                <td className="px-3 py-1.5">
                  <PlayerCell p={p} />
                  {(p.top5_gap || p.top5) && <span className="ml-1.5 rounded bg-accent/15 px-1 text-xs text-accent">top 5{p.top5_gap && p.top5 ? "" : p.top5_gap ? " gap" : " conf"}</span>}
                </td>
                <td className="px-3 py-1.5">{STAT_LABEL[p.stat]}</td>
                <td className="num px-3 py-1.5 text-right">
                  {fmt(p.line)}
                  {p.open_line != null && p.open_line !== p.line && <span className="ml-1 text-xs text-muted">(opened {fmt(p.open_line)})</span>}
                </td>
                <td className={`num px-3 py-1.5 text-right ${p.clv == null || p.clv === 0 ? "text-text-2" : p.clv > 0 ? "text-good" : "text-critical"}`}>
                  {p.clv == null ? "–" : p.clv === 0 ? "0" : `${p.clv > 0 ? "+" : ""}${fmt(p.clv)}`}
                </td>
                <td className="num px-3 py-1.5 text-right text-text-2">{oddsText(p.over_odds)} / {oddsText(p.under_odds)}</td>
                <td className="num px-3 py-1.5 text-right">{fmt(p.predicted)}</td>
                <td className="num px-3 py-1.5"><PickText pick={p.pick} line={p.line} /></td>
                <td className="num px-3 py-1.5 text-right">{pctText(p.prob)}</td>
                <td className="num px-3 py-1.5 text-right"><Units value={p.ev} /></td>
                <td className="num px-3 py-1.5 text-right font-medium">{p.actual ?? "–"}</td>
                <td className="px-3 py-1.5"><ResultBadge result={p.result} /></td>
                <td className="num px-4 py-1.5 text-right"><Units value={p.units} /></td>
              </tr>
            ))}
            {filtered.length === 0 && <tr><td colSpan={13} className="px-4 py-3 text-text-2">No picks match these filters.</td></tr>}
          </tbody>
        </table>
      </div>
      {filtered.length > limit && (
        <button type="button" onClick={() => setLimit((l) => l + PAGE)} className="w-full border-t border-border px-4 py-2 text-sm text-accent hover:bg-surface-2">
          Show {Math.min(PAGE, filtered.length - limit)} more of {filtered.length - limit} remaining
        </button>
      )}
      {noPick > 0 && (
        <p className="border-t border-border px-4 py-2 text-xs text-text-2">
          {noPick} prediction{noPick === 1 ? "" : "s"} landed exactly on the line and made no pick.
        </p>
      )}
    </section>
  );
}
