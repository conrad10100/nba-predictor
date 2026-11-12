"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { api, fmt, type PlayerPrediction, type Slate, type SlateGame, type SlateOverrides, type TeamSide } from "@/lib/api";

type SortKey = "pts" | "reb" | "ast" | "pra";
type ConferenceFilter = "all" | "same" | "cross";

const EMPTY: SlateOverrides = { out: [], available: [], b2b: {} };
const hasOverrides = (o: SlateOverrides) => o.out.length > 0 || o.available.length > 0 || Object.keys(o.b2b).length > 0;

export function SlateView() {
  const [dates, setDates] = useState<{ date: string; games: number }[]>([]);
  const [date, setDate] = useState<string | null>(null);
  const [base, setBase] = useState<Slate | null>(null);
  const [slate, setSlate] = useState<Slate | null>(null);
  const [overrides, setOverrides] = useState<SlateOverrides>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [conference, setConference] = useState<ConferenceFilter>("all");
  const [b2bOnly, setB2bOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("pts");
  const [minMinutes, setMinMinutes] = useState(15);

  const loadDate = useCallback((d: string) => {
    setDate(d);
    setLoading(true);
    setOverrides(EMPTY);
    api.slate(d)
      .then((s) => { setBase(s); setSlate(s); setError(null); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    api.dates()
      .then((d) => {
        setDates(d);
        loadDate(d[0]?.date ?? new Date().toISOString().slice(0, 10));
      })
      .catch((e) => { setError(String(e.message ?? e)); setLoading(false); });
  }, [loadDate]);

  const apply = useCallback((next: SlateOverrides) => {
    if (!date) return;
    setOverrides(next);
    if (!hasOverrides(next)) { setSlate(base); return; }
    setLoading(true);
    api.whatIf(date, next)
      .then((s) => { setSlate(s); setError(null); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [date, base]);

  const markOut = (id: number) =>
    apply({ ...overrides, out: [...overrides.out, id], available: overrides.available.filter((p) => p !== id) });
  const markAvailable = (id: number) =>
    apply({ ...overrides, out: overrides.out.filter((p) => p !== id), available: overrides.out.includes(id) ? overrides.available : [...overrides.available, id] });
  const toggleB2b = (team: TeamSide) => {
    const b2b = { ...overrides.b2b };
    if (team.b2b_overridden) delete b2b[team.id];
    else b2b[team.id] = !team.b2b;
    apply({ ...overrides, b2b });
  };

  const baseById = useMemo(() => new Map(base?.predictions.map((p) => [`${p.game_id}:${p.player_id}`, p]) ?? []), [base]);

  const games = useMemo(() => {
    if (!slate) return [];
    return slate.games.filter((g) => {
      if (conference === "same" && !g.same_conference) return false;
      if (conference === "cross" && g.same_conference) return false;
      if (b2bOnly && !g.home.b2b && !g.away.b2b) return false;
      return true;
    });
  }, [slate, conference, b2bOnly]);

  const playersByGame = useMemo(() => {
    const q = search.trim().toLowerCase();
    const map = new Map<string, PlayerPrediction[]>();
    for (const p of slate?.predictions ?? []) {
      if ((p.min_l10 ?? 0) < minMinutes) continue;
      if (q && !p.name.toLowerCase().includes(q)) continue;
      const list = map.get(p.game_id) ?? [];
      list.push(p);
      map.set(p.game_id, list);
    }
    const key = (p: PlayerPrediction) => (sortBy === "pra" ? p.pra_pred : p[`${sortBy}_pred`]) ?? 0;
    for (const list of map.values()) list.sort((a, b) => key(b) - key(a));
    return map;
  }, [slate, search, sortBy, minMinutes]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Player projections</h1>
          <p className="text-sm text-text-2">
            {slate ? `${slate.games.length} games` : "…"}
            {slate?.model_version && <> · model {slate.model_version}</>}
            {hasOverrides(overrides) && (
              <>
                {" · "}
                <span className="font-medium text-accent">what-if active</span>{" "}
                <button className="underline" onClick={() => apply(EMPTY)}>reset</button>
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap gap-1">
          {dates.map((d) => (
            <button
              key={d.date}
              onClick={() => loadDate(d.date)}
              className={`rounded-md border px-3 py-1.5 text-sm ${d.date === date ? "border-accent bg-accent text-white" : "border-border bg-surface hover:bg-surface-2"}`}
            >
              {new Date(`${d.date}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}
              <span className="ml-1 opacity-70">({d.games})</span>
            </button>
          ))}
        </div>
      </div>

      <Filters {...{ conference, setConference, b2bOnly, setB2bOnly, search, setSearch, sortBy, setSortBy, minMinutes, setMinMinutes }} />

      {error && <p className="rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-sm">{error}</p>}
      {loading && <p className="text-sm text-text-2">Updating projections…</p>}
      {slate && games.length === 0 && !loading && <p className="text-text-2">No games match these filters.</p>}

      <div className={`space-y-5 transition-opacity ${loading ? "opacity-60" : ""}`}>
        {games.map((g) => (
          <GameCard
            key={g.game_id}
            game={g}
            players={playersByGame.get(g.game_id) ?? []}
            baseById={hasOverrides(overrides) ? baseById : null}
            onOut={markOut}
            onAvailable={markAvailable}
            onToggleB2b={toggleB2b}
          />
        ))}
      </div>
    </div>
  );
}

function Filters(p: {
  conference: ConferenceFilter; setConference: (v: ConferenceFilter) => void;
  b2bOnly: boolean; setB2bOnly: (v: boolean) => void;
  search: string; setSearch: (v: string) => void;
  sortBy: SortKey; setSortBy: (v: SortKey) => void;
  minMinutes: number; setMinMinutes: (v: number) => void;
}) {
  const seg = (active: boolean) => `px-3 py-1.5 text-sm ${active ? "bg-surface-2 font-medium" : "hover:bg-surface-2"}`;
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
      <input
        value={p.search}
        onChange={(e) => p.setSearch(e.target.value)}
        placeholder="Search player"
        className="w-44 rounded-md border border-border bg-bg px-2.5 py-1.5 text-sm"
      />
      <div className="flex overflow-hidden rounded-md border border-border" role="group" aria-label="Conference matchup">
        {(["all", "same", "cross"] as const).map((c) => (
          <button key={c} className={seg(p.conference === c)} onClick={() => p.setConference(c)}>
            {c === "all" ? "All games" : c === "same" ? "Same conf" : "East vs West"}
          </button>
        ))}
      </div>
      <label className="flex items-center gap-1.5 text-sm">
        <input type="checkbox" checked={p.b2bOnly} onChange={(e) => p.setB2bOnly(e.target.checked)} />
        Back-to-backs only
      </label>
      <label className="flex items-center gap-1.5 text-sm">
        Min L10 minutes
        <input
          type="number" min={0} max={40} value={p.minMinutes}
          onChange={(e) => p.setMinMinutes(Number(e.target.value))}
          className="num w-16 rounded-md border border-border bg-bg px-2 py-1"
        />
      </label>
      <div className="ml-auto flex items-center gap-1.5 text-sm">
        Sort by
        <select value={p.sortBy} onChange={(e) => p.setSortBy(e.target.value as SortKey)} className="rounded-md border border-border bg-bg px-2 py-1">
          <option value="pts">Points</option>
          <option value="reb">Rebounds</option>
          <option value="ast">Assists</option>
          <option value="pra">PRA</option>
        </select>
      </div>
    </div>
  );
}

function TeamHeader({ team, opp, home, onToggleB2b }: { team: TeamSide; opp: TeamSide; home: boolean; onToggleB2b: (t: TeamSide) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-lg font-semibold">{team.abbr}</span>
      <span className="text-xs text-muted">{home ? "home" : "away"} · {team.conference}</span>
      <button
        onClick={() => onToggleB2b(team)}
        title={team.b2b_overridden ? "Overridden – click to reset" : "Click to override back-to-back"}
        className={`rounded-full border px-2 py-0.5 text-xs ${team.b2b ? "border-warning text-warning" : "border-border text-text-2"} ${team.b2b_overridden ? "border-dashed" : ""}`}
      >
        {team.b2b ? "B2B" : `${team.rest_days ?? "?"}d rest`}{team.b2b_overridden && " *"}
      </button>
      <span className="num text-xs text-text-2">pace {fmt(team.pace)} · DRtg {fmt(team.def_rtg)}</span>
      <span className="sr-only">vs {opp.abbr}</span>
    </div>
  );
}

function GameCard({ game, players, baseById, onOut, onAvailable, onToggleB2b }: {
  game: SlateGame;
  players: PlayerPrediction[];
  baseById: Map<string, PlayerPrediction> | null;
  onOut: (id: number) => void;
  onAvailable: (id: number) => void;
  onToggleB2b: (t: TeamSide) => void;
}) {
  const abbr = (id: number) => (id === game.home.id ? game.home.abbr : game.away.abbr);
  return (
    <section className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <TeamHeader team={game.away} opp={game.home} home={false} onToggleB2b={onToggleB2b} />
          <span className="text-muted">@</span>
          <TeamHeader team={game.home} opp={game.away} home onToggleB2b={onToggleB2b} />
        </div>
        <div className="flex items-center gap-2">
          {game.line && (
            <span className="num text-xs text-text-2" title="Sportsbook line (DraftKings via ESPN)">
              {game.line.favorite ? `${game.line.favorite} −${game.line.spread}` : "Even"} · O/U {game.line.total}
            </span>
          )}
          <span className={`rounded-full px-2 py-0.5 text-xs ${game.same_conference ? "bg-surface-2 font-medium" : "text-text-2"}`}>
            {game.same_conference ? "Conference game" : "East vs West"}
          </span>
        </div>
      </div>

      {game.out.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 border-b border-border px-4 py-2 text-xs">
          <span className="text-text-2">Out:</span>
          {game.out.map((o) => (
            <button
              key={o.player_id}
              onClick={() => onAvailable(o.player_id)}
              title="Mark as playing"
              className="rounded-full border border-border px-2 py-0.5 hover:bg-surface-2"
            >
              {o.name} <span className="text-muted">({abbr(o.team_id)}{o.manual ? ", manual" : `, ${o.status}`})</span> ✕
            </button>
          ))}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="text-left text-xs text-text-2">
            <tr className="border-b border-border">
              <th className="px-4 py-2 font-medium">Player</th>
              <th className="px-2 py-2 text-right font-medium">PTS</th>
              <th className="px-2 py-2 text-right font-medium">REB</th>
              <th className="px-2 py-2 text-right font-medium">AST</th>
              <th className="px-2 py-2 text-right font-medium">PRA</th>
              <th className="px-2 py-2 text-right font-medium" title="Last 10 games: points / rebounds / assists">L10 P/R/A</th>
              <th className="px-2 py-2 text-right font-medium">MIN</th>
              <th className="px-2 py-2 text-right font-medium" title="Usage rate, season">USG%</th>
              <th className="px-2 py-2 text-right font-medium" title="True shooting, season">TS%</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {players.map((p) => {
              const before = baseById?.get(`${p.game_id}:${p.player_id}`);
              return (
                <tr key={p.player_id} className="border-b border-border/60 last:border-0 hover:bg-surface-2/60">
                  <td className="px-4 py-2">
                    <Link href={`/player/${p.player_id}`} className="font-medium hover:underline">{p.name}</Link>
                    <span className="ml-1.5 text-xs text-muted">{abbr(p.team_id)} · {p.position ?? "–"}</span>
                    {p.injury_status && (
                      <span title={p.injury_detail ?? ""} className="ml-1.5 rounded border border-warning px-1 text-xs text-warning">
                        ⚠ {p.injury_status}
                      </span>
                    )}
                  </td>
                  {(["pts", "reb", "ast"] as const).map((t) => (
                    <td key={t} className="num px-2 py-2 text-right">
                      <div className="font-semibold">
                        {fmt(p[`${t}_pred`])}
                        {before && <Delta now={p[`${t}_pred`]} was={before[`${t}_pred`]} />}
                      </div>
                      <div className="text-xs text-muted">{fmt(p[`${t}_p10`], 0)}–{fmt(p[`${t}_p90`], 0)}</div>
                    </td>
                  ))}
                  <td className="num px-2 py-2 text-right font-semibold">{fmt(p.pra_pred)}</td>
                  <td className="num px-2 py-2 text-right text-text-2">{fmt(p.pts_l10)} / {fmt(p.reb_l10)} / {fmt(p.ast_l10)}</td>
                  <td className="num px-2 py-2 text-right text-text-2">{fmt(p.min_l10)}</td>
                  <td className="num px-2 py-2 text-right text-text-2">{fmt(p.usg_season)}</td>
                  <td className="num px-2 py-2 text-right text-text-2">{p.ts_season == null ? "–" : (p.ts_season * 100).toFixed(1)}</td>
                  <td className="px-4 py-2 text-right">
                    <button onClick={() => onOut(p.player_id)} className="rounded border border-border px-2 py-0.5 text-xs hover:bg-surface-2">
                      Mark out
                    </button>
                  </td>
                </tr>
              );
            })}
            {players.length === 0 && (
              <tr><td colSpan={10} className="px-4 py-3 text-text-2">No players match the filters.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Delta({ now, was }: { now: number | null; was: number | null }) {
  if (now == null || was == null) return null;
  const d = now - was;
  if (Math.abs(d) < 0.1) return null;
  return (
    <span className={`ml-1 text-xs font-normal ${d > 0 ? "text-good" : "text-critical"}`}>
      {d > 0 ? "▲" : "▼"}{Math.abs(d).toFixed(1)}
    </span>
  );
}
