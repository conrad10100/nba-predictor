export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Target = "pts" | "reb" | "ast";
export const TARGETS: Target[] = ["pts", "reb", "ast"];

export type TeamSide = {
  id: number;
  abbr: string;
  name: string;
  conference: "East" | "West";
  b2b: boolean | null;
  b2b_overridden: boolean;
  rest_days: number | null;
  pace: number | null;
  def_rtg: number | null;
};

export type OutPlayer = { player_id: number; name: string; team_id: number; status: string; manual: boolean };

export type GameLine = { favorite: string | null; spread: number; total: number };

export type SlateGame = {
  game_id: string; home: TeamSide; away: TeamSide; same_conference: boolean; out: OutPlayer[]; line: GameLine | null;
};

type PerTarget<S extends string> = { [K in `${Target}_${S}`]: number | null };

export type PlayerPrediction = {
  player_id: number;
  name: string;
  position: string | null;
  team_id: number;
  opp_team_id: number;
  game_id: string;
  injury_status: string | null;
  injury_detail: string | null;
  min_l10: number | null;
  usg_season: number | null;
  ts_season: number | null;
  missing_teammate_pts: number | null;
  pra_pred: number | null;
} & PerTarget<"pred"> & PerTarget<"p10"> & PerTarget<"p90"> & PerTarget<"l5"> & PerTarget<"l10"> & PerTarget<"season"> & PerTarget<"vs_opp">;

export type Slate = { date: string; model_version: string | null; games: SlateGame[]; predictions: PlayerPrediction[] };

export type SlateOverrides = { out: number[]; available: number[]; b2b: Record<number, boolean> };

export type PlayerGame = {
  date: string;
  game_id: string;
  opp: string;
  min: number;
  pts: number;
  reb: number;
  ast: number;
  predicted: Partial<Record<Target, { predicted: number; p10: number; p90: number }>> | null;
};

export type PlayerDetail = { id: number; name: string; position: string | null; team: string | null; games: PlayerGame[] };

export type ModelMetrics = {
  latest: {
    target: Target;
    version: string;
    trained_at: string;
    mae: number;
    baseline_mae: number;
    rmse: number;
    interval_coverage: number;
    promoted: boolean;
    n_train: number;
    n_holdout: number;
    top_features: Record<string, number>;
  }[];
  history: { target: Target; version: string; trained_at: string; mae: number; baseline_mae: number; promoted: boolean }[];
  live_30d: { target: Target; n: number; mae: number; baseline_mae: number }[];
  config: { target: Target; params: Record<string, number | string>; bias_correction: number; note: string | null; updated_at: string | null }[];
};

export type Segment = { n: number; mae?: number; bias?: number };

export type DayEval = {
  date: string;
  target: Target;
  model_version: string;
  n: number;
  mae: number;
  rmse: number;
  bias: number;
  baseline_mae: number | null;
  beat_baseline_pct: number | null;
  close_threshold: number;
  within_close_pct: number;
  within_wide_pct: number;
  interval_coverage: number;
  segments: Record<string, Segment>;
  kind: "live" | "backtest";
};

export type EvalTotals = Record<Target, {
  days: number; n: number; mae: number; bias: number; baseline_mae: number | null; beat_baseline_pct: number | null;
  within_close_pct: number; within_wide_pct: number; interval_coverage: number; close_threshold: number;
}>;

export type GradedStat = { predicted: number; actual: number; error: number; p10: number; p90: number; baseline: number | null };
export type GradedPlayer = { player_id: number; name: string; team: string | null } & Partial<Record<Target, GradedStat>>;

export type PickResult = "win" | "loss" | "push" | "no pick" | "void" | "pending";
export type Kind = "live" | "backtest";
export type StatKey = Target | "all";

export type Record_ = {
  wins: number; losses: number; pushes: number; picks: number; win_pct: number | null;
  units: number; roi: number | null; expected_units: number; avg_prob: number | null; odds_assumed: number;
};

export type Pick = {
  date: string; game_id: string; player_id: number; player: string | null; team: string | null; opp: string | null; stat: Target;
  line: number; open_line: number | null; over_odds: number | null; under_odds: number | null; predicted: number; p10: number; p90: number;
  edge: number; pick: "over" | "under" | null; odds: number; odds_assumed: boolean; prob: number | null; market_prob: number | null;
  ev: number | null; actual: number | null; result: PickResult; units: number | null; top5: boolean; confidence_rank: number | null;
  model_version: string; kind: Kind;
};

export type TopPerformer = {
  date: string; stat: Target; rank: number; player_id: number; player: string | null; team: string | null; actual: number;
  predicted: number | null; model_rank: number | null; line: number | null; pick: "over" | "under" | null; result: PickResult | null;
};

export type SimpleRecord = { wins: number; losses: number; pushes: number; picks: number; win_pct: number | null };

export type RunningRow = {
  date: string; stat: StatKey; day_wins: number; day_losses: number; day_pushes: number; day_units: number;
  wins: number; losses: number; pushes: number; units: number; expected_units: number; win_pct: number | null;
};

export type Performance = {
  kind: Kind;
  available: Record<Kind, number>;
  totals: Partial<Record<StatKey, Record_>>;
  no_pick: number;
  void: number;
  running: RunningRow[];
  by_edge: ({ label: string; min: number } & Record_)[];
  by_confidence: ({ label: string } & Record_)[];
  confident: { totals: Record_; running: RunningRow[]; picks: Pick[] };
  top_performers: TopPerformer[];
  top_summary: { nights: number; in_model_top5: Partial<Record<Target, number>>; record: Partial<Record<StatKey, SimpleRecord>> } | null;
  picks: Pick[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  dates: () => request<{ date: string; games: number }[]>("/api/dates"),
  slate: (date?: string) => request<Slate>(`/api/slate${date ? `?date=${date}` : ""}`),
  whatIf: (date: string, o: SlateOverrides) =>
    request<Slate>("/api/slate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ date, ...o }) }),
  player: (id: string) => request<PlayerDetail>(`/api/players/${id}`),
  metrics: () => request<ModelMetrics>("/api/model/metrics"),
  evaluation: () => request<{ days: DayEval[]; totals: Record<"live" | "backtest", Partial<EvalTotals>> }>("/api/evaluation"),
  evaluationDay: (date: string) => request<GradedPlayer[]>(`/api/evaluation/${date}`),
  performance: (kind: Kind) => request<Performance>(`/api/performance?kind=${kind}`),
};

export const performanceExportUrl = `${API_URL}/api/performance-export.csv`;

export const exportUrl = `${API_URL}/api/evaluation-export.csv`;
export const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(0)}%`);
export const STAT_LABEL: Record<Target, string> = { pts: "Points", reb: "Rebounds", ast: "Assists" };

export const fmt = (v: number | null | undefined, digits = 1) => (v == null ? "–" : v.toFixed(digits));
