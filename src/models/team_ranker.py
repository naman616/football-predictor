"""Custom team power ranking system (beyond FIFA rankings)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from typing import Optional

from config.settings import CONFEDERATIONS, PROCESSED_DIR


def get_confederation(team: str) -> str:
    for conf, teams in CONFEDERATIONS.items():
        if team in teams:
            return conf
    return "Other"


class TeamRanker:
    """
    Multi-dimensional team ranking system.

    Dimensions:
    - Overall Power: Composite of Elo + form + tournament performance
    - Attack Rating: Goals scored, xG proxy, win rate in high-scoring matches
    - Defense Rating: Goals conceded, clean sheet rate, defensive resilience
    - Form Rating: Recent performance decay-weighted
    """

    def __init__(self, df: pd.DataFrame, elo_ratings: pd.DataFrame):
        """
        df: matches DataFrame with Elo columns (matches_with_elo.parquet)
        elo_ratings: current Elo ratings DataFrame
        """
        self.df = df.sort_values("date").reset_index(drop=True)
        self.elo_ratings = elo_ratings
        self._rankings: Optional[pd.DataFrame] = None

    def _decay_weights(self, dates: pd.Series, half_life_days: int = 365) -> np.ndarray:
        """Exponential decay weights: recent matches count more."""
        most_recent = dates.max()
        days_ago = (most_recent - dates).dt.days
        weights = np.exp(-np.log(2) * days_ago / half_life_days)
        return weights / weights.sum()

    def _team_attack_rating(self, team: str, recent_n: int = 30) -> float:
        """Weighted goals scored per game (recent matches weighted more)."""
        mask = (self.df["home_team"] == team) | (self.df["away_team"] == team)
        matches = self.df[mask].tail(recent_n)
        if matches.empty:
            return 50.0

        gf = []
        weights = []
        for _, row in matches.iterrows():
            if row["home_team"] == team:
                gf.append(row["home_score"])
            else:
                gf.append(row["away_score"])
            weights.append(1.0)

        w = self._decay_weights(matches["date"]) if len(matches) > 1 else np.array([1.0])
        weighted_gf = np.dot(gf, w)
        return weighted_gf

    def _team_defense_rating(self, team: str, recent_n: int = 30) -> float:
        """Weighted goals conceded per game (lower is better)."""
        mask = (self.df["home_team"] == team) | (self.df["away_team"] == team)
        matches = self.df[mask].tail(recent_n)
        if matches.empty:
            return 50.0

        ga = []
        for _, row in matches.iterrows():
            if row["home_team"] == team:
                ga.append(row["away_score"])
            else:
                ga.append(row["home_score"])

        w = self._decay_weights(matches["date"]) if len(matches) > 1 else np.array([1.0])
        weighted_ga = np.dot(ga, w)
        return weighted_ga

    def _team_form_rating(self, team: str, recent_n: int = 10) -> float:
        """Recent form as points per game (0-3), decay weighted."""
        mask = (self.df["home_team"] == team) | (self.df["away_team"] == team)
        matches = self.df[mask].tail(recent_n)
        if matches.empty:
            return 1.0

        points = []
        for _, row in matches.iterrows():
            is_home = row["home_team"] == team
            res = row["result"]
            if is_home:
                pts = 3 if res == 0 else (1 if res == 1 else 0)
            else:
                pts = 3 if res == 2 else (1 if res == 1 else 0)
            points.append(pts)

        w = self._decay_weights(matches["date"]) if len(matches) > 1 else np.array([1.0])
        return float(np.dot(points, w) * 3)  # Scale to 0-3

    def _tournament_bonus(self, team: str) -> float:
        """
        Bonus for performance in major tournaments (last 4 years).
        Win WC final = +200, runner-up = +100, semi = +50, etc.
        """
        cutoff = self.df["date"].max() - pd.Timedelta(days=4 * 365)
        major_tournaments = [
            "FIFA World Cup", "UEFA Euro", "Copa America",
            "Africa Cup of Nations", "AFC Asian Cup",
        ]
        bonus = 0.0
        for tourn in major_tournaments:
            mask = (
                (self.df["date"] >= cutoff)
                & (self.df["tournament"].str.contains(tourn, case=False, na=False))
                & (
                    (self.df["home_team"] == team)
                    | (self.df["away_team"] == team)
                )
            )
            tourn_matches = self.df[mask]
            if tourn_matches.empty:
                continue

            wins = 0
            for _, row in tourn_matches.iterrows():
                is_home = row["home_team"] == team
                res = row["result"]
                if (is_home and res == 0) or (not is_home and res == 2):
                    wins += 1

            total = len(tourn_matches)
            if total > 0:
                win_rate = wins / total
                # Scale by number of tournament appearances
                bonus += win_rate * min(total, 7) * 5

        return bonus

    def build_rankings(self, min_matches: int = 5) -> pd.DataFrame:
        """Build comprehensive team rankings."""
        all_teams = set(self.df["home_team"].unique()) | set(self.df["away_team"].unique())

        records = []
        for team in all_teams:
            mask = (self.df["home_team"] == team) | (self.df["away_team"] == team)
            n_matches = mask.sum()
            if n_matches < min_matches:
                continue

            elo_row = self.elo_ratings[self.elo_ratings["team"] == team]
            elo = float(elo_row["elo"].iloc[0]) if not elo_row.empty else 1500.0

            attack_raw = self._team_attack_rating(team)
            defense_raw = self._team_defense_rating(team)
            form_raw = self._team_form_rating(team)
            tourn_bonus = self._tournament_bonus(team)

            records.append({
                "team": team,
                "confederation": get_confederation(team),
                "elo": round(elo, 1),
                "attack_raw": attack_raw,
                "defense_raw": defense_raw,  # Lower = better
                "form_raw": form_raw,
                "tournament_bonus": tourn_bonus,
                "n_matches": n_matches,
            })

        df_ranks = pd.DataFrame(records)
        if df_ranks.empty:
            return df_ranks

        # Normalize each dimension to 0-100
        def norm(series: pd.Series, invert: bool = False) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series([50.0] * len(series), index=series.index)
            normalized = (series - mn) / (mx - mn) * 100
            return 100 - normalized if invert else normalized

        df_ranks["attack_rating"] = norm(df_ranks["attack_raw"]).round(1)
        df_ranks["defense_rating"] = norm(df_ranks["defense_raw"], invert=True).round(1)
        df_ranks["form_rating"] = norm(df_ranks["form_raw"]).round(1)
        df_ranks["elo_norm"] = norm(df_ranks["elo"]).round(1)
        df_ranks["tourn_norm"] = norm(df_ranks["tournament_bonus"]).round(1)

        # Composite power rating
        df_ranks["power_rating"] = (
            0.35 * df_ranks["elo_norm"]
            + 0.20 * df_ranks["attack_rating"]
            + 0.20 * df_ranks["defense_rating"]
            + 0.15 * df_ranks["form_rating"]
            + 0.10 * df_ranks["tourn_norm"]
        ).round(1)

        df_ranks = df_ranks.sort_values("power_rating", ascending=False).reset_index(drop=True)
        df_ranks["rank"] = range(1, len(df_ranks) + 1)

        self._rankings = df_ranks
        df_ranks.to_parquet(PROCESSED_DIR / "team_rankings.parquet", index=False)
        return df_ranks

    def get_ranking_for_team(self, team: str) -> Optional[pd.Series]:
        if self._rankings is None:
            return None
        mask = self._rankings["team"] == team
        if not mask.any():
            return None
        return self._rankings[mask].iloc[0]

    def get_recent_form(self, team: str, n: int = 5) -> list[str]:
        """Return last N results as list of 'W', 'D', 'L'."""
        mask = (self.df["home_team"] == team) | (self.df["away_team"] == team)
        recent = self.df[mask].tail(n)
        results = []
        for _, row in recent.iterrows():
            is_home = row["home_team"] == team
            res = row["result"]
            if (is_home and res == 0) or (not is_home and res == 2):
                results.append("W")
            elif res == 1:
                results.append("D")
            else:
                results.append("L")
        return results

    def get_head_to_head_record(self, team1: str, team2: str) -> dict:
        """Return H2H record between two teams."""
        mask = (
            ((self.df["home_team"] == team1) & (self.df["away_team"] == team2))
            | ((self.df["home_team"] == team2) & (self.df["away_team"] == team1))
        )
        h2h = self.df[mask]
        t1_wins, t2_wins, draws = 0, 0, 0
        t1_gf, t2_gf = 0, 0
        for _, row in h2h.iterrows():
            if row["home_team"] == team1:
                t1_gf += row["home_score"]
                t2_gf += row["away_score"]
                if row["result"] == 0:
                    t1_wins += 1
                elif row["result"] == 1:
                    draws += 1
                else:
                    t2_wins += 1
            else:
                t1_gf += row["away_score"]
                t2_gf += row["home_score"]
                if row["result"] == 2:
                    t1_wins += 1
                elif row["result"] == 1:
                    draws += 1
                else:
                    t2_wins += 1

        return {
            "team1": team1,
            "team2": team2,
            "team1_wins": t1_wins,
            "draws": draws,
            "team2_wins": t2_wins,
            "team1_goals": t1_gf,
            "team2_goals": t2_gf,
            "total_matches": len(h2h),
        }
