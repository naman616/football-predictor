"""World Football Elo Rating System."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from typing import Optional

from config.settings import (
    ELO_INITIAL, ELO_HOME_ADVANTAGE,
    TOURNAMENT_K_FACTORS, DEFAULT_K_FACTOR,
    PROCESSED_DIR,
)


class EloSystem:
    """Computes and maintains Elo ratings for international football teams."""

    def __init__(self, initial_rating: float = ELO_INITIAL):
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}
        self.rating_history: list[dict] = []

    def get_k_factor(self, tournament: str) -> float:
        for key, k in TOURNAMENT_K_FACTORS.items():
            if key.lower() in tournament.lower():
                return k
        return DEFAULT_K_FACTOR

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A vs team B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
        date: Optional[pd.Timestamp] = None,
    ) -> tuple[float, float]:
        """Update Elo ratings after a match. Returns new (home_elo, away_elo)."""
        k = self.get_k_factor(tournament)

        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Apply home advantage (not for neutral venues)
        effective_home_elo = home_elo + (0 if neutral else ELO_HOME_ADVANTAGE)

        exp_home = self.expected_score(effective_home_elo, away_elo)
        exp_away = 1.0 - exp_home

        # Actual scores (1=win, 0.5=draw, 0=loss)
        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score == away_score:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        # Goal difference multiplier (encourages margin of victory)
        gd = abs(home_score - away_score)
        gd_multiplier = 1.0
        if gd == 2:
            gd_multiplier = 1.5
        elif gd == 3:
            gd_multiplier = 1.75
        elif gd >= 4:
            gd_multiplier = 1.75 + (gd - 3) * 0.125

        new_home_elo = home_elo + k * gd_multiplier * (actual_home - exp_home)
        new_away_elo = away_elo + k * gd_multiplier * (actual_away - exp_away)

        self.ratings[home_team] = new_home_elo
        self.ratings[away_team] = new_away_elo

        if date is not None:
            self.rating_history.append({
                "date": date,
                "team": home_team,
                "elo": new_home_elo,
            })
            self.rating_history.append({
                "date": date,
                "team": away_team,
                "elo": new_away_elo,
            })

        return new_home_elo, new_away_elo

    def process_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all matches chronologically and compute Elo for each."""
        df = df.sort_values("date").reset_index(drop=True)
        home_elos_before, away_elos_before = [], []
        home_elos_after, away_elos_after = [], []

        for _, row in df.iterrows():
            h_before = self.get_rating(row["home_team"])
            a_before = self.get_rating(row["away_team"])
            home_elos_before.append(h_before)
            away_elos_before.append(a_before)

            h_after, a_after = self.update(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                tournament=str(row.get("tournament", "Friendly")),
                neutral=bool(row.get("neutral", False)),
                date=row["date"],
            )
            home_elos_after.append(h_after)
            away_elos_after.append(a_after)

        df = df.copy()
        df["home_elo_before"] = home_elos_before
        df["away_elo_before"] = away_elos_before
        df["home_elo_after"] = home_elos_after
        df["away_elo_after"] = away_elos_after
        df["elo_diff_before"] = df["home_elo_before"] - df["away_elo_before"]
        return df

    def get_current_ratings(self) -> pd.DataFrame:
        """Return current ratings as a sorted DataFrame."""
        rows = [{"team": t, "elo": r} for t, r in self.ratings.items()]
        df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_team_history(self, team: str) -> pd.DataFrame:
        """Return Elo history for a specific team."""
        df = pd.DataFrame(self.rating_history)
        if df.empty:
            return pd.DataFrame(columns=["date", "team", "elo"])
        return df[df["team"] == team].sort_values("date").reset_index(drop=True)

    def predict_outcome(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> tuple[float, float, float]:
        """
        Predict probabilities using Elo ratings only.
        Returns (p_home_win, p_draw, p_away_win).
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        if not neutral:
            home_elo += ELO_HOME_ADVANTAGE

        # Convert Elo difference to win probabilities
        elo_diff = home_elo - away_elo

        # Dixon-Coles inspired conversion
        # Base draw probability ~27%, adjusted by Elo difference
        p_home_win = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # Draw probability shrinks with larger Elo differences
        p_draw_base = 0.27
        p_draw = p_draw_base * np.exp(-abs(elo_diff) / 800.0)
        p_draw = max(0.08, min(0.35, p_draw))

        p_home_win = p_home_win * (1 - p_draw)
        p_away_win = (1 - p_home_win) * (1 - p_draw)

        # Renormalize
        total = p_home_win + p_draw + p_away_win
        return p_home_win / total, p_draw / total, p_away_win / total


def build_and_save_elo(df: pd.DataFrame) -> tuple["EloSystem", pd.DataFrame]:
    """Build Elo system from match history and save results."""
    elo = EloSystem()
    df_with_elo = elo.process_matches(df)

    # Save Elo ratings
    ratings_df = elo.get_current_ratings()
    ratings_df.to_parquet(PROCESSED_DIR / "elo_ratings.parquet", index=False)

    # Save match history with Elo
    df_with_elo.to_parquet(PROCESSED_DIR / "matches_with_elo.parquet", index=False)

    return elo, df_with_elo


def load_elo_system(df: pd.DataFrame) -> "EloSystem":
    """Rebuild Elo system by replaying all matches."""
    elo = EloSystem()
    elo.process_matches(df)
    return elo
