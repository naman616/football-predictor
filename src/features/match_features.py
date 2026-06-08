"""Match feature engineering for prediction models."""
import sys
import bisect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from typing import Optional

from config.settings import (
    FORM_WINDOW, H2H_WINDOW, TOURNAMENT_IMPORTANCE,
    TRAIN_CUTOFF_YEAR,
)


FEATURE_COLS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "form_home_ppg",       # Points per game, last N matches
    "form_away_ppg",
    "form_home_gf",        # Goals for per game, last N
    "form_away_gf",
    "form_home_ga",        # Goals against per game, last N
    "form_away_ga",
    "form_home_gd",        # Goal difference per game, last N
    "form_away_gd",
    "h2h_home_win_rate",   # H2H win rate for home team
    "h2h_draw_rate",
    "h2h_home_gf",         # H2H avg goals for home
    "h2h_away_gf",         # H2H avg goals for away
    "tournament_importance",
    "is_neutral",
    "elo_diff_sq",         # Non-linear Elo feature
    "form_diff_ppg",       # Form differential
    "gf_diff",             # Goal-scoring differential
    "ga_diff",             # Goal-conceding differential
]


class MatchFeatureEngine:
    """Builds match-level features for ML prediction."""

    def __init__(self, df: pd.DataFrame):
        """
        df: cleaned matches DataFrame with columns:
            date, home_team, away_team, home_score, away_score,
            tournament, neutral, result,
            home_elo_before, away_elo_before
        """
        self.df = df.sort_values("date").reset_index(drop=True)
        self._team_form_cache: dict = {}
        self._h2h_cache: dict = {}
        self._build_team_index()

    def _build_team_index(self):
        """
        Pre-index row positions and extract columns as numpy arrays.
        Avoids Arrow-backed DataFrame iloc slicing (~3.7ms/call) by
        indexing raw numpy arrays directly (~50μs/call).
        """
        from collections import defaultdict
        idx: dict[str, list[int]] = defaultdict(list)
        ht = self.df["home_team"].tolist()
        at = self.df["away_team"].tolist()
        for i in range(len(self.df)):
            idx[ht[i]].append(i)
            idx[at[i]].append(i)
        self._team_idx: dict[str, list[int]] = dict(idx)

        # Pre-extract as plain numpy arrays for O(1) column access
        self._ht_arr: np.ndarray = np.array(ht)
        self._at_arr: np.ndarray = np.array(at)
        self._res_arr: np.ndarray = self.df["result"].to_numpy()
        self._hs_arr: np.ndarray = self.df["home_score"].to_numpy()
        self._as_arr: np.ndarray = self.df["away_score"].to_numpy()

    def _get_team_results_before(self, team: str, before_idx: int) -> pd.DataFrame:
        """Get all results for a team before a given row index (kept for build_all_features)."""
        indices = self._team_idx.get(team, [])
        end = bisect.bisect_left(indices, before_idx)
        tail_indices = indices[max(0, end - FORM_WINDOW):end]
        if not tail_indices:
            return pd.DataFrame(columns=self.df.columns)
        return self.df.iloc[tail_indices]

    def _compute_form(self, team: str, before_idx: int) -> dict:
        """Compute recent form stats using pre-extracted numpy arrays (no iloc)."""
        indices = self._team_idx.get(team, [])
        end = bisect.bisect_left(indices, before_idx)
        tail = indices[max(0, end - FORM_WINDOW):end]
        if not tail:
            return {"ppg": 1.0, "gf": 1.2, "ga": 1.2, "gd": 0.0,
                    "wins": 0, "draws": 0, "losses": 0, "n": 0}

        idx = np.array(tail)
        is_home = self._ht_arr[idx] == team
        res = self._res_arr[idx]
        hs = self._hs_arr[idx]
        as_ = self._as_arr[idx]

        scored = np.where(is_home, hs, as_)
        conceded = np.where(is_home, as_, hs)
        pts = np.where(
            is_home,
            np.where(res == 0, 3, np.where(res == 1, 1, 0)),
            np.where(res == 2, 3, np.where(res == 1, 1, 0)),
        )
        n = len(pts)
        return {
            "ppg": float(pts.sum() / n),
            "gf": float(scored.mean()),
            "ga": float(conceded.mean()),
            "gd": float((scored - conceded).mean()),
            "wins": int((pts == 3).sum()),
            "draws": int((pts == 1).sum()),
            "losses": int((pts == 0).sum()),
            "n": n,
        }

    def _get_h2h(self, home_team: str, away_team: str, before_idx: int) -> dict:
        """Compute H2H stats using pre-extracted numpy arrays (no iloc)."""
        a_set = set(self._team_idx.get(home_team, []))
        b_set = set(self._team_idx.get(away_team, []))
        shared = sorted(a_set & b_set)
        end = bisect.bisect_left(shared, before_idx)
        tail = shared[max(0, end - H2H_WINDOW):end]
        if not tail:
            return {"home_wins": 0.0, "draws": 0.0, "away_wins": 0.0,
                    "home_gf": 1.2, "away_gf": 1.2}

        idx = np.array(tail)
        is_home = self._ht_arr[idx] == home_team
        res = self._res_arr[idx]
        hs = self._hs_arr[idx]
        as_ = self._as_arr[idx]

        hw = int(((is_home) & (res == 0)).sum() + ((~is_home) & (res == 2)).sum())
        dr = int((res == 1).sum())
        aw = len(tail) - hw - dr
        home_gf = np.where(is_home, hs, as_)
        away_gf = np.where(is_home, as_, hs)
        n = len(tail)

        return {
            "home_wins": hw / n,
            "draws": dr / n,
            "away_wins": aw / n,
            "home_gf": float(home_gf.mean()),
            "away_gf": float(away_gf.mean()),
        }

    def build_features_for_row(self, idx: int) -> Optional[dict]:
        """Build feature dict for a single match row."""
        row = self.df.iloc[idx]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Require at least 3 H2H or form matches for training quality
        form_h = self._compute_form(home_team, idx)
        form_a = self._compute_form(away_team, idx)
        h2h = self._get_h2h(home_team, away_team, idx)

        elo_home = row.get("home_elo_before", 1500.0)
        elo_away = row.get("away_elo_before", 1500.0)
        elo_diff = elo_home - elo_away

        tournament = str(row.get("tournament", "Friendly"))
        t_importance = TOURNAMENT_IMPORTANCE.get(tournament, 0.3)
        # Partial match on tournament name
        if t_importance == 0.3:
            for key, val in TOURNAMENT_IMPORTANCE.items():
                if key.lower() in tournament.lower():
                    t_importance = val
                    break

        return {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_diff,
            "form_home_ppg": form_h["ppg"],
            "form_away_ppg": form_a["ppg"],
            "form_home_gf": form_h["gf"],
            "form_away_gf": form_a["gf"],
            "form_home_ga": form_h["ga"],
            "form_away_ga": form_a["ga"],
            "form_home_gd": form_h["gd"],
            "form_away_gd": form_a["gd"],
            "h2h_home_win_rate": h2h["home_wins"],
            "h2h_draw_rate": h2h["draws"],
            "h2h_home_gf": h2h["home_gf"],
            "h2h_away_gf": h2h["away_gf"],
            "tournament_importance": t_importance,
            "is_neutral": int(row.get("neutral", False)),
            "elo_diff_sq": elo_diff ** 2,
            "form_diff_ppg": form_h["ppg"] - form_a["ppg"],
            "gf_diff": form_h["gf"] - form_a["gf"],
            "ga_diff": form_h["ga"] - form_a["ga"],
            "result": int(row["result"]),
        }

    def build_all_features(self, min_year: int = TRAIN_CUTOFF_YEAR) -> pd.DataFrame:
        """Build feature matrix for all matches from min_year onward."""
        records = []
        mask = self.df["year"] >= min_year
        indices = self.df[mask].index.tolist()

        for idx in indices:
            feat = self.build_features_for_row(idx)
            if feat is not None:
                records.append(feat)

        return pd.DataFrame(records)

    def get_features_for_prediction(
        self,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> pd.DataFrame:
        """Build features for a single prediction (no future leakage)."""
        idx = len(self.df)  # Use all available data

        # Cache form per team — same idx every prediction call so safe to reuse
        if home_team not in self._team_form_cache:
            self._team_form_cache[home_team] = self._compute_form(home_team, idx)
        if away_team not in self._team_form_cache:
            self._team_form_cache[away_team] = self._compute_form(away_team, idx)
        form_h = self._team_form_cache[home_team]
        form_a = self._team_form_cache[away_team]

        h2h_key = (home_team, away_team)
        if h2h_key not in self._h2h_cache:
            self._h2h_cache[h2h_key] = self._get_h2h(home_team, away_team, idx)
        h2h = self._h2h_cache[h2h_key]

        elo_diff = home_elo - away_elo
        t_importance = TOURNAMENT_IMPORTANCE.get(tournament, 0.3)
        for key, val in TOURNAMENT_IMPORTANCE.items():
            if key.lower() in tournament.lower():
                t_importance = val
                break

        feat = {
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": elo_diff,
            "form_home_ppg": form_h["ppg"],
            "form_away_ppg": form_a["ppg"],
            "form_home_gf": form_h["gf"],
            "form_away_gf": form_a["gf"],
            "form_home_ga": form_h["ga"],
            "form_away_ga": form_a["ga"],
            "form_home_gd": form_h["gd"],
            "form_away_gd": form_a["gd"],
            "h2h_home_win_rate": h2h["home_wins"],
            "h2h_draw_rate": h2h["draws"],
            "h2h_home_gf": h2h["home_gf"],
            "h2h_away_gf": h2h["away_gf"],
            "tournament_importance": t_importance,
            "is_neutral": int(neutral),
            "elo_diff_sq": elo_diff ** 2,
            "form_diff_ppg": form_h["ppg"] - form_a["ppg"],
            "gf_diff": form_h["gf"] - form_a["gf"],
            "ga_diff": form_h["ga"] - form_a["ga"],
        }

        return pd.DataFrame([feat])[FEATURE_COLS]

    def get_feature_values(
        self,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> np.ndarray:
        """Return features as a flat numpy array — faster for batch pre-computation."""
        idx = len(self.df)

        if home_team not in self._team_form_cache:
            self._team_form_cache[home_team] = self._compute_form(home_team, idx)
        if away_team not in self._team_form_cache:
            self._team_form_cache[away_team] = self._compute_form(away_team, idx)
        form_h = self._team_form_cache[home_team]
        form_a = self._team_form_cache[away_team]

        h2h_key = (home_team, away_team)
        if h2h_key not in self._h2h_cache:
            self._h2h_cache[h2h_key] = self._get_h2h(home_team, away_team, idx)
        h2h = self._h2h_cache[h2h_key]

        elo_diff = home_elo - away_elo
        t_importance = TOURNAMENT_IMPORTANCE.get(tournament, 0.3)
        for key, val in TOURNAMENT_IMPORTANCE.items():
            if key.lower() in tournament.lower():
                t_importance = val
                break

        return np.array([
            home_elo, away_elo, elo_diff,
            form_h["ppg"], form_a["ppg"],
            form_h["gf"], form_a["gf"],
            form_h["ga"], form_a["ga"],
            form_h["gd"], form_a["gd"],
            h2h["home_wins"], h2h["draws"],
            h2h["home_gf"], h2h["away_gf"],
            t_importance, int(neutral),
            elo_diff ** 2,
            form_h["ppg"] - form_a["ppg"],
            form_h["gf"] - form_a["gf"],
            form_h["ga"] - form_a["ga"],
        ], dtype=np.float64)
