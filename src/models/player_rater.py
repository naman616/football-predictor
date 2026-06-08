"""Position-specific player rating system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from config.settings import PROCESSED_DIR


# Position-specific rating weights
FW_WEIGHTS = {
    "goals_p90": 0.22,
    "assists_p90": 0.15,
    "xG_p90": 0.14,
    "xA_p90": 0.12,
    "shots_on_target_p90": 0.07,
    "key_passes_p90": 0.10,
    "progressive_carries_p90": 0.10,
    "dribbles_completed_p90": 0.07,
    "conversion_rate": 0.03,
}

MF_WEIGHTS = {
    "progressive_passes_p90": 0.18,
    "xA_p90": 0.16,
    "assists_p90": 0.14,
    "key_passes_p90": 0.12,
    "pass_completion_pct": 0.10,
    "goals_p90": 0.14,
    "tackles_p90": 0.08,
    "interceptions_p90": 0.08,
}

DF_WEIGHTS = {
    "tackles_p90": 0.18,
    "interceptions_p90": 0.18,
    "clearances_p90": 0.14,
    "blocks_p90": 0.12,
    "aerial_win_pct": 0.14,
    "pass_completion_pct": 0.10,
    "progressive_passes_p90": 0.08,
    "progressive_carries_p90": 0.06,
}

GK_WEIGHTS = {
    "save_pct": 0.30,
    "psxg_ga_p90": 0.30,  # Goals prevented per 90 (higher = better)
    "clean_sheet_pct": 0.25,
    "pass_completion_pct": 0.15,
}

POSITION_WEIGHTS = {
    "FW": FW_WEIGHTS,
    "MF": MF_WEIGHTS,
    "DF": DF_WEIGHTS,
    "GK": GK_WEIGHTS,
}

ATTACK_WEIGHTS = {
    "FW": {"goals_p90": 0.4, "assists_p90": 0.2, "xG_p90": 0.25, "xA_p90": 0.15},
    "MF": {"goals_p90": 0.3, "assists_p90": 0.3, "xG_p90": 0.2, "xA_p90": 0.2},
    "DF": {"goals_p90": 0.4, "assists_p90": 0.35, "xG_p90": 0.15, "xA_p90": 0.1},
    "GK": {"goals_p90": 0.0, "assists_p90": 0.0, "xG_p90": 0.0, "xA_p90": 0.0},
}

DEFENSE_WEIGHTS = {
    "FW": {"tackles_p90": 0.4, "interceptions_p90": 0.4, "clearances_p90": 0.2},
    "MF": {"tackles_p90": 0.4, "interceptions_p90": 0.4, "clearances_p90": 0.2},
    "DF": {"tackles_p90": 0.3, "interceptions_p90": 0.3, "clearances_p90": 0.25, "blocks_p90": 0.15},
    "GK": {"save_pct": 0.5, "clean_sheet_pct": 0.3, "psxg_ga_p90": 0.2},
}


class PlayerRater:
    """
    Computes per-90, percentile, and composite ratings for players.

    Rating scale: 0-100 (normalized within position group).
    """

    def __init__(self, df: pd.DataFrame):
        """df: raw player statistics DataFrame."""
        self.raw = df.copy()
        self.rated: Optional[pd.DataFrame] = None
        self._per90: Optional[pd.DataFrame] = None

    def _compute_per90(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert counting stats to per-90-minute rates."""
        df = df.copy()
        mins = df["mins_played"].clip(lower=1)
        per90 = mins / 90.0

        counting_cols = [
            "goals", "assists", "xG", "xA", "shots", "shots_on_target",
            "key_passes", "progressive_passes", "progressive_carries",
            "dribbles_completed", "dribbles_attempted",
            "tackles", "tackles_won", "interceptions", "clearances",
            "blocks", "aerial_wins", "aerial_losses",
            "saves", "goals_conceded",
        ]

        for col in counting_cols:
            if col in df.columns:
                df[f"{col}_p90"] = (df[col] / per90).round(3)

        # Derived ratios
        df["conversion_rate"] = np.where(
            df["shots"] > 0, df["goals"] / df["shots"], 0.0
        ).round(3)
        df["aerial_win_pct"] = np.where(
            (df["aerial_wins"] + df["aerial_losses"]) > 0,
            df["aerial_wins"] / (df["aerial_wins"] + df["aerial_losses"]),
            0.5,
        ).round(3)
        df["save_pct"] = np.where(
            (df["saves"] + df["goals_conceded"]) > 0,
            df["saves"] / (df["saves"] + df["goals_conceded"]),
            0.0,
        ).round(3)
        df["clean_sheet_pct"] = np.where(
            df["apps"] > 0,
            df["clean_sheets"] / df["apps"],
            0.0,
        ).round(3)
        # psxg_ga positive = goals prevented (we flip sign for GK rating)
        df["psxg_ga_p90"] = np.where(
            df["position"] == "GK",
            df["psxg_ga"] / (mins / 90.0),
            0.0,
        ).round(3)

        return df

    def _percentile_rank(self, series: pd.Series) -> pd.Series:
        """Convert raw values to 0-100 percentile ranks within group."""
        return series.rank(pct=True, method="average") * 100

    def _scaled_score(self, series: pd.Series) -> pd.Series:
        """
        Scale values using z-score normalization clipped to ±2σ → [30, 95].

        Min-max lets a single outlier season (e.g. 43 goals) compress the entire
        pool toward 30. Z-score keeps the distribution around the mean (62.5)
        regardless of extremes, so elite-but-not-record-breaking players still
        score in the 65-80 range rather than being pushed to 40.
        """
        s = series.fillna(0).astype(float)
        std = s.std()
        if std == 0:
            return pd.Series(62.5, index=s.index)
        z = (s - s.mean()) / std
        z_clipped = z.clip(-2.0, 2.0)
        # Map [-2, 2] → [30, 95]: midpoint 62.5, half-range 32.5
        scaled = 62.5 + z_clipped * 16.25
        return scaled.round(1)

    def _compute_weighted_rating(
        self,
        df: pd.DataFrame,
        weights: dict,
        suffix: str = "",
    ) -> pd.Series:
        """Compute weighted sum of scaled feature scores."""
        scores = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for col, weight in weights.items():
            if col not in df.columns:
                continue
            scaled = self._scaled_score(df[col].fillna(0))
            scores += weight * scaled
            total_weight += weight

        if total_weight > 0:
            scores = scores / total_weight

        return scores.round(1)

    def compute_ratings(self, min_apps: int = 5, min_mins: int = 450) -> pd.DataFrame:
        """
        Compute all ratings for all players.

        Filters out players with too few appearances or minutes.
        Returns enriched DataFrame with rating columns.
        """
        df = self.raw.copy()

        # Aggregate multiple seasons per player
        df = self._aggregate_seasons(df)

        # Filter minimum thresholds
        df = df[(df["apps"] >= min_apps) & (df["mins_played"] >= min_mins)].copy()
        df = df.reset_index(drop=True)

        # Per-90 stats
        df = self._compute_per90(df)
        self._per90 = df.copy()

        # Compute ratings per position
        position_dfs = []
        for position, weights in POSITION_WEIGHTS.items():
            pos_df = df[df["position"] == position].copy()
            if pos_df.empty:
                continue

            pos_df["overall_rating"] = self._compute_weighted_rating(pos_df, weights)

            # Attack rating
            atk_w = ATTACK_WEIGHTS.get(position, {})
            if position == "GK":
                pos_df["attack_rating"] = 0.0
            else:
                pos_df["attack_rating"] = self._compute_weighted_rating(pos_df, atk_w)

            # Defense rating
            def_w = DEFENSE_WEIGHTS.get(position, {})
            pos_df["defense_rating"] = self._compute_weighted_rating(pos_df, def_w)

            # Form rating (use most recent season data as proxy)
            pos_df["form_rating"] = pos_df["overall_rating"]  # Simplified: same as overall for aggregated

            position_dfs.append(pos_df)

        if not position_dfs:
            return pd.DataFrame()

        rated = pd.concat(position_dfs, ignore_index=True)

        # Sort by overall rating descending
        rated = rated.sort_values("overall_rating", ascending=False).reset_index(drop=True)
        rated["global_rank"] = range(1, len(rated) + 1)

        self.rated = rated
        rated.to_parquet(PROCESSED_DIR / "player_ratings.parquet", index=False)
        return rated

    def _aggregate_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use each player's most recent season only.

        Averaging across seasons dilutes peak-season per-90 rates and makes
        multi-season veterans look worse than single-season breakout players.
        Using the most recent season gives current-form ratings.
        """
        df = df.copy()
        # Season strings like "2024-25" sort correctly lexicographically
        df = df.sort_values("season", ascending=False)
        result = df.groupby("name", sort=False).first().reset_index()
        return result

    def get_player_stats(self, name: str) -> Optional[pd.Series]:
        if self.rated is None:
            return None
        mask = self.rated["name"].str.lower() == name.lower()
        if not mask.any():
            # Try partial match
            mask = self.rated["name"].str.lower().str.contains(name.lower(), na=False)
        if not mask.any():
            return None
        return self.rated[mask].iloc[0]

    def get_top_players(
        self,
        position: Optional[str] = None,
        nationality: Optional[str] = None,
        n: int = 20,
    ) -> pd.DataFrame:
        if self.rated is None:
            return pd.DataFrame()
        df = self.rated.copy()
        if position:
            df = df[df["position"] == position]
        if nationality:
            df = df[df["nationality"].str.lower() == nationality.lower()]
        return df.head(n)

    def get_percentile_stats(self, player_name: str) -> Optional[dict]:
        """Return all per-90 stats as percentiles vs position peers."""
        if self._per90 is None:
            return None

        player_row = self._per90[self._per90["name"].str.lower() == player_name.lower()]
        if player_row.empty:
            return None

        player = player_row.iloc[0]
        position = player["position"]
        peers = self._per90[self._per90["position"] == position]

        stat_cols_p90 = [
            c for c in self._per90.columns
            if c.endswith("_p90") or c in [
                "conversion_rate", "aerial_win_pct", "save_pct",
                "clean_sheet_pct", "pass_completion_pct",
            ]
        ]

        result = {}
        for col in stat_cols_p90:
            if col not in peers.columns:
                continue
            val = player.get(col, 0)
            pct = float(stats.percentileofscore(peers[col].dropna(), val, kind="rank"))
            result[col] = {"value": round(float(val), 3), "percentile": round(pct, 1)}

        return result
