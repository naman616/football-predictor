"""Utility helpers for Football Predictor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Any

from config.settings import (
    PROCESSED_DIR, MODELS_DIR, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    OUTCOME_LABELS,
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def confidence_label(score: float) -> tuple[str, str]:
    """Return (label, color) for a confidence score."""
    if score >= CONFIDENCE_HIGH:
        return "High", "#2dc653"
    elif score >= CONFIDENCE_MEDIUM:
        return "Medium", "#f0c040"
    else:
        return "Low", "#e63946"


def format_probability(p: float) -> str:
    return f"{p * 100:.1f}%"


def outcome_label(code: int) -> str:
    return OUTCOME_LABELS.get(code, "Unknown")


@st.cache_data(ttl=3600, show_spinner=False)
def load_processed_results() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "matches_with_elo.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_elo_ratings() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "elo_ratings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_rankings() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "team_rankings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_ratings() -> Optional[pd.DataFrame]:
    p = PROCESSED_DIR / "player_ratings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_resource(show_spinner=False)
def load_match_predictor():
    """Load match predictor from disk (cached as resource)."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.match_predictor import MatchPredictor
    predictor = MatchPredictor()
    if predictor.load():
        return predictor
    return None


@st.cache_resource(show_spinner=False)
def get_elo_system(df: pd.DataFrame):
    """Build Elo system from match history."""
    from src.models.elo_system import EloSystem
    elo = EloSystem()
    elo.process_matches(df)
    return elo


@st.cache_resource(show_spinner=False)
def get_feature_engine(df: pd.DataFrame):
    """Build feature engine from match history."""
    from src.features.match_features import MatchFeatureEngine
    return MatchFeatureEngine(df)


@st.cache_resource(show_spinner=False)
def get_team_ranker(df: pd.DataFrame, elo_ratings: pd.DataFrame):
    """Build team ranker."""
    from src.models.team_ranker import TeamRanker
    ranker = TeamRanker(df, elo_ratings)
    return ranker


@st.cache_resource(show_spinner=False)
def get_player_rater(player_df: pd.DataFrame):
    """Build player rater."""
    from src.models.player_rater import PlayerRater
    rater = PlayerRater(player_df)
    return rater


def data_is_ready() -> bool:
    """Check if all required processed data files exist."""
    required = [
        PROCESSED_DIR / "matches_with_elo.parquet",
        PROCESSED_DIR / "elo_ratings.parquet",
        PROCESSED_DIR / "player_ratings.parquet",
    ]
    return all(p.exists() for p in required)


def model_is_ready() -> bool:
    return (MODELS_DIR / "match_predictor.joblib").exists()


def get_all_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted list of all teams in dataset."""
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    return sorted(teams)


def get_flag_emoji(country: str) -> str:
    """Return flag emoji for country (best-effort)."""
    FLAGS = {
        "Argentina": "🇦🇷", "France": "🇫🇷", "Brazil": "🇧🇷",
        "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪",
        "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Italy": "🇮🇹",
        "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾",
        "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Japan": "🇯🇵",
        "South Korea": "🇰🇷", "United States": "🇺🇸", "Mexico": "🇲🇽",
        "Colombia": "🇨🇴", "Denmark": "🇩🇰", "Switzerland": "🇨🇭",
        "Austria": "🇦🇹", "Poland": "🇵🇱", "Serbia": "🇷🇸",
        "Australia": "🇦🇺", "Ecuador": "🇪🇨", "Canada": "🇨🇦",
        "Norway": "🇳🇴", "Sweden": "🇸🇪", "Ghana": "🇬🇭",
        "Nigeria": "🇳🇬", "Egypt": "🇪🇬", "Cameroon": "🇨🇲",
        "Ivory Coast": "🇨🇮", "Algeria": "🇩🇿", "Tunisia": "🇹🇳",
        "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "Qatar": "🇶🇦",
        "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Greece": "🇬🇷", "Romania": "🇷🇴",
        "Hungary": "🇭🇺", "Slovakia": "🇸🇰", "Czech Republic": "🇨🇿",
        "Peru": "🇵🇪", "Chile": "🇨🇱", "Venezuela": "🇻🇪",
        "Bolivia": "🇧🇴", "Paraguay": "🇵🇾", "Costa Rica": "🇨🇷",
        "Jamaica": "🇯🇲", "Panama": "🇵🇦", "Honduras": "🇭🇳",
        "New Zealand": "🇳🇿", "United Arab Emirates": "🇦🇪",
        "Iraq": "🇮🇶", "Sweden": "🇸🇪", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "Georgia": "🇬🇪", "El Salvador": "🇸🇻", "Guatemala": "🇬🇹",
    }
    return FLAGS.get(country, "🌍")


def form_to_html(form: list[str]) -> str:
    """Convert form list to colored HTML badges."""
    color_map = {"W": "#2dc653", "D": "#f0c040", "L": "#e63946"}
    badges = []
    for r in form:
        c = color_map.get(r, "#888")
        badges.append(
            f'<span style="background:{c};color:white;padding:2px 7px;'
            f'border-radius:4px;font-weight:bold;margin:2px;font-size:13px;">{r}</span>'
        )
    return " ".join(badges)


def rating_badge(rating: float) -> str:
    """Format an overall rating as a colored badge."""
    if rating >= 80:
        color = "#f0c040"
    elif rating >= 65:
        color = "#2dc653"
    elif rating >= 50:
        color = "#457b9d"
    else:
        color = "#888"
    return f'<span style="background:{color};color:#000;padding:4px 10px;border-radius:6px;font-weight:bold;">{rating:.0f}</span>'


def safe_metric(val: Any, default: str = "N/A") -> str:
    """Safely format a metric value."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return str(val)
    except Exception:
        return default
