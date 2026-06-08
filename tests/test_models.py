"""Basic unit tests for Football Predictor models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


class TestEloSystem:
    def setup_method(self):
        from src.models.elo_system import EloSystem
        self.elo = EloSystem()

    def test_initial_rating(self):
        assert self.elo.get_rating("NewTeam") == 1500.0

    def test_update_win(self):
        h_new, a_new = self.elo.update(
            "TeamA", "TeamB", 2, 0, "Friendly", neutral=True
        )
        assert h_new > 1500.0
        assert a_new < 1500.0

    def test_update_draw(self):
        h_new, a_new = self.elo.update(
            "TeamA", "TeamB", 1, 1, "Friendly", neutral=True
        )
        # Equal teams drawing should stay near 1500
        assert abs(h_new - 1500.0) < 10
        assert abs(a_new - 1500.0) < 10

    def test_predict_outcome_sums_to_one(self):
        self.elo.ratings["TeamA"] = 1800.0
        self.elo.ratings["TeamB"] = 1400.0
        p1, p2, p3 = self.elo.predict_outcome("TeamA", "TeamB")
        assert abs(p1 + p2 + p3 - 1.0) < 1e-6

    def test_stronger_team_favored(self):
        self.elo.ratings["Strong"] = 1900.0
        self.elo.ratings["Weak"] = 1200.0
        p_win, _, _ = self.elo.predict_outcome("Strong", "Weak", neutral=True)
        assert p_win > 0.7

    def test_process_matches(self):
        df = pd.DataFrame([
            {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1, "tournament": "Friendly",
             "neutral": True, "result": 0, "year": 2020},
            {"date": pd.Timestamp("2020-02-01"), "home_team": "C", "away_team": "A",
             "home_score": 0, "away_score": 3, "tournament": "Friendly",
             "neutral": True, "result": 2, "year": 2020},
        ])
        result = self.elo.process_matches(df)
        assert "home_elo_before" in result.columns
        assert "away_elo_before" in result.columns
        assert len(result) == 2


class TestMatchFeatures:
    def setup_method(self):
        self.df = pd.DataFrame([
            {"date": pd.Timestamp(f"2020-{m:02d}-01"), "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1, "tournament": "Friendly",
             "neutral": True, "result": 0, "year": 2020,
             "home_elo_before": 1600.0, "away_elo_before": 1400.0}
            for m in range(1, 13)
        ])

    def test_feature_engine_creates_features(self):
        from src.features.match_features import MatchFeatureEngine, FEATURE_COLS
        engine = MatchFeatureEngine(self.df)
        feats = engine.get_features_for_prediction(
            "A", "B", 1600.0, 1400.0, "Friendly", True
        )
        assert isinstance(feats, pd.DataFrame)
        assert set(FEATURE_COLS).issubset(set(feats.columns))
        assert len(feats) == 1

    def test_elo_diff_correct(self):
        from src.features.match_features import MatchFeatureEngine
        engine = MatchFeatureEngine(self.df)
        feats = engine.get_features_for_prediction("A", "B", 1600.0, 1400.0)
        assert abs(feats["elo_diff"].iloc[0] - 200.0) < 1.0


class TestPlayerRater:
    def setup_method(self):
        from src.data.loader import _get_known_players
        self.player_df = pd.DataFrame(_get_known_players())

    def test_compute_ratings_returns_df(self):
        from src.models.player_rater import PlayerRater
        rater = PlayerRater(self.player_df)
        rated = rater.compute_ratings(min_apps=1, min_mins=100)
        assert isinstance(rated, pd.DataFrame)
        assert len(rated) > 0

    def test_overall_rating_range(self):
        from src.models.player_rater import PlayerRater
        rater = PlayerRater(self.player_df)
        rated = rater.compute_ratings(min_apps=1, min_mins=100)
        if not rated.empty:
            assert rated["overall_rating"].between(0, 100).all()

    def test_positions_covered(self):
        from src.models.player_rater import PlayerRater
        rater = PlayerRater(self.player_df)
        rated = rater.compute_ratings(min_apps=1, min_mins=100)
        if not rated.empty:
            positions = set(rated["position"].unique())
            assert len(positions) >= 3  # FW, MF, DF at minimum


class TestTournamentSimulator:
    def setup_method(self):
        # Simple predict function
        def predict_fn(h, a, neutral=True):
            return 0.45, 0.25, 0.30
        self.predict_fn = predict_fn

    def test_simulation_runs(self):
        from src.models.tournament_simulator import TournamentSimulator
        sim = TournamentSimulator(self.predict_fn)
        results = sim.simulate("FIFA World Cup 2022", n_sims=100)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 32  # 32 teams

    def test_champion_probabilities_sum_to_100(self):
        from src.models.tournament_simulator import TournamentSimulator
        sim = TournamentSimulator(self.predict_fn)
        results = sim.simulate("FIFA World Cup 2022", n_sims=100)
        # Champion probs should sum to ~100%
        champ_col = "p_champion"
        if champ_col in results.columns:
            total = results[champ_col].sum()
            assert 90 <= total <= 110  # Allows for rounding

    def test_all_teams_included(self):
        from src.models.tournament_simulator import TournamentSimulator, WORLD_CUP_2022
        sim = TournamentSimulator(self.predict_fn)
        results = sim.simulate("FIFA World Cup 2022", n_sims=50)
        all_teams = [t for teams in WORLD_CUP_2022["groups"].values() for t in teams]
        result_teams = set(results["team"].tolist())
        for team in all_teams:
            assert team in result_teams, f"{team} missing from results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
