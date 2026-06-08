"""Monte Carlo tournament simulator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Callable

from config.settings import N_SIMULATIONS


# ── Tournament definitions ────────────────────────────────────────────────────

WORLD_CUP_2022 = {
    "name": "FIFA World Cup 2022",
    "format": "32_team",
    "groups": {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
    "top_per_group": 2,
    "bracket": "standard_ko",
}

EURO_2024 = {
    "name": "UEFA Euro 2024",
    "format": "24_team",
    "groups": {
        "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
        "B": ["Spain", "Croatia", "Italy", "Albania"],
        "C": ["Slovenia", "Denmark", "Serbia", "England"],
        "D": ["Poland", "Netherlands", "Austria", "France"],
        "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
        "F": ["Turkey", "Georgia", "Portugal", "Czech Republic"],
    },
    "top_per_group": 2,
    "best_third": 4,
    "bracket": "euro_24",
}

COPA_AMERICA_2024 = {
    "name": "Copa América 2024",
    "format": "16_team",
    "groups": {
        "A": ["Argentina", "Peru", "Chile", "Canada"],
        "B": ["Mexico", "Ecuador", "Venezuela", "Jamaica"],
        "C": ["United States", "Uruguay", "Panama", "Bolivia"],
        "D": ["Brazil", "Colombia", "Paraguay", "Costa Rica"],
    },
    "top_per_group": 2,
    "bracket": "standard_ko",
}

WORLD_CUP_2026 = {
    "name": "FIFA World Cup 2026",
    "format": "48_team",
    "groups": {
        "A": ["United States", "Panama", "Honduras", "Serbia"],
        "B": ["Mexico", "Jamaica", "Canada", "Venezuela"],
        "C": ["Argentina", "Chile", "Peru", "New Zealand"],
        "D": ["France", "England", "Wales", "Cameroon"],
        "E": ["Spain", "Morocco", "Croatia", "Tunisia"],
        "F": ["Germany", "Japan", "Costa Rica", "South Korea"],
        "G": ["Brazil", "Colombia", "Ecuador", "Saudi Arabia"],
        "H": ["Portugal", "Uruguay", "Algeria", "United Arab Emirates"],
        "I": ["Netherlands", "Senegal", "Australia", "Poland"],
        "J": ["Italy", "Nigeria", "Denmark", "Iran"],
        "K": ["Belgium", "Ghana", "Sweden", "Iraq"],
        "L": ["Switzerland", "Turkey", "Ivory Coast", "Norway"],
    },
    "top_per_group": 2,
    "best_third": 8,
    "bracket": "48_team",
}

TOURNAMENTS = {
    "FIFA World Cup 2022": WORLD_CUP_2022,
    "UEFA Euro 2024": EURO_2024,
    "Copa América 2024": COPA_AMERICA_2024,
    "FIFA World Cup 2026": WORLD_CUP_2026,
}


class TournamentSimulator:
    """
    Monte Carlo tournament simulator.

    Uses a match predictor function to simulate each match.
    Runs N_SIMULATIONS and aggregates probabilities.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str, bool], tuple[float, float, float]],
    ):
        """
        predict_fn: callable(home_team, away_team, neutral=True)
                    returns (p_home_win, p_draw, p_away_win)
        """
        self.predict_fn = predict_fn
        self._rng = np.random.default_rng(42)

    def _simulate_match(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        allow_draw: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Simulate a single match. Returns winner team name (or None for draw)."""
        if rng is None:
            rng = self._rng

        p_home, p_draw, p_away = self.predict_fn(team_a, team_b, neutral)

        r = rng.random()
        if not allow_draw:
            p_a_wins = p_home + p_draw * (p_home / (p_home + p_away + 1e-9))
            return team_a if r < p_a_wins else team_b
        else:
            total = p_home + p_draw + p_away
            if r < p_home / total:
                return team_a
            elif r < (p_home + p_draw) / total:
                return "draw"
            else:
                return team_b

    def _simulate_group(
        self,
        teams: list[str],
        rng: np.random.Generator,
    ) -> list[dict]:
        """Simulate round-robin group and return standings."""
        table = {t: [0, 0, 0, 0, 0, 0, 0, 0] for t in teams}
        #          P  W  D  L  GF GA GD Pts

        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                p_home, p_draw, p_away = self.predict_fn(home, away, neutral=True)
                total = p_home + p_draw + p_away
                r = rng.random()

                # Simulate scoreline
                exp_hg = 1.5 * (p_home + 0.3 * p_draw)
                exp_ag = 1.5 * (p_away + 0.3 * p_draw)
                hg = rng.poisson(max(0.3, exp_hg))
                ag = rng.poisson(max(0.3, exp_ag))

                threshold_h = p_home / total
                threshold_d = (p_home + p_draw) / total

                if r < threshold_h:   # home win
                    if hg <= ag:
                        hg = ag + 1
                    table[home][1] += 1; table[home][7] += 3; table[away][3] += 1
                    outcome = 0
                elif r < threshold_d:  # draw
                    ag = hg
                    table[home][2] += 1; table[home][7] += 1
                    table[away][2] += 1; table[away][7] += 1
                    outcome = 1
                else:                  # away win
                    if ag <= hg:
                        ag = hg + 1
                    table[away][1] += 1; table[away][7] += 3; table[home][3] += 1
                    outcome = 2

                table[home][0] += 1; table[away][0] += 1
                table[home][4] += hg; table[home][5] += ag; table[home][6] += hg - ag
                table[away][4] += ag; table[away][5] += hg; table[away][6] += ag - hg

        standings = [
            {"team": t, "P": v[0], "W": v[1], "D": v[2], "L": v[3],
             "GF": v[4], "GA": v[5], "GD": v[6], "Pts": v[7]}
            for t, v in table.items()
        ]
        standings.sort(key=lambda x: (x["Pts"], x["GD"], x["GF"]), reverse=True)
        return standings

    def _run_single_simulation(
        self,
        tournament: dict,
        rng: np.random.Generator,
    ) -> dict[str, str]:
        """
        Run one tournament simulation.
        Returns dict mapping team -> furthest stage reached.
        """
        stages_reached = {}
        groups = tournament["groups"]
        top_n = tournament["top_per_group"]

        # Group stage
        group_winners = {}
        group_runners_up = {}
        all_third_place = []

        for group_name, teams in groups.items():
            standings = self._simulate_group(teams, rng)
            group_winners[group_name] = standings[0]["team"]
            group_runners_up[group_name] = standings[1]["team"]

            for team_row in standings:
                stages_reached[team_row["team"]] = "Group Stage"

            if len(standings) > 2:
                all_third_place.append({
                    "team": standings[2]["team"],
                    "pts": standings[2]["Pts"],
                    "gd": standings[2]["GD"],
                    "gf": standings[2]["GF"],
                    "group": group_name,
                })

        # Update group qualifiers
        qualifiers = list(group_winners.values()) + list(group_runners_up.values())
        for t in qualifiers:
            stages_reached[t] = "Round of 16"

        # Best third-place teams for Euro/WC2026 format
        best_third = tournament.get("best_third", 0)
        if best_third > 0 and all_third_place:
            all_third_place.sort(
                key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True
            )
            for rec in all_third_place[:best_third]:
                qualifiers.append(rec["team"])
                stages_reached[rec["team"]] = "Round of 16"

        # Knockout rounds
        ko_teams = qualifiers[:]
        ko_teams = self._pad_bracket(ko_teams)

        round_names = self._get_round_names(len(ko_teams))

        for round_name in round_names:
            next_round = []
            for i in range(0, len(ko_teams), 2):
                if i + 1 >= len(ko_teams):
                    next_round.append(ko_teams[i])
                    stages_reached[ko_teams[i]] = round_name
                    continue
                winner = self._simulate_match(
                    ko_teams[i], ko_teams[i + 1], neutral=True,
                    allow_draw=False, rng=rng,
                )
                loser = ko_teams[i + 1] if winner == ko_teams[i] else ko_teams[i]
                stages_reached[loser] = round_name
                next_round.append(winner)

            ko_teams = next_round

        # Champion
        if ko_teams:
            stages_reached[ko_teams[0]] = "Champion"

        return stages_reached

    def _pad_bracket(self, teams: list[str]) -> list[str]:
        """Pad bracket to next power of 2 if needed."""
        n = len(teams)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        while len(teams) < next_pow2:
            teams.append(f"__bye_{len(teams)}__")
        return teams

    def _get_round_names(self, n_teams: int) -> list[str]:
        """Return round names based on bracket size."""
        mapping = {
            64: ["Round of 64", "Round of 32", "Quarter-Final", "Semi-Final", "Final"],
            32: ["Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Final"],
            16: ["Round of 16", "Quarter-Final", "Semi-Final", "Final"],
            8: ["Quarter-Final", "Semi-Final", "Final"],
            4: ["Semi-Final", "Final"],
            2: ["Final"],
        }
        return mapping.get(n_teams, ["Round 1", "Semi-Final", "Final"])

    def _build_prob_cache(
        self,
        tournament: dict,
        batch_predict_fn: Optional[Callable] = None,
    ) -> dict:
        """
        Pre-compute all pairwise match probabilities for the tournament.

        For a 32-team tournament this is 32*31 = 992 pairs; for 48-team 2,256 pairs.
        Done once before the Monte Carlo loop so each simulation uses O(1) dict lookups
        instead of re-running the full ML pipeline (23 ms/call → 2.5 h for 5k sims).
        If batch_predict_fn is provided it receives a list of (home, away) and returns
        a list of (p_home, p_draw, p_away); otherwise falls back to single calls.
        """
        all_teams = list({t for teams in tournament["groups"].values() for t in teams})
        pairs = [(ta, tb) for ta in all_teams for tb in all_teams if ta != tb]
        cache: dict[tuple[str, str], tuple[float, float, float]] = {}

        if batch_predict_fn is not None:
            results = batch_predict_fn(pairs)
            for (ta, tb), probs in zip(pairs, results):
                cache[(ta, tb)] = probs
        else:
            for ta, tb in pairs:
                cache[(ta, tb)] = self.predict_fn(ta, tb, neutral=True)

        return cache

    def simulate(
        self,
        tournament_name: str,
        n_sims: int = N_SIMULATIONS,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        precompute_callback: Optional[Callable[[], None]] = None,
        batch_predict_fn: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Run N Monte Carlo simulations and return probability table.

        Returns DataFrame with columns:
            team, group, champion_pct, final_pct, semi_pct, ...
        """
        tournament = TOURNAMENTS.get(tournament_name)
        if tournament is None:
            raise ValueError(f"Unknown tournament: {tournament_name}")

        # Pre-compute all pairwise probabilities once — turns 2.5 h into ~40 s
        if precompute_callback:
            precompute_callback()
        prob_cache = self._build_prob_cache(tournament, batch_predict_fn)

        # Wrap predict_fn to use cache; fall back to live call for unseen pairs
        # NOTE: must NOT use dict.get(key, _orig(...)) — Python evaluates the
        # default eagerly, which would call ML inference on every iteration.
        _orig = self.predict_fn
        def _cached(home: str, away: str, neutral: bool = True):
            if home.startswith("__bye") or away.startswith("__bye"):
                return 0.99, 0.005, 0.005
            cached = prob_cache.get((home, away))
            if cached is not None:
                return cached
            return _orig(home, away, neutral)
        self.predict_fn = _cached

        try:
            stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            all_teams = []
            for grp, teams in tournament["groups"].items():
                for t in teams:
                    all_teams.append((t, grp))
                    stage_counts[t]  # Initialize

            rng = np.random.default_rng(2024)
            for sim_i in range(n_sims):
                if progress_callback and sim_i % max(1, n_sims // 20) == 0:
                    progress_callback(sim_i, n_sims)

                result = self._run_single_simulation(tournament, rng)
                for team, stage in result.items():
                    if not team.startswith("__bye"):
                        stage_counts[team][stage] += 1

        finally:
            self.predict_fn = _orig

        # Build results table
        stage_hierarchy = [
            "Group Stage", "Round of 64", "Round of 32", "Round of 16",
            "Quarter-Final", "Semi-Final", "Final", "Champion",
        ]

        records = []
        for team, group in all_teams:
            counts = stage_counts[team]
            row = {"team": team, "group": group}

            cumulative = 0
            for stage in reversed(stage_hierarchy):
                cumulative += counts.get(stage, 0)
                row[f"p_{stage.lower().replace(' ', '_').replace('-', '_')}"] = round(cumulative / n_sims * 100, 1)

            records.append(row)

        df = pd.DataFrame(records)
        return df

    def simulate_match_chain(
        self,
        bracket: list[tuple[str, str]],
        n_sims: int = 1000,
    ) -> pd.DataFrame:
        """Simulate a custom bracket and return win probabilities."""
        win_counts: dict[str, int] = defaultdict(int)
        rng = np.random.default_rng(2024)

        for _ in range(n_sims):
            remaining = list(bracket)
            winners = []
            for home, away in remaining:
                w = self._simulate_match(home, away, neutral=True, allow_draw=False, rng=rng)
                winners.append(w)
            for w in winners:
                win_counts[w] += 1

        records = []
        for (home, away) in bracket:
            for team in [home, away]:
                records.append({
                    "team": team,
                    "win_probability": round(win_counts[team] / n_sims * 100, 1),
                })
        return pd.DataFrame(records)
