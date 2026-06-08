"""Player Comparison Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Player Comparison · Football Predictor", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 20px 0 12px;
}
.compare-header {
    background: #1c1e26; border-radius: 12px; padding: 20px;
    text-align: center; border: 2px solid;
}
.p1-header { border-color: #52b788; }
.p2-header { border-color: #f0c040; }
.compare-header .rating { font-size: 3rem; font-weight: 900; }
.p1-header .rating { color: #52b788; }
.p2-header .rating { color: #f0c040; }
.compare-header .name { font-size: 1.2rem; font-weight: 600; color: #e0e0e0; }
.compare-header .meta { color: #888; font-size: 0.85rem; margin-top: 4px; }
.insight-card {
    background: #1c1e26; border-left: 3px solid #52b788;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_resources():
    from src.utils.helpers import load_player_ratings, get_player_rater
    from src.data.loader import load_player_data
    players = load_player_ratings()
    if players is not None and not players.empty:
        player_df = load_player_data()
        rater = get_player_rater(player_df)
        rater.rated = players
        rater._per90 = players
        return players, rater
    return None, None


players, rater = get_resources()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Player Comparison")
st.markdown("Compare any two players with radar charts, statistics, and percentile rankings.")

if players is None or players.empty:
    st.error("Player data unavailable.")
    st.stop()


# ── Player Selection ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Select Players to Compare</div>", unsafe_allow_html=True)

player_names = sorted(players["name"].unique().tolist())

col1, col_vs, col2 = st.columns([5, 1, 5])

# Defaults: Mbappé vs Haaland
default_p1 = next((i for i, n in enumerate(player_names) if "Mbapp" in n), 0)
default_p2 = next((i for i, n in enumerate(player_names) if "Haaland" in n), 1)

with col1:
    player1_name = st.selectbox("Player 1", player_names, index=default_p1, key="p1")
with col_vs:
    st.markdown("<br><br><center style='color:#52b788;font-size:1.3rem;font-weight:700'>VS</center>", unsafe_allow_html=True)
with col2:
    player2_name = st.selectbox("Player 2", player_names, index=default_p2, key="p2")

if player1_name == player2_name:
    st.warning("Select two different players.")
    st.stop()


# ── Get player data ────────────────────────────────────────────────────────────
p1_df = players[players["name"] == player1_name]
p2_df = players[players["name"] == player2_name]

if p1_df.empty or p2_df.empty:
    st.error("Player data not found.")
    st.stop()

p1 = p1_df.iloc[0]
p2 = p2_df.iloc[0]

from src.utils.helpers import get_flag_emoji

# ── Profile headers ────────────────────────────────────────────────────────────
st.markdown("---")
col_h1, col_mid, col_h2 = st.columns([5, 1, 5])

with col_h1:
    flag1 = get_flag_emoji(p1.get("nationality", ""))
    st.markdown(f"""
    <div class="compare-header p1-header">
        <div class="rating">{p1.get('overall_rating', 0):.0f}</div>
        <div class="name">{p1['name']}</div>
        <div class="meta">{flag1} {p1.get('nationality','')} · {p1.get('position','')}</div>
        <div class="meta">{p1.get('club','')} · {p1.get('league','')}</div>
    </div>""", unsafe_allow_html=True)

with col_mid:
    st.markdown("")

with col_h2:
    flag2 = get_flag_emoji(p2.get("nationality", ""))
    st.markdown(f"""
    <div class="compare-header p2-header">
        <div class="rating">{p2.get('overall_rating', 0):.0f}</div>
        <div class="name">{p2['name']}</div>
        <div class="meta">{flag2} {p2.get('nationality','')} · {p2.get('position','')}</div>
        <div class="meta">{p2.get('club','')} · {p2.get('league','')}</div>
    </div>""", unsafe_allow_html=True)


# ── Determine comparison stats based on positions ────────────────────────────
def get_comparison_stats(player1: pd.Series, player2: pd.Series) -> dict:
    """Select relevant comparison stats based on positions."""
    pos1 = player1.get("position", "FW")
    pos2 = player2.get("position", "FW")

    # Universal stats available for both
    common_stats = {
        "Goals/90": ("goals_p90", "goals_p90"),
        "Assists/90": ("assists_p90", "assists_p90"),
        "xG/90": ("xG_p90", "xG_p90"),
        "xA/90": ("xA_p90", "xA_p90"),
        "Key Passes/90": ("key_passes_p90", "key_passes_p90"),
        "Tackles/90": ("tackles_p90", "tackles_p90"),
        "Interceptions/90": ("interceptions_p90", "interceptions_p90"),
        "Pass Completion": ("pass_completion_pct", "pass_completion_pct"),
        "Prog. Carries/90": ("progressive_carries_p90", "progressive_carries_p90"),
        "Prog. Passes/90": ("progressive_passes_p90", "progressive_passes_p90"),
    }

    result = {}
    for label, (c1, c2) in common_stats.items():
        v1 = float(player1.get(c1, 0) or 0)
        v2 = float(player2.get(c2, 0) or 0)
        if v1 > 0 or v2 > 0:
            result[label] = {"p1": v1, "p2": v2}

    return result


comparison_stats = get_comparison_stats(p1, p2)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Radar Chart", "Statistics", "Percentile Comparison", "Analytics"])

with tab1:
    st.markdown("<div class='section-header'>Radar Comparison</div>", unsafe_allow_html=True)

    # Build radar data from overall, attack, defense + key stats
    radar_cats = ["Overall", "Attack", "Defense"]
    p1_radar = [
        p1.get("overall_rating", 50),
        p1.get("attack_rating", 50),
        p1.get("defense_rating", 50),
    ]
    p2_radar = [
        p2.get("overall_rating", 50),
        p2.get("attack_rating", 50),
        p2.get("defense_rating", 50),
    ]

    # Add position-specific stats as percentiles
    from scipy import stats as scipy_stats

    pos_p90_map = {
        "FW": ["goals_p90", "assists_p90", "xG_p90", "dribbles_completed_p90"],
        "MF": ["progressive_passes_p90", "xA_p90", "tackles_p90", "key_passes_p90"],
        "DF": ["tackles_p90", "interceptions_p90", "clearances_p90", "aerial_win_pct"],
        "GK": ["save_pct", "clean_sheet_pct", "psxg_ga_p90", "pass_completion_pct"],
    }
    stat_labels = {
        "goals_p90": "Goals/90", "assists_p90": "Assists/90",
        "xG_p90": "xG/90", "dribbles_completed_p90": "Dribbles/90",
        "progressive_passes_p90": "Prog.Passes/90", "xA_p90": "xA/90",
        "tackles_p90": "Tackles/90", "key_passes_p90": "Key Passes/90",
        "interceptions_p90": "Interceptions/90", "clearances_p90": "Clearances/90",
        "aerial_win_pct": "Aerial Win%",
        "save_pct": "Save%", "clean_sheet_pct": "CS%",
        "psxg_ga_p90": "Goals Prev/90", "pass_completion_pct": "Pass%",
    }

    # Use the position of player 1 for extra stats
    extra_cols = pos_p90_map.get(p1.get("position", "FW"), [])[:4]
    peers = players[players["position"] == p1.get("position", "FW")]

    for col in extra_cols:
        if col not in players.columns:
            continue
        # Convert to percentile
        p1_val = float(p1.get(col, 0) or 0)
        p2_val = float(p2.get(col, 0) or 0)
        peer_vals = peers[col].dropna().values

        if len(peer_vals) > 0:
            p1_pct = float(scipy_stats.percentileofscore(peer_vals, p1_val, kind="rank"))
            p2_pct = float(scipy_stats.percentileofscore(peer_vals, p2_val, kind="rank"))
            radar_cats.append(stat_labels.get(col, col))
            p1_radar.append(p1_pct)
            p2_radar.append(p2_pct)

    # Close radar
    p1_radar_c = p1_radar + [p1_radar[0]]
    p2_radar_c = p2_radar + [p2_radar[0]]
    cats_c = radar_cats + [radar_cats[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=p1_radar_c, theta=cats_c,
        fill="toself", name=player1_name,
        line_color="#52b788", fillcolor="rgba(82,183,136,0.25)",
        line_width=2,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=p2_radar_c, theta=cats_c,
        fill="toself", name=player2_name,
        line_color="#f0c040", fillcolor="rgba(240,192,64,0.2)",
        line_width=2,
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2a2d3a",
                            tickfont=dict(size=9, color="#888")),
            angularaxis=dict(gridcolor="#2a2d3a"),
            bgcolor="#0f1117",
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="#e0e0e0", family="Inter"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.3),
        margin=dict(l=60, r=60, t=60, b=80),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


with tab2:
    st.markdown("<div class='section-header'>Statistical Comparison</div>", unsafe_allow_html=True)

    if comparison_stats:
        from src.utils.charts import comparison_bar_chart
        fig_bars = comparison_bar_chart(
            player1_name, player2_name, comparison_stats,
            title="Key Statistics (per 90 minutes)",
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Side-by-side stats table
    st.markdown("**Raw Statistics:**")
    raw_stats_labels = {
        "Apps": ("apps", "apps"),
        "Goals": ("goals", "goals"),
        "Assists": ("assists", "assists"),
        "xG": ("xG", "xG"),
        "xA": ("xA", "xA"),
        "Minutes": ("mins_played", "mins_played"),
    }
    rows = []
    for label, (c1, c2) in raw_stats_labels.items():
        v1 = p1.get(c1, 0)
        v2 = p2.get(c2, 0)
        if pd.notna(v1) and pd.notna(v2):
            v1f = f"{int(v1)}" if isinstance(v1, (int, float)) and v1 == int(v1) else f"{float(v1):.1f}"
            v2f = f"{int(v2)}" if isinstance(v2, (int, float)) and v2 == int(v2) else f"{float(v2):.1f}"
            winner = "→" if float(v1) > float(v2) else ("←" if float(v2) > float(v1) else "=")
            rows.append({"Stat": label, player1_name: v1f, "": winner, player2_name: v2f})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab3:
    st.markdown("<div class='section-header'>Percentile Rankings vs Position Peers</div>", unsafe_allow_html=True)

    if rater is not None:
        pct1 = rater.get_percentile_stats(player1_name)
        pct2 = rater.get_percentile_stats(player2_name)

        col_pct1, col_pct2 = st.columns(2)
        with col_pct1:
            st.markdown(f"**{player1_name}** (vs {p1.get('position','?')} peers)")
            if pct1:
                from src.utils.charts import player_percentile_bar
                fig_p1 = player_percentile_bar(pct1, p1.get("position", "FW"))
                st.plotly_chart(fig_p1, use_container_width=True)
            else:
                st.info("No percentile data available.")

        with col_pct2:
            st.markdown(f"**{player2_name}** (vs {p2.get('position','?')} peers)")
            if pct2:
                from src.utils.charts import player_percentile_bar
                fig_p2 = player_percentile_bar(pct2, p2.get("position", "FW"))
                st.plotly_chart(fig_p2, use_container_width=True)
            else:
                st.info("No percentile data available.")

        # Combined percentile table
        if pct1 and pct2:
            st.markdown("**Combined Percentile Table:**")
            all_cats = sorted(set(pct1.keys()) | set(pct2.keys()))
            label_map = {
                "goals_p90": "Goals/90", "assists_p90": "Assists/90",
                "xG_p90": "xG/90", "xA_p90": "xA/90",
                "key_passes_p90": "Key Passes/90", "tackles_p90": "Tackles/90",
                "interceptions_p90": "Interceptions/90", "clearances_p90": "Clearances/90",
                "progressive_passes_p90": "Prog. Passes/90", "pass_completion_pct": "Pass%",
            }
            table_rows = []
            for cat in all_cats:
                if cat not in label_map:
                    continue
                v1 = pct1.get(cat, {}).get("percentile", "-")
                v2 = pct2.get(cat, {}).get("percentile", "-")
                if v1 != "-" and v2 != "-":
                    better = player1_name if float(v1) > float(v2) else (
                        player2_name if float(v2) > float(v1) else "Tied"
                    )
                    table_rows.append({
                        "Stat": label_map[cat],
                        f"{player1_name} %ile": f"{v1:.0f}th",
                        f"{player2_name} %ile": f"{v2:.0f}th",
                        "Edge": better,
                    })
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


with tab4:
    st.markdown("<div class='section-header'>Analytical Insights</div>", unsafe_allow_html=True)

    # Auto-generate comparison insights
    insights = []

    # Overall rating
    overall_diff = p1.get("overall_rating", 50) - p2.get("overall_rating", 50)
    if abs(overall_diff) < 3:
        insights.append(f"**Equally matched overall** — Both players sit within {abs(overall_diff):.1f} rating points of each other, making this an extremely close comparison.")
    elif overall_diff > 0:
        insights.append(f"**{player1_name} edges overall** — {overall_diff:.1f} point overall advantage, reflecting superior consistency across multiple metrics.")
    else:
        insights.append(f"**{player2_name} edges overall** — {abs(overall_diff):.1f} point overall advantage.")

    # Attacking comparison
    g1 = float(p1.get("goals_p90", 0) or 0)
    g2 = float(p2.get("goals_p90", 0) or 0)
    if g1 > 0 or g2 > 0:
        if abs(g1 - g2) > 0.1:
            better_scorer = player1_name if g1 > g2 else player2_name
            worse_scorer = player2_name if g1 > g2 else player1_name
            insights.append(
                f"**Scoring rate:** {better_scorer} scores at {max(g1,g2):.2f} goals/90 "
                f"vs {min(g1,g2):.2f} for {worse_scorer} — a meaningful difference at the top level."
            )

    # xG vs actual goals (finishing quality)
    xg1 = float(p1.get("xG_p90", 0) or 0)
    xg2 = float(p2.get("xG_p90", 0) or 0)
    if g1 > 0 and xg1 > 0:
        overperform1 = g1 - xg1
        overperform2 = g2 - xg2 if xg2 > 0 and g2 > 0 else 0
        if abs(overperform1) > 0.05:
            direction = "above" if overperform1 > 0 else "below"
            insights.append(
                f"**Finishing quality ({player1_name}):** Scoring {abs(overperform1):.2f} goals/90 "
                f"{direction} xG expectation — {'clinical finisher' if overperform1 > 0 else 'wasteful in front of goal'}."
            )
        if abs(overperform2) > 0.05:
            direction = "above" if overperform2 > 0 else "below"
            insights.append(
                f"**Finishing quality ({player2_name}):** Scoring {abs(overperform2):.2f} goals/90 "
                f"{direction} xG expectation."
            )

    # Defensive work rate
    t1 = float(p1.get("tackles_p90", 0) or 0)
    t2 = float(p2.get("tackles_p90", 0) or 0)
    if max(t1, t2) > 0.3:
        harder_worker = player1_name if t1 > t2 else player2_name
        insights.append(
            f"**Defensive work rate:** {harder_worker} contributes more defensively "
            f"({max(t1,t2):.2f} vs {min(t1,t2):.2f} tackles/90)."
        )

    # Age & career trajectory
    age1 = int(p1.get("age", 0) or 0)
    age2 = int(p2.get("age", 0) or 0)
    if age1 > 0 and age2 > 0:
        if abs(age1 - age2) >= 2:
            younger = player1_name if age1 < age2 else player2_name
            older = player2_name if age1 < age2 else player1_name
            younger_age = min(age1, age2)
            insights.append(
                f"**Career stage:** {younger} ({younger_age}) has more prime years ahead "
                f"vs {older} ({max(age1,age2)}). Long-term value favors {younger}."
            )

    # Pass quality
    pc1 = float(p1.get("pass_completion_pct", 0) or 0)
    pc2 = float(p2.get("pass_completion_pct", 0) or 0)
    if pc1 > 70 and pc2 > 70:
        better_passer = player1_name if pc1 > pc2 else player2_name
        insights.append(
            f"**Passing precision:** {better_passer} completes passes at {max(pc1,pc2):.1f}% "
            f"vs {min(pc1,pc2):.1f}% — higher recycling reliability."
        )

    if not insights:
        insights.append("Insufficient data for detailed insights. More statistical data would improve this analysis.")

    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # Quick verdict
    st.markdown("---")
    st.markdown("**Summary Verdict:**")
    overall1 = p1.get("overall_rating", 50)
    overall2 = p2.get("overall_rating", 50)
    attack1 = p1.get("attack_rating", 50)
    attack2 = p2.get("attack_rating", 50)
    defense1 = p1.get("defense_rating", 50)
    defense2 = p2.get("defense_rating", 50)

    p1_wins = sum([
        1 if overall1 > overall2 else 0,
        1 if attack1 > attack2 else 0,
        1 if defense1 > defense2 else 0,
        1 if g1 > g2 else 0,
        1 if float(p1.get("assists_p90", 0) or 0) > float(p2.get("assists_p90", 0) or 0) else 0,
    ])
    p2_wins = 5 - p1_wins

    if p1_wins > p2_wins:
        verdict = f"**{player1_name}** leads in {p1_wins}/5 measured categories."
        st.success(verdict)
    elif p2_wins > p1_wins:
        verdict = f"**{player2_name}** leads in {p2_wins}/5 measured categories."
        st.success(verdict)
    else:
        st.info("Genuinely even comparison — both players are statistically equivalent across key metrics.")
