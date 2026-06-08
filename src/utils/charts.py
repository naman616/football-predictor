"""Plotly chart utilities for Football Predictor."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1a472a",
    "secondary": "#2d6a4f",
    "accent": "#52b788",
    "gold": "#f0c040",
    "red": "#e63946",
    "blue": "#457b9d",
    "light": "#d8f3dc",
    "dark": "#081c15",
    "win": "#2dc653",
    "draw": "#f0c040",
    "loss": "#e63946",
    "bg": "#0f1117",
    "card_bg": "#1c1e26",
    "text": "#e0e0e0",
}

CONF_COLORS = {
    "UEFA": "#3a86ff",
    "CONMEBOL": "#ffbe0b",
    "CONCACAF": "#fb5607",
    "CAF": "#8338ec",
    "AFC": "#06d6a0",
    "OFC": "#ef233c",
    "Other": "#adb5bd",
}


def _dark_template() -> go.layout.Template:
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a"),
            yaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a"),
            colorway=[
                COLORS["accent"], COLORS["gold"], COLORS["blue"],
                COLORS["red"], "#ff99c8", "#9bf6ff",
            ],
        )
    )


DARK_TEMPLATE = _dark_template()


def prediction_gauge(p_home: float, p_draw: float, p_away: float,
                     home: str, away: str) -> go.Figure:
    """Three-way probability bar for match prediction."""
    fig = go.Figure()

    labels = [f"{home}\nWin", "Draw", f"{away}\nWin"]
    values = [p_home * 100, p_draw * 100, p_away * 100]
    colors = [COLORS["win"], COLORS["draw"], COLORS["loss"]]

    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        fig.add_trace(go.Bar(
            x=[val],
            y=[0],
            orientation="h",
            name=label,
            marker_color=color,
            marker_line_width=0,
            text=f"<b>{val:.1f}%</b>",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"{label}: {val:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        template=DARK_TEMPLATE,
        height=100,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            range=[0, 100],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def shap_waterfall(shap_values: np.ndarray, feature_names: list[str],
                   base_value: float, predicted_class: str) -> go.Figure:
    """SHAP waterfall chart showing feature contributions."""
    n = min(10, len(shap_values))
    # Sort by absolute value
    idx = np.argsort(np.abs(shap_values))[-n:][::-1]
    vals = shap_values[idx]
    names = [feature_names[i] for i in idx]

    # Rename features for readability
    rename = {
        "elo_diff": "Elo Difference",
        "elo_home": "Home Elo",
        "elo_away": "Away Elo",
        "form_home_ppg": "Home Form (PPG)",
        "form_away_ppg": "Away Form (PPG)",
        "form_home_gf": "Home Goals/Game",
        "form_away_gf": "Away Goals/Game",
        "form_home_ga": "Home GA/Game",
        "form_away_ga": "Away GA/Game",
        "h2h_home_win_rate": "H2H Win Rate",
        "h2h_draw_rate": "H2H Draw Rate",
        "tournament_importance": "Tournament Level",
        "is_neutral": "Neutral Venue",
        "form_diff_ppg": "Form Differential",
        "elo_diff_sq": "Elo Diff (squared)",
        "gf_diff": "Goals Diff",
        "ga_diff": "Defense Diff",
    }
    names = [rename.get(n, n.replace("_", " ").title()) for n in names]

    colors = [COLORS["win"] if v > 0 else COLORS["loss"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Feature Contributions → {predicted_class}", font_size=14),
        template=DARK_TEMPLATE,
        height=400,
        margin=dict(l=20, r=60, t=50, b=20),
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def feature_importance_bar(importances: pd.Series, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top = importances.head(top_n).sort_values()
    rename = {
        "elo_diff": "Elo Difference",
        "elo_home": "Home Elo Rating",
        "elo_away": "Away Elo Rating",
        "form_home_ppg": "Home Recent Form",
        "form_away_ppg": "Away Recent Form",
        "form_diff_ppg": "Form Differential",
        "tournament_importance": "Tournament Importance",
        "h2h_home_win_rate": "Head-to-Head Record",
        "elo_diff_sq": "Elo Gap (non-linear)",
        "form_home_gf": "Home Goals Scored",
        "form_away_gf": "Away Goals Scored",
        "is_neutral": "Neutral Venue",
        "gf_diff": "Goals Differential",
    }
    labels = [rename.get(n, n.replace("_", " ").title()) for n in top.index]

    fig = go.Figure(go.Bar(
        x=top.values,
        y=labels,
        orientation="h",
        marker_color=COLORS["accent"],
        marker_line_width=0,
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=350,
        margin=dict(l=10, r=30, t=10, b=30),
        xaxis_title="Importance",
        showlegend=False,
    )
    return fig


def elo_history_chart(history_df: pd.DataFrame, team: str) -> go.Figure:
    """Line chart of Elo rating over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["date"],
        y=history_df["elo"],
        mode="lines",
        name=team,
        line=dict(color=COLORS["accent"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.0f}<extra></extra>",
    ))
    fig.add_hline(
        y=1500, line_dash="dash", line_color="#555",
        annotation_text="Average (1500)", annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"{team} — Elo Rating History",
        template=DARK_TEMPLATE,
        height=350,
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis_title="Date",
        yaxis_title="Elo Rating",
    )
    return fig


def player_radar_chart(
    players: list[dict],
    categories: list[str],
    title: str = "Player Comparison",
) -> go.Figure:
    """Radar/spider chart for player attribute comparison."""
    fig = go.Figure()
    palette = [COLORS["accent"], COLORS["gold"], COLORS["red"], COLORS["blue"]]

    for i, player in enumerate(players):
        values = [player.get(c, 0) for c in categories]
        values += [values[0]]  # Close the loop
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill="toself",
            name=player["name"],
            line_color=palette[i % len(palette)],
            fillcolor=palette[i % len(palette)],
            opacity=0.3,
        ))
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            mode="lines+markers",
            name=player["name"],
            line_color=palette[i % len(palette)],
            showlegend=False,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9),
                gridcolor="#2a2d3a",
            ),
            angularaxis=dict(gridcolor="#2a2d3a"),
            bgcolor="#0f1117",
        ),
        template=DARK_TEMPLATE,
        title=dict(text=title, font_size=14),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def player_percentile_bar(stats: dict, position: str) -> go.Figure:
    """Horizontal bar chart showing player percentiles vs peers."""
    # Select relevant stats by position
    pos_stats = {
        "FW": ["goals_p90", "assists_p90", "xG_p90", "xA_p90",
               "shots_on_target_p90", "key_passes_p90", "dribbles_completed_p90"],
        "MF": ["progressive_passes_p90", "key_passes_p90", "assists_p90",
               "xA_p90", "pass_completion_pct", "goals_p90",
               "tackles_p90", "interceptions_p90"],
        "DF": ["tackles_p90", "interceptions_p90", "clearances_p90",
               "blocks_p90", "aerial_win_pct", "pass_completion_pct",
               "progressive_passes_p90"],
        "GK": ["save_pct", "clean_sheet_pct", "psxg_ga_p90", "pass_completion_pct"],
    }
    rename = {
        "goals_p90": "Goals p90",
        "assists_p90": "Assists p90",
        "xG_p90": "xG p90",
        "xA_p90": "xA p90",
        "shots_on_target_p90": "SoT p90",
        "key_passes_p90": "Key Passes p90",
        "progressive_passes_p90": "Prog. Passes p90",
        "progressive_carries_p90": "Prog. Carries p90",
        "dribbles_completed_p90": "Dribbles p90",
        "tackles_p90": "Tackles p90",
        "interceptions_p90": "Interceptions p90",
        "clearances_p90": "Clearances p90",
        "blocks_p90": "Blocks p90",
        "aerial_win_pct": "Aerial Win %",
        "pass_completion_pct": "Pass Completion %",
        "save_pct": "Save %",
        "clean_sheet_pct": "Clean Sheet %",
        "psxg_ga_p90": "Goals Prevented p90",
    }

    sel = pos_stats.get(position, [])
    labels, pcts, values_display = [], [], []

    for s in sel:
        if s in stats:
            labels.append(rename.get(s, s))
            pcts.append(stats[s]["percentile"])
            values_display.append(stats[s]["value"])

    if not labels:
        return go.Figure()

    colors = [
        COLORS["win"] if p >= 70 else (COLORS["gold"] if p >= 40 else COLORS["loss"])
        for p in pcts
    ]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}th" for p in pcts],
        textposition="outside",
        customdata=values_display,
        hovertemplate="%{y}: %{x:.1f}th percentile (val: %{customdata:.2f})<extra></extra>",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=max(250, 40 * len(labels)),
        margin=dict(l=20, r=60, t=10, b=20),
        xaxis=dict(range=[0, 110], title="Percentile vs Position Peers"),
        xaxis_tickvals=[0, 25, 50, 75, 100],
        xaxis_ticktext=["0th", "25th", "50th", "75th", "100th"],
        showlegend=False,
    )
    return fig


def rankings_scatter(df: pd.DataFrame) -> go.Figure:
    """Bubble chart of team rankings: attack vs defense, bubble=power rating."""
    fig = px.scatter(
        df.head(60),
        x="attack_rating",
        y="defense_rating",
        size="power_rating",
        color="confederation",
        text="team",
        hover_data={"power_rating": True, "elo": True, "form_rating": True},
        color_discrete_map=CONF_COLORS,
        template="plotly_dark",
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(opacity=0.8, line=dict(width=1, color="white")),
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=550,
        xaxis_title="Attack Rating",
        yaxis_title="Defense Rating",
        title="Team Power Rankings: Attack vs Defense",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def tournament_probability_table(df: pd.DataFrame, stage_col: str) -> go.Figure:
    """Heatmap-style table for tournament probabilities."""
    display_cols = [
        "team", "group",
        "p_round_of_16", "p_quarter_final",
        "p_semi_final", "p_final", "p_champion",
    ]
    present = [c for c in display_cols if c in df.columns]
    df_display = df[present].sort_values("p_champion", ascending=False).head(32)

    rename_cols = {
        "team": "Team", "group": "Group",
        "p_round_of_16": "R16 %",
        "p_quarter_final": "QF %",
        "p_semi_final": "SF %",
        "p_final": "Final %",
        "p_champion": "Champion %",
    }
    df_display = df_display.rename(columns=rename_cols)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df_display.columns],
            fill_color="#1a2a1a",
            font=dict(color="white", size=12),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[df_display[c] for c in df_display.columns],
            fill_color=[[
                "#0f1117" if i % 2 == 0 else "#161b1d"
                for i in range(len(df_display))
            ]],
            font=dict(color="white", size=11),
            align=["left", "center"] + ["center"] * (len(df_display.columns) - 2),
            height=30,
        ),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=min(1000, 60 + 32 * len(df_display)),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def comparison_bar_chart(
    player1_name: str,
    player2_name: str,
    stats: dict,
    title: str = "Player Comparison",
) -> go.Figure:
    """Side-by-side bar chart for player comparison."""
    categories = list(stats.keys())
    p1_vals = [stats[c]["p1"] for c in categories]
    p2_vals = [stats[c]["p2"] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=player1_name,
        x=categories,
        y=p1_vals,
        marker_color=COLORS["accent"],
        text=[f"{v:.2f}" for v in p1_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=player2_name,
        x=categories,
        y=p2_vals,
        marker_color=COLORS["gold"],
        text=[f"{v:.2f}" for v in p2_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        template=DARK_TEMPLATE,
        title=title,
        height=400,
        margin=dict(l=20, r=20, t=60, b=80),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def form_timeline(results: list[str], team: str) -> go.Figure:
    """Visual form strip (W/D/L) as colored blocks."""
    color_map = {"W": COLORS["win"], "D": COLORS["draw"], "L": COLORS["loss"]}
    n = len(results)
    fig = go.Figure()

    for i, r in enumerate(results):
        fig.add_trace(go.Bar(
            x=[1],
            y=[i],
            orientation="h",
            marker_color=color_map.get(r, "#888"),
            showlegend=False,
            text=r,
            textposition="inside",
            insidetextanchor="middle",
            hoverinfo="skip",
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        height=max(100, 35 * n),
        barmode="stack",
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def elo_distribution(ratings_df: pd.DataFrame) -> go.Figure:
    """Histogram of Elo ratings across all teams."""
    fig = px.histogram(
        ratings_df,
        x="elo",
        nbins=40,
        color_discrete_sequence=[COLORS["accent"]],
        template="plotly_dark",
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis_title="Elo Rating",
        yaxis_title="Number of Teams",
        showlegend=False,
    )
    return fig
