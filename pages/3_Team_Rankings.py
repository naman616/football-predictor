"""Team Rankings Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Team Rankings · Football Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 20px 0 12px;
}
.rank-bar {
    height: 8px; border-radius: 4px; background: #52b788;
    display: inline-block; margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_resources():
    from src.utils.helpers import (
        load_processed_results, load_elo_ratings, load_team_rankings,
        get_elo_system,
    )
    df = load_processed_results()
    if df is None:
        return None, None, None, None
    elo_df = load_elo_ratings()
    rankings = load_team_rankings()
    elo = get_elo_system(df)
    return df, elo_df, rankings, elo


df, elo_df, rankings, elo_system = get_resources()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Team Power Rankings")
st.markdown("Multi-dimensional team ranking combining Elo ratings, recent form, attack, defense, and tournament performance.")

if df is None:
    st.error("Data not initialized. Please run setup first.")
    st.stop()

# Build rankings if not cached
if rankings is None:
    st.info("Building team rankings...")
    from src.models.team_ranker import TeamRanker
    ranker = TeamRanker(df, elo_df if elo_df is not None else pd.DataFrame())
    rankings = ranker.build_rankings()

if rankings is None or rankings.empty:
    st.error("Could not build team rankings.")
    st.stop()


# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Filter & Sort</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    conf_options = ["All"] + sorted(rankings["confederation"].unique().tolist())
    selected_conf = st.selectbox("Confederation", conf_options)
with col2:
    sort_col = st.selectbox(
        "Sort by",
        ["power_rating", "elo", "attack_rating", "defense_rating", "form_rating"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
with col3:
    top_n = st.slider("Show top N teams", 10, 100, 50)

# Apply filters
filtered = rankings.copy()
if selected_conf != "All":
    filtered = filtered[filtered["confederation"] == selected_conf]
filtered = filtered.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
filtered["rank"] = range(1, len(filtered) + 1)


# ── Visualization tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Power Rankings", "Attack vs Defense", "Elo Distribution", "Team Deep-Dive"])

with tab1:
    st.markdown("<div class='section-header'>Power Rankings Table</div>", unsafe_allow_html=True)

    from src.utils.helpers import get_flag_emoji
    import plotly.graph_objects as go

    # Top 15 leaderboard bar chart
    top15 = filtered.head(15).copy()
    team_labels_r = [f"{get_flag_emoji(t)} {t}" for t in top15["team"]]
    fig_rank_bar = go.Figure()
    fig_rank_bar.add_trace(go.Bar(
        name="Power",
        y=team_labels_r[::-1],
        x=top15["power_rating"].iloc[::-1],
        orientation="h",
        marker_color="#52b788",
        text=[f"{v:.1f}" for v in top15["power_rating"].iloc[::-1]],
        textposition="outside",
    ))
    if "attack_rating" in top15.columns:
        fig_rank_bar.add_trace(go.Bar(
            name="Attack",
            y=team_labels_r[::-1],
            x=top15["attack_rating"].iloc[::-1],
            orientation="h",
            marker_color="#f0c040",
            text=[f"{v:.1f}" for v in top15["attack_rating"].iloc[::-1]],
            textposition="outside",
        ))
    if "defense_rating" in top15.columns:
        fig_rank_bar.add_trace(go.Bar(
            name="Defense",
            y=team_labels_r[::-1],
            x=top15["defense_rating"].iloc[::-1],
            orientation="h",
            marker_color="#457b9d",
            text=[f"{v:.1f}" for v in top15["defense_rating"].iloc[::-1]],
            textposition="outside",
        ))
    fig_rank_bar.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=500,
        margin=dict(l=20, r=70, t=20, b=20),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2d3a", title="Rating"),
        yaxis=dict(gridcolor="#2a2d3a"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.3),
    )
    st.plotly_chart(fig_rank_bar, use_container_width=True, config={"displayModeBar": False})

    # Display as formatted table
    display_cols = ["rank", "team", "confederation", "power_rating", "elo",
                    "attack_rating", "defense_rating", "form_rating"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].copy()
    display_df["team"] = display_df["team"].apply(
        lambda t: f"{get_flag_emoji(t)} {t}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(800, 40 * len(display_df) + 50),
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("#", width=50),
            "team": "Team",
            "confederation": "Confederation",
            "power_rating": st.column_config.ProgressColumn(
                "Power Rating", min_value=0, max_value=100, format="%.1f"
            ),
            "elo": st.column_config.NumberColumn("Elo", format="%.0f"),
            "attack_rating": st.column_config.ProgressColumn(
                "Attack", min_value=0, max_value=100, format="%.1f"
            ),
            "defense_rating": st.column_config.ProgressColumn(
                "Defense", min_value=0, max_value=100, format="%.1f"
            ),
            "form_rating": st.column_config.ProgressColumn(
                "Form", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button("Download Rankings CSV", csv, "team_rankings.csv", "text/csv")


with tab2:
    from src.utils.charts import rankings_scatter, CONF_COLORS

    # Add confederation colors
    fig = rankings_scatter(filtered)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Reading the chart:**
    - **X-axis:** Attack rating (higher = more dangerous going forward)
    - **Y-axis:** Defense rating (higher = better at stopping goals)
    - **Bubble size:** Overall power rating
    - **Color:** Confederation
    - **Ideal position:** Top-right (strong in both phases)
    """)


with tab3:
    from src.utils.charts import elo_distribution
    import plotly.express as px

    st.markdown("<div class='section-header'>Elo Rating Distribution</div>", unsafe_allow_html=True)

    fig_hist = elo_distribution(filtered[["elo"]])
    st.plotly_chart(fig_hist, use_container_width=True)

    # Elo percentiles by confederation
    st.markdown("<div class='section-header'>Average Elo by Confederation</div>", unsafe_allow_html=True)
    conf_stats = (
        rankings.groupby("confederation")["elo"]
        .agg(["mean", "max", "min", "count"])
        .round(0)
        .reset_index()
        .sort_values("mean", ascending=False)
        .rename(columns={"mean": "Avg Elo", "max": "Max Elo", "min": "Min Elo", "count": "Teams"})
    )

    fig_conf = px.bar(
        conf_stats,
        x="confederation",
        y="Avg Elo",
        color="confederation",
        color_discrete_map=CONF_COLORS,
        text="Avg Elo",
        template="plotly_dark",
    )
    fig_conf.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_conf.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        font=dict(family="Inter, sans-serif"),
        xaxis_title="",
        yaxis_title="Average Elo Rating",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.dataframe(conf_stats, use_container_width=True, hide_index=True)


with tab4:
    st.markdown("<div class='section-header'>Team Deep-Dive</div>", unsafe_allow_html=True)

    all_teams_list = sorted(rankings["team"].tolist())
    selected_team = st.selectbox("Select Team", all_teams_list, key="team_deepdive")

    flag = get_flag_emoji(selected_team)
    team_row = rankings[rankings["team"] == selected_team]

    if not team_row.empty:
        r = team_row.iloc[0]
        st.markdown(f"## {flag} {selected_team}")

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Power Rank", f"#{r['rank']:.0f}")
        col_m2.metric("Elo Rating", f"{r['elo']:.0f}")
        col_m3.metric("Power Rating", f"{r['power_rating']:.1f}")
        col_m4.metric("Attack", f"{r['attack_rating']:.1f}")
        col_m5.metric("Defense", f"{r['defense_rating']:.1f}")

        col_left, col_right = st.columns(2)

        with col_left:
            # Elo history chart
            from src.utils.charts import elo_history_chart
            if elo_system:
                history = elo_system.get_team_history(selected_team)
                if not history.empty:
                    fig_hist = elo_history_chart(history, selected_team)
                    st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            # Recent form
            from src.models.team_ranker import TeamRanker
            from src.utils.helpers import form_to_html, load_elo_ratings
            elo_ratings = load_elo_ratings()
            ranker = TeamRanker(df, elo_ratings if elo_ratings is not None else pd.DataFrame())
            form = ranker.get_recent_form(selected_team, n=10)

            st.markdown("**Recent Form (last 10):**")
            if form:
                st.markdown(form_to_html(form), unsafe_allow_html=True)
                wins = form.count("W")
                draws = form.count("D")
                losses = form.count("L")
                pts = wins * 3 + draws
                st.markdown(f"W{wins} D{draws} L{losses} — **{pts} pts**")

            # Radar chart for rankings dimensions
            import plotly.graph_objects as go
            radar_cats = ["Power", "Attack", "Defense", "Form", "Elo (norm)"]
            radar_vals = [
                r["power_rating"],
                r["attack_rating"],
                r["defense_rating"],
                r["form_rating"],
                r["elo_norm"] if "elo_norm" in r.index else 50,
            ]
            radar_vals += [radar_vals[0]]  # Close loop
            radar_cats_closed = radar_cats + [radar_cats[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals,
                theta=radar_cats_closed,
                fill="toself",
                line_color="#52b788",
                fillcolor="rgba(82, 183, 136, 0.2)",
                name=selected_team,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2a2d3a"),
                    angularaxis=dict(gridcolor="#2a2d3a"),
                    bgcolor="#0f1117",
                ),
                paper_bgcolor="#0f1117",
                font=dict(color="#e0e0e0", family="Inter"),
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Recent matches table
        st.markdown("**Recent Matches:**")
        mask = (df["home_team"] == selected_team) | (df["away_team"] == selected_team)
        recent_matches = df[mask].tail(15).sort_values("date", ascending=False)

        match_records = []
        for _, match in recent_matches.iterrows():
            is_home = match["home_team"] == selected_team
            opponent = match["away_team"] if is_home else match["home_team"]
            team_goals = match["home_score"] if is_home else match["away_score"]
            opp_goals = match["away_score"] if is_home else match["home_score"]
            res = match["result"]
            if (is_home and res == 0) or (not is_home and res == 2):
                result_str = "W"
            elif res == 1:
                result_str = "D"
            else:
                result_str = "L"

            match_records.append({
                "Date": match["date"].strftime("%Y-%m-%d"),
                "H/A": "H" if is_home else "A",
                "Opponent": f"{get_flag_emoji(opponent)} {opponent}",
                "Score": f"{team_goals}-{opp_goals}",
                "Result": result_str,
                "Tournament": match.get("tournament", ""),
            })

        match_df = pd.DataFrame(match_records)

        def color_result(val):
            colors = {"W": "#1a472a", "D": "#3a2a00", "L": "#3a0a0a"}
            return f"background-color: {colors.get(val, 'transparent')}; color: white;"

        st.dataframe(match_df, use_container_width=True, hide_index=True)
    else:
        st.info("Team not found in rankings.")
