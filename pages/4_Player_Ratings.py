"""Player Ratings Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Player Ratings · Football Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 20px 0 12px;
}
.player-card {
    background: #1c1e26; border-radius: 12px; padding: 16px;
    border: 1px solid #2a2d3a; margin: 6px 0;
}
.player-card h4 { margin: 0 0 4px; color: #e0e0e0; font-size: 1rem; }
.player-card .meta { color: #888; font-size: 0.8rem; }
.rating-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 700; margin-left: 8px;
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
        rater._per90 = players  # Use same for percentile lookups
        return players, rater
    # Build from scratch
    try:
        player_df = load_player_data()
        from src.models.player_rater import PlayerRater
        rater = PlayerRater(player_df)
        rated = rater.compute_ratings()
        return rated, rater
    except Exception as e:
        st.error(f"Player data error: {e}")
        return None, None


players, rater = get_resources()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Player Ratings")
st.markdown("Position-specific ratings based on per-90 statistics across Europe's top leagues and international competitions.")

if players is None or players.empty:
    st.error("Player data unavailable.")
    st.stop()


# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Filter Players</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    pos_options = ["All"] + sorted(players["position"].unique().tolist())
    selected_pos = st.selectbox("Position", pos_options)

with col2:
    # Get unique nationalities
    nat_options = ["All"] + sorted(players["nationality"].dropna().unique().tolist())
    selected_nat = st.selectbox("Nationality", nat_options)

with col3:
    # Clubs
    club_options = ["All"]
    if "club" in players.columns:
        club_options += sorted(players["club"].dropna().unique().tolist())
    selected_club = st.selectbox("Club", club_options)

with col4:
    top_n = st.slider("Show top N", 10, 60, 30)

# Apply filters
filtered = players.copy()
if selected_pos != "All":
    filtered = filtered[filtered["position"] == selected_pos]
if selected_nat != "All":
    filtered = filtered[filtered["nationality"] == selected_nat]
if selected_club != "All" and "club" in filtered.columns:
    filtered = filtered[filtered["club"] == selected_club]
filtered = filtered.sort_values("overall_rating", ascending=False).head(top_n)


# ── Rating tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Rankings", "By Position", "Top Stats", "Player Profile"])

with tab1:
    st.markdown("<div class='section-header'>Player Rankings</div>", unsafe_allow_html=True)

    from src.utils.helpers import get_flag_emoji

    # Overall rating distribution by position
    pos_order = ["FW", "MF", "DF", "GK"]
    pos_colors = {"FW": "#f0c040", "MF": "#52b788", "DF": "#457b9d", "GK": "#e63946"}
    fig_dist = go.Figure()
    for pos in pos_order:
        pos_data = players[players["position"] == pos]["overall_rating"].dropna()
        if not pos_data.empty:
            fig_dist.add_trace(go.Box(
                y=pos_data,
                name=pos,
                marker_color=pos_colors.get(pos, "#52b788"),
                boxmean="sd",
                jitter=0.3,
                pointpos=-1.8,
                boxpoints="outliers",
            ))
    fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=320,
        margin=dict(l=20, r=20, t=30, b=30),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        yaxis=dict(title="Overall Rating", gridcolor="#2a2d3a"),
        xaxis=dict(title="Position", gridcolor="#2a2d3a"),
        title=dict(text="Rating Distribution by Position", font_size=13, font_color="#52b788"),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    display_cols = ["global_rank", "name", "nationality", "position", "club",
                    "overall_rating", "attack_rating", "defense_rating"]
    if "league" in filtered.columns:
        display_cols.insert(5, "league")
    display_cols = [c for c in display_cols if c in filtered.columns]
    disp = filtered[display_cols].copy()

    if "nationality" in disp.columns:
        disp["nationality"] = disp["nationality"].apply(
            lambda n: f"{get_flag_emoji(n)} {n}" if pd.notna(n) else n
        )

    st.dataframe(
        disp,
        use_container_width=True,
        height=min(800, 45 * len(disp) + 60),
        hide_index=True,
        column_config={
            "global_rank": st.column_config.NumberColumn("#", width=60),
            "name": "Player",
            "nationality": "Country",
            "position": "Pos",
            "club": "Club",
            "league": "League",
            "overall_rating": st.column_config.ProgressColumn(
                "Overall", min_value=0, max_value=100, format="%.1f"
            ),
            "attack_rating": st.column_config.ProgressColumn(
                "Attack", min_value=0, max_value=100, format="%.1f"
            ),
            "defense_rating": st.column_config.ProgressColumn(
                "Defense", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    # Download
    st.download_button("Download CSV", filtered.to_csv(index=False), "player_ratings.csv", "text/csv")


with tab2:
    st.markdown("<div class='section-header'>Top 5 by Position</div>", unsafe_allow_html=True)

    for pos, pos_label in [
        ("FW", "Forwards"),
        ("MF", "Midfielders"),
        ("DF", "Defenders"),
        ("GK", "Goalkeepers"),
    ]:
        pos_players = players[players["position"] == pos].head(5)
        if pos_players.empty:
            continue

        st.markdown(f"**Top {pos_label}**")
        cols = st.columns(len(pos_players))
        for col, (_, p) in zip(cols, pos_players.iterrows()):
            with col:
                flag = get_flag_emoji(p.get("nationality", ""))
                rating = p.get("overall_rating", 0)
                r_color = "#f0c040" if rating >= 80 else "#52b788" if rating >= 65 else "#457b9d"
                club = p.get("club", "")
                nat = p.get("nationality", "")
                st.markdown(f"""
                <div class="player-card">
                    <div style="font-size:1.4rem;font-weight:700;color:{r_color}">{rating:.0f}</div>
                    <h4>{p['name']}</h4>
                    <div class="meta">{flag} {nat}</div>
                    <div class="meta">{club}</div>
                </div>""", unsafe_allow_html=True)

        # Horizontal bar of ratings
        import plotly.graph_objects as go
        fig_pos = go.Figure()
        colors_pos = ["#f0c040", "#52b788", "#52b788", "#457b9d", "#457b9d"]
        for i, (_, p) in enumerate(pos_players.iterrows()):
            fig_pos.add_trace(go.Bar(
                name=p["name"],
                x=[p.get("overall_rating", 0)],
                y=[p["name"]],
                orientation="h",
                marker_color=colors_pos[i],
                text=f"{p.get('overall_rating', 0):.1f}",
                textposition="outside",
                showlegend=False,
            ))
        fig_pos.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=200,
            margin=dict(l=20, r=60, t=0, b=0),
            xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
            yaxis=dict(autorange="reversed"),
            font=dict(family="Inter"),
            barmode="overlay",
        )
        st.plotly_chart(fig_pos, use_container_width=True)
        st.markdown("---")


with tab3:
    st.markdown("<div class='section-header'>Statistical Leaders</div>", unsafe_allow_html=True)

    # Find per-90 columns
    p90_cols = [c for c in players.columns if c.endswith("_p90")]
    stat_options = {
        "goals_p90": "Goals per 90",
        "assists_p90": "Assists per 90",
        "xG_p90": "xG per 90",
        "xA_p90": "xA per 90",
        "progressive_passes_p90": "Progressive Passes per 90",
        "progressive_carries_p90": "Progressive Carries per 90",
        "key_passes_p90": "Key Passes per 90",
        "tackles_p90": "Tackles per 90",
        "interceptions_p90": "Interceptions per 90",
        "clearances_p90": "Clearances per 90",
    }
    available_stats = {k: v for k, v in stat_options.items() if k in players.columns}

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stat_col = st.selectbox("Statistic", list(available_stats.keys()),
                                format_func=lambda x: available_stats.get(x, x))
    with col_s2:
        pos_filter = st.selectbox("Position filter", ["All", "FW", "MF", "DF", "GK"],
                                  key="stats_pos_filter")

    stat_df = players.copy()
    if pos_filter != "All":
        stat_df = stat_df[stat_df["position"] == pos_filter]

    if stat_col in stat_df.columns:
        top_stat = stat_df.nlargest(15, stat_col)[["name", "nationality", "position", "club", stat_col, "overall_rating"]]

        flags = [get_flag_emoji(n) for n in top_stat["nationality"]]
        labels = [f"{f} {n}" for f, n in zip(flags, top_stat["name"])]

        fig_stat = go.Figure(go.Bar(
            x=top_stat[stat_col],
            y=labels,
            orientation="h",
            marker=dict(
                color=top_stat["overall_rating"],
                colorscale=[[0, "#1a472a"], [0.5, "#52b788"], [1, "#f0c040"]],
                showscale=True,
                colorbar=dict(title="Overall Rating", len=0.8),
            ),
            text=[f"{v:.2f}" for v in top_stat[stat_col]],
            textposition="outside",
        ))
        fig_stat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=500,
            margin=dict(l=20, r=80, t=20, b=40),
            xaxis_title=available_stats.get(stat_col, stat_col),
            font=dict(family="Inter"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_stat, use_container_width=True)


with tab4:
    st.markdown("<div class='section-header'>Player Profile</div>", unsafe_allow_html=True)

    player_names = sorted(players["name"].unique().tolist())
    selected_player = st.selectbox("Select Player", player_names, key="profile_player")

    player_data = players[players["name"] == selected_player]
    if player_data.empty:
        st.info("Player not found.")
    else:
        p = player_data.iloc[0]
        flag = get_flag_emoji(p.get("nationality", ""))
        pos = p.get("position", "?")

        # Header
        col_l, col_r = st.columns([2, 3])
        with col_l:
            overall = p.get("overall_rating", 0)
            attack = p.get("attack_rating", 0)
            defense = p.get("defense_rating", 0)

            r_color = "#f0c040" if overall >= 80 else "#52b788" if overall >= 65 else "#457b9d"
            st.markdown(f"""
            <div style="background:#1c1e26;border-radius:16px;padding:24px;text-align:center;
                        border:3px solid {r_color};">
                <div style="font-size:3rem;font-weight:900;color:{r_color}">{overall:.0f}</div>
                <div style="font-size:1.2rem;font-weight:600;color:#e0e0e0">{p['name']}</div>
                <div style="color:#888;margin-top:4px">{flag} {p.get('nationality','')} · {pos}</div>
                <div style="color:#888;font-size:0.85rem">{p.get('club','')} · {p.get('league','')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            m1, m2 = st.columns(2)
            m1.metric("Attack", f"{attack:.1f}")
            m2.metric("Defense", f"{defense:.1f}")
            if "age" in p.index and not pd.isna(p["age"]):
                st.metric("Age", f"{int(p['age'])}")
            if "apps" in p.index and not pd.isna(p["apps"]):
                st.metric("Appearances", f"{int(p['apps'])}")
            if "goals" in p.index and not pd.isna(p["goals"]):
                g_col, a_col = st.columns(2)
                g_col.metric("Goals", f"{int(p.get('goals', 0))}")
                a_col.metric("Assists", f"{int(p.get('assists', 0))}")

        with col_r:
            # Percentile chart
            if rater is not None:
                try:
                    pct_stats = rater.get_percentile_stats(selected_player)
                    if pct_stats:
                        from src.utils.charts import player_percentile_bar
                        fig_pct = player_percentile_bar(pct_stats, pos)
                        st.plotly_chart(fig_pct, use_container_width=True)
                    else:
                        st.info("Percentile stats not available for this player.")
                except Exception as e:
                    st.info(f"Stats unavailable: {e}")

        # Key per-90 stats
        p90_display = {}
        p90_cols_show = {
            "FW": ["goals_p90", "assists_p90", "xG_p90", "xA_p90", "shots_on_target_p90"],
            "MF": ["goals_p90", "assists_p90", "progressive_passes_p90", "key_passes_p90", "tackles_p90"],
            "DF": ["tackles_p90", "interceptions_p90", "clearances_p90", "blocks_p90", "aerial_win_pct"],
            "GK": ["save_pct", "clean_sheet_pct", "goals_conceded_p90", "pass_completion_pct"],
        }
        show_cols = p90_cols_show.get(pos, ["goals_p90", "assists_p90"])

        rename_stat = {
            "goals_p90": "Goals/90", "assists_p90": "Assists/90",
            "xG_p90": "xG/90", "xA_p90": "xA/90",
            "shots_on_target_p90": "SoT/90", "key_passes_p90": "Key Passes/90",
            "progressive_passes_p90": "Prog. Passes/90", "tackles_p90": "Tackles/90",
            "interceptions_p90": "Interceptions/90", "clearances_p90": "Clearances/90",
            "blocks_p90": "Blocks/90", "aerial_win_pct": "Aerial Win %",
            "save_pct": "Save %", "clean_sheet_pct": "Clean Sheet %",
            "goals_conceded_p90": "Goals Conceded/90", "pass_completion_pct": "Pass Completion %",
        }

        stat_metrics = {rename_stat.get(c, c): round(float(p.get(c, 0)), 2)
                        for c in show_cols if c in p.index}

        if stat_metrics:
            st.markdown("**Key Statistics (per 90 min / per game):**")
            met_cols = st.columns(len(stat_metrics))
            for mc, (stat_name, val) in zip(met_cols, stat_metrics.items()):
                mc.metric(stat_name, f"{val:.2f}")
