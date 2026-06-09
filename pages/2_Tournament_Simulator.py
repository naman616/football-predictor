"""Tournament Simulator Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Tournament Simulator · Football Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 20px 0 12px;
}
.champion-card {
    background: linear-gradient(135deg, #1a2a1a, #2a3a00);
    border: 2px solid #f0c040; border-radius: 16px;
    padding: 24px; text-align: center;
}
.champion-card h2 { color: #f0c040; font-size: 2rem; margin: 0; }
.champion-card p { color: #ccc; margin: 6px 0 0; }
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_resources():
    from src.utils.helpers import (
        load_processed_results, load_elo_ratings,
        load_match_predictor, get_elo_system, get_feature_engine,
    )
    df = load_processed_results()
    if df is None:
        return None, None, None, None
    elo_df = load_elo_ratings()
    elo = get_elo_system(df)
    predictor = load_match_predictor()
    engine = get_feature_engine(df)
    return df, elo, predictor, engine


df, elo_system, predictor, feature_engine = get_resources()


def make_predict_fn(elo_sys, pred, feat_eng):
    """Create prediction function for tournament simulator."""
    def predict_fn(home_team: str, away_team: str, neutral: bool = True):
        if home_team.startswith("__bye") or away_team.startswith("__bye"):
            return 0.99, 0.005, 0.005

        home_elo = elo_sys.get_rating(home_team) if elo_sys else 1500.0
        away_elo = elo_sys.get_rating(away_team) if elo_sys else 1500.0

        if pred is not None and pred.is_trained and feat_eng is not None:
            try:
                import numpy as np
                feat = feat_eng.get_feature_values(
                    home_team, away_team, home_elo, away_elo,
                    tournament="FIFA World Cup", neutral=neutral,
                )
                proba = pred.predict_proba_matrix(feat.reshape(1, -1))[0]
                return float(proba[0]), float(proba[1]), float(proba[2])
            except Exception:
                pass
        if elo_sys:
            return elo_sys.predict_outcome(home_team, away_team, neutral)
        return 0.40, 0.25, 0.35

    return predict_fn


def make_batch_predict_fn(elo_sys, pred, feat_eng):
    """
    Batch version: pre-computes all feature rows as numpy arrays then runs one
    batched ML call — avoids per-pair DataFrame creation overhead (~2s for 48-team WC).
    """
    def batch_fn(pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        import numpy as np

        results = [None] * len(pairs)
        feature_rows = []
        valid_indices = []

        for i, (home, away) in enumerate(pairs):
            home_elo = elo_sys.get_rating(home) if elo_sys else 1500.0
            away_elo = elo_sys.get_rating(away) if elo_sys else 1500.0

            if pred is not None and pred.is_trained and feat_eng is not None:
                try:
                    row = feat_eng.get_feature_values(
                        home, away, home_elo, away_elo,
                        tournament="FIFA World Cup", neutral=True,
                    )
                    feature_rows.append(row)
                    valid_indices.append(i)
                    continue
                except Exception:
                    pass

            # Fallback for pairs without ML features
            results[i] = elo_sys.predict_outcome(home, away, True) if elo_sys else (0.40, 0.25, 0.35)

        # Single batched ML call for all valid pairs
        if feature_rows and pred is not None:
            X = np.vstack(feature_rows)
            proba = pred.predict_proba_matrix(X)
            for arr_i, orig_i in enumerate(valid_indices):
                p = proba[arr_i]
                results[orig_i] = (float(p[0]), float(p[1]), float(p[2]))

        return results

    return batch_fn


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Tournament Simulator")
st.markdown("Run Monte Carlo simulations to estimate each team's probability of winning the tournament.")

if df is None:
    st.error("Data not initialized. Please run setup first.")
    st.stop()


# ── Configuration ─────────────────────────────────────────────────────────────
from src.models.tournament_simulator import TOURNAMENTS

st.markdown("<div class='section-header'>Configure Simulation</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    tournament_name = st.selectbox("Tournament", options=list(TOURNAMENTS.keys()), index=0)
with col2:
    n_sims = st.select_slider(
        "Simulations",
        options=[1000, 2000, 5000, 10000, 20000],
        value=5000,
        help="More simulations = more accurate probabilities but slower",
    )
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)


# ── Show tournament participants ──────────────────────────────────────────────
tournament_config = TOURNAMENTS[tournament_name]
groups = tournament_config["groups"]

with st.expander(f"{tournament_name} — Group Stage Draw", expanded=False):
    from src.utils.helpers import get_flag_emoji
    n_groups = len(groups)
    cols_per_row = min(4, n_groups)
    group_names = list(groups.keys())

    for row_start in range(0, n_groups, cols_per_row):
        cols = st.columns(cols_per_row)
        for i, (col, gn) in enumerate(zip(cols, group_names[row_start:row_start + cols_per_row])):
            with col:
                st.markdown(f"**Group {gn}**")
                for t in groups[gn]:
                    flag = get_flag_emoji(t)
                    elo_val = elo_system.get_rating(t) if elo_system else 1500
                    st.markdown(f"{flag} {t} *(Elo: {elo_val:.0f})*")


# ── Run simulation ────────────────────────────────────────────────────────────
if run_btn:
    predict_fn = make_predict_fn(elo_system, predictor, feature_engine)
    batch_fn = make_batch_predict_fn(elo_system, predictor, feature_engine)

    from src.models.tournament_simulator import TournamentSimulator, TOURNAMENTS as T_MAP
    sim = TournamentSimulator(predict_fn)

    # Count unique pairs for progress display
    t_config = T_MAP[tournament_name]
    n_teams = sum(len(v) for v in t_config["groups"].values())
    n_pairs = n_teams * (n_teams - 1)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def precompute_callback():
        progress_bar.progress(0.0)
        status_text.text(f"Pre-computing {n_pairs:,} match probabilities for {n_teams} teams...")

    def progress_callback(done: int, total: int):
        pct = min(done / total, 1.0)
        progress_bar.progress(pct)
        status_text.text(f"Simulating... {done:,}/{total:,} ({pct*100:.0f}%)")

    with st.spinner(f"Running {n_sims:,} simulations..."):
        results = sim.simulate(
            tournament_name=tournament_name,
            n_sims=n_sims,
            progress_callback=progress_callback,
            precompute_callback=precompute_callback,
            batch_predict_fn=batch_fn,
        )

    progress_bar.progress(1.0)
    status_text.text(f"{n_sims:,} simulations complete!")

    st.session_state["sim_results"] = results
    st.session_state["sim_tournament"] = tournament_name

# ── Display results ───────────────────────────────────────────────────────────
if "sim_results" in st.session_state:
    results = st.session_state["sim_results"]
    t_name = st.session_state.get("sim_tournament", tournament_name)

    st.markdown("---")

    # ── Champion probabilities ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Championship Probabilities</div>", unsafe_allow_html=True)

    # Top 5 champion candidates
    champ_col = "p_champion"
    if champ_col not in results.columns:
        # Try to find the champion column
        champ_cols = [c for c in results.columns if "champion" in c.lower()]
        champ_col = champ_cols[0] if champ_cols else results.columns[-1]

    top_champs = results.nlargest(5, champ_col)
    cols = st.columns(5)
    for col, (_, row) in zip(cols, top_champs.iterrows()):
        with col:
            flag = get_flag_emoji(row["team"])
            pct = row[champ_col]
            bar_color = "#f0c040" if pct == top_champs[champ_col].max() else "#52b788"
            st.markdown(f"""
            <div style="background:#1c1e26;border-radius:10px;padding:16px;text-align:center;
                        border:2px solid {bar_color};">
                <div style="font-size:1.8rem;">{flag}</div>
                <div style="font-weight:600;font-size:0.9rem;color:#e0e0e0;">{row['team']}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{bar_color};">{pct:.1f}%</div>
                <div style="font-size:0.75rem;color:#888;">Champion</div>
            </div>""", unsafe_allow_html=True)

    # ── Full probability table ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Complete Tournament Probabilities</div>", unsafe_allow_html=True)

    # Find available stage columns
    stage_cols = [c for c in results.columns if c.startswith("p_")]
    display_cols = ["team", "group"] + stage_cols
    display_cols = [c for c in display_cols if c in results.columns]

    results_display = results[display_cols].sort_values(champ_col, ascending=False)

    # Rename columns for display
    rename = {
        "team": "Team", "group": "Group",
        "p_group_stage": "Group Stage %",
        "p_round_of_64": "R64 %",
        "p_round_of_32": "R32 %",
        "p_round_of_16": "R16 %",
        "p_quarter_final": "QF %",
        "p_semi_final": "SF %",
        "p_final": "Final %",
        "p_champion": "Champion %",
    }
    results_display = results_display.rename(columns={k: v for k, v in rename.items() if k in results_display.columns})

    st.dataframe(
        results_display,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Champion %": st.column_config.ProgressColumn("Champion %", min_value=0, max_value=100, format="%.1f%%"),
            "Final %": st.column_config.ProgressColumn("Final %", min_value=0, max_value=100, format="%.1f%%"),
            "SF %": st.column_config.ProgressColumn("SF %", min_value=0, max_value=100, format="%.1f%%"),
            "QF %": st.column_config.ProgressColumn("QF %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # ── Visualizations ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Visualizations</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Champion Probabilities", "Stage Progression", "Group Analysis"])

    with tab1:
        top20 = results.nlargest(20, champ_col).sort_values(champ_col)
        flags = [get_flag_emoji(t) for t in top20["team"]]
        labels = [f"{f} {t}" for f, t in zip(flags, top20["team"])]

        fig = go.Figure(go.Bar(
            x=top20[champ_col],
            y=labels,
            orientation="h",
            marker=dict(
                color=top20[champ_col],
                colorscale=[[0, "#1a472a"], [0.5, "#52b788"], [1.0, "#f0c040"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in top20[champ_col]],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=600,
            margin=dict(l=20, r=60, t=20, b=20),
            xaxis_title="Champion Probability (%)",
            font=dict(family="Inter, sans-serif"),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Stage progression heatmap
        stage_mapping = {
            "p_round_of_16": "R16", "p_quarter_final": "QF",
            "p_semi_final": "SF", "p_final": "Final", "p_champion": "Champion",
        }
        available_stages = {k: v for k, v in stage_mapping.items() if k in results.columns}

        top16 = results.nlargest(16, champ_col)
        heatmap_data = top16[["team"] + list(available_stages.keys())].set_index("team")
        heatmap_data.columns = list(available_stages.values())

        flags_h = [get_flag_emoji(t) for t in heatmap_data.index]
        y_labels = [f"{f} {t}" for f, t in zip(flags_h, heatmap_data.index)]

        fig2 = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=y_labels,
            colorscale=[[0, "#081c15"], [0.4, "#1a472a"], [0.7, "#52b788"], [1.0, "#f0c040"]],
            text=[[f"{v:.1f}%" for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
        ))
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        # Per-group champion probabilities
        if "group" in results.columns:
            group_list = sorted(results["group"].unique())
            g_cols = st.columns(min(4, len(group_list)))
            for gc, grp in zip(g_cols * 10, group_list):
                with gc:
                    grp_data = results[results["group"] == grp].sort_values(champ_col, ascending=False)
                    st.markdown(f"**Group {grp}**")
                    for _, row in grp_data.iterrows():
                        flag = get_flag_emoji(row["team"])
                        r16 = row.get("p_round_of_16", 0)
                        champ = row.get(champ_col, 0)
                        st.markdown(
                            f"{flag} {row['team']}: "
                            f"<span style='color:#52b788'>R16 {r16:.0f}%</span> | "
                            f"<span style='color:#f0c040'>{champ:.1f}%</span>",
                            unsafe_allow_html=True,
                        )

    # ── Download ──────────────────────────────────────────────────────────────
    csv = results_display.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name=f"{t_name.replace(' ', '_')}_simulation.csv",
        mime="text/csv",
    )
else:
    # Show instructions
    st.markdown("---")
    st.info(
        "Configure the tournament above and click **▶️ Run Simulation** to start.\n\n"
        f"The simulator will run Monte Carlo simulations using our match prediction model "
        f"to estimate each team's probability of winning the **{tournament_name}**."
    )

    # Show Elo-based quick preview
    if elo_system:
        st.markdown("<div class='section-header'>Pre-Tournament Elo Ratings</div>", unsafe_allow_html=True)
        t_config = TOURNAMENTS[tournament_name]
        rows = []
        for grp, teams in t_config["groups"].items():
            for team in teams:
                rows.append({
                    "group": grp,
                    "team": team,
                    "elo": elo_system.get_rating(team),
                })
        elo_preview = pd.DataFrame(rows).sort_values("elo", ascending=False)

        # Bar chart of Elo ratings coloured by group
        import plotly.express as px
        group_palette = px.colors.qualitative.Set2
        group_list = sorted(elo_preview["group"].unique())
        color_map = {g: group_palette[i % len(group_palette)] for i, g in enumerate(group_list)}
        elo_preview["color"] = elo_preview["group"].map(color_map)
        elo_sorted = elo_preview.sort_values("elo", ascending=True)
        team_labels_elo = [f"{get_flag_emoji(t)} {t}" for t in elo_sorted["team"]]

        fig_elo_prev = go.Figure(go.Bar(
            x=elo_sorted["elo"],
            y=team_labels_elo,
            orientation="h",
            marker_color=elo_sorted["color"],
            text=[f"{v:.0f}" for v in elo_sorted["elo"]],
            textposition="outside",
            customdata=elo_sorted["group"],
            hovertemplate="<b>%{y}</b><br>Elo: %{x:.0f}<br>Group: %{customdata}<extra></extra>",
        ))
        fig_elo_prev.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=max(400, 26 * len(elo_sorted)),
            margin=dict(l=20, r=60, t=20, b=20),
            xaxis=dict(title="Elo Rating", gridcolor="#2a2d3a"),
            yaxis=dict(gridcolor="#2a2d3a"),
            font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        )
        st.plotly_chart(fig_elo_prev, use_container_width=True, config={"displayModeBar": False})
