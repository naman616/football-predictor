"""Match Predictor Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Match Predictor · Football Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 20px 0 12px;
}
.prob-card {
    background: #1c1e26; border-radius: 12px; padding: 20px;
    text-align: center; border: 2px solid transparent;
}
.prob-card.win  { border-color: #2dc653; }
.prob-card.draw { border-color: #f0c040; }
.prob-card.loss { border-color: #e63946; }
.prob-card .pct { font-size: 2.4rem; font-weight: 700; }
.prob-card .lbl { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
.win .pct  { color: #2dc653; }
.draw .pct { color: #f0c040; }
.loss .pct { color: #e63946; }
.conf-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-top: 8px;
}
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
    elo_df = load_elo_ratings()
    predictor = load_match_predictor()
    if df is None:
        return None, None, None, None, None
    elo = get_elo_system(df)
    engine = get_feature_engine(df)
    return df, elo_df, elo, engine, predictor


df, elo_df, elo_system, feature_engine, predictor = get_resources()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Match Predictor")
st.markdown("Predict the outcome of any international football match using our AI model.")

if df is None:
    st.error("Data not initialized. Please run setup first.")
    st.stop()


# ── Team Selection ────────────────────────────────────────────────────────────
from src.utils.helpers import get_all_teams, get_flag_emoji
from config.settings import TOP_TEAMS, TOURNAMENT_IMPORTANCE

all_teams = get_all_teams(df)

st.markdown("<div class='section-header'>Select Match</div>", unsafe_allow_html=True)
col1, col_vs, col2 = st.columns([5, 1, 5])

# Default to a compelling fixture
default_home_idx = all_teams.index("Argentina") if "Argentina" in all_teams else 0
default_away_idx = all_teams.index("France") if "France" in all_teams else 1

with col1:
    home_team = st.selectbox("Home Team", all_teams, index=default_home_idx)
with col_vs:
    st.markdown("<br><br><center><b style='color:#52b788;font-size:1.2rem'>VS</b></center>", unsafe_allow_html=True)
with col2:
    away_team = st.selectbox("Away Team", all_teams, index=default_away_idx)

col3, col4, col5 = st.columns(3)
with col3:
    tournament = st.selectbox(
        "Tournament",
        options=list(TOURNAMENT_IMPORTANCE.keys()),
        index=0,
    )
with col4:
    neutral = st.checkbox("Neutral Venue", value=True)
with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict", type="primary", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn or True:  # Always show prediction
    if home_team == away_team:
        st.warning("Please select two different teams.")
        st.stop()

    # Get Elo ratings
    home_elo = elo_system.get_rating(home_team) if elo_system else 1500.0
    away_elo = elo_system.get_rating(away_team) if elo_system else 1500.0

    # Build features
    features_df = feature_engine.get_features_for_prediction(
        home_team=home_team,
        away_team=away_team,
        home_elo=home_elo,
        away_elo=away_elo,
        tournament=tournament,
        neutral=neutral,
    )

    # Predict
    if predictor is not None and predictor.is_trained:
        p_home, p_draw, p_away = predictor.predict(features_df)
    else:
        # Fallback to Elo-only prediction
        p_home, p_draw, p_away = elo_system.predict_outcome(home_team, away_team, neutral)

    from src.utils.helpers import confidence_label
    conf_score = max(p_home, p_draw, p_away)
    conf_label, conf_color = confidence_label(conf_score)
    predicted_outcome = ["Home Win", "Draw", "Away Win"][np.argmax([p_home, p_draw, p_away])]

    # ── Results display ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)

    col_h, col_d, col_a = st.columns(3)
    flag_h = get_flag_emoji(home_team)
    flag_a = get_flag_emoji(away_team)

    with col_h:
        st.markdown(f"""
        <div class="prob-card win">
            <div class="pct">{p_home*100:.1f}%</div>
            <div class="lbl">{flag_h} {home_team} Win</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="prob-card draw">
            <div class="pct">{p_draw*100:.1f}%</div>
            <div class="lbl">Draw</div>
        </div>""", unsafe_allow_html=True)
    with col_a:
        st.markdown(f"""
        <div class="prob-card loss">
            <div class="pct">{p_away*100:.1f}%</div>
            <div class="lbl">{flag_a} {away_team} Win</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    from src.utils.charts import prediction_gauge
    fig_gauge = prediction_gauge(p_home, p_draw, p_away, home_team, away_team)
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Confidence + outcome
    conf_css = f"background:{conf_color};color:{'#000' if conf_color=='#f0c040' else '#fff'}"
    st.markdown(
        f"**Predicted:** {predicted_outcome} &nbsp;&nbsp; "
        f'<span class="conf-badge" style="{conf_css}">{conf_label} Confidence ({conf_score*100:.1f}%)</span>',
        unsafe_allow_html=True,
    )

    # ── Team Context ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Team Context</div>", unsafe_allow_html=True)

    from src.models.team_ranker import TeamRanker
    from src.utils.helpers import load_elo_ratings, load_team_rankings, form_to_html

    elo_ratings = load_elo_ratings()
    team_rankings = load_team_rankings()

    col_left, col_right = st.columns(2)
    for col, team, elo_val in [(col_left, home_team, home_elo), (col_right, away_team, away_elo)]:
        with col:
            flag = get_flag_emoji(team)
            st.markdown(f"**{flag} {team}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Elo Rating", f"{elo_val:.0f}")

            if team_rankings is not None:
                rank_row = team_rankings[team_rankings["team"] == team]
                if not rank_row.empty:
                    r = rank_row.iloc[0]
                    m2.metric("Power Rating", f"{r['power_rating']:.1f}")
                    m3.metric("Form Rating", f"{r['form_rating']:.1f}")

            # Recent form
            ranker = TeamRanker(df, elo_ratings if elo_ratings is not None else pd.DataFrame())
            form = ranker.get_recent_form(team, n=7)
            if form:
                st.markdown(f"**Recent Form:** {form_to_html(form)}", unsafe_allow_html=True)

    # ── H2H record ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Head-to-Head Record (All Time)</div>", unsafe_allow_html=True)

    ranker = TeamRanker(df, pd.DataFrame())
    h2h = ranker.get_head_to_head_record(home_team, away_team)

    if h2h["total_matches"] > 0:
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric(f"{home_team} Wins", h2h["team1_wins"])
        hc2.metric("Draws", h2h["draws"])
        hc3.metric(f"{away_team} Wins", h2h["team2_wins"])
        hc4.metric("Total Meetings", h2h["total_matches"])

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_h2h = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Win/Draw/Loss Split", "Goals Scored"),
            specs=[[{"type": "pie"}, {"type": "bar"}]],
        )
        fig_h2h.add_trace(go.Pie(
            labels=[f"{home_team} Wins", "Draws", f"{away_team} Wins"],
            values=[h2h["team1_wins"], h2h["draws"], h2h["team2_wins"]],
            marker_colors=["#2dc653", "#f0c040", "#e63946"],
            hole=0.45,
            textinfo="label+percent",
            showlegend=False,
        ), row=1, col=1)
        fig_h2h.add_trace(go.Bar(
            x=[home_team, away_team],
            y=[h2h["team1_goals"], h2h["team2_goals"]],
            marker_color=["#2dc653", "#e63946"],
            text=[h2h["team1_goals"], h2h["team2_goals"]],
            textposition="outside",
            showlegend=False,
        ), row=1, col=2)
        fig_h2h.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        )
        fig_h2h.update_yaxes(gridcolor="#2a2d3a", row=1, col=2)
        st.plotly_chart(fig_h2h, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No recorded head-to-head meetings found in the dataset.")

    # ── Form timelines ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Recent Form Timeline</div>", unsafe_allow_html=True)
    from src.utils.charts import form_timeline

    ft_col1, ft_col2 = st.columns(2)
    ranker_ft = TeamRanker(df, pd.DataFrame())

    with ft_col1:
        flag_h2 = get_flag_emoji(home_team)
        st.markdown(f"**{flag_h2} {home_team}**")
        form_h = ranker_ft.get_recent_form(home_team, n=10)
        if form_h:
            import plotly.graph_objects as go
            color_map = {"W": "#2dc653", "D": "#f0c040", "L": "#e63946"}
            fig_form_h = go.Figure()
            for i, r in enumerate(form_h):
                fig_form_h.add_trace(go.Bar(
                    x=[i], y=[1],
                    marker_color=color_map.get(r, "#888"),
                    text=r, textposition="inside",
                    showlegend=False,
                    hovertemplate=f"Match {i+1}: {r}<extra></extra>",
                ))
            fig_form_h.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                height=100, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                barmode="stack",
            )
            st.plotly_chart(fig_form_h, use_container_width=True, config={"displayModeBar": False})

    with ft_col2:
        flag_a2 = get_flag_emoji(away_team)
        st.markdown(f"**{flag_a2} {away_team}**")
        form_a = ranker_ft.get_recent_form(away_team, n=10)
        if form_a:
            import plotly.graph_objects as go
            color_map = {"W": "#2dc653", "D": "#f0c040", "L": "#e63946"}
            fig_form_a = go.Figure()
            for i, r in enumerate(form_a):
                fig_form_a.add_trace(go.Bar(
                    x=[i], y=[1],
                    marker_color=color_map.get(r, "#888"),
                    text=r, textposition="inside",
                    showlegend=False,
                    hovertemplate=f"Match {i+1}: {r}<extra></extra>",
                ))
            fig_form_a.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                height=100, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                barmode="stack",
            )
            st.plotly_chart(fig_form_a, use_container_width=True, config={"displayModeBar": False})

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Why This Prediction?</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["SHAP Feature Contributions", "Feature Importances"])

    with tab1:
        if predictor is not None and predictor.is_trained:
            try:
                with st.spinner("Computing SHAP values..."):
                    shap_vals, base_val = predictor.get_shap_values(features_df)
                from src.utils.charts import shap_waterfall
                from src.features.match_features import FEATURE_COLS
                fig_shap = shap_waterfall(shap_vals, FEATURE_COLS, base_val, predicted_outcome)
                st.plotly_chart(fig_shap, use_container_width=True)

                # Natural language explanation
                from src.features.match_features import FEATURE_COLS
                feat_vals = features_df[FEATURE_COLS].iloc[0].to_dict()
                explanations = []

                elo_diff = feat_vals.get("elo_diff", 0)
                if abs(elo_diff) > 100:
                    stronger = home_team if elo_diff > 0 else away_team
                    explanations.append(
                        f"**Elo gap:** {stronger} is significantly stronger "
                        f"({abs(elo_diff):.0f} Elo points difference)"
                    )

                form_diff = feat_vals.get("form_diff_ppg", 0)
                if abs(form_diff) > 0.3:
                    better = home_team if form_diff > 0 else away_team
                    explanations.append(
                        f"**Recent form:** {better} has been in better form "
                        f"({feat_vals.get('form_home_ppg',0):.2f} vs "
                        f"{feat_vals.get('form_away_ppg',0):.2f} pts/game)"
                    )

                h2h_rate = feat_vals.get("h2h_home_win_rate", 0)
                if h2h_rate > 0.55:
                    explanations.append(f"**H2H advantage:** {home_team} has won {h2h_rate*100:.0f}% of recent meetings")
                elif h2h_rate < 0.35:
                    explanations.append(f"**H2H disadvantage:** {away_team} has dominated recent meetings")

                if not neutral:
                    explanations.append("**Home advantage:** Playing at home adds ~100 Elo points")

                if explanations:
                    st.markdown("**Key factors:**")
                    for exp in explanations:
                        st.markdown(f"- {exp}")

            except Exception as e:
                st.info(f"SHAP analysis unavailable: {e}")
        else:
            st.info("Train the model (run setup) for SHAP explanations.")

    with tab2:
        if predictor is not None and predictor.feature_importances_ is not None:
            from src.utils.charts import feature_importance_bar
            fig_fi = feature_importance_bar(predictor.feature_importances_)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            # Show Elo-based explanation
            st.markdown("**Prediction driven by:**")
            st.markdown(f"- Elo difference: **{home_elo - away_elo:.0f}** points")
            st.markdown(f"- {home_team} Elo: **{home_elo:.0f}**")
            st.markdown(f"- {away_team} Elo: **{away_elo:.0f}**")
            st.markdown(f"- Venue: {'Neutral' if neutral else 'Home advantage'}")

    # ── Model info ────────────────────────────────────────────────────────────
    with st.expander("Model Details"):
        if predictor is not None and predictor.is_trained:
            st.markdown("""
**Ensemble Composition:**
- XGBoost Classifier (45% weight) — n_estimators=400, max_depth=5
- LightGBM Classifier (40% weight) — n_estimators=400, max_depth=5
- Logistic Regression (15% weight) — multinomial, L2 regularization

**Training:**
- Dataset: 20+ years of international matches (2000–2022 train, 2022+ validation)
- Features: 21 engineered features including Elo, form, H2H, and tournament context
- Target: 3-class classification (Home Win / Draw / Away Win)

**Elo System:**
- Initial rating: 1500 for all teams
- K-factor varies by tournament (World Cup K=60, Friendly K=20)
- Goal-difference multiplier (rewards larger wins)
- Home advantage: +100 Elo points for non-neutral matches
""")
        else:
            st.markdown("Model not trained. Run setup to train the full model.")
