"""Football Predictor — Main Streamlit App (Home Page)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1c1e26 0%, #1a2a1a 100%);
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}
.metric-card h2 { color: #52b788; font-size: 2rem; margin: 0; }
.metric-card p  { color: #aaa; font-size: 0.85rem; margin: 4px 0 0; }

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #52b788, #f0c040);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-subtitle { font-size: 1.15rem; color: #aaa; margin-top: 8px; }

.feature-card {
    background: #1c1e26;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 18px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: #52b788; }
.feature-card h4 { color: #52b788; margin: 0 0 6px; font-size: 1rem; }
.feature-card p  { color: #bbb; font-size: 0.85rem; margin: 0; }

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-ready  { background: #1a472a; color: #52b788; }
.badge-setup  { background: #3a2a00; color: #f0c040; }
.badge-error  { background: #3a0a0a; color: #e63946; }

.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #52b788;
    border-bottom: 2px solid #1a472a;
    padding-bottom: 6px;
    margin: 20px 0 12px;
}

div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1a472a;
}
div[data-testid="metric-container"] {
    background: #1c1e26;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-setup on first run ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_app():
    """Run setup if data not present."""
    from src.utils.helpers import data_is_ready, model_is_ready
    if data_is_ready() and model_is_ready():
        return "ready"
    try:
        from scripts.setup_data import run_setup
        run_setup()
        return "just_setup"
    except Exception as e:
        return f"error:{e}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Football Predictor")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("app.py", label="Home")
    st.page_link("pages/1_Match_Predictor.py", label="Match Predictor")
    st.page_link("pages/2_Tournament_Simulator.py", label="Tournament Simulator")
    st.page_link("pages/3_Team_Rankings.py", label="Team Rankings")
    st.page_link("pages/4_Player_Ratings.py", label="Player Ratings")
    st.page_link("pages/5_Player_Comparison.py", label="Player Comparison")
    st.page_link("pages/6_About.py", label="About")
    st.markdown("---")

    # Data status
    with st.spinner("Initializing..."):
        init_status = initialize_app()

    from src.utils.helpers import data_is_ready, model_is_ready

    if data_is_ready():
        st.markdown('<span class="status-badge badge-ready">✓ Data Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-setup">⟳ Setting Up...</span>', unsafe_allow_html=True)

    if model_is_ready():
        st.markdown('<span class="status-badge badge-ready">✓ Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-setup">⟳ Training...</span>', unsafe_allow_html=True)

    if init_status.startswith("error"):
        st.error(f"Setup error: {init_status[6:]}")

    st.markdown("---")
    st.markdown(
        "<small style='color:#666'>Data: International Results 1872–2024<br>"
        "Model: XGBoost + LightGBM Ensemble<br>"
        "Elo: Custom World Football System</small>",
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 30px 10px 20px;">
    <div class="hero-title">Football Predictor</div>
    <div class="hero-subtitle">
        AI-powered football analytics — match predictions, tournament simulations,
        team rankings, and player ratings built on 150 years of international football data.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Quick Stats ───────────────────────────────────────────────────────────────
from src.utils.helpers import load_processed_results, load_player_ratings, load_team_rankings

df = load_processed_results()
players = load_player_ratings()
rankings = load_team_rankings()

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_matches = f"{len(df):,}" if df is not None else "—"
    st.markdown(f'<div class="metric-card"><h2>{n_matches}</h2><p>Matches Analysed</p></div>', unsafe_allow_html=True)
with col2:
    n_teams = f"{df['home_team'].nunique():,}" if df is not None else "—"
    st.markdown(f'<div class="metric-card"><h2>{n_teams}</h2><p>International Teams</p></div>', unsafe_allow_html=True)
with col3:
    n_players = f"{len(players):,}" if players is not None else "—"
    st.markdown(f'<div class="metric-card"><h2>{n_players}</h2><p>Players Rated</p></div>', unsafe_allow_html=True)
with col4:
    years = "150+" if df is not None else "—"
    st.markdown(f'<div class="metric-card"><h2>{years}</h2><p>Years of Data</p></div>', unsafe_allow_html=True)


st.markdown("<div class='section-header'>Platform Features</div>", unsafe_allow_html=True)

# Feature cards
cols = st.columns(3)
features = [
    ("Match Predictor",
     "Predict match outcomes using an XGBoost + LightGBM ensemble trained on 20+ years "
     "of international data. Features include Elo ratings, recent form, H2H records, "
     "and tournament level."),
    ("Tournament Simulator",
     "Run 10,000 Monte Carlo simulations for World Cup, Euro, Copa América, and more. "
     "Get champion, final, semi-final, and group-stage probabilities for every team."),
    ("Team Power Rankings",
     "Multi-dimensional ranking beyond FIFA. Combines Elo, recent form, attack and "
     "defense metrics, and major tournament performance into a composite power score."),
    ("Player Ratings",
     "Position-specific ratings derived from per-90 statistics across Europe's top "
     "leagues and international competitions. Covers forwards, midfielders, defenders, "
     "and goalkeepers."),
    ("Player Comparison",
     "Compare any two players side-by-side with radar charts, percentile rankings, "
     "statistical breakdowns, and analytical insights."),
    ("Explainable AI",
     "Every prediction comes with SHAP-powered explanations showing exactly which "
     "factors drove the model's decision — from Elo differences to head-to-head records."),
]
for i, (title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(
            f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )


# ── Live Rankings Snapshot ────────────────────────────────────────────────────
if rankings is not None and not rankings.empty:
    st.markdown("<div class='section-header'>Top 10 Teams by Power Rating</div>", unsafe_allow_html=True)
    top10 = rankings.head(10).copy()

    from src.utils.helpers import get_flag_emoji
    import plotly.graph_objects as go

    team_labels = [f"{get_flag_emoji(t)} {t}" for t in top10["team"]]

    fig_top10 = go.Figure()
    fig_top10.add_trace(go.Bar(
        name="Power Rating",
        x=team_labels,
        y=top10["power_rating"],
        marker_color="#52b788",
        text=[f"{v:.1f}" for v in top10["power_rating"]],
        textposition="outside",
    ))
    fig_top10.add_trace(go.Bar(
        name="Attack",
        x=team_labels,
        y=top10.get("attack_rating", top10["power_rating"] * 0),
        marker_color="#f0c040",
        text=[f"{v:.1f}" for v in top10.get("attack_rating", top10["power_rating"] * 0)],
        textposition="outside",
    ))
    fig_top10.add_trace(go.Bar(
        name="Defense",
        x=team_labels,
        y=top10.get("defense_rating", top10["power_rating"] * 0),
        marker_color="#457b9d",
        text=[f"{v:.1f}" for v in top10.get("defense_rating", top10["power_rating"] * 0)],
        textposition="outside",
    ))
    fig_top10.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=380,
        margin=dict(l=20, r=20, t=20, b=60),
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(tickangle=-25, gridcolor="#2a2d3a"),
        yaxis=dict(gridcolor="#2a2d3a", title="Rating"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.3),
    )
    st.plotly_chart(fig_top10, use_container_width=True, config={"displayModeBar": False})

    # Confederation breakdown pie chart
    if "confederation" in rankings.columns:
        import plotly.express as px
        CONF_COLORS = {
            "UEFA": "#3a86ff", "CONMEBOL": "#ffbe0b", "CONCACAF": "#fb5607",
            "CAF": "#8338ec", "AFC": "#06d6a0", "OFC": "#ef233c", "Other": "#adb5bd",
        }
        conf_counts = rankings["confederation"].value_counts().reset_index()
        conf_counts.columns = ["confederation", "teams"]
        fig_conf_pie = px.pie(
            conf_counts,
            names="confederation",
            values="teams",
            color="confederation",
            color_discrete_map=CONF_COLORS,
            hole=0.45,
            template="plotly_dark",
        )
        fig_conf_pie.update_layout(
            paper_bgcolor="#0f1117",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            title=dict(text="Teams by Confederation", font_size=13, font_color="#52b788"),
        )
        fig_conf_pie.update_traces(textinfo="label+percent", textfont_size=11)
        st.plotly_chart(fig_conf_pie, use_container_width=True, config={"displayModeBar": False})


# ── Top Players Snapshot ──────────────────────────────────────────────────────
if players is not None and not players.empty:
    st.markdown("<div class='section-header'>Top Rated Players</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Overall Top 5**")
        top5 = players.head(5)
        for _, p in top5.iterrows():
            from src.utils.helpers import rating_badge
            st.markdown(
                f"{rating_badge(p.get('overall_rating', 0))} &nbsp; "
                f"**{p['name']}** &nbsp; <small style='color:#aaa'>"
                f"{p.get('position','?')} · {p.get('nationality','?')} · "
                f"{p.get('club','?')}</small>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**Top Forwards**")
        fw = players[players["position"] == "FW"].head(5)
        for _, p in fw.iterrows():
            from src.utils.helpers import rating_badge
            st.markdown(
                f"{rating_badge(p.get('overall_rating', 0))} &nbsp; "
                f"**{p['name']}** &nbsp; <small style='color:#aaa'>"
                f"{p.get('nationality','?')} · {p.get('club','?')}</small>",
                unsafe_allow_html=True,
            )


# ── Methodology ───────────────────────────────────────────────────────────────
with st.expander("Methodology Overview", expanded=False):
    st.markdown("""
### Match Prediction
- **Elo Rating System** — Custom world football Elo with K-factor varying by tournament importance
  (World Cup K=60, Friendly K=20), goal-difference multiplier, and home advantage (+100 pts)
- **Features** — Elo difference, recent form (last 10 matches), head-to-head record,
  goals scored/conceded averages, tournament importance, neutral venue
- **Models** — XGBoost (45%) + LightGBM (40%) + Logistic Regression (15%) soft-voting ensemble
- **Explainability** — SHAP values computed via TreeExplainer for every prediction

### Tournament Simulation
- Monte Carlo with **10,000 simulations** per tournament
- Uses the match predictor as the outcome engine
- Simulates group stage with realistic scorelines (Poisson distribution)
- Handles tiebreakers: points → goal difference → goals scored
- Supports 32-team (WC2022), 24-team (Euro2024), 48-team (WC2026) formats

### Team Rankings
- Multi-dimensional: **Elo** (35%) + **Attack** (20%) + **Defense** (20%) + **Form** (15%) + **Tournament** (10%)
- Exponential decay weighting — recent matches count more
- Tournament bonus for major competition performance (last 4 years)

### Player Ratings
- Position-specific weighted formulas on per-90 statistics
- Percentile normalization within position group
- Covers forwards (goals/xG/assists/carries), midfielders (prog. passes/xA/tackles),
  defenders (tackles/interceptions/clearances/aerials), goalkeepers (save%/PSxG-GA)
""")

st.markdown("---")
st.markdown(
    "<center><small style='color:#555'>Football Predictor · Built with Streamlit, XGBoost, SHAP · "
    "Data: International Results 1872–2024 · Made for Hugging Face Spaces</small></center>",
    unsafe_allow_html=True,
)
