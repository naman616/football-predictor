"""About Page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="About · Football Predictor", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.3rem; font-weight: 600; color: #52b788;
    border-bottom: 2px solid #1a472a; padding-bottom: 6px; margin: 24px 0 14px;
}
.tech-card {
    background: #1c1e26; border: 1px solid #2a2d3a; border-radius: 10px;
    padding: 16px; margin: 8px 0;
}
.tech-card h4 { color: #52b788; margin: 0 0 6px; }
.tech-card p { color: #bbb; font-size: 0.88rem; margin: 0; }
.metric-highlight {
    background: linear-gradient(135deg, #1c1e26, #1a2a1a);
    border: 1px solid #2d6a4f; border-radius: 10px;
    padding: 16px; text-align: center;
}
.metric-highlight h3 { color: #52b788; font-size: 1.8rem; margin: 0; }
.metric-highlight p { color: #888; font-size: 0.8rem; margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ℹ️ About Football Predictor")
st.markdown("A production-quality football analytics platform built on 150+ years of international match data.")

st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)
st.markdown("""
Football Predictor is a data-driven analytics platform that applies machine learning,
statistical modeling, and football domain expertise to provide:

- **Accurate match predictions** using an ensemble of XGBoost, LightGBM, and Logistic Regression
- **Tournament simulations** via Monte Carlo methods running thousands of scenarios
- **Realistic team rankings** combining Elo ratings, recent form, and tournament performance
- **Position-specific player ratings** based on per-90 statistics
- **Explainable predictions** powered by SHAP values

Every number in this platform is derived from data and models — not arbitrary heuristics.
""")


st.markdown("<div class='section-header'>Data Sources</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**International Match Data**
- Source: [Martj42 International Results](https://github.com/martj42/international_results)
- Coverage: 1872–2024, ~47,000+ matches
- Includes: All international fixtures, scores, tournament context

**Elo Ratings**
- Computed from scratch using the complete match history
- Custom K-factor schedule by tournament importance
- Goal-difference multiplier for realistic rating changes
""")

with col2:
    st.markdown("""
**Player Statistics**
- Season-level statistics for top players (2022–2024)
- Covers: Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- Includes: Goals, assists, xG, xA, progressive passes/carries, defensive actions

**Tournament Data**
- Hardcoded group draws for major tournaments:
  - FIFA World Cup 2022 & 2026
  - UEFA Euro 2024
  - Copa América 2024
""")


st.markdown("<div class='section-header'>Technical Architecture</div>", unsafe_allow_html=True)
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.markdown("""
    <div class="tech-card">
        <h4>🤖 ML Models</h4>
        <p><b>XGBoost</b> (45%) — Gradient boosted trees, 400 estimators, max_depth=5<br><br>
        <b>LightGBM</b> (40%) — Light gradient boosting, fast inference<br><br>
        <b>Logistic Regression</b> (15%) — Regularized multinomial, calibrated probabilities</p>
    </div>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("""
    <div class="tech-card">
        <h4>📊 Elo System</h4>
        <p>Custom world football Elo with:<br><br>
        • Tournament K-factors (WC=60, Friendly=20)<br>
        • Goal-difference multiplier<br>
        • Home advantage (+100 pts)<br>
        • Processes 47k+ matches chronologically</p>
    </div>
    """, unsafe_allow_html=True)

with col_t3:
    st.markdown("""
    <div class="tech-card">
        <h4>🎲 Tournament Simulator</h4>
        <p>Monte Carlo simulation:<br><br>
        • Up to 20,000 iterations per tournament<br>
        • Poisson-distributed scorelines<br>
        • Proper tiebreaker rules<br>
        • Supports 32, 24, and 48-team formats</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div class='section-header'>Feature Engineering</div>", unsafe_allow_html=True)
with st.expander("Match Prediction Features (21 features)", expanded=False):
    features = {
        "Elo Features": [
            "elo_home — Home team current Elo rating",
            "elo_away — Away team current Elo rating",
            "elo_diff — Elo difference (home - away)",
            "elo_diff_sq — Squared Elo difference (captures non-linear effects)",
        ],
        "Form Features": [
            "form_home_ppg — Home team points per game (last 10 matches)",
            "form_away_ppg — Away team points per game (last 10 matches)",
            "form_home_gf — Home team avg goals scored (last 10)",
            "form_away_gf — Away team avg goals scored (last 10)",
            "form_home_ga — Home team avg goals conceded (last 10)",
            "form_away_ga — Away team avg goals conceded (last 10)",
            "form_home_gd — Home team goal difference per game",
            "form_away_gd — Away team goal difference per game",
            "form_diff_ppg — Form differential (home PPG - away PPG)",
            "gf_diff — Goals scored differential",
            "ga_diff — Goals conceded differential",
        ],
        "H2H Features": [
            "h2h_home_win_rate — Home team win rate in recent H2H meetings",
            "h2h_draw_rate — Draw rate in recent H2H meetings",
            "h2h_home_gf — Avg goals scored by home team in H2H",
            "h2h_away_gf — Avg goals scored by away team in H2H",
        ],
        "Context Features": [
            "tournament_importance — Tournament level (0.2 for friendly, 1.0 for World Cup)",
            "is_neutral — Whether match is at a neutral venue",
        ],
    }
    for category, feats in features.items():
        st.markdown(f"**{category}:**")
        for f in feats:
            st.markdown(f"  - `{f}`")


st.markdown("<div class='section-header'>Player Rating Methodology</div>", unsafe_allow_html=True)
with st.expander("Rating weights by position", expanded=False):
    col_pos1, col_pos2 = st.columns(2)
    with col_pos1:
        st.markdown("""
**Forwards (FW):**
| Metric | Weight |
|---|---|
| Goals/90 | 28% |
| xG/90 | 18% |
| Assists/90 | 12% |
| xA/90 | 10% |
| Shots on Target/90 | 8% |
| Key Passes/90 | 8% |
| Progressive Carries/90 | 7% |
| Dribbles Completed/90 | 5% |
| Conversion Rate | 4% |

**Midfielders (MF):**
| Metric | Weight |
|---|---|
| Progressive Passes/90 | 22% |
| xA/90 | 18% |
| Assists/90 | 14% |
| Key Passes/90 | 12% |
| Pass Completion % | 12% |
| Goals/90 | 8% |
| Tackles/90 | 7% |
| Interceptions/90 | 7% |
""")
    with col_pos2:
        st.markdown("""
**Defenders (DF):**
| Metric | Weight |
|---|---|
| Tackles/90 | 18% |
| Interceptions/90 | 18% |
| Clearances/90 | 14% |
| Blocks/90 | 12% |
| Aerial Win % | 14% |
| Pass Completion % | 10% |
| Progressive Passes/90 | 8% |
| Progressive Carries/90 | 6% |

**Goalkeepers (GK):**
| Metric | Weight |
|---|---|
| Save % | 30% |
| PSxG-GA per 90 | 30% |
| Clean Sheet % | 25% |
| Pass Completion % | 15% |
""")


st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown('<div class="metric-highlight"><h3>~54%</h3><p>Model Accuracy<br>(3-class, 2022–24)</p></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-highlight"><h3>~1.02</h3><p>Log-Loss Score<br>(lower = better)</p></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-highlight"><h3>10K</h3><p>Monte Carlo<br>Simulations</p></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown('<div class="metric-highlight"><h3>21</h3><p>Engineered<br>Features</p></div>', unsafe_allow_html=True)

st.markdown("""
> **Note on accuracy:** Predicting football is inherently difficult — even the best models achieve ~55% on 3-class (W/D/L)
> problems. The model's strength is in producing **well-calibrated probabilities**, not in picking exact outcomes.
> A 70% prediction means the event happens roughly 70% of the time.
""")


st.markdown("<div class='section-header'>Deployment</div>", unsafe_allow_html=True)
st.markdown("""
**Hosted on:** Hugging Face Spaces (Streamlit)

**Stack:**
- Frontend: Streamlit 1.32 + Plotly 5.20
- ML: XGBoost 2.0 + LightGBM 4.3 + scikit-learn 1.4
- Explainability: SHAP 0.44
- Data: pandas 2.2 + pyarrow (Parquet)

**Reproducibility:**
- All random seeds fixed for deterministic outputs
- Models saved as `.joblib` files
- Processed data cached as `.parquet` files

Run the setup script to reproduce everything from raw data:
```bash
python scripts/setup_data.py
```
""")

st.markdown("---")
st.markdown("""
<center>
<small style='color:#555'>
Football Predictor · Built with ❤️ for the football analytics community ·
Data: Martj42 International Results (1872–2024) ·
Hosted on Hugging Face Spaces
</small>
</center>
""", unsafe_allow_html=True)
