---
title: Football Predictor
emoji: ⚽
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered football analytics platform with match predictions, tournament simulations, team rankings, and player ratings.
---

# ⚽ Football Predictor

A production-quality football analytics platform built on 150+ years of international football data.

## Features

| Module | Description |
|---|---|
| 🎯 **Match Predictor** | XGBoost + LightGBM ensemble predicting Win/Draw/Loss probabilities with SHAP explainability |
| 🏆 **Tournament Simulator** | Monte Carlo (10,000 simulations) for World Cup, Euro, Copa América |
| 📊 **Team Rankings** | Custom multi-dimensional Elo + form + attack/defense power ratings |
| ⭐ **Player Ratings** | Position-specific per-90 ratings for 60+ top international players |
| ⚖️ **Player Comparison** | Radar charts, percentile rankings, and analytical insights |
| 🔍 **Explainability** | SHAP values for every match prediction |

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/football-predictor
cd football-predictor

# Install dependencies
pip install -r requirements.txt

# Download data, train models, compute ratings (run once)
python scripts/setup_data.py

# Launch the app
streamlit run app.py
```

### On Hugging Face Spaces

The app auto-initializes on first load:
1. Downloads international results from GitHub (~5MB)
2. Computes Elo ratings for all teams
3. Trains the match prediction ensemble
4. Builds team rankings and player ratings

This takes 2–5 minutes on first run. Subsequent loads use cached data.

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| International Results | [martj42/international_results](https://github.com/martj42/international_results) | 1872–2024, ~47k matches |
| Player Statistics | Embedded (top players 2022–24) | 60+ players, 5 leagues |
| Tournament Draws | Hardcoded | WC2022, WC2026, Euro2024, Copa2024 |

## Architecture

```
football-predictor/
├── app.py                    # Home page (Streamlit multi-page)
├── pages/
│   ├── 1_Match_Predictor.py
│   ├── 2_Tournament_Simulator.py
│   ├── 3_Team_Rankings.py
│   ├── 4_Player_Ratings.py
│   ├── 5_Player_Comparison.py
│   └── 6_About.py
├── src/
│   ├── data/
│   │   └── loader.py         # Data download & loading
│   ├── features/
│   │   └── match_features.py # Feature engineering (21 features)
│   ├── models/
│   │   ├── elo_system.py     # Custom Elo rating system
│   │   ├── match_predictor.py # XGBoost + LightGBM ensemble
│   │   ├── tournament_simulator.py # Monte Carlo simulator
│   │   ├── team_ranker.py    # Multi-dimensional rankings
│   │   └── player_rater.py  # Position-specific ratings
│   └── utils/
│       ├── charts.py         # Plotly visualizations
│       └── helpers.py        # Streamlit caching utilities
├── config/
│   └── settings.py           # Central configuration
└── scripts/
    └── setup_data.py         # One-time data setup
```

## Models

### Match Predictor
- **XGBoost** (45% weight) — n_estimators=400, max_depth=5, learning_rate=0.05
- **LightGBM** (40% weight) — n_estimators=400, max_depth=5
- **Logistic Regression** (15% weight) — multinomial, L2 regularization
- Training: Time-based split (pre-2022 train, 2022+ validation)
- Accuracy: ~54% (3-class W/D/L on unseen data)

### Elo System
- Initial rating: 1500
- K-factors: World Cup=60, Major Tournament=50, Qualification=40, Friendly=20
- Goal difference multiplier: 2-goal margin +50%, 3+ goals +75%
- Home advantage: +100 Elo points

### Team Rankings
Composite score = 35% Elo + 20% Attack + 20% Defense + 15% Form + 10% Tournament

### Player Ratings
Position-specific weighted sum of percentile-ranked per-90 statistics.
See the About page for full weight tables.

## Requirements

```
streamlit==1.32.0
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1
xgboost==2.0.3
lightgbm==4.3.0
shap==0.44.1
plotly==5.20.0
requests==2.31.0
joblib==1.3.2
scipy==1.12.0
pyarrow==15.0.2
```

## License

MIT License — feel free to use, modify, and distribute.

## Acknowledgments

- Match data: [Martj42 International Results](https://github.com/martj42/international_results)
- Elo methodology inspired by [World Football Elo Ratings](https://www.eloratings.net/)
- Player statistics curated from publicly available sources (2022–2024)
