"""Central configuration for Football Predictor."""
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data source URLs (public, freely available)
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)

# Elo system parameters
ELO_INITIAL = 1500
ELO_HOME_ADVANTAGE = 100  # points added for home team

# K-factors by tournament category (higher = more volatile)
TOURNAMENT_K_FACTORS = {
    "FIFA World Cup": 60,
    "Confederations Cup": 50,
    "UEFA Euro": 50,
    "Copa America": 50,
    "AFC Asian Cup": 50,
    "Africa Cup of Nations": 50,
    "CONCACAF Gold Cup": 40,
    "UEFA Nations League": 40,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 35,
    "Copa America qualification": 35,
    "AFC Asian Cup qualification": 30,
    "AFCON qualification": 30,
    "Friendly": 20,
}
DEFAULT_K_FACTOR = 30

# Tournament importance score (used as a feature)
TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.9,
    "Copa America": 0.9,
    "AFC Asian Cup": 0.8,
    "Africa Cup of Nations": 0.8,
    "CONCACAF Gold Cup": 0.7,
    "Confederations Cup": 0.7,
    "UEFA Nations League": 0.6,
    "FIFA World Cup qualification": 0.6,
    "UEFA Euro qualification": 0.5,
    "Copa America qualification": 0.5,
    "AFC Asian Cup qualification": 0.45,
    "AFCON qualification": 0.45,
    "Friendly": 0.2,
}

# Training config
TRAIN_CUTOFF_YEAR = 2000  # Only use matches from 2000 onward
TEST_CUTOFF_YEAR = 2022   # Matches from 2022 onward held out for test

# Form window
FORM_WINDOW = 10          # Last N matches for form calculation
H2H_WINDOW = 10           # Last N H2H meetings

# Outcome labels
OUTCOME_HOME_WIN = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY_WIN = 2
OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

# Prediction confidence thresholds
CONFIDENCE_HIGH = 0.65
CONFIDENCE_MEDIUM = 0.50

# Player position groups
POSITION_GROUPS = {
    "FW": "Forward",
    "MF": "Midfielder",
    "DF": "Defender",
    "GK": "Goalkeeper",
}

# UI config
APP_TITLE = "Football Predictor"
APP_ICON = "⚽"
THEME_COLOR = "#1a472a"  # Dark green

# Simulation config
N_SIMULATIONS = 10_000

# Top teams for quick-select
TOP_TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain",
    "Germany", "Portugal", "Netherlands", "Italy", "Belgium",
    "Croatia", "Uruguay", "Morocco", "Senegal", "Japan",
    "South Korea", "United States", "Mexico", "Colombia", "Denmark",
    "Switzerland", "Austria", "Poland", "Serbia", "Australia",
]

# Confederation mapping
CONFEDERATIONS = {
    "UEFA": [
        "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
        "Italy", "Belgium", "Croatia", "Denmark", "Switzerland", "Austria",
        "Poland", "Serbia", "Sweden", "Norway", "Czech Republic", "Hungary",
        "Slovakia", "Scotland", "Wales", "Turkey", "Ukraine", "Greece",
        "Romania", "Bulgaria", "Slovenia", "Albania", "Finland", "Ireland",
        "Bosnia and Herzegovina", "North Macedonia", "Kosovo", "Iceland",
        "Georgia", "Russia",
    ],
    "CONMEBOL": [
        "Argentina", "Brazil", "Uruguay", "Colombia", "Chile", "Ecuador",
        "Peru", "Paraguay", "Venezuela", "Bolivia",
    ],
    "CONCACAF": [
        "United States", "Mexico", "Canada", "Costa Rica", "Jamaica",
        "Honduras", "El Salvador", "Guatemala", "Panama", "Trinidad and Tobago",
        "Cuba", "Haiti",
    ],
    "CAF": [
        "Morocco", "Senegal", "Nigeria", "Egypt", "Cameroon", "Ghana",
        "Ivory Coast", "Algeria", "Tunisia", "South Africa", "Mali",
        "Burkina Faso", "Democratic Republic of the Congo", "Guinea",
        "Kenya", "Uganda", "Tanzania", "Zambia", "Zimbabwe",
    ],
    "AFC": [
        "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia",
        "Qatar", "China", "Iraq", "United Arab Emirates", "Oman",
        "Indonesia", "Vietnam", "Thailand", "India",
    ],
    "OFC": [
        "New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands",
    ],
}
