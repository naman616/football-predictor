"""Match outcome prediction model — ensemble of XGBoost + LightGBM + Logistic Regression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from config.settings import (
    MODELS_DIR, PROCESSED_DIR, TEST_CUTOFF_YEAR, OUTCOME_LABELS,
)
from src.features.match_features import FEATURE_COLS, MatchFeatureEngine

logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Ensemble match outcome predictor.

    Combines XGBoost, LightGBM, and Logistic Regression via soft voting.
    Trained on time-split data to avoid leakage.
    """

    MODEL_PATH = MODELS_DIR / "match_predictor.joblib"

    def __init__(self):
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.lr_pipeline: Optional[Pipeline] = None
        self.feature_importances_: Optional[pd.Series] = None
        self.is_trained: bool = False

        # Ensemble weights (tuned on validation set)
        self.weights = {"xgb": 0.45, "lgb": 0.40, "lr": 0.15}

    def _build_models(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
                solver="lbfgs",
            )),
        ])

    def train(self, features_df: pd.DataFrame) -> dict:
        """
        Train all models.

        features_df must have FEATURE_COLS + 'result' column.
        Uses time-based train/test split at TEST_CUTOFF_YEAR.
        """
        self._build_models()

        X = features_df[FEATURE_COLS].values
        y = features_df["result"].values

        # Time-based split
        n = len(features_df)
        train_size = int(n * 0.85)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")

        # Train XGBoost
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Train LightGBM
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        # Train Logistic Regression
        self.lr_pipeline.fit(X_train, y_train)

        # Compute feature importances from XGBoost
        self.feature_importances_ = pd.Series(
            self.xgb_model.feature_importances_,
            index=FEATURE_COLS,
        ).sort_values(ascending=False)

        # Validation metrics
        val_preds = self.predict_proba_matrix(X_val)
        val_acc = accuracy_score(y_val, np.argmax(val_preds, axis=1))
        val_ll = log_loss(y_val, val_preds)

        self.is_trained = True
        self.save()

        metrics = {
            "val_accuracy": val_acc,
            "val_log_loss": val_ll,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        logger.info(f"Trained: acc={val_acc:.3f}, log_loss={val_ll:.3f}")
        return metrics

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 3) probability matrix for input matrix X."""
        # Pass as DataFrame to give LightGBM feature names (avoids UserWarning)
        X_df = pd.DataFrame(X, columns=FEATURE_COLS)
        p_xgb = self.xgb_model.predict_proba(X_df)
        p_lgb = self.lgb_model.predict_proba(X_df)
        p_lr = self.lr_pipeline.predict_proba(X_df)
        ensemble = (
            self.weights["xgb"] * p_xgb
            + self.weights["lgb"] * p_lgb
            + self.weights["lr"] * p_lr
        )
        # Normalize rows to sum exactly to 1.0 (guard against floating-point drift)
        row_sums = ensemble.sum(axis=1, keepdims=True)
        return ensemble / row_sums

    def predict(
        self,
        feature_df: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Predict outcome probabilities for a single match.

        feature_df: single-row DataFrame with FEATURE_COLS columns.
        Returns: (p_home_win, p_draw, p_away_win)
        """
        X = feature_df[FEATURE_COLS].values
        probs = self.predict_proba_matrix(X)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def confidence_score(self, probs: tuple[float, float, float]) -> float:
        """Return max probability as confidence score."""
        return max(probs)

    def get_shap_values(self, feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values using XGBoost explainer.
        Returns (shap_values, base_values) for the predicted class.
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.xgb_model)
            X = feature_df[FEATURE_COLS].values
            shap_vals = explainer.shap_values(X)
            # shap_vals is (n_classes, n_samples, n_features)
            probs = self.predict_proba_matrix(X)[0]
            predicted_class = int(np.argmax(probs))
            if isinstance(shap_vals, list):
                return shap_vals[predicted_class][0], explainer.expected_value[predicted_class]
            return shap_vals[0], explainer.expected_value
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return np.zeros(len(FEATURE_COLS)), 0.0

    def save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Store feature_importances as plain dict to avoid pandas StringDtype
        # version incompatibility between pandas 2.x and 3.x
        fi_dict = (
            {str(k): float(v) for k, v in self.feature_importances_.items()}
            if self.feature_importances_ is not None
            else {}
        )
        joblib.dump(
            {
                "xgb": self.xgb_model,
                "lgb": self.lgb_model,
                "lr": self.lr_pipeline,
                "feature_importances": fi_dict,
                "weights": self.weights,
            },
            self.MODEL_PATH,
        )
        logger.info(f"Model saved to {self.MODEL_PATH}")

    def load(self) -> bool:
        if not self.MODEL_PATH.exists():
            return False
        data = joblib.load(self.MODEL_PATH)
        self.xgb_model = data["xgb"]
        self.lgb_model = data["lgb"]
        self.lr_pipeline = data["lr"]
        fi_raw = data["feature_importances"]
        if isinstance(fi_raw, dict):
            self.feature_importances_ = pd.Series(fi_raw, dtype=float).sort_values(ascending=False)
        else:
            self.feature_importances_ = fi_raw
        self.weights = data.get("weights", self.weights)
        self.is_trained = True
        logger.info("Model loaded from disk")
        return True


def build_and_train(
    df_with_elo: pd.DataFrame,
    feature_engine: "MatchFeatureEngine",
) -> tuple["MatchPredictor", dict]:
    """Full training pipeline: build features, train models, return predictor."""
    features_path = PROCESSED_DIR / "match_features.parquet"

    if features_path.exists():
        logger.info("Loading cached features")
        features_df = pd.read_parquet(features_path)
    else:
        logger.info("Building match features (this may take a few minutes)...")
        features_df = feature_engine.build_all_features()
        features_df.to_parquet(features_path, index=False)
        logger.info(f"Built {len(features_df)} feature rows")

    predictor = MatchPredictor()
    metrics = predictor.train(features_df)
    return predictor, metrics
