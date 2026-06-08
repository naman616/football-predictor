"""
Data setup script.

Run once to download raw data, compute Elo ratings, train models,
and generate player ratings. All outputs stored in data/processed/ and models/.

Usage:
    python scripts/setup_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_setup(force: bool = False):
    from config.settings import PROCESSED_DIR, MODELS_DIR

    logger.info("=" * 60)
    logger.info("Football Predictor — Data Setup")
    logger.info("=" * 60)

    # Step 1: Download international results
    logger.info("\n[1/6] Downloading international results...")
    from src.data.loader import load_international_results
    df_results = load_international_results(force_download=force)
    logger.info(f"Loaded {len(df_results):,} matches from {df_results['year'].min()} to {df_results['year'].max()}")

    # Step 2: Compute Elo ratings
    logger.info("\n[2/6] Computing Elo ratings...")
    t0 = time.time()
    from src.models.elo_system import build_and_save_elo
    elo_system, df_with_elo = build_and_save_elo(df_results)
    logger.info(f"Elo computed in {time.time() - t0:.1f}s. {len(elo_system.ratings)} teams rated.")

    # Top 10 teams by Elo
    top10 = elo_system.get_current_ratings().head(10)
    logger.info("\nTop 10 teams by Elo:")
    for _, row in top10.iterrows():
        logger.info(f"  {row['rank']:2d}. {row['team']:<25} {row['elo']:.0f}")

    # Step 3: Build match features
    logger.info("\n[3/6] Building match features...")
    features_path = PROCESSED_DIR / "match_features.parquet"
    if not features_path.exists() or force:
        t0 = time.time()
        from src.features.match_features import MatchFeatureEngine
        engine = MatchFeatureEngine(df_with_elo)
        features_df = engine.build_all_features()
        features_df.to_parquet(features_path, index=False)
        logger.info(f"Built {len(features_df):,} feature rows in {time.time() - t0:.1f}s")
    else:
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        logger.info(f"Loaded cached features: {len(features_df):,} rows")

    # Step 4: Train match predictor
    logger.info("\n[4/6] Training match prediction model...")
    model_path = MODELS_DIR / "match_predictor.joblib"
    if not model_path.exists() or force:
        t0 = time.time()
        from src.models.match_predictor import MatchPredictor
        predictor = MatchPredictor()
        metrics = predictor.train(features_df)
        logger.info(f"Model trained in {time.time() - t0:.1f}s")
        logger.info(f"  Validation accuracy: {metrics['val_accuracy']:.3f}")
        logger.info(f"  Validation log-loss: {metrics['val_log_loss']:.3f}")
        logger.info(f"  Train samples: {metrics['train_samples']:,}")
        logger.info(f"  Val samples: {metrics['val_samples']:,}")
    else:
        logger.info("Model already trained, skipping")

    # Step 5: Build team rankings
    logger.info("\n[5/6] Building team rankings...")
    rankings_path = PROCESSED_DIR / "team_rankings.parquet"
    if not rankings_path.exists() or force:
        t0 = time.time()
        from src.models.team_ranker import TeamRanker
        elo_ratings = elo_system.get_current_ratings()
        ranker = TeamRanker(df_with_elo, elo_ratings)
        rankings = ranker.build_rankings()
        logger.info(f"Rankings built in {time.time() - t0:.1f}s. {len(rankings)} teams ranked.")
        logger.info("\nTop 10 teams by Power Rating:")
        for _, row in rankings.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['team']:<25} Power: {row['power_rating']:.1f}")
    else:
        logger.info("Rankings already built, skipping")

    # Step 6: Compute player ratings
    logger.info("\n[6/6] Computing player ratings...")
    player_path = PROCESSED_DIR / "player_ratings.parquet"
    if not player_path.exists() or force:
        t0 = time.time()
        from src.data.loader import load_player_data
        from src.models.player_rater import PlayerRater
        player_df = load_player_data()
        rater = PlayerRater(player_df)
        rated = rater.compute_ratings()
        logger.info(f"Rated {len(rated)} players in {time.time() - t0:.1f}s")
        if not rated.empty:
            logger.info("\nTop 10 players by overall rating:")
            for _, row in rated.head(10).iterrows():
                logger.info(
                    f"  {row.get('global_rank', '?'):2}. {row['name']:<22} "
                    f"{row.get('position', '?'):<3} "
                    f"Overall: {row.get('overall_rating', 0):.1f}"
                )
    else:
        logger.info("Player ratings already computed, skipping")

    logger.info("\n" + "=" * 60)
    logger.info("Setup complete! Run: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Football Predictor data setup")
    parser.add_argument("--force", action="store_true", help="Re-download and re-process everything")
    args = parser.parse_args()
    run_setup(force=args.force)
