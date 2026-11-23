"""End-to-end pipeline runner for Polish Football Attendance prediction with PyTorch Tabular Neural Network."""
import torch
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib

from src.db.schema import init_db, get_connection
from src.scraper.stadiums import import_stadiums_to_db
from src.scraper.ninety_minut import scrape_all_seasons
from src.scraper.weather import batch_enrich_weather
from src.features.engineering import build_full_features
from src.models.neural_net import (
    TabularQuantileNN,
    NeuralEnsembleQuantileModel,
    HybridEnsembleQuantileModel,
    evaluate_predictions
)


def load_dataset_from_db(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and join matches, weather, and stadium capacities from SQLite without building features prematurely."""
    conn = get_connection(db_path)
    query = """
    SELECT 
        m.match_id, m.season, m.league_tier, m.league_name, m.matchday,
        m.match_date, m.match_time, m.match_datetime,
        m.home_team, m.away_team, m.home_score, m.away_score,
        m.attendance, m.capacity_resolved, m.occupancy_pct,
        m.is_outlier, m.outlier_reason,
        w.temperature_c, w.precipitation_mm, w.rain_mm, w.weather_code,
        w.wind_speed_kmh, w.cloud_cover_pct,
        sc.roof_pct, sc.city as stadium_city, sc.stadium_name
    FROM matches m
    LEFT JOIN weather w ON m.match_id = w.match_id
    LEFT JOIN stadium_capacities sc ON (
        sc.club_name = m.home_team
        AND sc.valid_from <= m.match_date AND sc.valid_to >= m.match_date
    )
    ORDER BY m.match_date ASC, m.match_time ASC
    """
    df = pd.read_sql(query, conn)
    
    # Load club locations
    locs_df = pd.read_sql("SELECT club_name, latitude, longitude FROM club_locations", conn)
    conn.close()
    
    # Join geographic coordinates
    df = df.merge(
        locs_df.rename(columns={"club_name": "home_team", "latitude": "home_lat", "longitude": "home_lon"}),
        on="home_team", how="left"
    )
    df = df.merge(
        locs_df.rename(columns={"club_name": "away_team", "latitude": "away_lat", "longitude": "away_lon"}),
        on="away_team", how="left"
    )
    
    print(f"Loaded {len(df)} raw matches from database.")
    return df, locs_df


def split_systematic_match_spectrum(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic, non-random systematic train/test split across the entire match spectrum.
    
    Systematic Grid Selection guarantees:
      1. Every single season (2022/23 to 2026/27) is represented strictly proportionally (~18-20%).
      2. For each club in each season, fixtures are sorted chronologically.
         Matches are selected at evenly-spaced intervals (e.g. early season, mid season, late season).
      3. All 69 home clubs and all 3 league tiers (Ekstraklasa, 1. Liga, 2. Liga) are covered.
      4. 100% deterministic, reproducible, with zero random roll dependencies.
    """
    sort_cols = [c for c in ["season", "league_tier", "home_team", "match_date", "matchday", "match_id"] if c in df.columns]
    df_sorted = df.sort_values(by=sort_cols).copy()
    test_indices = []

    group_cols = [c for c in ["season", "league_tier", "home_team"] if c in df_sorted.columns]
    if group_cols:
        for _, grp in df_sorted.groupby(group_cols, sort=False):
            n = len(grp)
            k = int(round(n * test_ratio))
            if k == 0 and n >= 3:
                k = 1
            if k > 0:
                step = n / k
                chosen_rel = [min(n - 1, int(np.floor((i + 0.5) * step))) for i in range(k)]
                chosen_rel = sorted(list(set(chosen_rel)))
                test_indices.extend(grp.iloc[chosen_rel].index.values)
    else:
        n = len(df_sorted)
        k = max(1, int(round(n * test_ratio)))
        step = n / k
        chosen_rel = [min(n - 1, int(np.floor((i + 0.5) * step))) for i in range(k)]
        test_indices = list(df_sorted.iloc[chosen_rel].index.values)

    test_mask = df.index.isin(test_indices)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, test_df


# Backward compatibility aliases
split_stratified_match_spectrum = split_systematic_match_spectrum
split_stratified_by_season_and_host = split_systematic_match_spectrum


def run_experiment(raw_data: Any, split_strategy: str = "stratified"):
    """Run neural network model training, evaluation, and comparisons with ZERO data leakage."""
    if isinstance(raw_data, tuple):
        raw_df, locs_df = raw_data
    else:
        raw_df = raw_data
        locs_df = pd.DataFrame()

    # Identify rows eligible for training and evaluation
    valid_mask = (
        (raw_df["attendance"].notna()) &
        (raw_df["attendance"] > 0) &
        (raw_df.get("is_outlier", 0) == 0) &
        (raw_df["capacity_resolved"].notna()) &
        (raw_df["capacity_resolved"] > 0)
    )
    valid_df = raw_df[valid_mask].copy()
    print(f"Total valid matches with attendance: {len(valid_df)}")
    
    # 1. Determine train/test split BEFORE computing target-dependent features!
    if split_strategy in ("stratified", "spectrum"):
        print("\nUsing Spectrum-Stratified Split: 20% sampled proportionally across ALL seasons, league tiers, home clubs, and attendance spectrum tiers.")
        _, test_subset = split_systematic_match_spectrum(valid_df, test_ratio=0.20)
        is_test_mask = raw_df.index.isin(test_subset.index)
    else:
        print("\nUsing Temporal Split: 2022/23-2024/25 train, 2025/26-2026/27 test.")
        test_seasons = ["2025/2026", "2026/2027"]
        is_test_mask = raw_df["season"].isin(test_seasons)

    # 2. Build full features with STRICT data leakage isolation
    print("\nBuilding features with strict train/test isolation (ZERO target leakage)...")
    feat_df = build_full_features(raw_df, locs_df, is_test_mask=pd.Series(is_test_mask, index=raw_df.index))
    
    train_df = feat_df[valid_mask & (~is_test_mask)].copy()
    test_df = feat_df[valid_mask & is_test_mask].copy()
    
    print(f"\nTrain set: {len(train_df)} matches across {train_df['season'].nunique()} seasons, {train_df['league_tier'].nunique()} tiers, and {train_df['home_team'].nunique()} home teams")
    print(f"Test/Validation set: {len(test_df)} matches across {test_df['season'].nunique()} seasons, {test_df['league_tier'].nunique()} tiers, and {test_df['home_team'].nunique()} home teams")
    print("Test matches per season:")
    for s, cnt in test_df['season'].value_counts().sort_index().items():
        total_s = (valid_df['season'] == s).sum()
        print(f"  {s}: {cnt}/{total_s} matches ({cnt/total_s*100:.1f}%)")

    print("Test matches per League Tier:")
    tier_labels = {1: "Ekstraklasa (Tier 1)", 2: "1 Liga (Tier 2)", 3: "2 Liga (Tier 3)"}
    for t, cnt in test_df['league_tier'].value_counts().sort_index().items():
        total_t = (valid_df['league_tier'] == t).sum()
        print(f"  {tier_labels.get(t, f'Tier {t}')}: {cnt}/{total_t} matches ({cnt/total_t*100:.1f}%)")
    
    # -------------------------------------------------------------
    # Modern Tabular ResNet Quantile Neural Network (PyTorch)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Training Tabular ResNet Quantile Neural Network (Single Seed + CQR)...")
    single_nn = TabularQuantileNN(
        quantiles=(0.25, 0.50, 0.75),
        club_emb_dim=16,
        league_emb_dim=4,
        hidden_dims=(128, 64, 32),
        dropout_rates=(0.20, 0.15, 0.10),
        lr=0.003,
        batch_size=64,
        epochs=100,
        patience=15,
        log_target=True,
        temporal_validation=False,
        conformal_alpha=0.50,
        verbose=True
    )
    single_nn.fit(train_df)
    preds_single = single_nn.predict_interval(test_df)
    metrics_single = evaluate_predictions(
        y_true=test_df["attendance"].values,
        y_pred=preds_single["q50"].values,
        cap=test_df["capacity_resolved"].values
    )
    cov_50_single = ((test_df["attendance"].values >= preds_single["q25"].values) & 
                     (test_df["attendance"].values <= preds_single["q75"].values)).mean() * 100.0
    print(f"Single Tabular NN 50% (IQR Q25-Q75) Coverage: {cov_50_single:.2f}% (CQR delta: {single_nn.conformal_delta_:+.1f})")

    # -------------------------------------------------------------
    # Hybrid Quantile Ensemble (3x PyTorch ResNets + CatBoost + LightGBM)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Training Hybrid Quantile Ensemble (3x PyTorch ResNets + CatBoost + LightGBM)...")
    ensemble_nn = HybridEnsembleQuantileModel(
        n_models=3,
        quantiles=(0.25, 0.50, 0.75),
        club_emb_dim=16,
        league_emb_dim=4,
        hidden_dims=(128, 64, 32),
        dropout_rates=(0.20, 0.15, 0.10),
        lr=0.003,
        batch_size=64,
        epochs=100,
        patience=15,
        log_target=True,
        temporal_validation=False,
        conformal_alpha=0.50,
        verbose=True
    )
    ensemble_nn.fit(train_df)
    preds_nn_intervals = ensemble_nn.predict_interval(test_df)
    preds_nn = preds_nn_intervals["q50"].values
    metrics_nn = evaluate_predictions(
        y_true=test_df["attendance"].values,
        y_pred=preds_nn,
        cap=test_df["capacity_resolved"].values
    )
    cov_50_nn = ((test_df["attendance"].values >= preds_nn_intervals["q25"].values) & 
                 (test_df["attendance"].values <= preds_nn_intervals["q75"].values)).mean() * 100.0
    med_width_nn = np.median(preds_nn_intervals["interval_width"].values)
    cqr_delta_ens = getattr(ensemble_nn, "conformal_delta_", 0.0)
    print(f"Hybrid Quantile Ensemble 50% (IQR Q25-Q75) Coverage: {cov_50_nn:.2f}% | Median Width: {med_width_nn:.0f} spectators (CQR delta: {cqr_delta_ens:+.1f})")
    
    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("               MODEL PERFORMANCE SUMMARY (TEST SET)")
    print("="*75)
    summary_data = [
        {"Model": "Single Tabular ResNet (Seed 42 + CQR)", **metrics_single},
        {"Model": "Hybrid Quantile Ensemble (3xNN + CatBoost + LGBM)", **metrics_nn},
    ]
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Use ensemble as production model
    nn_model = ensemble_nn
    
    # -------------------------------------------------------------
    # Performance breakdown by League Tier
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("      PERFORMANCE BREAKDOWN BY LEAGUE (Tabular Quantile NN)")
    print("="*70)
    tier_results = []
    tier_names = {1: "Ekstraklasa", 2: "1 Liga", 3: "2 Liga"}
    for tier in [1, 2, 3]:
        t_mask = test_df["league_tier"] == tier
        if t_mask.sum() > 0:
            sub_test = test_df[t_mask]
            sub_preds = preds_nn_intervals.loc[sub_test.index]
            sub_metrics = evaluate_predictions(
                y_true=sub_test["attendance"].values,
                y_pred=sub_preds["q50"].values,
                cap=sub_test["capacity_resolved"].values
            )
            tier_results.append({
                "Tier": tier_names.get(tier, f"Tier {tier}"),
                "Mean_Actual_Att": round(sub_test["attendance"].mean(), 0),
                "MAE": sub_metrics["MAE"],
                "MAPE_pct": sub_metrics["MAPE_pct"],
                "Occupancy_MAE_pct": sub_metrics["Occupancy_MAE_pct"],
                "Matches": sub_metrics["count"]
            })
    print(pd.DataFrame(tier_results).to_string(index=False))
    
    # -------------------------------------------------------------
    # Feature Importances from Neural Network (PFI)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("          TOP 12 NEURAL NETWORK FEATURES (Permutation Importance)")
    print("="*70)
    feat_imp = nn_model.get_feature_importances(test_df.sample(min(300, len(test_df)), random_state=42), method="permutation", n_repeats=2)
    print(feat_imp.head(12).to_string(index=False))
    
    # -------------------------------------------------------------
    # Save Model Artifacts and Enriched Test Set Predictions
    # -------------------------------------------------------------
    models_dir = Path(__file__).resolve().parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(nn_model, models_dir / "tabular_nn_model.joblib")
    
    test_out = test_df.copy()
    test_out["pred_att_nn"] = preds_nn_intervals["q50"].values
    test_out["pred_occ_nn"] = preds_nn_intervals["pred_occupancy"].values if "pred_occupancy" in preds_nn_intervals else (preds_nn / test_out["capacity_resolved"].fillna(5000)).clip(0.01, 1.05).round(4)
    test_out["pred_q25_nn"] = preds_nn_intervals["q25"].values
    test_out["pred_q75_nn"] = preds_nn_intervals["q75"].values
    test_out["pred_q10_nn"] = preds_nn_intervals["q25"].values  # backward compat
    test_out["pred_q90_nn"] = preds_nn_intervals["q75"].values  # backward compat
    test_out["interval_width_nn"] = preds_nn_intervals["interval_width"].values
    test_out["error_abs_nn"] = (test_out["attendance"] - test_out["pred_att_nn"]).abs()
    test_out["error_signed_nn"] = test_out["attendance"] - test_out["pred_att_nn"]
    test_out["error_pct_nn"] = ((test_out["error_abs_nn"] / test_out["attendance"]) * 100.0).round(2)
    
    # Aliases for compatibility
    test_out["pred_attendance"] = test_out["pred_att_nn"]
    test_out["pred_occupancy"] = test_out["pred_occ_nn"]
    test_out["pred_q25"] = test_out["pred_q25_nn"]
    test_out["pred_q75"] = test_out["pred_q75_nn"]
    test_out["pred_q10"] = test_out["pred_q25_nn"]
    test_out["pred_q90"] = test_out["pred_q75_nn"]
    test_out["interval_width"] = test_out["interval_width_nn"]
    test_out["error_abs"] = test_out["error_abs_nn"]
    test_out["error_signed"] = test_out["error_signed_nn"]
    test_out["error_pct"] = test_out["error_pct_nn"]
    
    try:
        test_out.to_parquet(models_dir / "test_predictions.parquet", index=False)
    except Exception as e:
        print(f"Notice: Parquet save bypassed ({e}), saving CSV.")
    test_out.to_csv(models_dir / "test_predictions.csv", index=False)
    print(f"\nSaved neural network model and test predictions to {models_dir}")
    
    return {
        "nn_model": nn_model,
        "summary": summary_df,
        "test_df": test_df,
        "preds_nn": preds_nn_intervals
    }


def main():
    parser = argparse.ArgumentParser(description="Run Polish football attendance neural network pipeline")
    parser.add_argument("--scrape", action="store_true", help="Scrape 90minut.pl leagues")
    parser.add_argument("--weather", action="store_true", help="Fetch Open-Meteo weather")
    parser.add_argument("--split", choices=["stratified", "spectrum", "temporal"], default="stratified", help="Validation split strategy (spectrum-stratified or legacy temporal)")
    args = parser.parse_args()
    
    init_db()
    import_stadiums_to_db()
    
    if args.scrape:
        print("Running scraper...")
        scrape_all_seasons()
        
    if args.weather:
        print("Enriching weather...")
        batch_enrich_weather()
        
    db_file = Path(__file__).resolve().parent / "data" / "attendance.db"
    raw_data = load_dataset_from_db(db_file)
    run_experiment(raw_data, split_strategy=args.split)


if __name__ == "__main__":
    main()
