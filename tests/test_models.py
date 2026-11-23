"""Tests for neural network model training, predictions, and evaluation metrics."""
import numpy as np
import pandas as pd
import pytest

from src.models.neural_net import TabularQuantileNN, NeuralEnsembleQuantileModel, evaluate_predictions


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 120
    clubs = ["Legia Warszawa", "Lech Poznań", "Widzew Łódź", "Warta Poznań"]
    df = pd.DataFrame({
        "match_id": list(range(n)),
        "home_team": np.random.choice(clubs, size=n),
        "away_team": np.random.choice(clubs, size=n),
        "league_tier": np.random.choice([1, 2], size=n),
        "season_progress": np.random.uniform(0.1, 0.9, size=n),
        "dow": np.random.choice([4, 5, 6], size=n),
        "hour_num": np.random.choice([15.0, 17.5, 20.0], size=n),
        "sin_hour": np.random.uniform(-1, 1, size=n),
        "cos_hour": np.random.uniform(-1, 1, size=n),
        "sin_dow": np.random.uniform(-1, 1, size=n),
        "cos_dow": np.random.uniform(-1, 1, size=n),
        "is_weekend": np.random.choice([0, 1], size=n),
        "is_friday": np.random.choice([0, 1], size=n),
        "is_holiday_proximate": np.random.choice([0, 1], size=n),
        "temperature_c": np.random.uniform(2.0, 26.0, size=n),
        "precipitation_mm": np.random.exponential(0.5, size=n),
        "rain_exposure_mm": np.random.exponential(0.2, size=n),
        "temp_comfort_dev": np.random.uniform(0, 15, size=n),
        "travel_distance_km": np.random.uniform(20, 450, size=n),
        "is_local_derby": np.random.choice([0, 1], size=n),
        "home_pts_last_5": np.random.randint(0, 15, size=n),
        "away_pts_last_5": np.random.randint(0, 15, size=n),
        "home_pts_last_3_home": np.random.randint(0, 9, size=n),
        "away_pts_last_3_away": np.random.randint(0, 9, size=n),
        "home_table_pos": np.random.randint(1, 18, size=n),
        "away_table_pos": np.random.randint(1, 18, size=n),
        "table_pos_diff": np.random.randint(-15, 15, size=n),
        "table_pts_diff": np.random.randint(-20, 20, size=n),
        "away_draw_factor": np.random.uniform(0.8, 1.6, size=n),
        "capacity_resolved": np.random.choice([15000, 31000, 42000], size=n),
        "roof_pct": np.random.choice([0.7, 1.0], size=n),
        "attendance": np.random.randint(2000, 30000, size=n),
        "occupancy_pct": np.random.uniform(0.2, 0.95, size=n),
    })
    return df


def test_evaluate_predictions():
    y_true = np.array([1000, 2000, 3000])
    y_pred = np.array([1100, 1900, 3100])
    cap = np.array([5000, 5000, 5000])
    metrics = evaluate_predictions(y_true, y_pred, cap)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE_pct" in metrics
    assert "R2" in metrics
    assert "Occupancy_MAE_pct" in metrics
    assert metrics["MAE"] == 100.0


def test_tabular_quantile_nn_model(sample_dataset):
    train_df = sample_dataset.iloc[:90]
    test_df = sample_dataset.iloc[90:]

    nn_model = TabularQuantileNN(epochs=5, batch_size=32, verbose=False)
    nn_model.fit(train_df)

    preds = nn_model.predict_interval(test_df)
    assert "q10" in preds.columns
    assert "q50" in preds.columns
    assert "q90" in preds.columns
    assert "interval_width" in preds.columns
    assert len(preds) == len(test_df)

    # Non-crossing quantiles
    assert (preds["q10"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q90"]).all()

    # Point predict returns q50
    p50 = nn_model.predict(test_df)
    assert len(p50) == len(test_df)
    assert (p50 == preds["q50"].values).all()


def test_tabular_nn_feature_importances(sample_dataset):
    train_df = sample_dataset.iloc[:90]
    test_df = sample_dataset.iloc[90:]

    nn_model = TabularQuantileNN(epochs=5, batch_size=32, verbose=False)
    nn_model.fit(train_df)

    # Weight method
    imps_w = nn_model.get_feature_importances(method="weights")
    assert "feature" in imps_w.columns
    assert "importance" in imps_w.columns
    assert len(imps_w) > 0
    assert "capacity_resolved" not in imps_w["feature"].values

    # Permutation method
    imps_p = nn_model.get_feature_importances(test_df, method="permutation", n_repeats=2)
    assert "feature" in imps_p.columns
    assert "importance" in imps_p.columns
    assert len(imps_p) > 0
    assert "capacity_resolved" not in imps_p["feature"].values


def test_neural_ensemble_quantile_model(sample_dataset):
    train_df = sample_dataset.iloc[:90]
    test_df = sample_dataset.iloc[90:]

    ens = NeuralEnsembleQuantileModel(n_models=3, epochs=4, batch_size=32, verbose=False)
    ens.fit(train_df)

    assert len(ens.models_) == 3

    preds = ens.predict_interval(test_df)
    assert "q10" in preds.columns
    assert "q50" in preds.columns
    assert "q90" in preds.columns
    assert "interval_width" in preds.columns
    assert len(preds) == len(test_df)

    # Monotonicity
    assert (preds["q10"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q90"]).all()

    # Point predictions match Q50
    p50 = ens.predict(test_df)
    assert (p50 == preds["q50"].values).all()

    # Permutation feature importances
    imps = ens.get_feature_importances(test_df, method="permutation", n_repeats=1)
    assert "feature" in imps.columns
    assert len(imps) > 0


def test_split_stratified_match_spectrum():
    from run_pipeline import split_stratified_match_spectrum, split_systematic_match_spectrum
    np.random.seed(42)
    n = 200
    seasons = ["2022/2023", "2023/2024", "2024/2025", "2025/2026", "2026/2027"]
    tiers = [1, 2, 3]
    clubs = ["Legia", "Lech", "Widzew", "Wisła", "Pogoń", "Górnik"]
    
    df = pd.DataFrame({
        "match_id": list(range(n)),
        "season": np.random.choice(seasons, size=n),
        "league_tier": np.random.choice(tiers, size=n),
        "home_team": np.random.choice(clubs, size=n),
        "attendance": np.random.exponential(5000, size=n) + 200,
        "capacity_resolved": np.random.choice([10000, 20000, 30000], size=n)
    })
    
    train_df, test_df = split_stratified_match_spectrum(df, test_ratio=0.20, n_att_bins=3, random_state=42)
    
    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) > 0
    test_ratio_actual = len(test_df) / len(df)
    assert 0.15 <= test_ratio_actual <= 0.25
    
    # All seasons represented in test
    for s in seasons:
        assert (test_df["season"] == s).sum() > 0
        
    # All tiers represented in test
    for t in tiers:
        assert (test_df["league_tier"] == t).sum() > 0

    # 100% Deterministic: two calls produce exact same test indices with zero randomness
    tr2, te2 = split_systematic_match_spectrum(df, test_ratio=0.20)
    assert list(test_df.index) == list(te2.index)


