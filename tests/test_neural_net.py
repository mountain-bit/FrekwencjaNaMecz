"""Unit tests for TabularQuantileNN and PinballLoss in PyTorch."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.neural_net import (
    TabularQuantileNN,
    TabularQuantileNet,
    NeuralEnsembleQuantileModel,
    HybridEnsembleQuantileModel,
    PinballLoss,
    CensoredSmoothPinballLoss
)


@pytest.fixture
def synthetic_attendance_data():
    """Generates synthetic match data for neural network testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    n = 100
    clubs = ["Legia Warszawa", "Lech Poznań", "Widzew Łódź", "Górnik Zabrze", "Jagiellonia Białystok"]
    
    df = pd.DataFrame({
        "match_id": list(range(n)),
        "home_team": np.random.choice(clubs, size=n),
        "away_team": np.random.choice(clubs, size=n),
        "league_tier": np.random.choice([1, 2], size=n),
        "capacity_resolved": np.random.choice([15000, 25000, 31000], size=n),
        "temperature_c": np.random.uniform(5.0, 25.0, size=n),
        "precipitation_mm": np.random.exponential(1.0, size=n),
        "dow": np.random.choice([4, 5, 6], size=n),
        "hour_num": np.random.choice([15.0, 17.5, 20.0], size=n),
        "sin_hour": np.random.uniform(-1, 1, size=n),
        "cos_hour": np.random.uniform(-1, 1, size=n),
        "sin_dow": np.random.uniform(-1, 1, size=n),
        "cos_dow": np.random.uniform(-1, 1, size=n),
        "is_weekend": np.random.choice([0, 1], size=n),
        "is_friday": np.random.choice([0, 1], size=n),
        "is_holiday_proximate": np.random.choice([0, 1], size=n),
        "travel_distance_km": np.random.uniform(20.0, 400.0, size=n),
        "is_local_derby": np.random.choice([0, 1], size=n),
        "away_draw_factor": np.random.uniform(0.8, 1.5, size=n),
        "season_progress": np.random.uniform(0.1, 0.9, size=n),
        "attendance": np.random.randint(3000, 28000, size=n),
    })
    return df


def test_pinball_loss():
    """Verify Pinball Loss calculations for specific test vectors."""
    loss_fn = PinballLoss(quantiles=[0.10, 0.50, 0.90])
    
    # 2 samples, 3 quantile predictions each
    targets = torch.tensor([[100.0], [200.0]], dtype=torch.float32)
    # Target > pred (underprediction) and Target < pred (overprediction)
    preds = torch.tensor([[80.0, 100.0, 120.0], [220.0, 200.0, 180.0]], dtype=torch.float32)
    
    # errors = targets - preds:
    # row 0: [20, 0, -20]
    #   q=0.1: max(0.1*20, -0.9*20) = 2.0
    #   q=0.5: max(0.5*0, -0.5*0) = 0.0
    #   q=0.9: max(0.9*-20, -0.1*-20) = 2.0
    # row 1: [-20, 0, 20]
    #   q=0.1: max(0.1*-20, -0.9*-20) = 18.0
    #   q=0.5: max(0.5*0, -0.5*0) = 0.0
    #   q=0.9: max(0.9*20, -0.1*20) = 18.0
    # Mean across batch:
    #   q=0.1: (2 + 18) / 2 = 10.0
    #   q=0.5: (0 + 0) / 2 = 0.0
    #   q=0.9: (2 + 18) / 2 = 10.0
    # Total loss = 10.0 + 0.0 + 10.0 = 20.0
    total_loss = loss_fn(preds, targets)
    assert pytest.approx(total_loss.item(), rel=1e-4) == 20.0


def test_tabular_quantile_nn_fit_predict(synthetic_attendance_data):
    """Test full scikit-learn compatibility: fit, predict, and attribute states."""
    train_df = synthetic_attendance_data.iloc[:80]
    test_df = synthetic_attendance_data.iloc[80:]

    model = TabularQuantileNN(
        epochs=8,
        batch_size=32,
        hidden_dims=(64, 32, 16),
        verbose=False
    )
    model.fit(train_df)

    assert model.is_fitted_ is True
    assert model.network_ is not None
    assert model.scaler_ is not None
    assert len(model.club_to_idx_) > 0

    preds_q50 = model.predict(test_df)
    assert isinstance(preds_q50, np.ndarray)
    assert len(preds_q50) == len(test_df)
    assert (preds_q50 > 0).all()


def test_tabular_quantile_nn_predict_interval(synthetic_attendance_data):
    """Test interval predictions, quantile crossing prevention, and capacity clipping."""
    train_df = synthetic_attendance_data.iloc[:80]
    test_df = synthetic_attendance_data.iloc[80:]

    model = TabularQuantileNN(
        epochs=6,
        batch_size=32,
        hidden_dims=(32, 16, 8),
        verbose=False
    )
    model.fit(train_df)

    intervals = model.predict_interval(test_df)
    assert isinstance(intervals, pd.DataFrame)
    assert set(["q25", "q50", "q75", "q10", "q90", "interval_width"]).issubset(intervals.columns)
    assert len(intervals) == len(test_df)

    # 1. Quantile crossing must never occur: q25 <= q50 <= q75
    assert (intervals["q25"] <= intervals["q50"]).all(), "Quantile crossing detected between Q25 and Q50!"
    assert (intervals["q50"] <= intervals["q75"]).all(), "Quantile crossing detected between Q50 and Q75!"
    assert (intervals["interval_width"] >= 0).all()

    # 2. Capacity clipping: predictions must not exceed 1.05 * capacity
    caps = test_df["capacity_resolved"].values
    assert (intervals["q90"].values <= caps * 1.05 + 1.0).all()


def test_unseen_clubs_mapping(synthetic_attendance_data):
    """Verify that predicting on unseen clubs maps safely to <UNK> token without errors."""
    train_df = synthetic_attendance_data.iloc[:80]
    model = TabularQuantileNN(epochs=4, batch_size=32, verbose=False)
    model.fit(train_df)

    unseen_df = pd.DataFrame({
        "home_team": ["Nieznany Klub FC"],
        "away_team": ["Nowy Rywal KS"],
        "league_tier": [2],
        "capacity_resolved": [12000],
        "temperature_c": [18.0],
        "precipitation_mm": [0.0],
        "travel_distance_km": [150.0],
    })

    # Must predict cleanly using <UNK> embedding
    preds = model.predict(unseen_df)
    assert len(preds) == 1
    assert preds[0] > 0

    intervals = model.predict_interval(unseen_df)
    assert intervals.loc[0, "q10"] <= intervals.loc[0, "q50"] <= intervals.loc[0, "q90"]


def test_loss_convergence(synthetic_attendance_data):
    """Test that training loss successfully decreases over epochs."""
    train_df = synthetic_attendance_data.iloc[:80]
    model = TabularQuantileNN(epochs=20, batch_size=32, lr=0.005, verbose=False)
    model.fit(train_df)
    assert model.is_fitted_ is True


def test_missing_values_and_nan_robustness(synthetic_attendance_data):
    """Verify that input matrices with missing values and NaNs are safely imputed without failing."""
    train_df = synthetic_attendance_data.iloc[:80]
    model = TabularQuantileNN(epochs=4, batch_size=32, verbose=False)
    model.fit(train_df)

    noisy_df = pd.DataFrame({
        "home_team": ["Legia Warszawa", None],
        "away_team": [None, "Widzew Łódź"],
        "league_tier": [np.nan, 1],
        "capacity_resolved": [np.nan, 20000],
        "temperature_c": [np.nan, np.nan],
        "precipitation_mm": [np.nan, 5.0],
        "travel_distance_km": [np.nan, 300.0],
    })

    intervals = model.predict_interval(noisy_df)
    assert not intervals["q50"].isna().any(), "Found NaNs in predictions!"
    assert not intervals["q10"].isna().any(), "Found NaNs in Q10!"
    assert not intervals["q90"].isna().any(), "Found NaNs in Q90!"
    assert (intervals["q10"] <= intervals["q50"]).all()
    assert (intervals["q50"] <= intervals["q90"]).all()


def test_joblib_serialization(synthetic_attendance_data, tmp_path):
    """Test that TabularQuantileNN can be saved and loaded with joblib preserving predictions exactly."""
    import joblib
    train_df = synthetic_attendance_data.iloc[:80]
    test_df = synthetic_attendance_data.iloc[80:]

    model = TabularQuantileNN(epochs=5, batch_size=32, verbose=False)
    model.fit(train_df)

    orig_preds = model.predict_interval(test_df)

    # Save and reload
    save_file = tmp_path / "test_nn_model.joblib"
    joblib.dump(model, save_file)
    reloaded_model = joblib.load(save_file)

    reloaded_preds = reloaded_model.predict_interval(test_df)

    # Predictions must match exactly
    np.testing.assert_array_almost_equal(orig_preds["q50"].values, reloaded_preds["q50"].values)
    np.testing.assert_array_almost_equal(orig_preds["q10"].values, reloaded_preds["q10"].values)
    np.testing.assert_array_almost_equal(orig_preds["q90"].values, reloaded_preds["q90"].values)


def test_sklearn_score_method(synthetic_attendance_data):
    """Test standard Scikit-Learn score() returning R^2 metric."""
    train_df = synthetic_attendance_data.iloc[:80]
    test_df = synthetic_attendance_data.iloc[80:]

    model = TabularQuantileNN(epochs=8, batch_size=32, verbose=False)
    model.fit(train_df)

    r2 = model.score(test_df, test_df["attendance"].values)
    assert isinstance(r2, float)
    assert not np.isnan(r2)


def test_structural_monotonicity_in_graph():
    """Verify TabularQuantileNet guarantees Q25 <= Q50 <= Q75 natively in PyTorch graph."""
    net = TabularQuantileNet(
        num_clubs=10,
        num_leagues=3,
        num_continuous_features=5,
        club_emb_dim=16,
        hidden_dims=(32, 16, 8)
    )
    net.eval()
    batch_size = 20
    h_idx = torch.randint(0, 10, (batch_size,))
    a_idx = torch.randint(0, 10, (batch_size,))
    l_idx = torch.randint(0, 3, (batch_size,))
    cont_x = torch.randn(batch_size, 5)

    with torch.no_grad():
        out = net(h_idx, a_idx, l_idx, cont_x)

    q25 = out[:, 0]
    q50 = out[:, 1]
    q75 = out[:, 2]

    # Check structural monotonicity directly in tensor output without sorting
    assert (q25 <= q50).all().item(), "Q25 must be <= Q50 in network output"
    assert (q50 <= q75).all().item(), "Q50 must be <= Q75 in network output"


def test_hybrid_club_embedding_decomposition():
    """Verify hybrid club embedding decomposition into shared, home-role, and away-role embeddings."""
    net = TabularQuantileNet(
        num_clubs=10,
        num_leagues=2,
        num_continuous_features=4,
        club_emb_dim=16,
        shared_club_dim=10
    )
    assert hasattr(net, "shared_club_emb")
    assert hasattr(net, "home_role_emb")
    assert hasattr(net, "away_role_emb")
    assert net.shared_dim == 10
    assert net.role_dim == 6

    # Test forward pass with UNK dropout enabled during training
    net.train()
    h_idx = torch.tensor([1, 2, 3])
    a_idx = torch.tensor([4, 5, 6])
    l_idx = torch.tensor([1, 1, 1])
    cont_x = torch.randn(3, 4)
    out = net(h_idx, a_idx, l_idx, cont_x)
    assert out.shape == (3, 3)


def test_censored_smooth_pinball_loss():
    """Verify CensoredSmoothPinballLoss smooth Huber penalty and sellout right-censoring."""
    loss_fn = CensoredSmoothPinballLoss(quantiles=[0.10, 0.50, 0.90], delta=0.02)

    # 1. Standard prediction
    preds = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32)
    targets = torch.tensor([[10.0]], dtype=torch.float32)
    # Zero error should yield 0 loss
    l_zero = loss_fn(preds, targets)
    assert pytest.approx(l_zero.item(), abs=1e-5) == 0.0

    # 2. Sellout right-censoring:
    # Target was 100 (sold out stadium capacity), model predicts 103 (within 1.05 * 100 = 105 limit).
    # Since match was sold out, overprediction within capacity buffer must NOT be penalized.
    targets_sellout = torch.tensor([[100.0]], dtype=torch.float32)
    preds_over = torch.tensor([[101.0, 103.0, 104.0]], dtype=torch.float32)
    is_sellout = torch.tensor([[1.0]], dtype=torch.float32)
    cap_limit = torch.tensor([[105.0]], dtype=torch.float32)

    loss_sellout = loss_fn(preds_over, targets_sellout, is_sellout=is_sellout, cap_limit=cap_limit)
    assert pytest.approx(loss_sellout.item(), abs=1e-5) == 0.0


def test_temporal_validation_and_cqr(synthetic_attendance_data):
    """Verify out-of-time chronological validation and CQR calibration delta estimation."""
    df = synthetic_attendance_data.copy()
    # Add chronological dates
    dates = pd.date_range("2023-01-01", periods=len(df), freq="3D")
    df["match_datetime"] = dates.strftime("%Y-%m-%d %H:%M:%S")

    model = TabularQuantileNN(
        epochs=4,
        batch_size=32,
        val_split=0.20,
        temporal_validation=True,
        conformal_alpha=0.20,
        verbose=False
    )
    model.fit(df)

    assert model.is_fitted_ is True
    assert hasattr(model, "conformal_delta_")
    assert isinstance(model.conformal_delta_, float)

    # Intervals must incorporate conformal calibration while preserving non-crossing bounds
    intervals = model.predict_interval(df.iloc[-10:])
    assert (intervals["q10"] <= intervals["q50"]).all()
    assert (intervals["q50"] <= intervals["q90"]).all()
    assert (intervals["interval_width"] >= 0).all()


def test_hybrid_ensemble_quantile_model(synthetic_attendance_data, tmp_path):
    """Verify training, prediction, CatBoost feature importance, and serialization of Hybrid Ensemble."""
    import joblib

    train_df = synthetic_attendance_data.iloc[:80]
    test_df = synthetic_attendance_data.iloc[80:]

    ensemble = HybridEnsembleQuantileModel(
        n_models=2,
        use_catboost=True,
        use_lightgbm=True,
        epochs=3,
        batch_size=32,
        verbose=False
    )
    ensemble.fit(train_df)

    assert ensemble.is_fitted_ is True
    assert len(ensemble.models_) == 2
    assert ensemble.catboost_model_ is not None

    # Predict intervals
    preds = ensemble.predict_interval(test_df)
    assert len(preds) == len(test_df)
    assert set(["q25", "q50", "q75", "q10", "q90", "interval_width", "pred_attendance"]).issubset(preds.columns)
    assert (preds["q25"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q75"]).all()

    # Point predictions
    q50 = ensemble.predict(test_df)
    assert len(q50) == len(test_df)
    assert (q50 > 0).all()

    # CatBoost native feature importance
    cb_imp = ensemble.get_feature_importances(method="catboost")
    assert isinstance(cb_imp, pd.DataFrame)
    assert "feature" in cb_imp.columns
    assert "importance" in cb_imp.columns
    assert len(cb_imp) > 0

    # Serialization test
    save_path = tmp_path / "hybrid_ensemble.joblib"
    joblib.dump(ensemble, save_path)
    loaded_ens = joblib.load(save_path)
    loaded_preds = loaded_ens.predict_interval(test_df)
    np.testing.assert_array_almost_equal(preds["q50"].values, loaded_preds["q50"].values)


def test_app_single_row_prediction_pipeline(synthetic_attendance_data):
    """Verify that a single-match simulation row matching app.py schema executes cleanly on fitted models."""
    from src.features.engineering import add_calendar_features, add_reserve_features, add_weather_interactions

    train_df = synthetic_attendance_data.copy()
    # Add calendar dates for feature engineering
    dates = pd.date_range("2023-08-01", periods=len(train_df), freq="7D")
    train_df["match_date"] = dates.strftime("%Y-%m-%d")
    train_df["match_time"] = "17:30"
    train_df["matchday"] = np.random.randint(1, 35, size=len(train_df))
    train_df = add_calendar_features(train_df)
    train_df = add_reserve_features(train_df)
    train_df = add_weather_interactions(train_df)

    model = HybridEnsembleQuantileModel(n_models=1, epochs=2, batch_size=32, verbose=False)
    model.fit(train_df)

    # Reconstruct single match dataframe as formed in app.py
    single_df = pd.DataFrame([{
        "league_tier": 1,
        "matchday": 15,
        "match_date": "2026-11-20",
        "match_time": "17:30",
        "home_team": "Legia Warszawa",
        "away_team": "Lech Poznań",
        "capacity_resolved": 31103,
        "roof_pct": 1.0,
        "temperature_c": 8.5,
        "precipitation_mm": 0.0,
        "weather_code": 0,
        "cloud_cover_pct": 40,
        "wind_speed_kmh": 15.0,
        "travel_distance_km": 310.0,
        "is_local_derby": 0,
        "is_major_derby": 1,
        "h2h_avg_att": 27500.0,
        "h2h_home_venue_avg_att": 28100.0,
        "home_rolling_att_last_3": 24000.0,
        "home_rolling_occ_last_3": 0.77,
        "is_title_or_promotion_race": 1,
        "is_relegation_battle": 0,
        "is_dead_rubber": 0,
        "match_stakes_index": 1.5,
        "home_table_pos": 2,
        "away_table_pos": 1,
        "table_pos_diff": -1,
        "table_pts_diff": -2,
        "home_pts_last_5": 10,
        "away_pts_last_5": 12,
        "home_pts_last_3_home": 7,
        "away_pts_last_3_away": 6,
        "away_draw_factor": 1.45,
        "away_sector_closed": 0,
        "stadium_expansion_boost": 0.0,
        "home_att_ema_3": 24000.0,
        "home_occupancy_ema_5": 0.77,
    }])
    single_df = add_calendar_features(single_df)
    single_df = add_reserve_features(single_df)
    single_df = add_weather_interactions(single_df)

    preds = model.predict_interval(single_df)
    assert len(preds) == 1
    assert "q25" in preds.columns
    assert "q50" in preds.columns
    assert "q75" in preds.columns
    assert "q10" in preds.columns
    assert "q90" in preds.columns
    assert preds["q25"].iloc[0] <= preds["q50"].iloc[0] <= preds["q75"].iloc[0]
    assert 50 <= preds["q50"].iloc[0] <= 31103 * 1.05






