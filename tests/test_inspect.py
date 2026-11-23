import os, sys
from pathlib import Path
import pytest
import joblib
import pandas as pd
import numpy as np

def test_multiple_predictions():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    model_path = PROJECT_ROOT / "data" / "models" / "tabular_nn_model.joblib"
    model = joblib.load(model_path)

    from src.features.engineering import add_calendar_features, add_weather_interactions, add_reserve_features

    # Przypadek 1: Znicz Pruszków vs Kotwica Kołobrzeg (1. Liga, pojemność 2000, realna frekwencja ~800)
    # W app.py: h2h brak -> capacity * 0.45 = 900
    df_znicz = pd.DataFrame([{
        "league_tier": 2, "matchday": 8, "match_date": "2026-09-20", "match_time": "15:00",
        "match_datetime": "2026-09-20 15:00:00", "home_team": "Znicz Pruszków", "away_team": "Kotwica Kołobrzeg",
        "capacity_resolved": 1977, "roof_pct": 0.5, "temperature_c": 16.0, "precipitation_mm": 0.0,
        "weather_code": 0, "cloud_cover_pct": 20, "wind_speed_kmh": 10.0, "travel_distance_km": 500.0,
        "is_local_derby": 0, "is_major_derby": 0, "h2h_avg_att": 900.0, "h2h_home_venue_avg_att": 900.0,
        "home_rolling_att_last_3": 850.0, "home_rolling_occ_last_3": 850.0 / 1977.0,
        "is_title_or_promotion_race": 0, "is_relegation_battle": 0, "is_dead_rubber": 0,
        "match_stakes_index": 1.0, "home_table_pos": 11, "away_table_pos": 13,
        "table_pos_diff": 2, "table_pts_diff": 2, "home_pts_last_5": 5, "away_pts_last_5": 4,
        "home_pts_last_3_home": 3, "away_pts_last_3_away": 1, "away_draw_factor": 1.0,
        "away_sector_closed": 0, "stadium_expansion_boost": 0.0, "home_att_ema_3": 850.0,
        "home_occupancy_ema_5": 850.0 / 1977.0,
    }])
    df_znicz = add_weather_interactions(add_reserve_features(add_calendar_features(df_znicz)))
    p_znicz = model.predict_interval(df_znicz)
    print(f"\nZnicz vs Kotwica: Q25={p_znicz['q25'].iloc[0]:.0f}, Q50={p_znicz['q50'].iloc[0]:.0f}, Q75={p_znicz['q75'].iloc[0]:.0f}")

    # Przypadek 2: Śląsk Wrocław na stadionie 45k, ale gdy home_rolling_att_last_3 = 8000 (słaba frekwencja)
    # Sprawdźmy czy model zareaguje, czy zawyży do średniej klubu (17k)
    df_slask_slaby = pd.DataFrame([{
        "league_tier": 1, "matchday": 5, "match_date": "2026-08-15", "match_time": "12:30",
        "match_datetime": "2026-08-15 12:30:00", "home_team": "Śląsk Wrocław", "away_team": "Puszcza Niepołomice",
        "capacity_resolved": 45105, "roof_pct": 1.0, "temperature_c": 22.0, "precipitation_mm": 0.0,
        "weather_code": 0, "cloud_cover_pct": 10, "wind_speed_kmh": 10.0, "travel_distance_km": 300.0,
        "is_local_derby": 0, "is_major_derby": 0, "h2h_avg_att": 10000.0, "h2h_home_venue_avg_att": 10000.0,
        "home_rolling_att_last_3": 8000.0, "home_rolling_occ_last_3": 8000.0 / 45105.0,
        "is_title_or_promotion_race": 0, "is_relegation_battle": 0, "is_dead_rubber": 0,
        "match_stakes_index": 1.0, "home_table_pos": 15, "away_table_pos": 16,
        "table_pos_diff": 1, "table_pts_diff": 0, "home_pts_last_5": 2, "away_pts_last_5": 3,
        "home_pts_last_3_home": 1, "away_pts_last_3_away": 1, "away_draw_factor": 0.65,
        "away_sector_closed": 0, "stadium_expansion_boost": 0.0, "home_att_ema_3": 8000.0,
        "home_occupancy_ema_5": 8000.0 / 45105.0,
    }])
    df_slask_slaby = add_weather_interactions(add_reserve_features(add_calendar_features(df_slask_slaby)))
    p_slask = model.predict_interval(df_slask_slaby)
    print(f"Śląsk vs Puszcza (słaby okres): Q25={p_slask['q25'].iloc[0]:.0f}, Q50={p_slask['q50'].iloc[0]:.0f}, Q75={p_slask['q75'].iloc[0]:.0f}")

    # Przypadek 3: Mecz rezerw - Śląsk II Wrocław (pojemność 1000, frekwencja realna 150-250)
    df_res = pd.DataFrame([{
        "league_tier": 3, "matchday": 5, "match_date": "2026-08-15", "match_time": "12:00",
        "match_datetime": "2026-08-15 12:00:00", "home_team": "Śląsk II Wrocław", "away_team": "Zagłębie II Lubin",
        "capacity_resolved": 1100, "roof_pct": 0.5, "temperature_c": 20.0, "precipitation_mm": 0.0,
        "weather_code": 0, "cloud_cover_pct": 10, "wind_speed_kmh": 10.0, "travel_distance_km": 70.0,
        "is_local_derby": 0, "is_major_derby": 0, "h2h_avg_att": 300.0, "h2h_home_venue_avg_att": 300.0,
        "home_rolling_att_last_3": 250.0, "home_rolling_occ_last_3": 250.0 / 1100.0,
        "is_title_or_promotion_race": 0, "is_relegation_battle": 0, "is_dead_rubber": 0,
        "match_stakes_index": 1.0, "home_table_pos": 8, "away_table_pos": 9,
        "table_pos_diff": 1, "table_pts_diff": 0, "home_pts_last_5": 5, "away_pts_last_5": 5,
        "home_pts_last_3_home": 3, "away_pts_last_3_away": 2, "away_draw_factor": 1.0,
        "away_sector_closed": 0, "stadium_expansion_boost": 0.0, "home_att_ema_3": 250.0,
        "home_occupancy_ema_5": 250.0 / 1100.0,
    }])
    df_res = add_weather_interactions(add_reserve_features(add_calendar_features(df_res)))
    p_res = model.predict_interval(df_res)
    print(f"Śląsk II vs Zagłębie II (Rezerwy): Q25={p_res['q25'].iloc[0]:.0f}, Q50={p_res['q50'].iloc[0]:.0f}, Q75={p_res['q75'].iloc[0]:.0f}")