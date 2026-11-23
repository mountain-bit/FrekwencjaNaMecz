"""Tests for feature engineering calculations and no data leakage."""
import math
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    haversine_distance,
    add_calendar_features,
    calculate_form_and_standings,
    add_weather_interactions,
    add_reserve_features,
    calculate_bidirectional_oof_features,
)


def test_haversine_distance():
    # Warsaw (52.23, 21.01) to Kraków (50.06, 19.94) is ~250-300 km
    d = haversine_distance(52.23, 21.01, 50.06, 19.94)
    assert 240 < d < 300


def test_add_calendar_features():
    df = pd.DataFrame({
        "match_datetime": ["2023-09-15T20:30:00", "2023-09-17T12:30:00"],
        "match_time": ["20:30", "12:30"],
        "matchday": [8, 8]
    })
    res = add_calendar_features(df)
    assert "sin_hour" in res.columns
    assert "cos_hour" in res.columns
    assert "is_weekend" in res.columns
    assert res.loc[0, "is_friday"] == 1
    assert res.loc[1, "is_weekend"] == 1
    assert -1.0 <= res.loc[0, "sin_hour"] <= 1.0


def test_form_and_standings_no_leakage():
    # Construct 3 sequential matches for Team A and Team B
    # Match 1: Team A 2-0 Team B
    # Match 2: Team A 1-1 Team C
    # Match 3: Team B 3-0 Team A
    df = pd.DataFrame({
        "match_id": [1, 2, 3],
        "match_date": ["2023-08-01", "2023-08-08", "2023-08-15"],
        "match_time": ["17:30", "17:30", "17:30"],
        "season": ["2023/2024", "2023/2024", "2023/2024"],
        "league_tier": [1, 1, 1],
        "home_team": ["Team A", "Team A", "Team B"],
        "away_team": ["Team B", "Team C", "Team A"],
        "home_score": [2, 1, 3],
        "away_score": [0, 1, 0],
        "attendance": [10000, 12000, 8000]
    })
    
    res = calculate_form_and_standings(df)
    
    # On match 1 (first game of season), form and points prior to game must be 0!
    assert res.loc[0, "home_pts_last_5"] == 0
    assert res.loc[0, "away_pts_last_5"] == 0
    assert res.loc[0, "home_table_pts"] == 0
    assert res.loc[0, "away_table_pts"] == 0
    
    # On match 2, Team A won match 1, so home_pts_last_5 should be 3!
    assert res.loc[1, "home_pts_last_5"] == 3
    assert res.loc[1, "home_table_pts"] == 3
    
    # On match 3:
    # Team B lost match 1 (0 pts)
    # Team A won match 1 (3 pts) and drew match 2 (1 pt) -> 4 pts!
    assert res.loc[2, "home_table_pts"] == 0  # Team B is home
    assert res.loc[2, "away_table_pts"] == 4  # Team A is away
    assert res.loc[2, "away_pts_last_5"] == 4


def test_weather_interactions():
    df = pd.DataFrame({
        "temperature_c": [10.0, 20.0],
        "precipitation_mm": [2.0, 5.0],
        "roof_pct": [1.0, 0.4]  # 100% roof vs 40% roof
    })
    res = add_weather_interactions(df)
    # With 100% roof, rain exposure should be 0.0
    assert res.loc[0, "rain_exposure_mm"] == 0.0
    # With 40% roof, 60% of rain hits fans: 5.0 * 0.6 = 3.0
    assert pytest.approx(res.loc[1, "rain_exposure_mm"]) == 3.0


def test_add_reserve_features():
    df = pd.DataFrame({
        "home_team": ["ŁKS Łódź", "ŁKS II Łódź", "Hutnik Kraków", "Lech II Poznań"],
        "away_team": ["Widzew Łódź", "Pogoń Siedlce", "Zagłębie II Lubin", "Kotwica Kołobrzeg"],
        "is_major_derby": [1, 1, 0, 0],
        "is_local_derby": [1, 1, 0, 0]
    })
    res = add_reserve_features(df)
    
    # First match: first teams -> not reserve match, derby remains 1
    assert res.loc[0, "is_home_reserve"] == 0
    assert res.loc[0, "is_away_reserve"] == 0
    assert res.loc[0, "is_reserve_fixture"] == 0
    assert res.loc[0, "is_major_derby"] == 1
    assert res.loc[0, "is_local_derby"] == 1
    
    # Second match: ŁKS II home -> is_home_reserve=1, derbies disabled
    assert res.loc[1, "is_home_reserve"] == 1
    assert res.loc[1, "is_away_reserve"] == 0
    assert res.loc[1, "is_reserve_fixture"] == 1
    assert res.loc[1, "is_major_derby"] == 0
    assert res.loc[1, "is_local_derby"] == 0
    
    # Third match: Zagłębie II away -> is_away_reserve=1, is_reserve_fixture=1
    assert res.loc[2, "is_home_reserve"] == 0
    assert res.loc[2, "is_away_reserve"] == 1
    assert res.loc[2, "is_reserve_fixture"] == 1


def test_bidirectional_oof_features_no_leakage():
    # 2 matches between Lech and Legia in different seasons
    # Match 1: 2022-10-01 (Attendance: 35,000)
    # Match 2: 2023-04-16 (Attendance: 39,000)
    # Match 3: 2023-05-01 (Radomiak vs Warta, Attendance: 4,000, sole encounter)
    df = pd.DataFrame({
        "match_id": [1, 2, 3],
        "season": ["2022/2023", "2022/2023", "2022/2023"],
        "home_team": ["Lech Poznań", "Legia Warszawa", "Radomiak Radom"],
        "away_team": ["Legia Warszawa", "Lech Poznań", "Warta Poznań"],
        "attendance": [35000.0, 39000.0, 4000.0],
        "capacity_resolved": [42837.0, 31103.0, 8500.0],
        "is_major_derby": [1, 1, 0],
        "is_local_derby": [0, 0, 0],
        "is_outlier": [0, 0, 0]
    })
    
    res = calculate_bidirectional_oof_features(df)
    
    # 1. Match 1 (early match) MUST see Match 2's attendance (39,000) bidirectionally!
    # And its own attendance (35,000) MUST NOT leak into its own feature.
    assert res.loc[0, "h2h_avg_att"] == 39000.0
    
    # 2. Match 2 MUST see Match 1's attendance (35,000), not its own 39,000!
    assert res.loc[1, "h2h_avg_att"] == 35000.0
    
    # 3. Match 3 is a sole encounter -> other_pair_cnt == 0, so it must fall back to prior!
    # Non-derby prior is 0.40 * 8500 = 3400.0
    assert pytest.approx(res.loc[2, "h2h_avg_att"]) == 3400.0


def test_round_opener_and_weather_discomfort():
    df_cal = pd.DataFrame({
        "match_datetime": ["2023-07-22T17:30:00", "2024-02-10T15:00:00", "2023-10-15T15:00:00"],
        "match_time": ["17:30", "15:00", "15:00"],
        "matchday": [1, 20, 11]
    })
    res_cal = add_calendar_features(df_cal)
    assert res_cal.loc[0, "is_round_opener"] == 1  # Summer season opener
    assert res_cal.loc[1, "is_round_opener"] == 1  # Winter break return opener
    assert res_cal.loc[2, "is_round_opener"] == 0  # Regular mid-season match

    df_weather = pd.DataFrame({
        "temperature_c": [5.0, 18.0],
        "precipitation_mm": [3.0, 0.0],
        "roof_pct": [0.5, 1.0]
    })
    res_weather = add_weather_interactions(df_weather)
    assert "effective_weather_discomfort" in res_weather.columns
    # Match 0: rain_exposure = 3.0 * 0.5 = 1.5; cold_penalty = (10 - 5) * 0.5 = 2.5; total = 4.0
    assert pytest.approx(res_weather.loc[0, "effective_weather_discomfort"]) == 4.0
    # Match 1: rain_exposure = 0; cold_penalty = 0; total = 0.0
    assert pytest.approx(res_weather.loc[1, "effective_weather_discomfort"]) == 0.0


def test_ema_and_stadium_expansion_boost():
    # 4 matches in 2 seasons for Team X
    df = pd.DataFrame({
        "match_id": [1, 2, 3, 4],
        "match_date": ["2022-09-01", "2022-09-15", "2022-10-01", "2023-08-01"],
        "match_time": ["18:00", "18:00", "18:00", "18:00"],
        "season": ["2022/2023", "2022/2023", "2022/2023", "2023/2024"],
        "league_tier": [1, 1, 1, 1],
        "home_team": ["Team X", "Team X", "Team X", "Team X"],
        "away_team": ["Team Y", "Team Z", "Team W", "Team Y"],
        "home_score": [1, 1, 1, 1],
        "away_score": [0, 0, 0, 0],
        "attendance": [5000, 7000, 9000, 18000],
        "capacity_resolved": [10000, 10000, 10000, 20000],
        "outlier_reason": ["", "", "sektor gości zamknięty", ""]
    })
    res = calculate_form_and_standings(df)
    assert "home_att_ema_3" in res.columns
    assert "home_occupancy_ema_5" in res.columns
    assert "stadium_expansion_boost" in res.columns
    assert "away_sector_closed" in res.columns

    # Match 3 has "sektor gości zamknięty"
    assert res.loc[2, "away_sector_closed"] == 1
    assert res.loc[0, "away_sector_closed"] == 0

    # Match 4: In 2023/24 capacity doubled to 20,000 (expansion boost > 1.0)
    assert res.loc[3, "stadium_expansion_boost"] >= 1.5
    # Prior to Match 4, EMA attendance should weight recent games more
    assert res.loc[3, "home_att_ema_3"] > 7000  # weights favor 9000



