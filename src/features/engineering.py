"""Feature engineering module for Polish football attendance prediction."""
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import holidays

POLAND_HOLIDAYS_CACHE: Dict[int, Set[datetime.date]] = {}

# Major Polish derbies and prestige fixtures with known huge attendance spikes
MAJOR_DERBIES = {
    frozenset({"Lechia Gdańsk", "Arka Gdynia"}),        # Derby Trójmiasta
    frozenset({"Górnik Zabrze", "Ruch Chorzów"}),      # Wielkie Derby Śląska
    frozenset({"Wisła Kraków", "Cracovia"}),           # Święta Wojna
    frozenset({"Widzew Łódź", "ŁKS Łódź"}),            # Derby Łodzi
    frozenset({"Legia Warszawa", "Polonia Warszawa"}),  # Derby Warszawy
    frozenset({"Śląsk Wrocław", "Zagłębie Lubin"}),    # Derby Dolnego Śląska
    frozenset({"Stal Rzeszów", "Resovia Rzeszów"}),    # Derby Rzeszowa
    frozenset({"Legia Warszawa", "Lech Poznań"}),      # Klasyk Polski
    frozenset({"Legia Warszawa", "Widzew Łódź"}),      # Klasyk Polski
    frozenset({"GKS Katowice", "Górnik Zabrze"}),      # Śląski klasyk
    frozenset({"GKS Katowice", "Ruch Chorzów"}),       # Śląski klasyk
    frozenset({"Polonia Bytom", "Ruch Chorzów"}),      # Śląski klasyk
}


def get_poland_holidays(years: List[int]) -> Set[datetime.date]:
    """Retrieve Polish public holidays for the specified years."""
    all_hols = set()
    for y in years:
        if y not in POLAND_HOLIDAYS_CACHE:
            h = holidays.Poland(years=y)
            POLAND_HOLIDAYS_CACHE[y] = set(h.keys())
        all_hols.update(POLAND_HOLIDAYS_CACHE[y])
    return all_hols


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers."""
    R = 6371.0  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, cyclical and holiday features."""
    df = df.copy()
    
    if "match_datetime" in df.columns:
        dt_series = pd.to_datetime(df["match_datetime"], errors="coerce")
    elif "match_date" in df.columns:
        if "match_time" in df.columns:
            dt_series = pd.to_datetime(df["match_date"].astype(str) + " " + df["match_time"].astype(str), errors="coerce")
        else:
            dt_series = pd.to_datetime(df["match_date"], errors="coerce")
    else:
        dt_series = pd.Series(pd.Timestamp.now(), index=df.index)
    
    df["dt"] = dt_series
    df["dow"] = dt_series.dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = dt_series.dt.month
    df["year"] = dt_series.dt.year
    
    def parse_hour_float(row):
        t = str(row["match_time"]) if pd.notna(row["match_time"]) else "17:00"
        parts = t.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return h + m / 60.0
        except Exception:
            return 17.0

    df["hour_num"] = df.apply(parse_hour_float, axis=1)
    
    # Cyclical hour encoding (period = 24h)
    df["sin_hour"] = np.sin(2.0 * np.pi * df["hour_num"] / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * df["hour_num"] / 24.0)
    
    # Cyclical day-of-week encoding (period = 7 days)
    df["sin_dow"] = np.sin(2.0 * np.pi * df["dow"] / 7.0)
    df["cos_dow"] = np.cos(2.0 * np.pi * df["dow"] / 7.0)
    
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)  # Sat, Sun
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_prime_time"] = ((df["dow"].isin([4, 5, 6])) & (df["hour_num"] >= 17.0)).astype(int)
    
    # Polish public holidays
    valid_years = [int(y) for y in df["year"].dropna().unique() if y > 2000]
    hol_set = get_poland_holidays(valid_years)
    
    def check_holiday(d):
        if pd.isna(d):
            return 0
        date_obj = d.date()
        if date_obj in hol_set or (date_obj + timedelta(days=1)) in hol_set:
            return 1
        return 0

    df["is_holiday_proximate"] = df["dt"].apply(check_holiday)
    if "matchday" in df.columns:
        df["season_progress"] = df["matchday"].fillna(17) / 34.0
    elif "season_progress" not in df.columns:
        df["season_progress"] = 0.5

    # Season opener (summer) & spring resumption opener after winter break
    def check_round_opener(row):
        m = row.get("month", 0)
        mday = row.get("matchday", 0)
        try:
            m = int(m) if pd.notna(m) else 0
            mday = int(mday) if pd.notna(mday) else 0
        except Exception:
            return 0
        # Summer opener in July/August (matchdays 1-2)
        if m in [7, 8] and mday in [1, 2]:
            return 1
        # Spring resumption after winter break in February (typically matchdays 18-22)
        if m == 2 and 18 <= mday <= 22:
            return 1
        return 0

    df["is_round_opener"] = df.apply(check_round_opener, axis=1)
    return df


def add_distance_features(df: pd.DataFrame, club_locations: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Calculate geographic distance between home and away clubs in kilometers."""
    df = df.copy()
    
    def get_dist(row):
        h = row["home_team"]
        a = row["away_team"]
        if h in club_locations and a in club_locations:
            lat1, lon1 = club_locations[h]
            lat2, lon2 = club_locations[a]
            return round(haversine_distance(lat1, lon1, lat2, lon2), 1)
        return 200.0

    df["travel_distance_km"] = df.apply(get_dist, axis=1)
    df["is_local_derby"] = (df["travel_distance_km"] <= 45.0).astype(int)
    
    # Check major prestige derbies
    def check_prestige_derby(row):
        pair = frozenset({row["home_team"], row["away_team"]})
        return int(pair in MAJOR_DERBIES)

    df["is_major_derby"] = df.apply(check_prestige_derby, axis=1)
    return df


def add_reserve_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify reserve teams (II zespoły) and their fixture impact.
    Reserve matches have drastically smaller attendance (~100-300 fans)
    and derbies do not mobilize crowds for reserve fixtures.
    """
    df = df.copy()
    
    def check_reserve(name):
        if not isinstance(name, str):
            return 0
        return int(bool(re.search(r"\b(ii|2)\b", name.lower())))
        
    df["is_home_reserve"] = df["home_team"].apply(check_reserve)
    df["is_away_reserve"] = df["away_team"].apply(check_reserve)
    df["is_reserve_fixture"] = ((df["is_home_reserve"] == 1) | (df["is_away_reserve"] == 1)).astype(int)
    
    # Derbies never apply to reserve fixtures
    if "is_major_derby" in df.columns:
        df.loc[df["is_reserve_fixture"] == 1, "is_major_derby"] = 0
    if "is_local_derby" in df.columns:
        df.loc[df["is_home_reserve"] == 1, "is_local_derby"] = 0
        
    return df


def calculate_form_and_standings(df: pd.DataFrame, is_test_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Chronologically compute rolling form, league table positions,
    and fixture H2H history before kickoff (ZERO data leakage).
    
    If is_test_mask is provided, test matches query history from preceding
    training matches, but NEVER leak their own attendance into history.
    """
    if is_test_mask is not None:
        df["is_test"] = is_test_mask.astype(int)
    elif "is_test" not in df.columns:
        df["is_test"] = 0

    df = df.sort_values(by=["match_date", "match_time", "match_id"]).copy().reset_index(drop=True)
    
    n = len(df)
    home_pts_last_5 = np.zeros(n, dtype=float)
    home_pts_last_3_home = np.zeros(n, dtype=float)
    home_goal_diff_last_5 = np.zeros(n, dtype=float)
    
    away_pts_last_5 = np.zeros(n, dtype=float)
    away_pts_last_3_away = np.zeros(n, dtype=float)
    away_goal_diff_last_5 = np.zeros(n, dtype=float)
    
    home_table_pos = np.full(n, 10.0, dtype=float)
    away_table_pos = np.full(n, 10.0, dtype=float)
    home_table_pts = np.zeros(n, dtype=float)
    away_table_pts = np.zeros(n, dtype=float)
    
    h2h_matches_prior = np.zeros(n, dtype=int)
    h2h_avg_att = np.zeros(n, dtype=float)
    h2h_home_venue_avg_att = np.zeros(n, dtype=float)
    h2h_home_venue_matches = np.zeros(n, dtype=int)
    
    # Fan Attendance Momentum & Stakes
    home_rolling_att_last_3 = np.zeros(n, dtype=float)
    home_rolling_occ_last_3 = np.zeros(n, dtype=float)
    home_att_ema_3 = np.zeros(n, dtype=float)
    home_occupancy_ema_5 = np.zeros(n, dtype=float)
    stadium_expansion_boost = np.ones(n, dtype=float)
    away_sector_closed = np.zeros(n, dtype=int)
    is_title_or_promotion_race = np.zeros(n, dtype=int)
    is_relegation_battle = np.zeros(n, dtype=int)
    is_dead_rubber = np.zeros(n, dtype=int)
    match_stakes_index = np.full(n, 0.30, dtype=float)
    
    team_history: Dict[str, List[dict]] = {}
    season_tables: Dict[Tuple[str, int], Dict[str, dict]] = {}
    h2h_history: Dict[Tuple[str, str], List[dict]] = {}
    h2h_home_venue_history: Dict[Tuple[str, str], List[float]] = {}
    home_att_history: Dict[str, List[float]] = {}
    home_occ_history: Dict[str, List[float]] = {}
    home_season_capacities: Dict[str, Dict[str, List[float]]] = {}
    
    # Precompute reasonable fallback means per host club (excluding test matches if specified)
    valid_train_pre = (df["is_test"] == 0) & (df["attendance"].notna()) & (df["attendance"] > 0)
    club_hist_means = df[valid_train_pre].groupby("home_team")["attendance"].mean().to_dict()
    tier_fallbacks = {1: 8500.0, 2: 2500.0, 3: 800.0}

    for i, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        season = row["season"]
        tier = row["league_tier"]
        att = row["attendance"]
        cap = row.get("capacity_resolved", 5000)
        is_test_match = bool(row.get("is_test", 0) == 1)

        # Baseline attendance prior for this club
        club_baseline = club_hist_means.get(h, tier_fallbacks.get(tier, 2000.0))
        club_baseline = min(float(cap), max(50.0, club_baseline))
        
        # 1. Standings before this match
        table_key = (season, tier)
        if table_key not in season_tables:
            season_tables[table_key] = {}
        curr_table = season_tables[table_key]
        
        if curr_table:
            sorted_teams = sorted(
                curr_table.items(),
                key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
                reverse=True
            )
            rank_map = {team_name: rank + 1 for rank, (team_name, _) in enumerate(sorted_teams)}
            home_table_pos[i] = rank_map.get(h, len(sorted_teams) + 1)
            away_table_pos[i] = rank_map.get(a, len(sorted_teams) + 1)
            home_table_pts[i] = curr_table.get(h, {}).get("pts", 0)
            away_table_pts[i] = curr_table.get(a, {}).get("pts", 0)
        else:
            home_table_pos[i] = 1.0
            away_table_pos[i] = 2.0
            
        # 2. Form before this match
        h_hist = team_history.get(h, [])
        a_hist = team_history.get(a, [])
        
        last_5_h = h_hist[-5:]
        if last_5_h:
            home_pts_last_5[i] = sum(m["pts"] for m in last_5_h)
            home_goal_diff_last_5[i] = sum(m["gf"] - m["ga"] for m in last_5_h)
            
        last_3_home = [m for m in h_hist if m["is_home"]][-3:]
        if last_3_home:
            home_pts_last_3_home[i] = sum(m["pts"] for m in last_3_home)
            
        last_5_a = a_hist[-5:]
        if last_5_a:
            away_pts_last_5[i] = sum(m["pts"] for m in last_5_a)
            away_goal_diff_last_5[i] = sum(m["gf"] - m["ga"] for m in last_5_a)
            
        last_3_away = [m for m in a_hist if not m["is_home"]][-3:]
        if last_3_away:
            away_pts_last_3_away[i] = sum(m["pts"] for m in last_3_away)
            
        # 3. Overall H2H and Specific Home Venue H2H
        pair = (min(h, a), max(h, a))
        prior_meetings = h2h_history.get(pair, [])
        h2h_matches_prior[i] = len(prior_meetings)
        
        # Cold-start prior for fixture pairs with no prior records in DB
        is_maj_d = int(row.get("is_major_derby", 0))
        is_loc_d = int(row.get("is_local_derby", 0))
        derby_mult = 1.35 if is_maj_d else (1.15 if is_loc_d else 1.0)
        default_pair_att = min(float(cap), max(50.0, club_baseline * derby_mult))
        
        if prior_meetings:
            valid_atts = [m["att"] for m in prior_meetings if m.get("att") and m["att"] > 0]
            if valid_atts:
                h2h_avg_att[i] = np.mean(valid_atts)
            else:
                h2h_avg_att[i] = default_pair_att
        else:
            h2h_avg_att[i] = default_pair_att
                
        # Home venue specific H2H (e.g. Lechia hosting Arka in Gdańsk)
        home_pair = (h, a)
        prior_home_atts = h2h_home_venue_history.get(home_pair, [])
        h2h_home_venue_matches[i] = len(prior_home_atts)
        if prior_home_atts:
            h2h_home_venue_avg_att[i] = np.mean(prior_home_atts)
        else:
            h2h_home_venue_avg_att[i] = default_pair_att
            
        # 4. Fan Attendance Momentum & EMAs
        prior_home_atts_list = home_att_history.get(h, [])
        prior_home_occs_list = home_occ_history.get(h, [])
        curr_cap = float(cap)

        if prior_home_atts_list:
            last_3 = prior_home_atts_list[-3:]
            home_rolling_att_last_3[i] = float(np.mean(last_3))
            home_rolling_occ_last_3[i] = float(np.mean(last_3) / (curr_cap + 1e-5))

            # Exponentially weighted moving average of last 3 matches (weights: 0.20, 0.30, 0.50)
            if len(last_3) == 3:
                w3 = [0.20, 0.30, 0.50]
            elif len(last_3) == 2:
                w3 = [0.35, 0.65]
            else:
                w3 = [1.0]
            home_att_ema_3[i] = float(np.sum(np.array(w3) * np.array(last_3)))
        else:
            home_rolling_att_last_3[i] = club_baseline
            home_rolling_occ_last_3[i] = float(club_baseline / (curr_cap + 1e-5))
            home_att_ema_3[i] = club_baseline

        if prior_home_occs_list:
            last_5 = prior_home_occs_list[-5:]
            k = len(last_5)
            raw_w = [math.exp(0.4 * j) for j in range(k)]
            w5 = np.array(raw_w) / sum(raw_w)
            home_occupancy_ema_5[i] = float(np.sum(w5 * np.array(last_5)))
        else:
            home_occupancy_ema_5[i] = float(club_baseline / (curr_cap + 1e-5))

        # 4b. Stadium expansion boost & away sector check
        past_season_caps = [
            np.mean(clist) for s_k, clist in home_season_capacities.get(h, {}).items()
            if s_k != season
        ]
        if past_season_caps and curr_cap > 0:
            hist_cap_mean = float(np.mean(past_season_caps))
            if hist_cap_mean > 0 and curr_cap >= 1.15 * hist_cap_mean:
                stadium_expansion_boost[i] = round(float(np.clip(curr_cap / hist_cap_mean, 1.0, 3.0)), 3)
            else:
                stadium_expansion_boost[i] = 1.0
        else:
            stadium_expansion_boost[i] = 1.0

        txt_info = f"{str(row.get('outlier_reason', ''))} {str(row.get('notes', ''))}".lower()
        if any(kw in txt_info for kw in ["sektor gości", "zakaz wyjazdowy", "bez kibiców gości", "zamknięty sektor", "sektora gości"]):
            away_sector_closed[i] = 1
        elif "away_sector_closed" in row and pd.notna(row["away_sector_closed"]):
            away_sector_closed[i] = int(row["away_sector_closed"])
        else:
            away_sector_closed[i] = 0

        # 5. Match Stakes Index & Season Endgame Context
        prog = float(row.get("season_progress", 0.5))
        h_pos = home_table_pos[i]
        a_pos = away_table_pos[i]

        top_limit = 3.0 if tier == 1 else 6.0
        releg_limit = 14.0
        if prog >= 0.65:
            if h_pos <= top_limit or a_pos <= top_limit:
                is_title_or_promotion_race[i] = 1
            if h_pos >= releg_limit or a_pos >= releg_limit:
                is_relegation_battle[i] = 1
            if (h_pos > top_limit and h_pos < releg_limit) and (a_pos > top_limit and a_pos < releg_limit):
                is_dead_rubber[i] = 1

        s_val = 0.30
        if is_title_or_promotion_race[i]:
            s_val += 0.40 + 0.20 * max(0.0, (prog - 0.65) / 0.35)
        elif is_relegation_battle[i]:
            s_val += 0.35 if (h_pos >= releg_limit and a_pos >= releg_limit) else 0.25
        elif is_dead_rubber[i]:
            s_val -= 0.20

        if row.get("is_major_derby", 0) == 1:
            s_val += 0.30
        elif row.get("is_local_derby", 0) == 1:
            s_val += 0.15

        match_stakes_index[i] = round(float(np.clip(s_val, 0.05, 1.50)), 3)
            
        # 6. Update state with current match result (sports table updates for all matches)
        hs = row["home_score"]
        as_ = row["away_score"]
        if pd.notna(hs) and pd.notna(as_):
            hs = int(hs)
            as_ = int(as_)
            if hs > as_:
                h_pts, a_pts = 3, 0
                winner = h
            elif hs == as_:
                h_pts, a_pts = 1, 1
                winner = "draw"
            else:
                h_pts, a_pts = 0, 3
                winner = a
                
            if h not in team_history:
                team_history[h] = []
            team_history[h].append({"pts": h_pts, "gf": hs, "ga": as_, "is_home": True})
            
            if a not in team_history:
                team_history[a] = []
            team_history[a].append({"pts": a_pts, "gf": as_, "ga": hs, "is_home": False})
            
            for team, pts, gf, ga in [(h, h_pts, hs, as_), (a, a_pts, as_, hs)]:
                if team not in curr_table:
                    curr_table[team] = {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0}
                curr_table[team]["pts"] += pts
                curr_table[team]["gf"] += gf
                curr_table[team]["ga"] += ga
                curr_table[team]["gd"] += (gf - ga)
                curr_table[team]["played"] += 1
            
        if h not in home_season_capacities:
            home_season_capacities[h] = {}
        if season not in home_season_capacities[h]:
            home_season_capacities[h][season] = []
        home_season_capacities[h][season].append(curr_cap)

        if att and att > 0 and not is_test_match:
            if pair not in h2h_history and pd.notna(hs) and pd.notna(as_):
                h2h_history[pair] = []
            if pd.notna(hs) and pd.notna(as_):
                h2h_history[pair].append({"winner": winner, "att": att})

            if home_pair not in h2h_home_venue_history:
                h2h_home_venue_history[home_pair] = []
            h2h_home_venue_history[home_pair].append(att)
            if h not in home_att_history:
                home_att_history[h] = []
            home_att_history[h].append(float(att))
            if h not in home_occ_history:
                home_occ_history[h] = []
            home_occ_history[h].append(float(np.clip(att / (curr_cap + 1e-5), 0.01, 1.05)))
            
    df["home_pts_last_5"] = home_pts_last_5
    df["home_pts_last_3_home"] = home_pts_last_3_home
    df["home_goal_diff_last_5"] = home_goal_diff_last_5
    
    df["away_pts_last_5"] = away_pts_last_5
    df["away_pts_last_3_away"] = away_pts_last_3_away
    df["away_goal_diff_last_5"] = away_goal_diff_last_5
    df["form_diff_pts"] = home_pts_last_5 - away_pts_last_5
    
    df["home_table_pos"] = home_table_pos
    df["away_table_pos"] = away_table_pos
    df["home_table_pts"] = home_table_pts
    df["away_table_pts"] = away_table_pts
    df["table_pos_diff"] = away_table_pos - home_table_pos
    df["table_pts_diff"] = home_table_pts - away_table_pts
    
    df["h2h_matches_prior"] = h2h_matches_prior
    df["h2h_avg_att"] = h2h_avg_att
    df["h2h_home_venue_avg_att"] = h2h_home_venue_avg_att
    df["h2h_home_venue_matches"] = h2h_home_venue_matches
    
    df["home_rolling_att_last_3"] = home_rolling_att_last_3
    df["home_rolling_occ_last_3"] = home_rolling_occ_last_3
    df["home_att_ema_3"] = home_att_ema_3
    df["home_occupancy_ema_5"] = home_occupancy_ema_5
    df["stadium_expansion_boost"] = stadium_expansion_boost
    df["away_sector_closed"] = away_sector_closed
    df["is_title_or_promotion_race"] = is_title_or_promotion_race
    df["is_relegation_battle"] = is_relegation_battle
    df["is_dead_rubber"] = is_dead_rubber
    df["match_stakes_index"] = match_stakes_index
    return df



def calculate_bidirectional_oof_features(
    df: pd.DataFrame,
    is_test_mask: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Calculate bidirectional out-of-fold (leave-one-out) target encoding for:
    1. h2h_avg_att: Average attendance for this fixture pair across all seasons,
       excluding current match (leave-one-out) to prevent data leakage.
    2. h2h_home_venue_avg_att: Average attendance for this exact home fixture across all seasons,
       excluding current match.
    3. away_draw_factor: Out-of-fold draw factor of the away club across all seasons,
       excluding current match.

    ZERO DATA LEAKAGE:
    If is_test_mask is provided, test matches DO NOT contribute to statistical sums,
    receiving clean aggregates learned strictly from training matches.
    Cold-start defaults rely on host club historical baseline, not nominal stadium capacity.
    """
    df = df.copy()
    
    if is_test_mask is not None:
        df["is_test"] = is_test_mask.astype(int)
    elif "is_test" not in df.columns:
        df["is_test"] = 0

    # Only training matches with valid non-outlier attendance contribute to statistics
    calc_mask = (
        (df["is_test"] == 0) &
        df["attendance"].notna() &
        (df["attendance"] > 0) &
        (df.get("is_outlier", 0) == 0)
    )
    
    val_att = np.where(calc_mask, df["attendance"], 0.0)
    val_cnt = np.where(calc_mask, 1.0, 0.0)
    
    cap = df["capacity_resolved"].fillna(5000).values if "capacity_resolved" in df.columns else np.full(len(df), 5000.0)
    is_maj = df["is_major_derby"].values if "is_major_derby" in df.columns else np.zeros(len(df))
    is_loc = df["is_local_derby"].values if "is_local_derby" in df.columns else np.zeros(len(df))
    
    derby_prior = np.where(
        is_maj == 1, cap * 0.75,
        np.where(is_loc == 1, cap * 0.55, cap * 0.40)
    )

    # 1. Fixture Pair H2H
    pair_series = df.apply(lambda r: (min(str(r["home_team"]), str(r["away_team"])), max(str(r["home_team"]), str(r["away_team"]))), axis=1)
    df["_pair"] = pair_series
    
    pair_sum = df.assign(_v=val_att).groupby("_pair")["_v"].transform("sum")
    pair_cnt = df.assign(_c=val_cnt).groupby("_pair")["_c"].transform("sum")
    
    # For training rows: leave-one-out (subtract own match)
    # For test rows: clean mean from training set (no subtraction since val_att was 0)
    other_pair_sum = np.where(calc_mask, pair_sum - df["attendance"], pair_sum)
    other_pair_cnt = np.where(calc_mask, pair_cnt - 1.0, pair_cnt)
    
    df["h2h_avg_att"] = np.where(other_pair_cnt > 0, other_pair_sum / np.maximum(other_pair_cnt, 1.0), derby_prior)
    df["h2h_matches_prior"] = np.maximum(0, other_pair_cnt.astype(int))
    
    # 2. Home Venue Specific H2H
    home_pair_series = df.apply(lambda r: (str(r["home_team"]), str(r["away_team"])), axis=1)
    df["_home_pair"] = home_pair_series
    
    venue_sum = df.assign(_v=val_att).groupby("_home_pair")["_v"].transform("sum")
    venue_cnt = df.assign(_c=val_cnt).groupby("_home_pair")["_c"].transform("sum")
    
    other_venue_sum = np.where(calc_mask, venue_sum - df["attendance"], venue_sum)
    other_venue_cnt = np.where(calc_mask, venue_cnt - 1.0, venue_cnt)
    
    df["h2h_home_venue_avg_att"] = np.where(
        other_venue_cnt > 0,
        other_venue_sum / np.maximum(other_venue_cnt, 1.0),
        df["h2h_avg_att"]
    )
    df["h2h_home_venue_matches"] = np.maximum(0, other_venue_cnt.astype(int))
    
    # 3. Away Opponent Draw Factor (Strictly on Training Matches)
    if "season" in df.columns and "home_team" in df.columns and "away_team" in df.columns:
        host_means = df[calc_mask].groupby(["season", "home_team"])["attendance"].transform("mean")
        df.loc[calc_mask, "_host_season_mean"] = host_means
        club_overall_means = df[calc_mask].groupby("home_team")["attendance"].mean()
        global_att_mean = df.loc[calc_mask, "attendance"].mean() if calc_mask.any() else 5000.0
        df["_host_season_mean"] = (
            df["_host_season_mean"]
            .fillna(df["home_team"].map(club_overall_means))
            .fillna(global_att_mean)
        )
        
        att_ratio = np.where(calc_mask, df["attendance"] / (df["_host_season_mean"] + 1e-5), 0.0)
        df["_att_ratio"] = att_ratio
        
        away_ratio_sum = df.groupby("away_team")["_att_ratio"].transform("sum")
        away_matches_cnt = df.assign(_c=val_cnt).groupby("away_team")["_c"].transform("sum")
        
        other_away_sum = np.where(calc_mask, away_ratio_sum - df["_att_ratio"], away_ratio_sum)
        other_away_cnt = np.where(calc_mask, away_matches_cnt - 1.0, away_matches_cnt)
        
        df["away_draw_factor"] = np.where(other_away_cnt > 0, other_away_sum / np.maximum(other_away_cnt, 1.0), 1.0)
        df.drop(columns=["_host_season_mean", "_att_ratio"], inplace=True, errors="ignore")
    else:
        df["away_draw_factor"] = 1.0
        
    df.drop(columns=["_pair", "_home_pair"], inplace=True, errors="ignore")
    return df


def calculate_away_draw_factor(df: pd.DataFrame, is_test_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """Calculate away opponent draw factor (backward-compatible wrapper)."""
    return calculate_bidirectional_oof_features(df, is_test_mask=is_test_mask)


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive weather condition features:
    temperature, rain, snow, sunny conditions, cloud cover, and wind speed.
    """
    df = df.copy()
    
    temp = df["temperature_c"].fillna(15.0) if "temperature_c" in df.columns else pd.Series(15.0, index=df.index)
    precip = df["precipitation_mm"].fillna(0.0) if "precipitation_mm" in df.columns else pd.Series(0.0, index=df.index)
    rain = df["rain_mm"].fillna(0.0) if "rain_mm" in df.columns else pd.Series(0.0, index=df.index)
    roof = df["roof_pct"].fillna(1.0) if "roof_pct" in df.columns else pd.Series(1.0, index=df.index)
    wcode = df["weather_code"].fillna(0).astype(int) if "weather_code" in df.columns else pd.Series(0, index=df.index)
    clouds = df["cloud_cover_pct"].fillna(50.0) if "cloud_cover_pct" in df.columns else pd.Series(50.0, index=df.index)
    wind = df["wind_speed_kmh"].fillna(10.0) if "wind_speed_kmh" in df.columns else pd.Series(10.0, index=df.index)
    
    # 1. Rain hitting fans: 0 if 100% roofed, full if uncovered
    df["rain_exposure_mm"] = precip * (1.0 - roof)
    
    # 2. Comfort penalty: deviation from ideal 18°C match temperature
    df["temp_comfort_dev"] = (temp - 18.0).abs()

    # 3. Combined effective weather discomfort (interaction of rain/roof with harsh cold)
    df["effective_weather_discomfort"] = df["rain_exposure_mm"] + np.maximum(0.0, 10.0 - temp) * 0.5
    
    # 4. Weather Condition Flags
    # Sunny: low cloud cover (< 30%) and zero precipitation
    df["is_sunny"] = ((clouds < 30.0) & (precip == 0.0)).astype(int)
    
    # Overcast / gloomy: heavy clouds and no precipitation
    df["is_overcast"] = ((clouds >= 80.0) & (precip == 0.0)).astype(int)
    
    # Snow: WMO codes 71, 73, 75, 77, 85, 86 or cold with precipitation
    df["is_snow"] = (wcode.isin([71, 73, 75, 77, 85, 86]) | ((temp <= 1.5) & (precip > 0.0))).astype(int)
    
    # Heavy rain / storm: precip >= 2.5 mm or WMO codes 63, 65, 81, 82, 95
    df["is_heavy_rain"] = ((precip >= 2.5) | wcode.isin([63, 65, 81, 82, 95])).astype(int)
    
    # High wind speed flag (> 30 km/h)
    df["is_high_wind"] = (wind > 30.0).astype(int)
    df["wind_speed_kmh"] = wind
    df["cloud_cover_pct"] = clouds
    
    # General adverse weather flag
    df["is_adverse_weather"] = ((precip > 1.5) | (temp < 2.0) | (temp > 30.0) | (wind > 35.0)).astype(int)
    
    return df


def build_full_features(
    df: pd.DataFrame,
    club_locations: Dict[str, Tuple[float, float]],
    is_test_mask: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Run complete feature engineering pipeline with ZERO data leakage."""
    df = add_calendar_features(df)
    df = add_distance_features(df, club_locations)
    df = add_reserve_features(df)
    df = calculate_form_and_standings(df, is_test_mask=is_test_mask)
    df = calculate_bidirectional_oof_features(df, is_test_mask=is_test_mask)
    df = add_weather_interactions(df)
    
    df["is_valid_train"] = (
        (df["attendance"].notna()) &
        (df["attendance"] > 0) &
        (df["is_outlier"] == 0) &
        (df["capacity_resolved"].notna()) &
        (df["capacity_resolved"] > 0)
    ).astype(int)
    
    return df
