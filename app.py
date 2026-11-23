"""Streamlit web application for Polish football attendance prediction with Deep Tabular Neural Network."""
import torch
import sys
from pathlib import Path
from datetime import datetime, date, time
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import get_connection
from src.scraper.stadiums import resolve_stadium, normalize_club_name
from src.features.engineering import haversine_distance, add_calendar_features,add_weather_interactions,add_reserve_features,get_poland_holidays

from src.models.neural_net import TabularQuantileNN, evaluate_predictions
from src.models.neural_net import (
    TabularQuantileNN,
    NeuralEnsembleQuantileModel,
    HybridEnsembleQuantileModel,
    evaluate_predictions
)

st.set_page_config(
    page_title="Predykcja Frekwencji - Ekstraklasa, 1. i 2. Liga",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_data():
    """Load serialized neural network model, test predictions, and SQLite metadata."""
    models_dir = PROJECT_ROOT / "data" / "models"
    nn_path = models_dir / "tabular_nn_model.joblib"
    val_path = models_dir / "test_predictions.parquet"
    
    nn_model = joblib.load(nn_path) if nn_path.exists() else None
    
    if val_path.exists():
        val_df = pd.read_parquet(val_path)
    else:
        csv_fallback = models_dir / "test_predictions.csv"
        val_df = pd.read_csv(csv_fallback) if csv_fallback.exists() else pd.DataFrame()
        
    # Get club list and locations from SQLite
    conn = get_connection()
    clubs_df = pd.read_sql("SELECT DISTINCT club_name, city, latitude, longitude FROM club_locations ORDER BY club_name", conn)
    club_dict = {r["club_name"]: (r["latitude"], r["longitude"], r["city"]) for _, r in clubs_df.iterrows()}
    
    # Historical stats per club (excluding flagged outliers like boycotts/closed doors)
    hist_stats = pd.read_sql("""
        SELECT home_team, 
               AVG(attendance) as avg_att, 
               MAX(attendance) as max_att, 
               MIN(attendance) as min_att,
               COUNT(*) as match_count
        FROM matches 
        WHERE attendance IS NOT NULL AND attendance > 0 AND is_outlier = 0
        GROUP BY home_team
    """, conn)
    hist_map = hist_stats.set_index("home_team").to_dict(orient="index")
    
    # Empirical away draw factor per away_team (excluding outliers)
    draw_df = pd.read_sql("""
        SELECT season, home_team, away_team, attendance
        FROM matches
        WHERE attendance IS NOT NULL AND attendance > 0 AND is_outlier = 0
    """, conn)
    host_means = draw_df.groupby(["season", "home_team"])["attendance"].transform("mean")
    draw_df["ratio"] = draw_df["attendance"] / (host_means + 1e-5)
    away_draw_map = draw_df.groupby("away_team")["ratio"].mean().to_dict()
    
    # Latest standings per league tier in 2026/2027
    latest_df = pd.read_sql("""
        SELECT match_id, league_tier, home_team, away_team, home_score, away_score
        FROM matches
        WHERE season = '2026/2027' AND home_score IS NOT NULL
    """, conn)
    standings_map = {}
    for tier, group in latest_df.groupby("league_tier"):
        pts, gd = {}, {}
        for _, row in group.iterrows():
            h, a = row["home_team"], row["away_team"]
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            pts[h] = pts.get(h, 0) + (3 if hs > as_ else (1 if hs == as_ else 0))
            pts[a] = pts.get(a, 0) + (3 if as_ > hs else (1 if hs == as_ else 0))
            gd[h] = gd.get(h, 0) + (hs - as_)
            gd[a] = gd.get(a, 0) + (as_ - hs)
        sorted_teams = sorted(pts.keys(), key=lambda t: (pts[t], gd[t]), reverse=True)
        for rank, t in enumerate(sorted_teams):
            standings_map[t] = rank + 1
    
    return nn_model, val_df, club_dict, hist_map, away_draw_map, standings_map


def get_club_recent_form(club: str, conn):
    """Calculate points earned in the last 5 matches."""
    rows = conn.execute("""
        SELECT match_date, home_team, away_team, home_score, away_score
        FROM matches
        WHERE (home_team = ? OR away_team = ?) AND home_score IS NOT NULL
        ORDER BY match_date DESC, match_time DESC
        LIMIT 5
    """, (club, club)).fetchall()
    pts = 0
    for r in rows:
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        if r["home_team"] == club:
            pts += 3 if hs > as_ else (1 if hs == as_ else 0)
        else:
            pts += 3 if as_ > hs else (1 if hs == as_ else 0)
    return pts if rows else 8


def compute_local_feature_contributions(model, row_df: pd.DataFrame, base_pred_att: float) -> pd.DataFrame:
    """
    Computes local feature contributions for a single fixture prediction (Explainable AI / Counterfactual Analysis).
    Measures how much key match parameters increased (+) or decreased (-) the predicted attendance.
    """
    if model is None or row_df.empty:
        return pd.DataFrame()

    contributions = []

    # 1. Medialność i prestiż rywala (away_draw_factor vs neutralny 1.00)
    df_neutral_opp = row_df.copy()
    df_neutral_opp["away_draw_factor"] = 1.00
    pred_opp = float(model.predict_interval(df_neutral_opp)["q50"].iloc[0])
    opp_impact = int(round(base_pred_att - pred_opp))
    draw_f = float(row_df["away_draw_factor"].iloc[0])
    contributions.append({
        "Czynnik": "Medialność i marka rywala",
        "Wpływ": opp_impact,
        "Wartość": f"Współczynnik: {draw_f:.2f}x ({row_df['away_team'].iloc[0]})",
        "Interpretacja": "Przyciąganie kibiców przez renomę i atrakcyjność gości" if opp_impact >= 0 else "Mniejsza siła przyciągania kibiców przez rywala"
    })

    # 2. Ranga i stawka meczu (derby / walka o cel vs zwykły mecz ligowy 0.30)
    df_neutral_stakes = row_df.copy()
    df_neutral_stakes["match_stakes_index"] = 0.30
    df_neutral_stakes["is_major_derby"] = 0
    df_neutral_stakes["is_local_derby"] = 0
    df_neutral_stakes["is_title_or_promotion_race"] = 0
    df_neutral_stakes["is_relegation_battle"] = 0
    pred_stakes = float(model.predict_interval(df_neutral_stakes)["q50"].iloc[0])
    stakes_impact = int(round(base_pred_att - pred_stakes))
    is_derby = int(row_df.get("is_major_derby", pd.Series([0])).iloc[0]) or int(row_df.get("is_local_derby", pd.Series([0])).iloc[0])
    derby_txt = "Derby / Mecz prestiżowy" if is_derby else ("Gra o awans/mistrzostwo" if row_df.get("is_title_or_promotion_race", pd.Series([0])).iloc[0] else "Standardowa stawka ligowa")
    contributions.append({
        "Czynnik": "Ranga i stawka meczu",
        "Wpływ": stakes_impact,
        "Wartość": f"{derby_txt} ({row_df['match_stakes_index'].iloc[0]:.2f}x)",
        "Interpretacja": "Zwiększona mobilizacja kibicowska z uwagi na stawkę spotkania" if stakes_impact > 0 else "Neutralny wpływ stawki ligowej"
    })

    # 3. Termin meczu (dzień tygodnia i godzina vs neutralny dzień roboczy czwartek 17:00)
    df_neutral_time = row_df.copy()
    df_neutral_time["dow"] = 3
    df_neutral_time["sin_dow"] = np.sin(2.0 * np.pi * 3 / 7.0)
    df_neutral_time["cos_dow"] = np.cos(2.0 * np.pi * 3 / 7.0)
    df_neutral_time["is_weekend"] = 0
    df_neutral_time["is_friday"] = 0
    df_neutral_time["is_prime_time"] = 0
    pred_time = float(model.predict_interval(df_neutral_time)["q50"].iloc[0])
    time_impact = int(round(base_pred_att - pred_time))
    day_names = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
    dow_val = int(row_df["dow"].iloc[0]) if "dow" in row_df else 5
    hour_val = row_df.get("match_time", pd.Series(["17:30"])).iloc[0]
    contributions.append({
        "Czynnik": "Termin spotkania",
        "Wpływ": time_impact,
        "Wartość": f"{day_names[dow_val]}, godz. {hour_val}",
        "Interpretacja": "Dogodny termin weekendowy sprzyjający frekwencji" if time_impact > 0 else "Mniej dogodny termin w środku tygodnia"
    })

    # 4. Warunki atmosferyczne i zadaszenie (vs komfortowe 18°C, 0mm opadów)
    df_neutral_weather = row_df.copy()
    df_neutral_weather["temperature_c"] = 18.0
    df_neutral_weather["precipitation_mm"] = 0.0
    df_neutral_weather["weather_code"] = 0
    df_neutral_weather["effective_weather_discomfort"] = 0.0
    df_neutral_weather["rain_exposure_mm"] = 0.0
    df_neutral_weather["temp_comfort_dev"] = 0.0
    df_neutral_weather["is_adverse_weather"] = 0
    df_neutral_weather["is_heavy_rain"] = 0
    df_neutral_weather["is_snow"] = 0
    pred_weather = float(model.predict_interval(df_neutral_weather)["q50"].iloc[0])
    weather_impact = int(round(base_pred_att - pred_weather))
    temp_val = float(row_df["temperature_c"].iloc[0])
    rain_val = float(row_df["precipitation_mm"].iloc[0])
    roof_val = int(row_df["roof_pct"].iloc[0] * 100)
    contributions.append({
        "Czynnik": "Warunki atmosferyczne i dach",
        "Wpływ": weather_impact,
        "Wartość": f"{temp_val:.1f}°C, {rain_val:.1f} mm deszczu (dach: {roof_val}%)",
        "Interpretacja": "Komfortowe warunki pogodowe dla kibiców" if weather_impact >= 0 else "Ujemny wpływ opadów lub chłodu na frekwencję"
    })

    # 5. Dystans kibiców gości i logistyka (vs bliski wyjazd 60 km)
    df_neutral_dist = row_df.copy()
    df_neutral_dist["travel_distance_km"] = 60.0
    pred_dist = float(model.predict_interval(df_neutral_dist)["q50"].iloc[0])
    dist_impact = int(round(base_pred_att - pred_dist))
    dist_val = float(row_df["travel_distance_km"].iloc[0])
    contributions.append({
        "Czynnik": "Odległość wyjazdowa rywala",
        "Wpływ": dist_impact,
        "Wartość": f"{dist_val:.0f} km",
        "Interpretacja": "Krótka trasa ułatwiająca przyjazd kibicom gości" if dist_impact >= 0 else "Daleka trasa ograniczająca liczbę kibiców przyjezdnych"
    })

    # 6. Status sektora gości (zamknięty vs otwarty)
    if int(row_df.get("away_sector_closed", pd.Series([0])).iloc[0]) == 1:
        df_sec = row_df.copy()
        df_sec["away_sector_closed"] = 0
        pred_sec = float(model.predict_interval(df_sec)["q50"].iloc[0])
        sec_impact = int(round(base_pred_att - pred_sec))
        contributions.append({
            "Czynnik": "Sektor gości zamknięty",
            "Wpływ": sec_impact,
            "Wartość": "Brak kibiców gości (zakaz / remont)",
            "Interpretacja": "Utrata puli widzów z sektora gości"
        })

    # 7. Aktualna forma sportowa obu ekip (vs neutralne 6 pkt w 5 meczach)
    df_neutral_form = row_df.copy()
    df_neutral_form["home_pts_last_5"] = 6
    df_neutral_form["away_pts_last_5"] = 6
    df_neutral_form["form_diff_pts"] = 0
    pred_form = float(model.predict_interval(df_neutral_form)["q50"].iloc[0])
    form_impact = int(round(base_pred_att - pred_form))
    h_pts = int(row_df["home_pts_last_5"].iloc[0])
    a_pts = int(row_df["away_pts_last_5"].iloc[0])
    contributions.append({
        "Czynnik": "Forma sportowa (ost. 5 meczów)",
        "Wpływ": form_impact,
        "Wartość": f"Gospodarz: {h_pts} pkt | Gość: {a_pts} pkt",
        "Interpretacja": "Dobra seria punktowa podbijająca popyt na bilety" if form_impact >= 0 else "Spadek zainteresowania słabszą formą sportową"
    })

    df_res = pd.DataFrame(contributions)
    df_res["Wpływ_abs"] = df_res["Wpływ"].abs()
    df_res = df_res.sort_values(by="Wpływ_abs", ascending=False).reset_index(drop=True)
    df_res.drop(columns=["Wpływ_abs"], inplace=True)
    return df_res


nn_model, val_df, club_dict, hist_map, away_draw_map, standings_map = load_models_and_data()

st.markdown('<div class="main-title">⚽ Model Predykcji Frekwencji Piłkarskiej</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">System estymacji i predykcji kwantylowej (Q25, Q50, Q75) w oparciu o głęboką sieć neuronową PyTorch</div>', unsafe_allow_html=True)

# Tabs
tab_predict, tab_validate, tab_info = st.tabs([
    "🎯 Sprawdź i Zaprognozuj Mecz",
    "📊 Weryfikacja na Zbiorze Walidacyjnym (Test Set)",
    "ℹ️ O Architekturze Sieci Neuronowej"
])


# ==============================================================================
# TAB 1: KALKULATOR MECZU (PREDICTION SIMULATOR)
# ==============================================================================
with tab_predict:
    st.subheader("Wybierz parametry spotkania")
    
    club_list = sorted(list(club_dict.keys()))
    if not club_list:
        club_list = ["Legia Warszawa", "Lech Poznań", "Widzew Łódź", "Górnik Zabrze", "Pogoń Szczecin", "Wisła Kraków"]
        
    def format_team(c: str) -> str:
        if bool(re.search(r"\b(ii|2)\b", c.lower())):
            return f"{c} 🛡️ (Drużyna Rezerw)"
        return c

    col_team1, col_team2, col_league = st.columns([1.2, 1.2, 1.0])
    with col_team1:
        default_home = "Legia Warszawa" if "Legia Warszawa" in club_list else club_list[0]
        home_team = st.selectbox("Gospodarz (Home Team)", club_list, index=club_list.index(default_home), format_func=format_team)
        
    with col_team2:
        default_away = "Lech Poznań" if "Lech Poznań" in club_list else club_list[1]
        away_team = st.selectbox("Gość (Away Team)", [c for c in club_list if c != home_team], index=0, format_func=format_team)
        
    is_home_res = int(bool(re.search(r"\b(ii|2)\b", home_team.lower())))
    is_away_res = int(bool(re.search(r"\b(ii|2)\b", away_team.lower())))
    is_res_fixture = int(is_home_res or is_away_res)
    default_tier_idx = 2 if is_res_fixture else 0

    with col_league:
        tier_choice = st.selectbox("Poziom ligowy", ["Ekstraklasa (Tier 1)", "1. Liga (Tier 2)", "2. Liga (Tier 3)"], index=default_tier_idx)
        league_tier = 1 if "Ekstraklasa" in tier_choice else (2 if "1. Liga" in tier_choice else 3)
        
    col_d1, col_d2, col_d3 = st.columns([1.0, 1.0, 1.2])
    with col_d1:
        today = date.today()
        from datetime import timedelta
        days_to_sat = (5 - today.weekday()) % 7
        if days_to_sat == 0 and today.weekday() != 5:
            days_to_sat = 7
        default_match_date = today + timedelta(days=days_to_sat)
        match_date = st.date_input("Data meczu", value=default_match_date)
        if match_date.weekday() < 4:
            st.caption("⚠️ *Dzień roboczy (pn–czw): mecze w środku tygodnia gromadzą statystycznie o 10–15% mniej kibiców niż w weekend.*")
        else:
            st.caption("🟢 *Weekend (pt–nd): optymalny termin ligowy.*")
        
    with col_d2:
        match_time = st.selectbox("Godzina rozpoczęcia", [
            "12:30 (Niedzielny obiad)",
            "14:45 (Wczesne popołudnie)",
            "15:00 (Klasyczna sobota)",
            "17:30 (Sobotni / niedzielny hit)",
            "18:00 (Piątkowy inaugurator)",
            "20:00 (Piątek / sobota prime-time)",
            "20:30 (Weekendowy wieczór)",
            "19:00 (Poniedziałkowy finisz)"
        ], index=3)
        time_str = match_time.split()[0]
        
    with col_d3:
        match_round = st.slider("Numer kolejki w sezonie (1-34)", min_value=1, max_value=34, value=15)
        
    # Resolve stadium dynamically
    stadium_info = resolve_stadium(home_team, str(match_date))
    capacity = stadium_info["capacity"]
    stadium_name = stadium_info["stadium_name"]
    stadium_city = stadium_info["city"]
    roof_pct = stadium_info["roof_pct"]
    
    # Distance and Derby checks
    lat1, lon1 = stadium_info["latitude"], stadium_info["longitude"]
    lat2, lon2 = club_dict.get(away_team, (lat1, lon1, ""))[:2]
    dist_km = round(haversine_distance(lat1, lon1, lat2, lon2), 1)
    
    # Major Derbies check
    major_derbies = [
        ("Lech Poznań", "Legia Warszawa"),
        ("Wisła Kraków", "Cracovia"),
        ("Górnik Zabrze", "Ruch Chorzów"),
        ("Widzew Łódź", "ŁKS Łódź"),
        ("Śląsk Wrocław", "Zagłębie Lubin"),
        ("Arka Gdynia", "Lechia Gdańsk"),
        ("GKS Katowice", "Górnik Zabrze"),
        ("GKS Katowice", "Ruch Chorzów"),
        ("Polonia Bytom", "Ruch Chorzów"),
        ("Resovia Rzeszów", "Stal Rzeszów")
    ]
    is_derby = int(
        (home_team, away_team) in major_derbies or 
        (away_team, home_team) in major_derbies or 
        (dist_km <= 40.0 and not is_res_fixture)
    )
    
    with st.expander("🏟️ Informacje o stadionie i logistyce wyjazdowej", expanded=True):
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.write(f"**Obiekt:** {stadium_name}")
        scol2.write(f"**Pojemność nominalna:** {capacity:,} miejsc")
        scol3.write(f"**Zadaszenie trybun:** {int(roof_pct*100)}%")
        derby_badge = "🔥 **MECZ DERBOWY / HIT**" if is_derby else "Standardowy mecz"
        scol4.write(f"**Dystans rywala:** {dist_km} km ({derby_badge})")
        away_sector_closed = st.checkbox("⛔ Sektor gości zamknięty / zakaz wyjazdowy kibiców gości", value=False)
        
    st.markdown("#### Czynniki sportowe i tabela")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        default_h_pos = standings_map.get(home_team, 5)
        home_pos = st.number_input("Pozycja gospodarza w tabeli", min_value=1, max_value=18, value=min(18, default_h_pos))
    with col_s2:
        default_a_pos = standings_map.get(away_team, 8)
        away_pos = st.number_input("Pozycja gościa w tabeli", min_value=1, max_value=18, value=min(18, default_a_pos))
    with col_s3:
        conn = get_connection()
        h_form = get_club_recent_form(home_team, conn)
        a_form = get_club_recent_form(away_team, conn)
        home_pts_last_5 = st.slider("Punkty gospodarza w ost. 5 meczach", 0, 15, value=h_form)
    with col_s4:
        away_pts_last_5 = st.slider("Punkty gościa w ost. 5 meczach", 0, 15, value=a_form)
        
    # Match Stakes - dokładnie zgodny ze wzorem z modelu (src/features/engineering.py)
    prog = min(1.0, max(0.0, match_round / 34.0))
    top_limit = 3 if league_tier == 1 else 6
    releg_limit = 14
    is_title_or_promotion_race = int((prog >= 0.65) and (home_pos <= top_limit or away_pos <= top_limit))
    is_relegation_battle = int((prog >= 0.65) and (home_pos >= releg_limit or away_pos >= releg_limit))
    is_dead_rubber = int((prog >= 0.65) and (home_pos > top_limit and home_pos < releg_limit) and (away_pos > top_limit and away_pos < releg_limit) and not is_derby)
    
    s_val = 0.30
    if is_title_or_promotion_race:
        s_val += 0.40 + 0.20 * max(0.0, (prog - 0.65) / 0.35)
    elif is_relegation_battle:
        s_val += 0.35 if (home_pos >= releg_limit and away_pos >= releg_limit) else 0.25
    elif is_dead_rubber:
        s_val -= 0.20

    if is_derby:
        s_val += 0.30

    match_stakes_index = round(float(np.clip(s_val, 0.05, 1.50)), 3)

    st.markdown("#### Prognoza pogody w porze meczu")
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        season_month = match_date.month
        def_temp = 18.0 if season_month in [5, 6, 7, 8, 9] else (10.0 if season_month in [4, 10] else 3.0)
        temperature_c = st.slider("Temperatura (°C)", min_value=-10.0, max_value=35.0, value=def_temp, step=0.5)
    with col_w2:
        precipitation_mm = st.slider("Opady deszczu (mm/h)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)
    with col_w3:
        cloud_cover_pct = st.slider("Zachmurzenie nieba (%)", min_value=0, max_value=100, value=30, step=5)
    with col_w4:
        wind_speed_kmh = st.slider("Prędkość wiatru (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=1.0)
        
    weather_code = 0 if precipitation_mm == 0 else (61 if precipitation_mm < 3.0 else 63)
    
    # Calculate H2H Out-Of-Fold statistics from database
    h_stats_init = hist_map.get(home_team, {"avg_att": capacity * 0.40, "max_att": capacity})
    hist_avg = float(h_stats_init.get("avg_att", capacity * 0.40))
    away_draw_factor = away_draw_map.get(away_team, 1.0)
    default_pair_att = float(np.clip(hist_avg * (1.30 if is_derby else away_draw_factor), 50.0, capacity))

    h2h_query = """
        SELECT attendance 
        FROM matches 
        WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
          AND attendance IS NOT NULL AND attendance > 0 AND is_outlier = 0
    """
    h2h_matches = conn.execute(h2h_query, (home_team, away_team, away_team, home_team)).fetchall()
    h2h_avg = float(np.mean([r[0] for r in h2h_matches])) if h2h_matches else default_pair_att
    
    h2h_venue_query = """
        SELECT attendance 
        FROM matches 
        WHERE home_team = ? AND away_team = ? 
          AND attendance IS NOT NULL AND attendance > 0 AND is_outlier = 0
    """
    h2h_venue_matches = conn.execute(h2h_venue_query, (home_team, away_team)).fetchall()
    h2h_venue_avg = float(np.mean([r[0] for r in h2h_venue_matches])) if h2h_venue_matches else default_pair_att
    
    # Recent club attendance
    home_rolling = conn.execute("""
        SELECT attendance FROM matches 
        WHERE home_team = ? AND attendance IS NOT NULL AND attendance > 0 AND is_outlier = 0
        ORDER BY match_date DESC, match_time DESC LIMIT 3
    """, (home_team,)).fetchall()
    home_rolling_att = float(np.mean([r[0] for r in home_rolling])) if home_rolling else hist_avg
    
    if st.button("🚀 Oblicz przewidywaną frekwencję (Sieć Neuronowa PyTorch)", type="primary"):
        # Construct input DataFrame
        single_df = pd.DataFrame([{
            "league_tier": league_tier,
            "matchday": match_round,
            "match_date": str(match_date),
            "match_time": time_str,
            "match_datetime": f"{match_date} {time_str}:00",
            "home_team": home_team,
            "away_team": away_team,
            "capacity_resolved": capacity,
            "roof_pct": roof_pct,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "weather_code": weather_code,
            "cloud_cover_pct": cloud_cover_pct,
            "wind_speed_kmh": wind_speed_kmh,
            "travel_distance_km": dist_km,
            "is_local_derby": int(dist_km <= 45.0),
            "is_major_derby": is_derby,
            "h2h_avg_att": h2h_avg,
            "h2h_home_venue_avg_att": h2h_venue_avg,
            "home_rolling_att_last_3": float(home_rolling_att),
            "home_rolling_occ_last_3": float(home_rolling_att) / max(1.0, float(capacity)),
            "is_title_or_promotion_race": int(is_title_or_promotion_race),
            "is_relegation_battle": int(is_relegation_battle),
            "is_dead_rubber": int(is_dead_rubber),
            "match_stakes_index": float(match_stakes_index),
            "home_table_pos": home_pos,
            "away_table_pos": away_pos,
            "table_pos_diff": away_pos - home_pos,
            "table_pts_diff": (19 - home_pos)*2 - (19 - away_pos)*2,
            "home_pts_last_5": home_pts_last_5,
            "away_pts_last_5": away_pts_last_5,
            "home_pts_last_3_home": int(home_pts_last_5 * 0.6),
            "away_pts_last_3_away": int(away_pts_last_5 * 0.5),
            "away_draw_factor": away_draw_factor,
            "away_sector_closed": int(away_sector_closed),
            "stadium_expansion_boost": 1.0,
            "home_att_ema_3": float(home_rolling_att),
            "home_occupancy_ema_5": float(home_rolling_att) / max(1.0, float(capacity)),
        }])
        
        single_df = add_calendar_features(single_df)
        single_df = add_reserve_features(single_df)
        single_df = add_weather_interactions(single_df)
        
        # Predictions with PyTorch Neural Network
        preds_nn = nn_model.predict_interval(single_df) if nn_model else None
        main_pred_att = int(preds_nn["q50"].iloc[0]) if preds_nn is not None else 0
        main_pred_occ = float(preds_nn["pred_occupancy"].iloc[0]) if (preds_nn is not None and "pred_occupancy" in preds_nn) else float(main_pred_att / (capacity + 1e-5))
        
        pred_q25 = int(preds_nn["q25"].iloc[0]) if (preds_nn is not None and "q25" in preds_nn) else (int(preds_nn["q10"].iloc[0]) if (preds_nn is not None and "q10" in preds_nn) else int(main_pred_att * 0.88))
        pred_q75 = int(preds_nn["q75"].iloc[0]) if (preds_nn is not None and "q75" in preds_nn) else (int(preds_nn["q90"].iloc[0]) if (preds_nn is not None and "q90" in preds_nn) else int(main_pred_att * 1.12))
        q_width = int(preds_nn["interval_width"].iloc[0]) if preds_nn is not None else (pred_q75 - pred_q25)
        
        # Historical context
        h_stats = hist_map.get(home_team, {"avg_att": capacity * 0.5, "max_att": capacity})
        hist_avg = int(h_stats["avg_att"])
        
        st.markdown("---")
        st.markdown("### 🧠 Wynik Predykcji (Tabular Quantile NN & Hybrid Ensemble)")
        
        cqr_badge = ""
        if hasattr(nn_model, "conformal_delta_"):
            cqr_badge = f" | 🛡️ **CQR Kalibracja (IQR)**: Δ = {nn_model.conformal_delta_:+.0f} widzów"
        is_hybrid = isinstance(nn_model, (HybridEnsembleQuantileModel, NeuralEnsembleQuantileModel)) or hasattr(nn_model, "models_")
        model_type_label = "Komitet Hybrydowy (PyTorch + CatBoost + LightGBM)" if is_hybrid else "Sieć Neuronowa TabularQuantileNet (PyTorch)"
        st.caption(f"Architektura predyktora: **{model_type_label}**{cqr_badge}")
        
        col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 1.4])
        with col_res1:
            st.metric(
                label="Przewidywana liczba widzów (Mediana Q50)",
                value=f"{main_pred_att:,} widzów",
                delta=f"{main_pred_att - hist_avg:+d} vs średnia klubu ({hist_avg:,})"
            )
        with col_res2:
            st.metric(
                label="Szacowane zapełnienie obiektu",
                value=f"{main_pred_occ*100:.1f}%",
                delta=f"{main_pred_att:,} / {capacity:,} miejsc"
            )
        with col_res3:
            st.metric(
                label="Wąski przedział typowej frekwencji (Q25 – Q75)",
                value=f"{pred_q25:,} – {pred_q75:,}",
                delta=f"Rozpiętość IQR: {q_width:,} widzów",
                delta_color="off"
            )
            
        # Stakes and Momentum Badge
        stakes_desc = []
        if is_derby:
            stakes_desc.append("🔥 Mecz Derbowy / Klasyk")
        if is_title_or_promotion_race:
            stakes_desc.append("🏆 Walka o Mistrzostwo / Awans")
        elif is_relegation_battle:
            stakes_desc.append("⚠️ Bój o utrzymanie")
        elif is_dead_rubber:
            stakes_desc.append("🥒 Mecz o pietruszkę (koniec sezonu)")
        
        stakes_info = " | ".join(stakes_desc) if stakes_desc else "Standardowa stawka ligowa"
        st.caption(f"📌 **Parametry meczowe**: Medialność rywala: **{away_draw_factor:.2f}x** | H2H na tym stadionie (OOF): **{h2h_venue_avg:,.0f} widzów** | Wskaźnik stawki: **{match_stakes_index:.2f}x** ({stakes_info})")
            
        # Progress bar
        st.progress(min(1.0, max(0.0, main_pred_occ)))
        
        # Summary metrics
        st.markdown("#### Podsumowanie predykcji kwantylowej (Wąski przedział IQR)")
        comp_df = pd.DataFrame([
            {"Kwantyl": "Dolny próg typowy (Q25)", "Liczba widzów": f"{pred_q25:,} widzów", "Prawdopodobieństwo": "75% szans, że frekwencja będzie wyższa", "Zastosowanie": "Realistyczna dolna granica przychodów biletowych"},
            {"Kwantyl": "Mediana prognozy (Q50)", "Liczba widzów": f"{main_pred_att:,} widzów", "Prawdopodobieństwo": "Najbardziej prawdopodobna frekwencja meczowa", "Zastosowanie": "Główny punkt odniesienia do logistyki meczu"},
            {"Kwantyl": "Górny próg typowy (Q75)", "Liczba widzów": f"{pred_q75:,} widzów", "Prawdopodobieństwo": "25% szans, że frekwencja przewyższy ten próg", "Zastosowanie": "Planowanie obsługi stadionu i cateringu bez przeszacowań"}
        ])
        st.table(comp_df)

        # Local Feature Impact Breakdown (Explainable AI / Counterfactual Sensitivity)
        st.markdown("---")
        st.markdown("### 🔍 Co wpłynęło na tę prognozę? (Wpływ kluczowych czynników)")
        st.caption("Analiza wyjaśnialności (Explainable AI / XAI): zestawienie pokazuje, o ile widzów poszczególne parametry tego meczu podwyższyły (+) lub obniżyły (-) prognozę frekwencji w odniesieniu do warunków neutralnych.")

        feat_contrib_df = compute_local_feature_contributions(nn_model, single_df, main_pred_att)
        if not feat_contrib_df.empty:
            exp_col1, exp_col2 = st.columns([1.5, 1.5])
            with exp_col1:
                # Plotly horizontal bar chart
                df_plot = feat_contrib_df.iloc[::-1].copy()
                bar_colors = ["#10B981" if w >= 0 else "#EF4444" for w in df_plot["Wpływ"]]
                
                fig_exp = go.Figure(go.Bar(
                    x=df_plot["Wpływ"],
                    y=df_plot["Czynnik"],
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{w:+d} widzów" for w in df_plot["Wpływ"]],
                    textposition="auto"
                ))
                fig_exp.update_layout(
                    title="Wpływ poszczególnych parametrów na frekwencję",
                    xaxis_title="Wpływ na liczbę widzów (+/-)",
                    yaxis_title="",
                    height=360,
                    margin=dict(l=10, r=20, t=40, b=30)
                )
                st.plotly_chart(fig_exp, use_container_width=True)

            with exp_col2:
                st.markdown("##### 📋 Szczegółowe zestawienie czynników")
                for _, r in feat_contrib_df.iterrows():
                    badge = "🟢 +" if r["Wpływ"] >= 0 else "🔴 "
                    st.markdown(f"**{badge}{r['Wpływ']:,} widzów** — **{r['Czynnik']}**")
                    st.caption(f"Stan parametru: *{r['Wartość']}* • {r['Interpretacja']}")


# ==============================================================================
# TAB 2: WERYFIKACJA NA ZBIORZE WALIDACYJNYM / TESTOWYM
# ==============================================================================
with tab_validate:
    st.subheader("Weryfikacja na zbiorze walidacyjnym (stratyfikacja wg sezonów i klubów)")
    st.caption("Zbiór walidacyjny zawiera 20% losowo wybranych meczów z każdego z 5 sezonów (od 2022/23 do 2026/27), stratyfikowany według każdego klubu gospodarza. Żaden z poniższych meczów NIE BYŁ widziany przez sieć neuronową podczas treningu.")
    
    if val_df.empty:
        st.warning("Brak pliku `test_predictions.parquet`. Uruchom najpierw pipeline treningowy.")
    else:
        # Filter controls
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        with f_col1:
            tier_filter = st.selectbox("Filtruj wg ligi", ["Wszystkie ligi", "Ekstraklasa (Tier 1)", "1. Liga (Tier 2)", "2. Liga (Tier 3)"])
        with f_col2:
            available_seasons = ["Wszystkie sezony"] + sorted(list(val_df["season"].unique()), reverse=True)
            season_filter = st.selectbox("Filtruj wg sezonu", available_seasons)
        with f_col3:
            club_filter = st.selectbox("Filtruj wg klubu", ["Wszystkie kluby"] + sorted(list(val_df["home_team"].unique())))
        with f_col4:
            res_filter = st.selectbox("Typ meczu", ["Wszystkie mecze", "Tylko mecze rezerw (II zespoły)", "Tylko pierwsze zespoły (bez rezerw)"])
        with f_col5:
            sort_by = st.selectbox("Sortuj tabelę", [
                "Błąd bezwzględny (od największych)",
                "Błąd bezwzględny (od najmniejszych - idealne)",
                "Frekwencja rzeczywista (malejąco)",
                "Data meczu (od najnowszych)"
            ])
            
        p_col = "pred_att_nn" if "pred_att_nn" in val_df.columns else "pred_attendance"
        err_abs_col = "error_abs_nn" if "error_abs_nn" in val_df.columns else "error_abs"
        err_pct_col = "error_pct_nn" if "error_pct_nn" in val_df.columns else "error_pct"
        err_sgn_col = "error_signed_nn" if "error_signed_nn" in val_df.columns else "error_signed"

        # Apply filters
        df_filtered = val_df.copy()
            
        if "Ekstraklasa" in tier_filter:
            df_filtered = df_filtered[df_filtered["league_tier"] == 1]
        elif "1. Liga" in tier_filter:
            df_filtered = df_filtered[df_filtered["league_tier"] == 2]
        elif "2. Liga" in tier_filter:
            df_filtered = df_filtered[df_filtered["league_tier"] == 3]
            
        if season_filter != "Wszystkie sezony":
            df_filtered = df_filtered[df_filtered["season"] == season_filter]
            
        if club_filter != "Wszystkie kluby":
            df_filtered = df_filtered[(df_filtered["home_team"] == club_filter) | (df_filtered["away_team"] == club_filter)]
            
        if res_filter == "Tylko mecze rezerw (II zespoły)":
            if "is_reserve_fixture" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["is_reserve_fixture"] == 1]
            else:
                df_filtered = df_filtered[df_filtered["home_team"].str.contains(r"\bII\b") | df_filtered["away_team"].str.contains(r"\bII\b")]
        elif res_filter == "Tylko pierwsze zespoły (bez rezerw)":
            if "is_reserve_fixture" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["is_reserve_fixture"] == 0]
            else:
                df_filtered = df_filtered[~df_filtered["home_team"].str.contains(r"\bII\b") & ~df_filtered["away_team"].str.contains(r"\bII\b")]
            
        # Metrics summary on filtered set
        m1, m2, m3, m4, m5 = st.columns(5)
        mae_filtered = (df_filtered["attendance"] - df_filtered[p_col]).abs().mean()
        mape_filtered = ((df_filtered["attendance"] - df_filtered[p_col]).abs() / df_filtered["attendance"]).mean() * 100
        mean_actual = df_filtered["attendance"].mean()
        occ_mae = (df_filtered["attendance"] / df_filtered["capacity_resolved"] - df_filtered[p_col] / df_filtered["capacity_resolved"]).abs().mean() * 100
        
        with m1:
            st.metric("Liczba meczów", f"{len(df_filtered):,}")
        with m2:
            st.metric("Średnia frekwencja", f"{mean_actual:,.0f}")
        with m3:
            st.metric("Średni błąd (MAE)", f"{mae_filtered:,.0f} widzów")
        with m4:
            st.metric("Błąd procentowy (MAPE)", f"{mape_filtered:.1f}%")
        with m5:
            st.metric("Średni błąd zapełnienia", f"{occ_mae:.1f}% stadionu")
            
        st.markdown("---")
        
        # Visualizations
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### 🎯 Frekwencja Rzeczywista vs Predykcja Sieci PyTorch")
            fig_scatter = px.scatter(
                df_filtered,
                x="attendance",
                y=p_col,
                color="league_name",
                hover_data=["match_date", "home_team", "away_team"],
                labels={"attendance": "Rzeczywista liczba widzów", p_col: "Predykcja sieci neuronowej (Q50)"},
                opacity=0.75
            )
            # Add y=x ideal line
            max_val = max(df_filtered["attendance"].max(), df_filtered[p_col].max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Idealna predykcja (y=x)",
                line=dict(color="red", dash="dash")
            ))
            fig_scatter.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_scatter, width="stretch")
            
        with chart_col2:
            st.markdown("##### 📉 Rozkład Błędów Predykcji (Rzeczywista - Sieć PyTorch)")
            df_filtered["current_error_signed"] = df_filtered["attendance"] - df_filtered[p_col]
            fig_hist = px.histogram(
                df_filtered,
                x="current_error_signed",
                nbins=40,
                color="league_name",
                labels={"current_error_signed": "Różnica (Rzeczywista - Predykcja)"},
                marginal="box"
            )
            fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hist, width="stretch")
            
        # Highlight: Best predictions vs Worst Outliers
        st.markdown("---")
        st.markdown("#### 🔍 Analiza Skrajnych Przypadków")
        out_col1, out_col2 = st.columns(2)
        
        with out_col1:
            st.markdown("##### 🟢 Najbardziej trafione predykcje (Top 5)")
            top_accurate = df_filtered.sort_values(by=err_abs_col).head(5)[[
                "match_date", "league_name", "home_team", "away_team", "attendance", p_col, err_abs_col, err_pct_col
            ]].copy()
            top_accurate.columns = ["Data", "Liga", "Gospodarz", "Gość", "Realna", "Predykcja NN", "Błąd (widzowie)", "Błąd (%)"]
            st.dataframe(top_accurate.style.format({"Realna": "{:,}", "Predykcja NN": "{:,.0f}", "Błąd (widzowie)": "{:,.0f}", "Błąd (%)": "{:.1f}%"}))
            
        with out_col2:
            st.markdown("##### 🔴 Największe odchyły / anomalie (Top 5)")
            top_err = df_filtered.sort_values(by=err_abs_col, ascending=False).head(5)[[
                "match_date", "league_name", "home_team", "away_team", "attendance", p_col, err_abs_col, err_pct_col
            ]].copy()
            top_err.columns = ["Data", "Liga", "Gospodarz", "Gość", "Realna", "Predykcja NN", "Błąd (widzowie)", "Błąd (%)"]
            st.dataframe(top_err.style.format({"Realna": "{:,}", "Predykcja NN": "{:,.0f}", "Błąd (widzowie)": "{:,.0f}", "Błąd (%)": "{:.1f}%"}))
            
        # Detailed Table
        st.markdown("---")
        st.markdown("#### 📋 Szczegółowe zestawienie wszystkich meczów w próbie")
        
        # Sorting
        if "od największych" in sort_by:
            df_display = df_filtered.sort_values(by=err_abs_col, ascending=False)
        elif "idealne trafienia" in sort_by or "od najmniejszych" in sort_by:
            df_display = df_filtered.sort_values(by=err_abs_col, ascending=True)
        elif "Frekwencja rzeczywista" in sort_by:
            df_display = df_filtered.sort_values(by="attendance", ascending=False)
        else:
            df_display = df_filtered.sort_values(by=["match_date", "match_time"], ascending=False)
            
        display_cols = [
            "match_date", "match_time", "league_name", "home_team", "away_team", 
            "attendance", p_col, err_abs_col, err_pct_col,
            "capacity_resolved"
        ]
        q_low_col = "pred_q25_nn" if "pred_q25_nn" in df_display.columns else ("pred_q25" if "pred_q25" in df_display.columns else ("pred_q10_nn" if "pred_q10_nn" in df_display.columns else "pred_q10"))
        q_high_col = "pred_q75_nn" if "pred_q75_nn" in df_display.columns else ("pred_q75" if "pred_q75" in df_display.columns else ("pred_q90_nn" if "pred_q90_nn" in df_display.columns else "pred_q90"))
        has_quantiles = q_low_col in df_display.columns and q_high_col in df_display.columns
        if has_quantiles:
            display_cols.extend([q_low_col, q_high_col])
            
        table_view = df_display[display_cols].copy()
        table_view["real_occ_pct"] = (table_view["attendance"] / table_view["capacity_resolved"]) * 100
        table_view["pred_occ_pct"] = (table_view[p_col] / table_view["capacity_resolved"]) * 100
        
        if has_quantiles:
            table_view["interval_iqr"] = table_view.apply(lambda r: f"{int(r[q_low_col]):,} – {int(r[q_high_col]):,}", axis=1)
            final_cols = [
                "match_date", "match_time", "league_name", "home_team", "away_team",
                "attendance", p_col, err_abs_col, err_pct_col, "interval_iqr",
                "real_occ_pct", "pred_occ_pct"
            ]
            col_names = [
                "Data", "Godz.", "Liga", "Gospodarz", "Gość",
                "Frekwencja Realna", "Predykcja NN", "Różnica", "Błąd %", "Przedział IQR (Q25 - Q75)",
                "Zapełnienie Realne %", "Zapełnienie Predykcja %"
            ]
        else:
            final_cols = [
                "match_date", "match_time", "league_name", "home_team", "away_team",
                "attendance", p_col, err_abs_col, err_pct_col,
                "real_occ_pct", "pred_occ_pct"
            ]
            col_names = [
                "Data", "Godz.", "Liga", "Gospodarz", "Gość",
                "Frekwencja Realna", "Predykcja NN", "Różnica", "Błąd %",
                "Zapełnienie Realne %", "Zapełnienie Predykcja %"
            ]
            
        table_view = table_view[final_cols]
        table_view.columns = col_names
        
        st.dataframe(
            table_view.style.format({
                "Frekwencja Realna": "{:,}",
                "Predykcja NN": "{:,.0f}",
                "Różnica": "{:,.0f}",
                "Błąd %": "{:.1f}%",
                "Zapełnienie Realne %": "{:.1f}%",
                "Zapełnienie Predykcja %": "{:.1f}%"
            }),
            height=400,
            width="stretch"
        )


# ==============================================================================
# TAB 3: INFORMACJE O MODELU I INŻYNIERII CECH
# ==============================================================================
with tab_info:
    st.subheader("Architektura Głębokiej Sieci Neuronowej (PyTorch)")
    st.subheader("Architektura Głębokiej Sieci Neuronowej i Komitetu Hybrydowego")
    
    st.markdown(r"""
    Projekt w całości opiera się na dedykowanej sieci neuronowej dla danych tabelarycznych: **TabularQuantileNN w PyTorch**.
    Projekt opiera się na nowoczesnej architekturze regresji kwantylowej dla danych tabelarycznych: **TabularQuantileNet w PyTorch** oraz **Komitecie Hybrydowym (PyTorch + CatBoost + LightGBM)** z certyfikacją **Conformalized Quantile Regression (CQR)**.
    
    ##### 1. Architektura Sieci Neuronowej (TabularQuantileNet):
    * **Entity Embeddings**: Kluby gospodarzy i gości reprezentowane są przez wektory w przestrzeni wielowymiarowej ($d=16$). Poziom ligi reprezentowany jest przez wektor ($d=4$). Pozwala to sieci uczyć się unikalnych relacji klubowych, bazy kibiców i renomy rywala.
    * **Zintegrowany StandardScaler**: Cechy numeryczne (kalendarzowe, pogodowe, odległościowe i sportowe) są automatycznie normalizowane wewnątrz tensora przed wejściem do warstw gęstych.
    * **Bloki gęste Dense (MLP)**: Ukryte warstwy o wymiarach **128 $\to$ 64 $\to$ 32**, gdzie każda warstwa posiada:
      - Połączenie liniowe (`nn.Linear`)
      - Normalizację wsadową (`nn.BatchNorm1d`) stabilizującą gradienty
      - Aktywację `ReLU`
      - Regularyzację `Dropout` (0.20 $\to$ 0.15 $\to$ 0.10) chroniącą przed przeuczeniem
    * **Jednoczesna Regresja Wielokwantylowa (Pinball Loss)**: Głowica wyjściowa sieci posiada 3 neurony odpowiadające kwantylom $[Q_{25}, Q_{50}, Q_{75}]$ (wąski przedział IQR).
      $$L_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$$
    * **Monotoniczne sortowanie wierszowe (Brak quantile crossing)**: Wektory wyjściowe są sortowane wierszowo, gwarantując $Q_{25} \le Q_{50} \le Q_{75}$.
    * **Niezależność od pojemności stadionu**: Pojemność obiektu została celowo wykluczona z cech uczących sieci – sieć prognozuje czysty popyt społeczny na mecz na bazie tożsamości klubów, medialności rywala i czynników sportowych.
    * **Strukturalnie Monotoniczna Głowica Wyjściowa (Brak Quantile Crossing)**:
      Sieć zamiast 3 niezależnych wyjść predykuje medianę $z_1 = \hat{y}_{Q50}$ oraz nieujemne spready modelowane przez funkcję Softplus:
      $$\hat{y}_{Q25} = z_1 - (\text{Softplus}(z_2) + 10^{-4})$$
      $$\hat{y}_{Q75} = z_1 + (\text{Softplus}(z_3) + 10^{-4})$$
      Dzięki temu $Q_{25} \le Q_{50} \le Q_{75}$ zachodzi w każdym punkcie przestrzeni wejściowej bezpośrednio z topologii grafu obliczeniowego PyTorch.
    * **Dekompozycja Embeddingów Klubowych**:
      Rozbicie reprezentacji każdego klubu na:
      - Wektor bazowej tożsamości i renomy klubu `shared_club_emb` ($d=10$)
      - Rola gospodarza `home_role_emb` ($d=6$)
      - Rola gościa `away_role_emb` ($d=6$)
      Zapewnia to transfer wiedzy między rolami oraz UNK dropout (5%) uodparniający model na beniaminków i kluby z niższych lig.
    * **Bloki Rezydualne (TabularResidualBlock) i Aktywacja SiLU**:
      Gęste warstwy z połączeniami rezydualnymi (skip connections), normalizacją wsadową `BatchNorm1d`, aktywacją `SiLU` (Swish) oraz regularyzacją `Dropout` (0.20 $\to$ 0.15 $\to$ 0.10).
    * **Funkcja Straty CensoredSmoothPinballLoss**:
      Gładka funkcja Pinball Loss z aproksymacją Hubera wokół zera (stabilne gradienty w punkcie minimum) oraz jednostronną cenzurą prawostronną dla meczów wyprzedanych do pojemności nominalnej obiektu.
    * **Kalibracja CQR (Conformalized Quantile Regression)**:
      Ocena finite-sample błędu niepokrycia na zbiorze walidacyjnym stratyfikowanym z całego spektrum meczowego i korekta granic $\hat{\Delta}$, gwarantująca certyfikowane pokrycie 50% wąskiego przedziału ufności IQR na nieznanych meczach.

    ##### 2. Komitet Hybrydowy (HybridEnsembleQuantileModel):
    * **3x TabularQuantileNet (PyTorch, waga: 0.50)** – różnorodność inicjalizacji wag i losowości optymalizacji AdamW.
    * **1x CatBoostRegressor (waga: 0.30)** – z funkcją celu `MultiQuantile:alpha=0.25,0.5,0.75` i natywną obsługą relacji kategorycznych.
    * **1x LightGBMRegressor (waga: 0.20)** – z estymacją kwantyli Q25, Q50, Q75 i fallbackiem.
    
    ##### 2. Kluczowe cechy wejściowe sieci:
    * **Tożsamość klubowa i rywal**: Entity Embeddings dla gospodarza i gościa.
    * **Dwukierunkowe Kodowanie Relacyjne H2H (Out-of-Fold)**: Średnia frekwencja danej pary rywali oraz na danym obiekcie liczona bez wycieku danych.
    * **Wskaźnik medialności gościa (Away Draw Factor OOF)**: Mnożnik zainteresowania przyjazdem rywala (np. Wisła: 1.87x, Lech: 1.40x, Legia: 1.35x).
    ##### 3. Kluczowe cechy inżynierii dziedzinowej:
    * **Tożsamość klubowa i rywal**: Wektory embeddingów dla gospodarza, gościa i poziomu ligi.
    * **Cechy dynamiczne formy i frekwencji**: Wykładnicze średnie ruchome `home_att_ema_3` i `home_occupancy_ema_5`.
    * **Efekt nowej trybuny / stadionu**: Wskaźnik `stadium_expansion_boost`.
    * **Sektor gości i zakazy**: Flaga `away_sector_closed`.
    * **Kalendarz i inauguracja**: Flaga `is_round_opener` (pierwsze kolejki po przerwie zimowej i letniej).
    * **Komfort pogodowy**: Wskaźnik `effective_weather_discomfort` (interakcja chłodu, deszczu i stopnia zadaszenia stadionu).
    * **Dwukierunkowe Kodowanie Relacyjne H2H (Out-of-Fold)**: Średnia frekwencja danej pary rywali oraz na danym obiekcie.
    * **Wskaźnik medialności gościa (Away Draw Factor OOF)**: Mnożnik zainteresowania przyjazdem rywala.
    * **Stawka meczu (`match_stakes_index`)**: Indeks uwzględniający walkę o mistrzostwo/awans, bój o utrzymanie, mecze o pietruszkę oraz derby (+40%).
    * **Drużyny Rezerw (II zespoły)**: Ścisłe rozróżnienie pierwszego zespołu od rezerw flagą `is_reserve_fixture` i osobnymi indeksami klubowymi.
    * **Warunki pogodowe (Open-Meteo)**: Temperatura, deszcz, wiatr, zachmurzenie i stopień zadaszenia stadionu.
    * **Dystans kibiców gości**: Odległość geograficzna między miastami w km (formuła Haversine'a).
    * **Drużyny Rezerw (II zespoły)**: Ścisłe rozróżnienie pierwszego zespołu od rezerw flagą `is_reserve_fixture`.
    """)
    
    if nn_model:
        st.markdown("##### Ranking Ważności Cech w Sieci Neuronowej (Permutation Feature Importance):")
        st.caption("Ważność cech wyznaczona metodą Permutation Feature Importance (PFI) – wskazuje wzrost błędu MAE [w liczbie widzów] po wymieszaniu danej cechy na próbie testowej.")
        
        name_map = {
            "home_team": "Gospodarz meczu (Entity Embedding)",
            "league_tier": "Poziom ligi (Ekstraklasa / 1. Liga / 2. Liga)",
            "roof_pct": "Zadaszenie stadionu (% dachu)",
            "h2h_avg_att": "Historia meczów H2H (Średnia frekwencja)",
            "h2h_home_venue_avg_att": "Historia meczów H2H na tym stadionie",
            "away_team": "Gość meczu (Entity Embedding)",
            "away_draw_factor": "Współczynnik medialności gościa (Away Draw Factor)",
            "home_pts_last_3_home": "Forma gospodarza u siebie (punkty w 3 meczach)",
            "home_table_pos": "Pozycja gospodarza w tabeli",
            "home_pts_last_5": "Forma gospodarza (punkty w 5 meczach)",
            "temperature_c": "Temperatura powietrza (°C)",
            "away_table_pos": "Pozycja gościa w tabeli",
            "is_home_reserve": "Flaga drużyny rezerw (II zespół)",
            "wind_speed_kmh": "Prędkość wiatru (km/h)",
            "hour_num": "Godzina rozpoczęcia meczu",
            "travel_distance_km": "Dystans wyjazdowy kibiców gości (km)",
            "is_major_derby": "Wielkie derby / Mecz prestiżowy",
            "match_stakes_index": "Indeks stawki meczu (tytuł/spadek)",
            "precipitation_mm": "Opady deszczu (mm)",
            "is_weekend": "Mecz w weekend (sobota/niedziela)",
            "cloud_cover_pct": "Zachmurzenie (%)",
            "home_att_ema_3": "Wykładnicza średnia frekwencji gospodarza (EMA 3)",
            "home_occupancy_ema_5": "Wykładnicza średnia zapełnienia stadionu (EMA 5)",
            "effective_weather_discomfort": "Dyskomfort pogodowy (deszcz × chłód × brak dachu)",
            "is_round_opener": "Inauguracja rundy (początek sezonu / runda wiosenna)",
            "away_sector_closed": "Zamknięty sektor gości / zakaz wyjazdowy",
            "stadium_expansion_boost": "Efekt otwarcia nowego obiektu / nowej trybuny"
        }
        
        try:
            sample_val = val_df.sample(min(300, len(val_df)), random_state=42) if not val_df.empty else pd.DataFrame()
            df_imp_nn = nn_model.get_feature_importances(sample_val, method="permutation", n_repeats=2).head(12)
            df_imp_nn["feature_name"] = df_imp_nn["feature"].map(lambda x: name_map.get(x, x))
            
            fig_bar_nn = px.bar(
                df_imp_nn,
                x="importance",
                y="feature_name",
                orientation="h",
                title="Ważność cech w Sieci Neuronowej PyTorch (Wzrost MAE po zaburzeniu cechy)",
                labels={"importance": "Wzrost średniego błędu MAE (widzowie)", "feature_name": "Cecha"},
                color="importance",
                color_continuous_scale="Viridis"
            )
            fig_bar_nn.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig_bar_nn, width="stretch")
        except Exception as e:
            st.caption(f"Wykres ważności cech niedostępny dla tego modelu ({e}).")
