"""Open-Meteo Historical Weather API client with caching and batch city fetching."""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import requests
import sqlite3

from src.db.schema import get_connection, DEFAULT_DB_PATH
from src.scraper.stadiums import resolve_stadium

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "weather"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_range(latitude: float, longitude: float, start_date: str, end_date: str) -> Optional[dict]:
    """Fetch hourly weather for a date range from Open-Meteo Historical API."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}_{start_date}_{end_date}.json"
    cache_file = CACHE_DIR / cache_key
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    params = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "rain", "weather_code", "wind_speed_10m", "cloud_cover"],
        "timezone": "Europe/Warsaw"
    }
    
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            time.sleep(0.3)
            return data
        else:
            print(f"Weather API status {resp.status_code} for {latitude},{longitude} ({start_date} to {end_date})")
            return None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def batch_enrich_weather(db_path: Optional[Path] = None, max_cities: Optional[int] = None) -> int:
    """
    Fetch and populate historical weather for all matches in the database
    by querying Open-Meteo once per unique city/stadium coordinate.
    """
    conn = get_connection(db_path)
    
    # 1. Find all matches needing weather
    query = """
    SELECT m.match_id, m.home_team, m.match_date, m.match_time, m.match_datetime
    FROM matches m
    LEFT JOIN weather w ON m.match_id = w.match_id
    WHERE w.match_id IS NULL AND m.match_date IS NOT NULL
    ORDER BY m.match_date ASC
    """
    matches_to_enrich = conn.execute(query).fetchall()
    if not matches_to_enrich:
        print("All matches already have weather data.")
        return 0
        
    print(f"Found {len(matches_to_enrich)} matches needing weather data.")
    
    # 2. Group matches by rounded coordinates
    loc_groups: Dict[Tuple[float, float], List[dict]] = {}
    for m in matches_to_enrich:
        st = resolve_stadium(m["home_team"], m["match_date"], db_path=db_path)
        lat = round(st["latitude"], 3)
        lon = round(st["longitude"], 3)
        key = (lat, lon)
        if key not in loc_groups:
            loc_groups[key] = []
        loc_groups[key].append(dict(m))
        
    total_locations = len(loc_groups)
    print(f"Matches grouped into {total_locations} distinct locations.")
    
    weather_insert_batch = []
    processed_locations = 0
    
    for (lat, lon), m_list in loc_groups.items():
        if max_cities and processed_locations >= max_cities:
            break
            
        processed_locations += 1
        dates = [m["match_date"] for m in m_list]
        min_date = min(dates)
        max_date = max(dates)
        
        # Split into yearly chunks if range > 365 days
        d_min = datetime.strptime(min_date, "%Y-%m-%d")
        d_max = datetime.strptime(max_date, "%Y-%m-%d")
        
        hourly_map = {}
        curr_start = d_min
        while curr_start <= d_max:
            curr_end = min(curr_start + timedelta(days=365), d_max)
            s_str = curr_start.strftime("%Y-%m-%d")
            e_str = curr_end.strftime("%Y-%m-%d")
            
            data = fetch_weather_range(lat, lon, s_str, e_str)
            if data and "hourly" in data:
                h = data["hourly"]
                t_list = h.get("time", [])
                temps = h.get("temperature_2m", [])
                precips = h.get("precipitation", [])
                rains = h.get("rain", [])
                wcodes = h.get("weather_code", [])
                winds = h.get("wind_speed_10m", [])
                clouds = h.get("cloud_cover", [])
                
                for i, t in enumerate(t_list):
                    time_prefix = t[:13]
                    hourly_map[time_prefix] = {
                        "temperature_c": temps[i] if i < len(temps) else None,
                        "precipitation_mm": precips[i] if i < len(precips) else 0.0,
                        "rain_mm": rains[i] if i < len(rains) else 0.0,
                        "weather_code": wcodes[i] if i < len(wcodes) else 0,
                        "wind_speed_kmh": winds[i] if i < len(winds) else 0.0,
                        "cloud_cover_pct": clouds[i] if i < len(clouds) else 0.0,
                    }
                    
            if curr_end >= d_max:
                break
            curr_start = curr_end + timedelta(days=1)
                
        # Now match each game to its weather
        for m in m_list:
            m_id = m["match_id"]
            m_date = m["match_date"]
            m_time = m["match_time"] or "17:00"
            try:
                hour = int(m_time.split(":")[0])
            except Exception:
                hour = 17
            target_key = f"{m_date}T{hour:02d}"
            
            w_rec = hourly_map.get(target_key)
            if not w_rec:
                for alt_h in [17, 18, 15, 12, 19]:
                    w_rec = hourly_map.get(f"{m_date}T{alt_h:02d}")
                    if w_rec:
                        break
                        
            if w_rec:
                weather_insert_batch.append((
                    m_id,
                    w_rec["temperature_c"],
                    w_rec["precipitation_mm"],
                    w_rec["rain_mm"],
                    w_rec["weather_code"],
                    w_rec["wind_speed_kmh"],
                    w_rec["cloud_cover_pct"],
                    "open-meteo"
                ))
                
        print(f"[{processed_locations}/{total_locations}] Processed location ({lat}, {lon}) - {len(m_list)} matches.")
        
    # Bulk insert
    if weather_insert_batch:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO weather
                (match_id, temperature_c, precipitation_mm, rain_mm, weather_code, wind_speed_kmh, cloud_cover_pct, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                weather_insert_batch
            )
        print(f"Successfully inserted weather for {len(weather_insert_batch)} matches.")
        
    return len(weather_insert_batch)


if __name__ == "__main__":
    batch_enrich_weather()
