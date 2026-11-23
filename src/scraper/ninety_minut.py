"""Scraper for 90minut.pl match data and reports across Polish leagues."""
import hashlib
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
from bs4 import BeautifulSoup
import sqlite3

from src.db.schema import get_connection, DEFAULT_DB_PATH
from src.scraper.stadiums import resolve_stadium, normalize_club_name

BASE_URL = "http://www.90minut.pl"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}

# League configurations for all target seasons (standardized naming across eras)
LEAGUE_CONFIG = {
    "2022/2023": {
        1: {"name": "Ekstraklasa", "path": "/liga/1/liga12330.html"},
        2: {"name": "1. Liga", "path": "/liga/1/liga12331.html"},
        3: {"name": "2. Liga", "path": "/liga/1/liga12332.html"},
    },
    "2023/2024": {
        1: {"name": "Ekstraklasa", "path": "/liga/1/liga12904.html"},
        2: {"name": "1. Liga", "path": "/liga/1/liga12905.html"},
        3: {"name": "2. Liga", "path": "/liga/1/liga12906.html"},
    },
    "2024/2025": {
        1: {"name": "Ekstraklasa", "path": "/liga/1/liga13482.html"},
        2: {"name": "1. Liga", "path": "/liga/1/liga13483.html"},
        3: {"name": "2. Liga", "path": "/liga/1/liga13484.html"},
    },
    "2025/2026": {
        1: {"name": "Ekstraklasa", "path": "/liga/1/liga14072.html"},
        2: {"name": "1. Liga", "path": "/liga/1/liga14073.html"},
        3: {"name": "2. Liga", "path": "/liga/1/liga14074.html"},
    },
    "2026/2027": {
        1: {"name": "Ekstraklasa", "path": "/liga/1/liga14675.html"},
        2: {"name": "1. Liga", "path": "/liga/1/liga14676.html"},
        3: {"name": "2. Liga", "path": "/liga/1/liga14677.html"},
    },
}

POLISH_MONTHS = {
    "stycznia": 1,
    "lutego": 2,
    "marca": 3,
    "kwietnia": 4,
    "maja": 5,
    "czerwca": 6,
    "lipca": 7,
    "sierpnia": 8,
    "września": 9,
    "wrzenia": 9,
    "października": 10,
    "padziernika": 10,
    "listopada": 11,
    "grudnia": 12,
}


def _get_cache_path(url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{url_hash}.html"


def fetch_url(url: str, use_cache: bool = True, delay: float = 0.25) -> Optional[str]:
    """Fetch URL with disk caching and ISO-8859-2 encoding support."""
    full_url = url if url.startswith("http") else f"{BASE_URL}{url}"
    cache_path = _get_cache_path(full_url)
    
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
            
    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            resp.encoding = "iso-8859-2"
            content = resp.text
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
            time.sleep(delay)
            return content
        else:
            print(f"Failed to fetch {full_url}: status {resp.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {full_url}: {e}")
        return None


def parse_date_and_time(raw_dt: str, season: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse raw date string e.g. '18 lipca, 19:00' or '11 września 2022, 17:30'
    Returns (YYYY-MM-DD, HH:MM, ISO datetime).
    """
    clean = raw_dt.replace("\xa0", " ").strip()
    
    # 1. Full date with explicit year: e.g. 11 września 2022, 17:30
    match_full = re.search(r"(\d{1,2})\s+([^\s\d,]+)\s+(\d{4}),?\s+(\d{1,2}):(\d{2})", clean)
    if match_full:
        day, month_str, year, hour, minute = match_full.groups()
        month = POLISH_MONTHS.get(month_str.lower())
        if month:
            d_str = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            t_str = f"{int(hour):02d}:{int(minute):02d}"
            return d_str, t_str, f"{d_str}T{t_str}:00"
            
    # 2. League round table date: e.g. '18 lipca, 19:00'
    match_partial = re.search(r"(\d{1,2})\s+([^\s\d,]+),?\s+(\d{1,2}):(\d{2})", clean)
    if match_partial:
        day, month_str, hour, minute = match_partial.groups()
        month = POLISH_MONTHS.get(month_str.lower())
        if month:
            years = [int(y) for y in season.split("/")]
            # In Polish football calendar: July-Dec is years[0], Jan-June is years[1]
            year = years[0] if month >= 7 else years[1]
            d_str = f"{year:04d}-{int(month):02d}-{int(day):02d}"
            t_str = f"{int(hour):02d}:{int(minute):02d}"
            return d_str, t_str, f"{d_str}T{t_str}:00"
            
    # 3. Only date without time: e.g. '18 lipca'
    match_date_only = re.search(r"(\d{1,2})\s+([^\s\d,]+)", clean)
    if match_date_only:
        day, month_str = match_date_only.groups()
        month = POLISH_MONTHS.get(month_str.lower())
        if month:
            years = [int(y) for y in season.split("/")]
            year = years[0] if month >= 7 else years[1]
            d_str = f"{year:04d}-{int(month):02d}-{int(day):02d}"
            return d_str, "17:00", f"{d_str}T17:00:00"
            
    return None, None, None


def parse_attendance_string(raw: str) -> Optional[int]:
    """
    Extract attendance integer from raw text like '( 16 748 )' or '(1014)'.
    On 90minut.pl league round tables, attendance is STRICTLY enclosed in parentheses!
    Never match raw digits outside parentheses, otherwise the day of the date (e.g. '3 listopada')
    gets falsely parsed as attendance!
    """
    clean = raw.replace("\xa0", " ").strip()
    m = re.search(r"\(\s*([\d\s]+)\s*\)", clean)
    if m:
        num_str = m.group(1).replace(" ", "")
        if num_str.isdigit():
            val = int(num_str)
            if val > 0:
                return val
    return None


def parse_match_details_page(match_id: int, use_cache: bool = True) -> Dict[str, Any]:
    """Parse match report from mecz.php to get attendance, referee, reported temp, news url."""
    url = f"/mecz.php?id_mecz={match_id}"
    html = fetch_url(url, use_cache=use_cache)
    if not html:
        return {}
        
    soup = BeautifulSoup(html, "html.parser")
    res = {}
    
    # 1. Closed doors / bez udziału publiczności check
    text_lower = soup.get_text(" ").lower()
    if "bez udziału publiczności" in text_lower or "bez publiczności" in text_lower:
        res["attendance"] = 0
        res["is_outlier"] = 1
        res["outlier_reason"] = "Mecz bez udziału publiczności"
    else:
        # 2. Official attendance next to att.gif icon
        att_img = soup.find("img", src=lambda s: s and "att.gif" in s)
        if att_img:
            ns = att_img.next_sibling
            if ns:
                m = re.search(r"(\d[\d\s]*\d|\d+)", str(ns))
                if m:
                    res["attendance"] = int(m.group(1).replace(" ", ""))
            if "attendance" not in res and att_img.parent:
                m_parent = re.search(r"att\.gif[^>]*>[^\d]*(\d[\d\s]*\d|\d+)", str(att_img.parent))
                if m_parent:
                    res["attendance"] = int(m_parent.group(1).replace(" ", ""))
    
    # Referee
    ref_img = soup.find("img", src=lambda s: s and "ref.jpg" in s)
    if ref_img:
        ref_a = ref_img.find_next("a")
        if ref_a:
            res["referee"] = ref_a.text.strip()
            
    # News link
    news_a = soup.find("a", href=lambda s: s and "/news/" in s)
    if news_a:
        res["news_url"] = news_a["href"]
        
    return res


def parse_news_page(news_url: str, use_cache: bool = True) -> Dict[str, Any]:
    """Parse news article for stadium and city."""
    html = fetch_url(news_url, use_cache=use_cache)
    if not html:
        return {}
        
    soup = BeautifulSoup(html, "html.parser")
    res = {}
    
    for line in soup.get_text("\n").split("\n"):
        line = line.strip()
        if "(" in line and ")" in line and any(k in line.lower() for k in ["stadion", "arena", "miejski", "sportow", "mosir", "gosir"]) and len(line) < 120:
            res["stadium_line"] = line
            # Check city vs stadium: e.g. "Częstochowa (Miejski Stadion Piłkarski Raków)"
            m = re.match(r"([^()]+)\s*\(([^()]+)\)", line)
            if m:
                res["city_reported"] = m.group(1).strip()
                res["stadium_reported"] = m.group(2).strip()
            break
            
    return res


def scrape_league_season(
    season: str,
    tier: int,
    enrich_news: bool = False,
    db_path: Optional[Path] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Scrape all matches for a given league and season and insert/update in SQLite.
    Returns list of parsed match records.
    """
    cfg = LEAGUE_CONFIG.get(season, {}).get(tier)
    if not cfg:
        raise ValueError(f"Unknown configuration for season {season}, tier {tier}")
        
    league_name = cfg["name"]
    league_path = cfg["path"]
    
    print(f"Scraping {season} - Tier {tier} ({league_name})...")
    html = fetch_url(league_path, use_cache=use_cache)
    if not html:
        print(f"Failed to fetch league page {league_path}")
        return []
        
    soup = BeautifulSoup(html, "html.parser")
    kolejka_matches = []
    
    # Locate all Kolejka headers
    kolejka_elements = soup.find_all(string=re.compile(r"Kolejka\s+\d+"))
    
    conn = get_connection(db_path)
    inserted_matches = []
    
    for k_elem in kolejka_elements:
        k_text = k_elem.strip()
        m_num = re.search(r"Kolejka\s+(\d+)", k_text)
        kolejka_num = int(m_num.group(1)) if m_num else None
        
        # The matches table follows the kolejka header
        parent_tr = k_elem.find_parent("tr")
        if not parent_tr:
            continue
        parent_table = parent_tr.find_parent("table")
        if not parent_table:
            continue
            
        matches_table = parent_table.find_next("table")
        if not matches_table:
            continue
            
        rows = matches_table.find_all("tr")
        for row in rows:
            score_a = row.find("a", href=lambda h: h and "id_mecz=" in h)
            if not score_a:
                continue
                
            tds = row.find_all("td")
            if len(tds) < 4:
                continue
                
            # Extract match ID
            m_id_match = re.search(r"id_mecz=(\d+)", score_a["href"])
            if not m_id_match:
                continue
            match_id = int(m_id_match.group(1))
            
            home_team = tds[0].get_text(" ", strip=True)
            score_text = tds[1].get_text(" ", strip=True)
            away_team = tds[2].get_text(" ", strip=True)
            dt_att_text = tds[3].get_text(" ", strip=True)
            
            # Scores
            home_score, away_score = None, None
            score_m = re.match(r"(\d+)\s*-\s*(\d+)", score_text)
            if score_m:
                home_score = int(score_m.group(1))
                away_score = int(score_m.group(2))
                
            # Date & time
            m_date, m_time, m_datetime = parse_date_and_time(dt_att_text, season)
            attendance = parse_attendance_string(dt_att_text)
            
            # If match has not been played yet (e.g. ongoing season), attendance may be None
            match_record = {
                "match_id": match_id,
                "season": season,
                "league_tier": tier,
                "league_name": league_name,
                "matchday": kolejka_num,
                "match_date": m_date,
                "match_time": m_time,
                "match_datetime": m_datetime,
                "home_team": normalize_club_name(home_team),
                "away_team": normalize_club_name(away_team),
                "home_score": home_score,
                "away_score": away_score,
                "attendance": attendance,
                "stadium_reported": None,
                "city_reported": None,
                "capacity_resolved": None,
                "occupancy_pct": None,
                "referee": None,
                "news_url": None,
                "is_outlier": 0,
                "outlier_reason": None,
            }
            
            # Enrich from details if attendance or date needs verification
            if enrich_news or attendance is None:
                details = parse_match_details_page(match_id, use_cache=use_cache)
                if details.get("attendance") is not None:
                    attendance = details["attendance"]
                    match_record["attendance"] = attendance
                if details.get("is_outlier"):
                    match_record["is_outlier"] = details["is_outlier"]
                    match_record["outlier_reason"] = details.get("outlier_reason")
                if details.get("referee"):
                    match_record["referee"] = details["referee"]
                if details.get("news_url"):
                    match_record["news_url"] = details["news_url"]
                    if enrich_news:
                        news_info = parse_news_page(details["news_url"], use_cache=use_cache)
                        match_record["city_reported"] = news_info.get("city_reported")
                        match_record["stadium_reported"] = news_info.get("stadium_reported")
                        
            # Resolve stadium capacity
            if m_date:
                stadium_info = resolve_stadium(
                    match_record["home_team"],
                    m_date,
                    match_record["stadium_reported"],
                    db_path=db_path
                )
                cap = stadium_info["capacity"]
                match_record["capacity_resolved"] = cap
                if not match_record["city_reported"]:
                    match_record["city_reported"] = stadium_info["city"]
                if not match_record["stadium_reported"]:
                    match_record["stadium_reported"] = stadium_info["stadium_name"]
                    
                if attendance is not None and cap > 0:
                    match_record["occupancy_pct"] = round(attendance / cap, 4)
                    
            # Outlier detection: 0 attendance or closed stadium
            if attendance == 0:
                match_record["is_outlier"] = 1
                match_record["outlier_reason"] = "Empty stadium / 0 attendance reported"
                
            # Upsert into database
            with conn:
                conn.execute(
                    """
                    INSERT INTO matches 
                    (match_id, season, league_tier, league_name, matchday, match_date, match_time, match_datetime,
                     home_team, away_team, home_score, away_score, stadium_reported, city_reported,
                     attendance, capacity_resolved, occupancy_pct, referee, news_url, is_outlier, outlier_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id) DO UPDATE SET
                      season=excluded.season,
                      league_tier=excluded.league_tier,
                      matchday=excluded.matchday,
                      match_date=excluded.match_date,
                      match_time=excluded.match_time,
                      match_datetime=excluded.match_datetime,
                      home_team=excluded.home_team,
                      away_team=excluded.away_team,
                      home_score=excluded.home_score,
                      away_score=excluded.away_score,
                      stadium_reported=COALESCE(excluded.stadium_reported, matches.stadium_reported),
                      city_reported=COALESCE(excluded.city_reported, matches.city_reported),
                      attendance=COALESCE(excluded.attendance, matches.attendance),
                      capacity_resolved=COALESCE(excluded.capacity_resolved, matches.capacity_resolved),
                      occupancy_pct=COALESCE(excluded.occupancy_pct, matches.occupancy_pct),
                      referee=COALESCE(excluded.referee, matches.referee),
                      news_url=COALESCE(excluded.news_url, matches.news_url),
                      is_outlier=excluded.is_outlier,
                      outlier_reason=excluded.outlier_reason
                    """,
                    (
                        match_record["match_id"],
                        match_record["season"],
                        match_record["league_tier"],
                        match_record["league_name"],
                        match_record["matchday"],
                        match_record["match_date"],
                        match_record["match_time"],
                        match_record["match_datetime"],
                        match_record["home_team"],
                        match_record["away_team"],
                        match_record["home_score"],
                        match_record["away_score"],
                        match_record["stadium_reported"],
                        match_record["city_reported"],
                        match_record["attendance"],
                        match_record["capacity_resolved"],
                        match_record["occupancy_pct"],
                        match_record["referee"],
                        match_record["news_url"],
                        match_record["is_outlier"],
                        match_record["outlier_reason"],
                    )
                )
            inserted_matches.append(match_record)
            
    print(f"Finished {season} - Tier {tier}: {len(inserted_matches)} matches parsed & saved.")
    return inserted_matches


def scrape_all_seasons(seasons: Optional[List[str]] = None, tiers: Optional[List[int]] = None, db_path: Optional[Path] = None) -> int:
    """Scrape all configured seasons and league tiers."""
    target_seasons = seasons or list(LEAGUE_CONFIG.keys())
    target_tiers = tiers or [1, 2, 3]
    total = 0
    for s in target_seasons:
        for t in target_tiers:
            matches = scrape_league_season(s, t, enrich_news=False, db_path=db_path)
            total += len(matches)
    return total


if __name__ == "__main__":
    # Test scraping on 2022/2023 Ekstraklasa
    res = scrape_league_season("2022/2023", 1)
    print(f"Sample scraped {len(res)} matches.")
    if res:
        print("Sample match 0:", res[0])

