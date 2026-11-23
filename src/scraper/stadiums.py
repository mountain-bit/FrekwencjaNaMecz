"""Stadium capacities and locations manager."""
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import sqlite3

from src.db.schema import get_connection, DEFAULT_DB_PATH

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "stadium_capacities.csv"

# Explicit mapping of short names used in 90minut to full standardized names
# MUST be exact full string matches, NEVER loose substring matches!
EXACT_CLUB_MAPPINGS = {
    # Ekstraklasa & 1 Liga & 2 Liga
    "raków częstochowa": "Raków Częstochowa",
    "raków": "Raków Częstochowa",
    "legia warszawa": "Legia Warszawa",
    "legia": "Legia Warszawa",
    "lech poznań": "Lech Poznań",
    "lechia gdańsk": "Lechia Gdańsk",
    "widzew łódź": "Widzew Łódź",
    "widzew": "Widzew Łódź",
    "śląsk wrocław": "Śląsk Wrocław",
    "śląsk": "Śląsk Wrocław",
    "górnik zabrze": "Górnik Zabrze",
    "górnik łęczna": "Górnik Łęczna",
    "górnik ł.": "Górnik Łęczna",
    "jagiellonia białystok": "Jagiellonia Białystok",
    "jagiellonia": "Jagiellonia Białystok",
    "cracovia": "Cracovia",
    "wisła kraków": "Wisła Kraków",
    "wisła płock": "Wisła Płock",
    "wisła puławy": "Wisła Puławy",
    "wisła p.": "Wisła Puławy",
    "zagłębie lubin": "Zagłębie Lubin",
    "zagłębie l.": "Zagłębie Lubin",
    "zagłębie sosnowiec": "Zagłębie Sosnowiec",
    "zagłębie s.": "Zagłębie Sosnowiec",
    "piast gliwice": "Piast Gliwice",
    "piast": "Piast Gliwice",
    "korona kielce": "Korona Kielce",
    "korona": "Korona Kielce",
    "stal mielec": "Stal Mielec",
    "stal m.": "Stal Mielec",
    "stal rzeszów": "Stal Rzeszów",
    "stal rz.": "Stal Rzeszów",
    "stal stalowa wola": "Stal Stalowa Wola",
    "stal st. w.": "Stal Stalowa Wola",
    "warta poznań": "Warta Poznań",
    "warta": "Warta Poznań",
    "radomiak radom": "Radomiak Radom",
    "radomiak": "Radomiak Radom",
    "miedź legnica": "Miedź Legnica",
    "miedź": "Miedź Legnica",
    "ruch chorzów": "Ruch Chorzów",
    "ruch": "Ruch Chorzów",
    "puszcza niepołomice": "Puszcza Niepołomice",
    "puszcza": "Puszcza Niepołomice",
    "łks łódź": "ŁKS Łódź",
    "łks": "ŁKS Łódź",
    "motor lublin": "Motor Lublin",
    "motor": "Motor Lublin",
    "gks katowice": "GKS Katowice",
    "gks k.": "GKS Katowice",
    "gks tychy": "GKS Tychy",
    "gks jastrzębie": "GKS Jastrzębie",
    "gks jastrzębie (jastrzębie zdrój)": "GKS Jastrzębie",
    "gks jastrzębie (jastrzębie-zdrój)": "GKS Jastrzębie",
    "gks j.": "GKS Jastrzębie",
    "avia świdnik": "Avia Świdnik",
    "garbarnia kraków": "Garbarnia Kraków",
    "górnik polkowice": "Górnik Polkowice",
    "lechia zielona góra": "Lechia Zielona Góra",
    "podhale nowy targ": "Podhale Nowy Targ",
    "sokół kleczew": "Sokół Kleczew",
    "unia skierniewice": "Unia Skierniewice",
    "zawisza bydgoszcz": "Zawisza Bydgoszcz",
    "arka gdynia": "Arka Gdynia",
    "arka": "Arka Gdynia",
    "bruk-bet termalica nieciecza": "Bruk-Bet Termalica Nieciecza",
    "bruk-bet termalica": "Bruk-Bet Termalica Nieciecza",
    "termalica": "Bruk-Bet Termalica Nieciecza",
    "nieciecza": "Bruk-Bet Termalica Nieciecza",
    "podbeskidzie bielsko-biała": "Podbeskidzie Bielsko-Biała",
    "podbeskidzie": "Podbeskidzie Bielsko-Biała",
    "odra opole": "Odra Opole",
    "odra o.": "Odra Opole",
    "odra": "Odra Opole",
    "polonia warszawa": "Polonia Warszawa",
    "polonia w.": "Polonia Warszawa",
    "polonia bytom": "Polonia Bytom",
    "polonia b.": "Polonia Bytom",
    "resovia rzeszów": "Resovia Rzeszów",
    "resovia": "Resovia Rzeszów",
    "chrobry głogów": "Chrobry Głogów",
    "chrobry": "Chrobry Głogów",
    "znicz pruszków": "Znicz Pruszków",
    "znicz": "Znicz Pruszków",
    "kotwica kołobrzeg": "Kotwica Kołobrzeg",
    "kotwica": "Kotwica Kołobrzeg",
    "pogoń szczecin": "Pogoń Szczecin",
    "pogoń siedlce": "Pogoń Siedlce",
    "pogoń s.": "Pogoń Siedlce",
    "pogoń grodzisk mazowiecki": "Pogoń Grodzisk Mazowiecki",
    "pogoń g.": "Pogoń Grodzisk Mazowiecki",
    "sandecja nowy sącz": "Sandecja Nowy Sącz",
    "sandecja": "Sandecja Nowy Sącz",
    "chojniczanka chojnice": "Chojniczanka Chojnice",
    "chojniczanka": "Chojniczanka Chojnice",
    "olimpia elbląg": "Olimpia Elbląg",
    "olimpia e.": "Olimpia Elbląg",
    "olimpia grudziądz": "Olimpia Grudziądz",
    "olimpia g.": "Olimpia Grudziądz",
    "kks 1925 kalisz": "KKS 1925 Kalisz",
    "kks kalisz": "KKS 1925 Kalisz",
    "hutnik kraków": "Hutnik Kraków",
    "hutnik": "Hutnik Kraków",
    "stomil olsztyn": "Stomil Olsztyn",
    "stomil": "Stomil Olsztyn",
    "radunia stężyca": "Radunia Stężyca",
    "radunia": "Radunia Stężyca",
    "wieczysta kraków": "Wieczysta Kraków",
    "wieczysta": "Wieczysta Kraków",
    "rekord bielsko-biała": "Rekord Bielsko-Biała",
    "rekord": "Rekord Bielsko-Biała",
    "skra częstochowa": "Skra Częstochowa",
    "skra": "Skra Częstochowa",
    "siarka tarnobrzeg": "Siarka Tarnobrzeg",
    "siarka": "Siarka Tarnobrzeg",
    "świt szczecin": "Świt Szczecin",
    "świt": "Świt Szczecin",
    # Reserve teams (must remain strictly distinct from first teams!)
    "lech ii poznań": "Lech II Poznań",
    "lech poznań ii": "Lech II Poznań",
    "lech ii": "Lech II Poznań",
    "zagłębie ii lubin": "Zagłębie II Lubin",
    "zagłębie lubin ii": "Zagłębie II Lubin",
    "zagłębie ii": "Zagłębie II Lubin",
    "śląsk ii wrocław": "Śląsk II Wrocław",
    "śląsk wrocław ii": "Śląsk II Wrocław",
    "śląsk ii": "Śląsk II Wrocław",
    "łks ii łódź": "ŁKS II Łódź",
    "łks łódź ii": "ŁKS II Łódź",
    "łks ii": "ŁKS II Łódź",
    "legia ii warszawa": "Legia II Warszawa",
    "legia warszawa ii": "Legia II Warszawa",
    "legia ii": "Legia II Warszawa",
}


def normalize_club_name(name: str) -> str:
    """Normalize club name using exact / tokenized mapping without accidental substring bleed."""
    clean = name.strip()
    clean_lower = clean.lower()
    
    # 1. Exact match in predefined dictionary
    if clean_lower in EXACT_CLUB_MAPPINGS:
        return EXACT_CLUB_MAPPINGS[clean_lower]
        
    # 2. Check for reserve teams flag "ii" or "2" explicitly
    is_reserve = bool(re.search(r"\b(ii|2)\b", clean_lower))
    if is_reserve:
        if "łks" in clean_lower or "lks" in clean_lower:
            return "ŁKS II Łódź"
        if "lech" in clean_lower:
            return "Lech II Poznań"
        if "śląsk" in clean_lower or "slask" in clean_lower:
            return "Śląsk II Wrocław"
        if "zagłębie" in clean_lower or "zaglebie" in clean_lower:
            return "Zagłębie II Lubin"
        if "legia" in clean_lower:
            return "Legia II Warszawa"

    # 3. If no exact match, return original cleaned name (which on 90minut is already proper!)
    return clean


def import_stadiums_to_db(db_path: Optional[Path] = None, csv_path: Optional[Path] = None) -> int:
    """Load stadium capacities CSV into SQLite stadium_capacities and club_locations tables."""
    c_path = csv_path or CSV_PATH
    if not c_path.exists():
        raise FileNotFoundError(f"Stadium CSV not found at {c_path}")
        
    conn = get_connection(db_path)
    count = 0
    unique_locations = {}
    
    with open(c_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with conn:
            conn.execute("DELETE FROM stadium_capacities;")
            for row in reader:
                club = row["club_name"].strip()
                stadium = row["stadium_name"].strip()
                city = row["city"].strip()
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                v_from = row["valid_from"].strip()
                v_to = row["valid_to"].strip()
                cap = int(row["capacity"])
                roof = float(row.get("roof_pct", 1.0))
                notes = row.get("notes", "").strip()
                
                conn.execute(
                    """
                    INSERT INTO stadium_capacities 
                    (club_name, stadium_name, city, latitude, longitude, valid_from, valid_to, capacity, roof_pct, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (club, stadium, city, lat, lon, v_from, v_to, cap, roof, notes)
                )
                count += 1
                unique_locations[club] = (city, lat, lon)
                
            # Populate club_locations
            conn.execute("DELETE FROM club_locations;")
            for club, (city, lat, lon) in unique_locations.items():
                conn.execute(
                    "INSERT INTO club_locations (club_name, city, latitude, longitude) VALUES (?, ?, ?, ?)",
                    (club, city, lat, lon)
                )
                
    return count


def resolve_stadium(
    club_name: str,
    match_date: str,
    reported_stadium: Optional[str] = None,
    away_team: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Resolve stadium capacity, roof coverage, city and GPS coordinates for a match."""
    norm_club = normalize_club_name(club_name)
    norm_away = normalize_club_name(away_team) if away_team else ""
    
    # Special venue derbies / large municipal stadium hostings
    if norm_club == "Warta Poznań" and ("Lech" in norm_away):
        return {
            "stadium_name": "Enea Stadion",
            "city": "Poznań",
            "latitude": 52.3975,
            "longitude": 16.8583,
            "capacity": 42837,
            "roof_pct": 1.0,
            "notes": "Derby Poznania na Enea Stadion"
        }
    if norm_club == "Wieczysta Kraków" and ("Wisła" in norm_away):
        return {
            "stadium_name": "Stadion Miejski im. Henryka Reymana",
            "city": "Kraków",
            "latitude": 50.0633,
            "longitude": 19.9122,
            "capacity": 33130,
            "roof_pct": 1.0,
            "notes": "Mecz na Stadionie Miejskim im. Reymana"
        }
    if norm_club == "Rekord Bielsko-Biała" and ("Podbeskidzie" in norm_away):
        return {
            "stadium_name": "Stadion Miejski w Bielsku-Białej",
            "city": "Bielsko-Biała",
            "latitude": 49.8183,
            "longitude": 19.0494,
            "capacity": 15076,
            "roof_pct": 1.0,
            "notes": "Derby Bielska-Białej na Stadionie Miejskim"
        }
        
    conn = get_connection(db_path)
    
    # 1. Match by exact club name and validity window
    row = conn.execute(
        """
        SELECT stadium_name, city, latitude, longitude, capacity, roof_pct, notes
        FROM stadium_capacities
        WHERE club_name = ?
          AND valid_from <= ? AND valid_to >= ?
        ORDER BY capacity DESC
        LIMIT 1
        """,
        (norm_club, match_date, match_date)
    ).fetchone()
    
    if row:
        return dict(row)
        
    # 2. Match without date constraint (fallback to latest)
    row = conn.execute(
        """
        SELECT stadium_name, city, latitude, longitude, capacity, roof_pct, notes
        FROM stadium_capacities
        WHERE club_name = ?
        ORDER BY valid_to DESC
        LIMIT 1
        """,
        (norm_club,)
    ).fetchone()
    
    if row:
        return dict(row)
        
    # 3. Default fallback for unknown smaller clubs
    return {
        "stadium_name": reported_stadium or f"Stadion {norm_club}",
        "city": "Polska",
        "latitude": 52.0,
        "longitude": 19.0,
        "capacity": 2000,
        "roof_pct": 0.3,
        "notes": "Default fallback"
    }
