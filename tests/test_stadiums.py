"""Tests for stadium resolver and club normalization."""
import tempfile
from pathlib import Path
import pytest

from src.db.schema import init_db
from src.scraper.stadiums import (
    normalize_club_name,
    resolve_stadium,
    import_stadiums_to_db,
)


@pytest.fixture(scope="module")
def populated_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    import_stadiums_to_db(db_path=temp_path)
    yield temp_path
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception:
        pass


def test_normalize_club_name():
    assert normalize_club_name("Raków") == "Raków Częstochowa"
    assert normalize_club_name("raków częstochowa") == "Raków Częstochowa"
    assert normalize_club_name("Legia") == "Legia Warszawa"
    assert normalize_club_name("Widzew") == "Widzew Łódź"
    assert normalize_club_name("GKS Katowice") == "GKS Katowice"


def test_resolve_stadium_dynamic_pogon(populated_db):
    # Pogoń before Oct 2022 had limited capacity
    res_old = resolve_stadium("Pogoń Szczecin", "2022-08-15", db_path=populated_db)
    assert res_old["capacity"] < 15000
    
    # Pogoń after Oct 2022 has 21k capacity
    res_new = resolve_stadium("Pogoń Szczecin", "2023-05-01", db_path=populated_db)
    assert res_new["capacity"] == 21163


def test_resolve_stadium_dynamic_radomiak(populated_db):
    # Radomiak played at MOSiR (ul. Narutowicza) until July 2023 (~4066)
    res_mosir = resolve_stadium("Radomiak Radom", "2022-09-01", db_path=populated_db)
    assert res_mosir["capacity"] == 4066
    
    # Radomiak opened Struga 63 in Aug 2023 (~8846)
    res_struga = resolve_stadium("Radomiak Radom", "2023-11-05", db_path=populated_db)
    assert res_struga["capacity"] == 8846


def test_resolve_stadium_puszcza_relocation(populated_db):
    # In Ekstraklasa (2023+), Puszcza played in Kraków at Cracovia stadium (15016)
    res_puszcza = resolve_stadium("Puszcza Niepołomice", "2023-08-11", db_path=populated_db)
    assert res_puszcza["city"] == "Kraków"
    assert res_puszcza["capacity"] == 15016


def test_reserve_teams_distinction(populated_db):
    # Test normalization of first team vs reserve aliases
    assert normalize_club_name("ŁKS Łódź") == "ŁKS Łódź"
    assert normalize_club_name("ŁKS Łódź II") == "ŁKS II Łódź"
    assert normalize_club_name("ŁKS II Łódź") == "ŁKS II Łódź"
    assert normalize_club_name("ŁKS II") == "ŁKS II Łódź"
    assert normalize_club_name("Lech Poznań") == "Lech Poznań"
    assert normalize_club_name("Lech Poznań II") == "Lech II Poznań"
    assert normalize_club_name("Lech II") == "Lech II Poznań"

    # Test stadium capacity distinction
    res_lks_first = resolve_stadium("ŁKS Łódź", "2024-05-01", db_path=populated_db)
    assert res_lks_first["capacity"] == 18029
    
    res_lks_reserve = resolve_stadium("ŁKS II Łódź", "2024-05-01", db_path=populated_db)
    assert res_lks_reserve["capacity"] == 900

    res_lks_reserve_alias = resolve_stadium("ŁKS Łódź II", "2024-05-01", db_path=populated_db)
    assert res_lks_reserve_alias["capacity"] == 900

    res_lech_first = resolve_stadium("Lech Poznań", "2024-05-01", db_path=populated_db)
    assert res_lech_first["capacity"] == 42837

    res_lech_reserve = resolve_stadium("Lech II Poznań", "2024-05-01", db_path=populated_db)
    assert res_lech_reserve["capacity"] == 1000
