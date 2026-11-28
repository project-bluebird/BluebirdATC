import json

import pytest

from bluebird_dt.utility import performance


def test_get_aircraft_key_mapping(tmp_path):
    """
    Test that get_aircraft_key_mapping can write and read files and that contents are cached, throws for erroneous path supplied
    """
    filename = "testing_synonyms_db7a07c2-2601-4495-a73b-a4c9bf0bbdad.json"
    path = tmp_path / filename
    path.write_text(json.dumps({"DEFAULT": "B753", "ABC": "ABC"}))

    first = performance.get_aircraft_key_mapping(str(path))
    path.write_text(json.dumps({"DEFAULT": "CHANGED"}))
    second = performance.get_aircraft_key_mapping(str(path))
    # We've changed the data but the cache should mean we get the same back
    assert first == second

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Aircraft synonym data file could not be found!"):
        performance.get_aircraft_key_mapping(str(missing))


def test_get_speed_table(tmp_path):
    """
    Test that get_performance_table can write and read files and that contents are cached, throws for erroneous path supplied
    """
    filename = "testing_aircraft_4dd8564d-5a63-4e82-a550-84f0f3593ec5.json"
    path = tmp_path / filename
    path.write_text(json.dumps({"aircraft": {"TEST": {"cas": [], "rocd": []}}}))
    first = performance.get_performance_table(str(path))

    path.write_text(json.dumps({"aircraft": {"NONSENSE": {"cas": [], "rocd": []}}}))
    second = performance.get_performance_table(str(path))
    assert "TEST" in second
    assert first == second

    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Speed profile data file could not be found!"):
        performance.get_performance_table(str(missing))


def test_get_performance_table_with_maxima():
    """
    Each table is get_performance_table's, plus the largest value of each CAS and Mach column under "max_<column>"
    (None if the column has no values)
    """
    tables = performance.get_performance_table(None)
    with_maxima = performance.get_performance_table_with_maxima(None)
    assert with_maxima.keys() == tables.keys()
    for aircraft_type, table in tables.items():
        speed_keys = [
            key for key, value in table.items() if key.startswith(("cas_", "mach_")) and isinstance(value, list)
        ]
        assert speed_keys
        extended = with_maxima[aircraft_type]
        assert extended.keys() == table.keys() | {f"max_{key}" for key in speed_keys}
        assert all(extended[key] == value for key, value in table.items())
        for key in speed_keys:
            values = [speed for speed in table[key] if speed is not None]
            assert extended[f"max_{key}"] == (max(values) if values else None)
