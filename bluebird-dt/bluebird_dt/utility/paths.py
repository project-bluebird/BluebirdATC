import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# We have the data in the bluebird_dt package itself.
BASE_DATA_DIR = ROOT_DIR.joinpath("scenario_data")
LOG_DIR: str = os.path.join(BASE_DATA_DIR, "scenario_logs")
SPRINGFIELD_DIR: str = os.path.join(BASE_DATA_DIR, "Springfield")

AIRCRAFT_DATA_DIR = ROOT_DIR.joinpath("aircraft_data")
AIRCRAFT_WEIGHT_MAPPING_FILE: str = os.path.join(AIRCRAFT_DATA_DIR, "aircraft_weight_map.json")
SIMPLE_PERFORMANCE_PROFILE_FILE = os.path.join(AIRCRAFT_DATA_DIR, "simple_performance_profile_data.json")
SIMPLE_PERFORMANCE_UNCERTAINTY_FILE = os.path.join(AIRCRAFT_DATA_DIR, "simple_performance_uncertainty_data.json")
