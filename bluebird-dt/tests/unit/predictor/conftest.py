import json

import pytest

from bluebird_dt.predictor import LinearPredictor


@pytest.fixture
def type_a_performance_files(tmp_path) -> tuple[str, str, str]:
    performance_profile_path = tmp_path / "performance_profile.json"
    performance_uncertainty_path = tmp_path / "performance_uncertainty.json"
    synonym_map_path = tmp_path / "synonyms.json"

    performance_profile = {
        "aircraft": {
            "TYPEA": {
                "flight_level": [100, 200, 300],
                "cas_cl": [250.0, 260.0, 270.0],
                "cas_cr": [260.0, 270.0, 280.0],
                "cas_des": [240.0, 250.0, 260.0],
                "mach_cl": [0.50, 0.60, 0.70],
                "mach_cr": [0.55, 0.65, 0.75],
                "mach_des": [0.50, 0.60, 0.70],
                "rocd_cl": [1500.0, 1400.0, 1300.0],
                "rocd_des": [1200.0, 1100.0, 1000.0],
            }
        }
    }
    performance_uncertainty = {
        "aircraft": {
            "TYPEA": {
                "cas_cl": {"sigma": 5.0, "minimum": None, "maximum": None, "norm_mean": 260.0},
                "cas_cr": {"sigma": 5.0, "minimum": None, "maximum": None, "norm_mean": 270.0},
                "cas_des": {"sigma": 5.0, "minimum": None, "maximum": None, "norm_mean": 250.0},
                "mach_cl": {"sigma": 0.02, "minimum": 0.56, "maximum": 0.64, "norm_mean": 0.60},
                "mach_cr": {"sigma": 0.02, "minimum": 0.61, "maximum": 0.69, "norm_mean": 0.65},
                "mach_des": {"sigma": 0.02, "minimum": 0.56, "maximum": 0.64, "norm_mean": 0.60},
                "rocd_cl": {"sigma": 100.0, "minimum": 500.0, "maximum": None, "norm_mean": 1400.0},
                "rocd_des": {"sigma": 100.0, "minimum": 500.0, "maximum": None, "norm_mean": 1100.0},
            }
        }
    }
    synonyms = {"DEFAULT": "TYPEA", "TYPEA": "TYPEA"}

    performance_profile_path.write_text(json.dumps(performance_profile), encoding="utf-8")
    performance_uncertainty_path.write_text(json.dumps(performance_uncertainty), encoding="utf-8")
    synonym_map_path.write_text(json.dumps(synonyms), encoding="utf-8")

    return str(performance_profile_path), str(performance_uncertainty_path), str(synonym_map_path)


@pytest.fixture
def b753_performance_files(tmp_path) -> tuple[str, str, str]:
    speed_profile_path = tmp_path / "speed_profile_legacy.json"
    speed_uncertainty_path = tmp_path / "speed_uncertainty_legacy.json"
    synonym_map_path = tmp_path / "synonyms_legacy.json"

    speed_profile = {
        "aircraft": {
            "B753": {
                "flight_level": [20, 120, 220, 320],
                "cas_cl": [None, 250.0, 280.0, 300.0],
                "cas_cr": [220.0, 250.0, 300.0, 330.0],
                "cas_des": [None, 240.0, 280.0, 310.0],
                "mach_cl": [None, 0.45, 0.55, 0.65],
                "mach_cr": [0.35, 0.47, 0.58, 0.68],
                "mach_des": [None, 0.44, 0.54, 0.64],
                "rocd_cl": [2500.0, 2200.0, 1800.0, 1400.0],
                "rocd_des": [None, 1200.0, 1500.0, 1700.0],
            }
        }
    }
    speed_uncertainty = {
        "aircraft": {
            "B753": {
                "cas_cl": {"sigma": 8.0, "minimum": None, "maximum": None, "norm_mean": 260.0},
                "cas_cr": {"sigma": 8.0, "minimum": None, "maximum": None, "norm_mean": 280.0},
                "cas_des": {"sigma": 8.0, "minimum": None, "maximum": None, "norm_mean": 270.0},
                "mach_cl": {"sigma": 0.02, "minimum": 0.51, "maximum": 0.59, "norm_mean": 0.55},
                "mach_cr": {"sigma": 0.02, "minimum": 0.54, "maximum": 0.62, "norm_mean": 0.58},
                "mach_des": {"sigma": 0.02, "minimum": 0.50, "maximum": 0.58, "norm_mean": 0.54},
                "rocd_cl": {"sigma": 120.0, "minimum": 500.0, "maximum": None, "norm_mean": 1900.0},
                "rocd_des": {"sigma": 120.0, "minimum": 500.0, "maximum": None, "norm_mean": 1400.0},
            }
        }
    }
    synonyms = {"DEFAULT": "B753", "B753": "B753", "B738": "B753", "NULL": "B753", "MEDIUM": "B753"}

    speed_profile_path.write_text(json.dumps(speed_profile), encoding="utf-8")
    speed_uncertainty_path.write_text(json.dumps(speed_uncertainty), encoding="utf-8")
    synonym_map_path.write_text(json.dumps(synonyms), encoding="utf-8")

    return str(speed_profile_path), str(speed_uncertainty_path), str(synonym_map_path)


@pytest.fixture
def build_linear_predictor_with_user_supplied_performance_data(generate_simple_environment, b753_performance_files):
    performance_profile_path, performance_uncertainty_path, synonym_map_path = b753_performance_files

    def _build(dt: float = 1.0, fix_proximity_threshold: float = 2.0, **kwargs) -> LinearPredictor:
        return LinearPredictor(
            dt,
            fix_proximity_threshold,
            fixes=generate_simple_environment.airspace.fixes,
            performance_profile_data_path=performance_profile_path,
            performance_uncertainty_data_path=performance_uncertainty_path,
            aircraft_mapping_path=synonym_map_path,
            **kwargs,
        )

    return _build
