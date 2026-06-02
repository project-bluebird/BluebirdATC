from __future__ import annotations

import datetime
from typing import Any

from bluebird_gymnasium.envs import EnvConfig, ViewType
from bluebird_gymnasium.envs.sector_xplus import SectorXPlusEnv


class FlightSchoolEnv(SectorXPlusEnv):
    """Gymnasium environment for the Flight School scenario."""

    def _generate_scenario(self): Any
        category = "Flight School"
        scenario = "Xplus-Sector"
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        suffix = self.config.simulation_log_config.get("log_suffix", None)
        if suffix is None or suffix == "":
            suffix = ""
        else:
            suffix = f"__{suffix}"
        log_filename = f"{category}_{scenario}_{timestamp}{suffix}"

        return self.scenario_manager.to_simulator(
            category=category,
            scenario_name=scenario,
            save_log_to_file=False,
            log_filename=log_filename,
            predictor=None,
        )

    @classmethod
    def get_default_env_config(
        cls, view_type: ViewType | str = ViewType.CENTRALIZED
    ) -> EnvConfig:
        config = super().get_default_env_config(view_type)
        config.scenario_config = {
            "cls": "infinite",
            "args": {
                "random_seed": None,
                "num_starter_aircraft": 2,
                "initial_spawn_rate": 0.002,
                "spawn_rate_increment": 0.002,
                "spawn_rate_increase_interval": 60,
                "max_spawn_rate": 0.03,
                "total_time_seconds": 3600.0,
            },
        }
        config.reward_config = {
            "fns": [
                "position_status_const",
                "lateral_centreline_distance_shaped",
                "safety_simple_avoidance_exp",
            ],
            "coeffs": [1.0, 1.0, 1.2],
        }
        config.scenario_duration = 10 * 60
        return config
