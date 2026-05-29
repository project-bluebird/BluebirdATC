from __future__ import annotations

import datetime
import typing

# simulator package
from bluebird_dt.airspace_generator.airspace_loader import AirspaceLoader
from bluebird_dt.predictor import LinearPredictor
from bluebird_dt.scenario_manager.infinite import Infinite

# simulator gymnasium wrapper
from bluebird_gymnasium.envs import CentralizedSampler, EnvConfig, ViewType
from bluebird_gymnasium.envs.base import BaseEnv, ScenarioGenSeedMode, _configure_airspace_metadata

# constants
from bluebird_gymnasium.utils.constants import (
    DEFAULT_RENDER_DIR,
)
from bluebird_gymnasium.utils.types import StrEnum

if typing.TYPE_CHECKING:
    from bluebird_dt.simulator import Simulator

    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self


class ScenarioName(StrEnum):
    sector_i = "I-Sector"
    sector_x = "X-Sector"
    sector_xplus = "Xplus-Sector"
    sector_y = "Y-Sector"




class InfiniteEnv(BaseEnv):
    """gymnasium environment for the Air Traffic Control Game.

    Args:
        render_mode: the mode to visualise (render) the simulator. It can
            only be set to None or one of the following:
            'human', 'rgb_array', 'file'.
            Defaults to `None`.
        config: defines the configuration parameters for the gymnasium
            environment and the underlying simulator.
    """

    scenario_seed_mode = ScenarioGenSeedMode.RESET_SEED_ATTRIBUTE

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
    ):
        super().__init__(
            render_mode,
            config,
        )

        ####### scenario manager
        self.scenario_manager = None  # set in `_generate_scenario`

        ####### airspace metadata
        _configure_airspace_metadata(self, self.config.scenario_config["scenario_name"])

        ####### reset env
        self.reset()

    def _generate_scenario(self) -> Simulator:
        """Generate a scenario in the simulator."""

        # set up simulation log name
        category = "Game"
        scenario_name = self.config.scenario_config["scenario_name"]
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        suffix = self.config.simulation_log_config.get("log_suffix", None)
        suffix = "" if suffix is None or suffix == "" else f"__{suffix}"
        log_filename = f"{category}_{scenario_name}_{timestamp}{suffix}"

        # set up simulator manager
        sim = Infinite.setup(
            scenario_name=scenario_name,
            random_seed=self._reset_seed,
            autosave=False,
            save_log_to_file=False,
            log_filename=log_filename,
            predictor=None,
        )
        self.scenario_manager = sim.scenario_manager

        return sim

    @classmethod
    def get_default_env_config(cls: type[Self], view_type: ViewType | str = ViewType.CENTRALIZED) -> EnvConfig:
        """Class method: Get the default config for an environment instance.

        Defined in each child class that inherits this base class.

        Args:
            cls: the class
            view_type: the type of agent view, centralised (single agent) or
                decentralised (multi-agent).
                Defaults to "centralized".

        Returns:
            dict, the default config.
        """

        if view_type not in ViewType:
            raise ValueError(f"{view_type} is not a valid value of {ViewType}")
        if isinstance(view_type, ViewType):
            view_type = view_type.value

        # airspace
        airspace_height = 60  # nautical miles

        # radar
        ## relative scale: for adjusting airspace and aircraft size
        rel_scale = 0.0
        ## aspect ratio
        aspect_ratio = 1.89
        ## width scale in nautical miles for rendered frames
        radar_scale = (airspace_height * (1.0 + rel_scale)) * aspect_ratio
        ## aircraft scale
        radar_aircraft_scale = 1.0 * (1.0 - rel_scale)
        lateral_separation = 5.0  # nautical miles

        config_dict = {
            "action_config": {
                "simple_heading_left": [
                    10,
                ],
                "simple_heading_right": [
                    10,
                ],
                "simple_heading_route_parallel": True,
                "simple_fl_descent": False,
                "simple_fl_climb": False,
                "simple_fl_intermediate": False,
                "simple_fl_exit": False,
                "simple_speed_increase": False,
                "simple_speed_decrease": False,
                "simple_route_direct": False,
                "simple_outcomm": False,
            },
            "airspace_config": {
                "exit_window_width": None,  # the default width will be set.
                "out_sector_control": False,
            },
            "radar_config": {
                "scale": radar_scale,
                "aspect_ratio": aspect_ratio,
                "aircraft_scale": radar_aircraft_scale,
                "render_fixes": True,
                "render_routes": True,
                "render_sep_bound": True,
                "aircraft_lateral_sep": lateral_separation,
                "prefix": "frame",
                "render_dir": DEFAULT_RENDER_DIR,
                "show_spines": False,
                "display_units": "lonlat",
                "scaled": False,
                "display_actions": False,
            },
            "reward_config": {
                "fns": [
                    "position_status_const",
                    "lateral_centreline_distance_shaped",
                ],
                "coeffs": [1.0, 1.0],
            },
            "scenario_config": {"scenario_name": ScenarioName.sector_xplus.value},
            "simulation_log_config": {
                "save_simulation": False,
                "log_suffix": "",
            },
            "state_repr_config": {
                "encoder_cls": "relative",
                "k_nearest_aircraft": 0,
            },
            "view_config": {
                "type": view_type,
                "centralized_params": {
                    "num_sampled_aircraft": 5,
                    "sample_strategy": CentralizedSampler.LATEST.value,
                },
                "decentralized_params": {},
            },
            "forward_fixes_config": {
                "num_fixes": 1,
                "use_filed_route": True,
            },
            "scenario_duration": 3600,
            "scenario_sec_per_step": 6,
            "diagnostics_level": None,
        }

        return EnvConfig(**config_dict)


class CustomInfiniteEnv(BaseEnv):
    """gymnasium environment for the Air Traffic Control Game.

    This implementation contains enables the customisation of low level
    parameters in the simulator such as the aircraft spawn rate. For the
    standard scenario, use the `InfiniteEnv` class above.

    When `.reset(seed=...)` is called with a seed, that seed overrides
    `config.scenario_config["random_seed"]` for the generated scenario.
    Calling `.reset()` or `.reset(seed=None)` preserves the configured
    scenario seed.

    Args:
        render_mode: the mode to visualise (render) the simulator. It can
            only be set to None or one of the following:
            'human', 'rgb_array', 'file'.
            Defaults to `None`.
        config: defines the configuration parameters for the gymnasium
            environment and the underlying simulator.
    """

    scenario_seed_mode = ScenarioGenSeedMode.RESET_SEED_ATTRIBUTE

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
    ):
        super().__init__(
            render_mode,
            config,
        )

        ####### scenario manager
        self.scenario_manager = None  # set in `_generate_scenario`

        ####### airspace metadata
        _configure_airspace_metadata(self, self.config.scenario_config["scenario_name"])

        ####### reset env
        self.reset()

    def _generate_scenario(self) -> Simulator:
        """Generate a scenario in the simulator."""

        # set up simulation log name
        category = "Game"
        scenario_name = self.config.scenario_config["scenario_name"]
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        suffix = self.config.simulation_log_config.get("log_suffix", None)
        suffix = "" if suffix is None or suffix == "" else f"__{suffix}"
        log_filename = f"{category}_{scenario_name}_{timestamp}{suffix}"

        # set up simulator manager
        sim = Infinite.setup(
            scenario_name=self.config.scenario_config["scenario_name"],
            random_seed=(
                self._reset_seed if self._reset_seed is not None else self.config.scenario_config["random_seed"]
            ),
            num_starter_aircraft=self.config.scenario_config["num_starter_aircraft"],
            initial_spawn_rate=self.config.scenario_config["initial_spawn_rate"],
            spawn_rate_increment=self.config.scenario_config["spawn_rate_increment"],
            spawn_rate_increase_interval=self.config.scenario_config["spawn_rate_increase_interval"],
            max_spawn_rate=self.config.scenario_config["max_spawn_rate"],
            spawn_distance_threshold=self.config.scenario_config["spawn_distance_threshold"],
            use_wind=self.config.scenario_config["use_wind"],
            use_forecast=self.config.scenario_config["use_forecast"],
            autosave=False,
            save_log_to_file=False,
            log_filename=log_filename,
            predictor=None,
        )
        self.scenario_manager = sim.scenario_manager

        return sim

    @classmethod
    def get_default_env_config(cls: type[Self], view_type: ViewType | str = ViewType.CENTRALIZED) -> EnvConfig:
        """Class method: Get the default config for an environment instance.

        Defined in each child class that inherits this base class.

        Args:
            cls: the class
            view_type: the type of agent view, centralised (single agent) or
                decentralised (multi-agent).
                Defaults to "centralized".

        Returns:
            dict, the default config.
        """

        if view_type not in ViewType:
            raise ValueError(f"{view_type} is not a valid value of {ViewType}")
        if isinstance(view_type, ViewType):
            view_type = view_type.value

        # airspace
        airspace_height = 60  # nautical miles

        # radar
        ## relative scale: for adjusting airspace and aircraft size
        rel_scale = 0.0
        ## aspect ratio
        aspect_ratio = 1.89
        ## width scale in nautical miles for rendered frames
        radar_scale = (airspace_height * (1.0 + rel_scale)) * aspect_ratio
        ## aircraft scale
        radar_aircraft_scale = 1.0 * (1.0 - rel_scale)
        lateral_separation = 5.0  # nautical miles

        config_dict = {
            "action_config": {
                "simple_heading_left": [
                    10,
                ],
                "simple_heading_right": [
                    10,
                ],
                "simple_heading_route_parallel": True,
                "simple_fl_descent": False,
                "simple_fl_climb": False,
                "simple_fl_intermediate": False,
                "simple_fl_exit": False,
                "simple_speed_increase": False,
                "simple_speed_decrease": False,
                "simple_route_direct": False,
                "simple_outcomm": False,
            },
            "airspace_config": {
                "exit_window_width": None,  # the default width will be set.
                "out_sector_control": False,
            },
            "radar_config": {
                "scale": radar_scale,
                "aspect_ratio": aspect_ratio,
                "aircraft_scale": radar_aircraft_scale,
                "render_fixes": True,
                "render_routes": True,
                "render_sep_bound": True,
                "aircraft_lateral_sep": lateral_separation,
                "prefix": "frame",
                "render_dir": DEFAULT_RENDER_DIR,
                "show_spines": False,
                "display_units": "lonlat",
                "scaled": False,
                "display_actions": False,
            },
            "reward_config": {
                "fns": [
                    "position_status_const",
                    "lateral_centreline_distance_shaped",
                ],
                "coeffs": [1.0, 1.0],
            },
            "scenario_config": {
                "scenario_name": ScenarioName.sector_xplus.value,
                "initial_spawn_rate": 0.01,
                "max_spawn_rate": 0.1,
                "spawn_rate_increment": 0.0,
                "spawn_rate_increase_interval": 0.0,
                "spawn_distance_threshold": 10,
                "random_seed": 1251,
                "num_starter_aircraft": 2,
                "use_wind": True,
                "use_forecast": True,
            },
            "simulation_log_config": {
                "save_simulation": False,
                "log_suffix": "",
            },
            "state_repr_config": {
                "encoder_cls": "relative",
                "k_nearest_aircraft": 0,
            },
            "view_config": {
                "type": view_type,
                "centralized_params": {
                    "num_sampled_aircraft": 5,
                    "sample_strategy": CentralizedSampler.LATEST.value,
                },
                "decentralized_params": {},
            },
            "forward_fixes_config": {
                "num_fixes": 1,
                "use_filed_route": True,
            },
            "scenario_duration": 3600,
            "scenario_sec_per_step": 6,
            "diagnostics_level": None,
        }

        return EnvConfig(**config_dict)
