from __future__ import annotations

import datetime
import string
import typing

# simulator package
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager
from bluebird_dt.predictor import LinearPredictor

# simulator gymnasium wrapper
from bluebird_gymnasium.envs import CentralizedSampler, EnvConfig, ViewType
from bluebird_gymnasium.envs.base import BaseEnv

# constants
from bluebird_gymnasium.utils.constants import (
    DEFAULT_RENDER_DIR,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.simulator import Simulator

    from bluebird_gymnasium.envs import (
        ActionType,
        DoneType,
        InfoType,
        ObsType,
        RewardType,
        TruncatedType,
    )


SPRINGFIELD_SECTOR_NAME = "SPRINGFIELD"


class SpringfieldEnv(BaseEnv):
    """gymnasium environment for the Springfield airspace.

    gym environment for the Springfield training sector/airspace, defined in
    simulation framework. Designed to simulate challenging air traffic control
    scenarios. In addition, the aircraft scenarios can be based on either
    real-world (historical) data or artificially generated. Based on the
    simulator.

    Args:
        render_mode: the mode to visualize (render) the simulator. It can
            only be set to None or one of the following:
            'human', 'rgb_array', 'file'.
            Defaults to `None`.
        config: defines the configuration parameters for the gymnasium
            environment and the underlying simulator.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
    ):
        super(SpringfieldEnv, self).__init__(
            render_mode,
            config,
        )

        # if the `exit_window_width` value was originally None, override the
        # default set in the parent class with a new default here (based on
        # the sector/airspace geometry).
        exit_window_width = self.config.airspace_config.get(
            "exit_window_width", None
        )
        if exit_window_width is None:
            # the width of each springfield exit boundary is 10 nautical miles
            self.exit_window_width = 10 // 2
        else:
            self.exit_window_width = exit_window_width

        ####### airspace
        sim = SpringfieldScenarioManager.setup(
            scenario_name=self.config.scenario_config["scenario"],
            use_wind=self.config.scenario_config["use_wind"],
            use_forecast=self.config.scenario_config["use_forecast"],
            autosave=False,
            save_log_to_file=False,
        )
        airspace = sim.manager.environment.airspace

        ####### scenario manager
        self.scenario_manager = sim.scenario_manager
        del sim

        ####### trajectory predictor (world model)
        # trajectory predictor for computing an estimated
        # future (rollout) trajectories. used in safety reward functions.
        self.rollout_predictor = LinearPredictor(
            dt=12,
            fix_proximity_threshold=2.0,
            fixes=airspace.fixes,
        )

        ####### active airspace sector
        airspace_sectors = list(airspace.sectors.keys())
        if (
            len(airspace_sectors) >= 1
            and SPRINGFIELD_SECTOR_NAME in airspace_sectors
        ):
            self.active_airspace_sector = SPRINGFIELD_SECTOR_NAME
        else:
            raise ValueError("Could not initialise sector")

        ####### reset env
        self.reset()

    def _generate_scenario(self) -> Simulator:
        # set up simulation log name
        category = string.capwords(SPRINGFIELD_SECTOR_NAME)
        scenario = self.config.scenario_config["scenario"]
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        suffix = self.config.simulation_log_config.get("log_suffix", None)
        if suffix is None or suffix == "":
            suffix = ""
        else:
            suffix = f"__{suffix}"
        log_filename = f"{category}_{scenario}_{timestamp}{suffix}"

        # set up simulator manager
        sim = self.scenario_manager.to_simulator(
            category=category,
            use_wind=self.config.scenario_config["use_wind"],
            use_forecast=self.config.scenario_config["use_forecast"],
            autosave=False,
            save_log_to_file=False,
            log_filename=log_filename,
            predictor=None,  # use the default in the scenario
        )
        return sim

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, RewardType, DoneType, TruncatedType, InfoType]:
        obs, reward, done, truncated, info = super().step(action)

        # Springfield can start before any aircraft is controllable; keep an
        # empty decentralized done dict from ending loops that call all(...).
        if (
            self.config.view_config["type"] == ViewType.DECENTRALIZED
            and isinstance(done, dict)
            and not done
            and self.timestep < self.maxstep
        ):
            done = {"__all__": False}
            truncated = {"__all__": False}

        return obs, reward, done, truncated, info

    @classmethod
    def get_default_env_config(
        cls, view_type: ViewType | str = ViewType.CENTRALIZED
    ) -> EnvConfig:
        """Class method: Get the default config for an environment instance.

        Defined in each child class that inherits this base class.

        Args:
            cls: the class
            view_type: the type of agent view, centralized (single agent) or
                decentralized (multi-agent).
                Defaults to "centralized".

        Returns:
            dict, the default config.
        """

        if view_type not in ViewType:
            raise ValueError(f"{view_type} is not a valid value of {ViewType}")
        elif isinstance(view_type, ViewType):
            view_type = view_type.value

        # airspace
        airspace_width = 280  # approximate. in nautical miles
        airspace_height = 270  # approximate. in nautical miles

        # radar
        ## relative scale: for adjusting airspace and aircraft size
        rel_scale = 0.0
        ## aspect ratio
        aspect_ratio = airspace_width / airspace_height
        ## width scale in nautical miles for rendered frames
        radar_scale = (airspace_height * (1.0 + rel_scale)) * aspect_ratio
        ## aircraft scale
        radar_aircraft_scale = 1.0 * (1.0 - rel_scale)
        lateral_separation = 5.0  # nautical miles

        # note order: lon, lat
        origin = SpringfieldScenarioManager.projection_centre
        # now reverse it to lat, lon
        origin = (origin[1], origin[0])

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
                "origin": origin,
                "exit_window_width": None,  # the default width will be set.
                "out_sector_control": False,
            },
            "radar_config": {
                "scale": radar_scale,
                "aspect_ratio": aspect_ratio,
                "aircraft_scale": radar_aircraft_scale,
                "render_fixes": [
                    "BERTY",
                    "GARRY",
                    "SPOUT",
                    "SIMPS",
                    "FIYFE",
                    "JIMMI",
                    "NATOR",
                    "LEXIE",
                    "TABSY",
                    "LEGGO",
                    "WINDY",
                    "CAKES",
                    "SCOTT",
                    "SCIFI",
                    "TARRA",
                    "STEPP",
                    "WISKY",
                    "SMUDJ",
                ],
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
                "scenario": "example-scenario",  # llm-scenario
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
            "scenario_duration": 1800,
            "scenario_sec_per_step": 6,
            "diagnostics_level": None,
        }

        return EnvConfig(**config_dict)


if __name__ == "__main__":
    from pprint import pprint

    view = ViewType.CENTRALIZED
    config = SpringfieldEnv.get_default_env_config(view)
    env = SpringfieldEnv(config=config)
    if view == ViewType.CENTRALIZED:
        obs, info = env.reset()
        print(obs)
        pprint(info)
        for idx in range(50):
            obs, reward, done, truncated, info = env.step(0)
            print(idx, obs)
            pprint(info)
            print("-" * 50)
    else:
        obs, info = env.reset()
        print(obs)
        pprint(info)
        for idx in range(50):
            actions = {callsign: 0 for callsign in obs.keys()}
            obs, reward, done, truncated, info = env.step(actions)
            print(idx, obs)
            pprint(info)
            print("-" * 50)
