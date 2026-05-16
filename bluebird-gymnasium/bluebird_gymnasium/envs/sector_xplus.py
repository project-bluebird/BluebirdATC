from __future__ import annotations
import datetime
import typing

# simulator package
from bluebird_dt.airspace_generator import SectorXPlus
from bluebird_dt.utility.geo_helper import GeoHelper
from bluebird_dt.predictor import LinearPredictor

# simulator gymnasium wrapper
from bluebird_gymnasium.envs import (
    CentralizedSampler,
    EnvConfig,
    ViewType,
    SCENARIO_CLS,
)
from bluebird_gymnasium.envs.base import BaseEnv

# constants
from bluebird_gymnasium.utils.constants import (
    DEFAULT_RENDER_DIR,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.simulator import Simulator


class SectorXPlusEnv(BaseEnv):
    """gymnasium environment for the X-plus sector airspace.

    Artificially generated aircraft scenarios and a configurable X-plus sector
    airspace, defined in the simulation framework.

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
        super(SectorXPlusEnv, self).__init__(
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
            # TODO: fix hardcode using airspace config. either
            # W or half_width_nmi
            self.exit_window_width = 10
        else:
            self.exit_window_width = exit_window_width

        ####### airspace
        # the airspace generator expects the origin in reverse order
        # i.e., lon, lat
        origin = (
            self.config.airspace_config["origin"][1],
            self.config.airspace_config["origin"][0],
        )
        airspace, routes = SectorXPlus(
            origin=origin,
            fl_limits=self.config.airspace_config["fl_limits"],
            rotation_deg=self.config.airspace_config["rotation_deg"],
            half_width_nmi=self.config.airspace_config["half_width_nmi"],
            L1=self.config.airspace_config["L1"],
            L2=self.config.airspace_config["L2"],
            L3=self.config.airspace_config["L3"],
            L4=self.config.airspace_config["L4"],
            W=self.config.airspace_config["W"],
            D=self.config.airspace_config["D"],
            F=self.config.airspace_config["F"],
            southern_leg_rotation_deg=self.config.airspace_config[
                "southern_leg_rotation_deg"
            ],
            max_turn_angle_deg=self.config.airspace_config[
                "max_turn_angle_deg"
            ],
        ).generate_airspace()
        airspace.geo_helper = GeoHelper(self.config.airspace_config["origin"])

        ####### scenario manager
        _scenario_cls = SCENARIO_CLS[self.config.scenario_config["cls"]]
        self.scenario_manager = _scenario_cls(
            airspace=airspace,
            routes=routes,
            **self.config.scenario_config["args"],
        )

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
        if len(airspace_sectors) == 1 and airspace_sectors[0] == "sector_xplus":
            self.active_airspace_sector = airspace_sectors[0]
        else:
            raise ValueError("Could not initialise sector")

        ####### reset env
        self.reset()

    def _generate_scenario(self) -> Simulator:
        # set up simulation log name
        category = "Artificial"
        scenario = "XPlus-Sector-Custom-Scenario"
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        suffix = self.config.simulation_log_config.get("log_suffix", None)
        if suffix is None or suffix == "":
            suffix = ""
        else:
            suffix = f"__{suffix}"
        log_filename = f"{category}_{scenario}_{timestamp}{suffix}"

        ####### setup the sim env manager
        sim = self.scenario_manager.to_simulator(
            category=category,
            scenario_name=scenario,
            save_log_to_file=False,
            log_filename=log_filename,
            predictor=None,  # use the default in the scenario.
        )
        return sim

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
        airspace_height = 120  # nautical miles

        # radar
        ## relative scale: for adjusting airspace and aircraft size
        rel_scale = 0.0
        ## aspect ratio
        aspect_ratio = 1.33
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
                "sector": "SectorXPlus",
                "origin": [0.0, 0.0],  # note order: lat, lon
                "fl_limits": [200, 400],
                "rotation_deg": -45.0,
                "half_width_nmi": 10.0,
                "L1": 3,
                "L2": 6,
                "L3": 5,
                "L4": 4.5,
                "W": 1,
                "D": 2.5,
                "F": 0.5,
                "southern_leg_rotation_deg": 30.0,
                "max_turn_angle_deg": 90.0,
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
                "cls": "tactical",
                "args": {
                    "num_aircraft": 1,
                    "vertical_buffer_distance": 100,
                    "lateral_buffer_distance": 200,
                },
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
    config = SectorXPlusEnv.get_default_env_config(view)
    env = SectorXPlusEnv(config=config)
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
        for idx in range(20):
            actions = {callsign: 0 for callsign in obs.keys()}
            obs, reward, done, truncated, info = env.step(actions)
            print(idx, obs)
            pprint(info)
            print("-" * 50)
