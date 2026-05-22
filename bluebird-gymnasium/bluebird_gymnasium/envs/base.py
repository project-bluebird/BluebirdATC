from __future__ import annotations

import contextlib
import copy
import os
import random
import typing
from dataclasses import asdict
from enum import Enum, auto

import gymnasium as gym
import numpy
import numpy as np

# simulator package
from bluebird_dt.core import Pos2D, Pos3D, Pos4D
from bluebird_dt.render.radar import Radar
from gymnasium import spaces

# constants
from bluebird_gymnasium.actions import ACTION_NOOP

# simulator gymnasium wrapper
from bluebird_gymnasium.actions.action_parser import ActionParser
from bluebird_gymnasium.envs import (
    CentralizedSampler,
    Diagnostics,
    EnvConfig,
    SuccessMetric,
    ViewType,
)
from bluebird_gymnasium.rewards import registry_reward_fn
from bluebird_gymnasium.state_repr import registry_repr
from bluebird_gymnasium.state_repr.custom.state_repr_drlan import (
    DrlanRepresentation,
    DrlanRepresentationRaw,
)
from bluebird_gymnasium.utils.constants import (
    DEFAULT_RENDER_DIR,
    DISTANCE_AWAY_FROM_EXIT_THRESHOLD,
    DISTANCE_AWAY_FROM_INCORRECT_EXIT_THRESHOLD,
    DISTANCE_TO_ENTRY_THRESHOLD,
    DISTANCE_TO_EXIT_THRESHOLD,
    DUMMY_CALLSIGN_PREFIX,
    EXIT_WINDOW_WIDTH_DEFAULT,
    FUTURE_TRAJ_DURATION,
    SECTOR_BACKGROUND,
    STEPS_SINCE_ACTION_MAX,
)
from bluebird_gymnasium.utils.geo_utils import (
    at_exit_window,
    get_centreline_distance,
    nearest_360_boundary_position,
    nearest_forward_boundary_position,
)
from bluebird_gymnasium.utils.interaction_utils import TrafficMonitor
from bluebird_gymnasium.utils.policy_utils import default_outcomm_policy
from bluebird_gymnasium.utils.simulator_utils import (
    aircraft_entry_coordination,
    aircraft_exit_coordination,
    basic_distances,
    get_aircraft_selected_cas,
    get_aircraft_selected_flight_level,
    get_aircraft_selected_heading,
    infer_aircraft_speed,
    predict_trajectory,
    prev_next_fixes,
)
from bluebird_gymnasium.utils.types import (
    ACPositionInfo,
    ACStateTracker,
    ForwardFixesInfo,
    PositionStatus,
)

if typing.TYPE_CHECKING:
    import matplotlib
    from bluebird_dt.core.action import Action as SimAction
    from bluebird_dt.core.coordination import Coordination
    from bluebird_dt.core.environment import Environment as SimulatorEnv
    from bluebird_dt.core.predictors import Predictor
    from bluebird_dt.manager import EnvironmentManager as SimulatorEnvManager
    from bluebird_dt.simulator import Simulator
    from numpy.typing import NDArray

    from bluebird_gymnasium.envs import (
        ActionType,
        DoneType,
        InfoType,
        ObsType,
        RewardType,
        TruncatedType,
    )


class ScenarioGenSeedMode(Enum):
    """How `reset(seed=...)` is applied during scenario generation.

    NONE: leave scenario generation untouched; fixed scenario files use this
        so reset seeds do not change their aircraft.
    RESET_SEED_ATTRIBUTE: store the reset seed on `self._reset_seed` while
        `_generate_scenario()` runs, for scenario managers that accept an
        explicit seed argument.
    LEGACY_MODULE_RNGS: temporarily seed module-level `random` and
        `np.random` while `_generate_scenario()` runs, for older scenario
        managers that still read global RNG state.
    """

    NONE = auto()
    RESET_SEED_ATTRIBUTE = auto()
    LEGACY_MODULE_RNGS = auto()


def _concat_state_action(state, action, num_actions):
    assert isinstance(action, int)
    assert action < num_actions

    # convert discrete action to one-hot
    action_one_hot = np.zeros(num_actions, dtype=np.float32)
    action_one_hot[action] = 1.0

    return np.concatenate([state, action_one_hot], dtype=state.dtype)


class BaseEnv(gym.Env):
    """(Gymnasium) Base Environment for the simulator.

    The simulator wrapped around `gymnasium` (gym) API.

    Args:
        render_mode: the mode to visualize (render) the simulator. It can
            only be set to None or one of the following:
            'human', 'rgb_array', 'file'.
            Defaults to `None`.
        config: defines the configuration parameters for the gymnasium
            environment and the underlying simulator.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "file"],
    }

    # how `reset(seed=...)` is applied while generating a scenario
    scenario_seed_mode = ScenarioGenSeedMode.NONE

    def __init__(
        self,
        render_mode: str | None = None,
        config: EnvConfig | None = None,
    ):
        super(BaseEnv, self).__init__()

        self.render_mode = render_mode
        self.radar = None

        # set up configurations
        default_config: EnvConfig = self.get_default_env_config()
        if config is None:
            config = default_config
            self.config = default_config
        else:
            self.config = config

        if self.config.action_config is None:
            self.config.action_config = default_config.action_config

        if self.config.airspace_config is None:
            self.config.airspace_config = default_config.airspace_config

        if self.config.radar_config is None:
            self.config.radar_config = default_config.radar_config

        if self.config.reward_config is None:
            self.config.reward_config = default_config.reward_config

        if self.config.scenario_config is None:
            self.config.scenario_config = default_config.scenario_config

        if self.config.simulation_log_config is None:
            self.config.simulation_log_config = (
                default_config.simulation_log_config
            )

        if self.config.state_repr_config is None:
            self.config.state_repr_config = default_config.state_repr_config

        if self.config.view_config is None:
            self.config.view_config = default_config.view_config

        if self.config.forward_fixes_config is None:
            self.config.forward_fixes_config = (
                default_config.forward_fixes_config
            )

        # used in state representation and route direct actions
        self.forward_fixes_info = ForwardFixesInfo(
            **self.config.forward_fixes_config
        )

        # set the radar render directory to the default if it is unset.
        if self.config.radar_config.get("render_dir", None) is None:
            self.config.radar_config["render_dir"] = DEFAULT_RENDER_DIR

        # reward co-efficient(s)
        if "coeffs" not in self.config.reward_config.keys():
            # default: all reward functions/components are weighted equally
            self.config.reward_config["coeffs"] = [
                1 for _ in self.config.reward_config["fns"]
            ]

        # time steps
        self.scenario_duration = self.config.scenario_duration
        self.scenario_sec_per_step = self.config.scenario_sec_per_step
        self.maxstep = self.scenario_duration // self.scenario_sec_per_step

        # diagnostics
        if self.config.diagnostics_level is None:
            self._diagnostics = lambda callsign: {}
        elif self.config.diagnostics_level == Diagnostics.MINIMAL:
            self._diagnostics = self._minimal_diagnostics
        elif self.config.diagnostics_level == Diagnostics.FULL:
            self._diagnostics = self._full_diagnostics
        else:
            _msg = "diagnostics_level can only be set to one of the "
            "following: {0}".format(list(Diagnostics))
            raise ValueError(_msg)

        # exit window width
        # it defines the maximum lateral deviation (in nautical miles) from
        # a sector exit position (either left or right) that is allowed for
        # an aircraft. If the aircraft exits the sector outside the bound,
        # then such exit is considered unsuccessful
        exit_window_width = self.config.airspace_config.get(
            "exit_window_width", None
        )
        if exit_window_width is None:
            self.exit_window_width = EXIT_WINDOW_WIDTH_DEFAULT
        else:
            self.exit_window_width = exit_window_width

        # use default outcomm policy: it is only valid for use when "outcomm"
        # in action config is set to False.
        if (
            self.config.action_config.get("outcomm", False) is True
            and self.config.use_default_outcomm_policy is True
        ):
            _msg = "`config.use_default_outcomm_policy` can only be set "
            "to True when `config.action_config['outcomm']` is set to False"
            raise ValueError(_msg)
        self.use_default_outcomm_policy = config.use_default_outcomm_policy

        # out-of-sector control
        # check out-of-sector control and default outcomm policy usage
        if (
            self.config.airspace_config.get("out_sector_control", False) is True
            and self.use_default_outcomm_policy is True
        ):
            _msg = "`use_default_outcomm_policy` was set to `True` when "
            "'out_sector_control` in `config.airspace_config` is set to "
            "`True`. "
            "set `use_default_outcomm_policy` to `False` to allow out of "
            "sector control in the airpsace."
            raise ValueError(_msg)
        self._out_sector_control = self.config.airspace_config.get(
            "out_sector_control", False
        )

        # other attributes
        self.timestep = 0
        self.traffic_monitor = TrafficMonitor()

        ## stores the name of the (active) sector in the airspace
        ## set in each child class implementation
        self.active_airspace_sector = None

        ## stores a metric that judges whether an aircraft successfully
        ## navigated through the airspace and exited at the correct fix
        ## and flight level without incorrectly exiting the sector
        self.ac_success_metric: dict[str, SuccessMetric] = {}

        ## stores aircraft flying through the sector, tracking their states
        self.ac_tracker: dict[str, ACStateTracker] = {}
        self.ac_tracker_prev_step: dict[str, ACStateTracker] = {}

        ## store aircraft that has transferred out of sector. aircraft in
        ## `ac_tracker` that have transferred flag set to True are moved here
        self.ac_archived: dict[str, ACStateTracker] = {}

        ## store aircraft that are background traffic (i.e., aircraft whose
        ## flight plan does not go through the sector/airspace). instead,
        ## they fly in past neighborough airspace but not through the current
        ## airspace.
        self.ac_background_traffic: list[str] = []

        ## a buffer to hold a list of callsigns for which an outcomm action
        ## has been issued for a specific time step. it is reset at the end
        ## of each step
        self._current_time_step_outcomm_buffer: list[str] = []

        ## list of chosen callsign per `step(...)`
        self.selected_aircraft: list[str] = []
        self.selected_aircraft_prev_step: list[str] = []

        ## logged simulator actions: log the list actions taken each timestep
        self._logged_actions: list[SimAction] = []

        ## store the history of passed to the gymnasium environment within an
        ## episode run. useful for debugging purpose.
        self._episode_actions: list[ActionType] = []

        ## position status filter for states
        self._pos_status_filter_for_states = []
        if self._out_sector_control:
            self._pos_status_filter_for_states = [
                PositionStatus.BEFORE_ENTRY,
                PositionStatus.IN_SECTOR,
                PositionStatus.OUT_SECTOR,
            ]
        else:
            self._pos_status_filter_for_states = [
                PositionStatus.BEFORE_ENTRY,
                PositionStatus.IN_SECTOR,
            ]

        # state representation, view, action space, and step function
        self.maybe_concat_action = None  # will be set below
        if self.config.view_config["type"] == ViewType.CENTRALIZED:
            ## validate parameters for centralized view
            ## and set it to default if does not exist
            _vp_config = self.config.view_config.get("centralized_params", {})
            _vp_default = default_config.view_config["centralized_params"]
            for k, v in _vp_default.items():
                if _vp_config.get(k, None) is None:
                    _vp_config[k] = _vp_default[k]
            self.config.view_config["centralized_params"] = _vp_config

            if _vp_config["sample_strategy"] not in CentralizedSampler:
                _msg = (
                    "'sample_strategy' in 'view_config' should set "
                    "to one of the following: {0}"
                )
                raise ValueError(_msg.format(list(CentralizedSampler)))

            num_sampled_ac = _vp_config["num_sampled_aircraft"]

            ## action space
            self.action_p = ActionParser(
                self.config.action_config,
                self.forward_fixes_info,
                num_sampled_ac,
            )
            self.action_space = spaces.Discrete(
                self.action_p.get_total_num_actions()
            )

            ## state encoder/representation per aircraft
            knn = self.config.state_repr_config["k_nearest_aircraft"]
            encoder_cls = registry_repr[
                self.config.state_repr_config["encoder_cls"]
            ]
            if isinstance(encoder_cls, DrlanRepresentation) or isinstance(
                encoder_cls, DrlanRepresentationRaw
            ):
                self.state_encoder = encoder_cls(
                    knn=knn,
                    num_forward_fixes=self.forward_fixes_info.num_fixes,
                    use_filed_route=self.forward_fixes_info.use_filed_route,
                    num_actions=self.action_p.get_num_actions_per_aircraft(
                        exclude_noop_action=False,
                    ),
                )
            else:
                self.state_encoder = encoder_cls(
                    knn=knn,
                    num_forward_fixes=self.forward_fixes_info.num_fixes,
                    use_filed_route=self.forward_fixes_info.use_filed_route,
                )

            ## final state view of all aircraft representation
            self.state_formatter = self._state_formatter_centralized

            ## define observation space: define lower and upper limits
            if num_sampled_ac > 1:
                _low = np.concatenate(
                    [self.state_encoder.low for _ in range(num_sampled_ac)],
                    dtype=np.float32,
                )

                _high = np.concatenate(
                    [self.state_encoder.high for _ in range(num_sampled_ac)],
                    dtype=np.float32,
                )
            elif num_sampled_ac == 1:
                _low = self.state_encoder.low
                _high = self.state_encoder.high
            else:
                raise ValueError(
                    (
                        "incorrect config `num_sampled_aircraft` in "
                        "view_config['centralized_params']. the set value"
                        "should be greater than 0"
                    )
                )

            if self.config.view_config.get("concat_prev_action", False):
                # discrete actions will be represented as a one-hot vector
                tmp = np.zeros(
                    self.action_p.get_total_num_actions(),
                )
                _low = np.concatenate([_low, tmp], dtype=np.float32)
                tmp = np.ones(
                    self.action_p.get_total_num_actions(),
                )
                _high = np.concatenate([_high, tmp], dtype=np.float32)

                self.maybe_concat_action = _concat_state_action
            else:
                self.maybe_concat_action = lambda state, act, num_acts: state

            self.observation_space = spaces.Box(
                low=_low,
                high=_high,
                dtype=np.float32,
            )

            ## step function
            self.step_fn = self._step_centralized

            self._prev_timestep_reward = 0.0
            self._prev_timestep_obs = None

        elif self.config.view_config["type"] == ViewType.DECENTRALIZED:
            ## action space
            self.action_p = ActionParser(
                self.config.action_config, self.forward_fixes_info, None
            )
            self.action_space = spaces.Discrete(
                self.action_p.get_total_num_actions()
            )

            ## state encoder/representation per aircraft
            knn = self.config.state_repr_config["k_nearest_aircraft"]
            encoder_cls = registry_repr[
                self.config.state_repr_config["encoder_cls"]
            ]
            if isinstance(encoder_cls, DrlanRepresentation) or isinstance(
                encoder_cls, DrlanRepresentationRaw
            ):
                self.state_encoder = encoder_cls(
                    knn=knn,
                    num_forward_fixes=self.forward_fixes_info.num_fixes,
                    use_filed_route=self.forward_fixes_info.use_filed_route,
                    num_actions=self.action_p.get_num_actions_per_aircraft(
                        exclude_noop_action=False,
                    ),
                )
            else:
                self.state_encoder = encoder_cls(
                    knn=knn,
                    num_forward_fixes=self.forward_fixes_info.num_fixes,
                    use_filed_route=self.forward_fixes_info.use_filed_route,
                )

            ## final state view of all aircraft representation
            self.state_formatter = self._state_formatter_decentralized

            ## define observation space: define lower and upper limits
            _low = self.state_encoder.low
            _high = self.state_encoder.high
            if self.config.view_config.get("concat_prev_action", False):
                # discrete actions will be represented as a one-hot vector
                tmp = np.zeros(
                    self.action_p.get_total_num_actions(),
                )
                _low = np.concatenate([_low, tmp], dtype=np.float32)
                tmp = np.ones(
                    self.action_p.get_total_num_actions(),
                )
                _high = np.concatenate([_high, tmp], dtype=np.float32)

                self.maybe_concat_action = _concat_state_action
            else:
                self.maybe_concat_action = lambda state, act, num_acts: state

            self.observation_space = spaces.Box(
                low=_low,
                high=_high,
                dtype=np.float32,
            )

            ## step function
            self.step_fn = self._step_decentralized

            self._prev_timestep_reward = {}
            self._prev_timestep_obs = None
        else:
            _msg = "Unknown state category. Please specify one of {0}"
            raise ValueError(_msg.format(list(registry_repr.keys())))

        # set in child class
        self.scenario_manager = None
        self.rollout_predictor = None

        # set on every call to `self.reset(...)`
        self.manager = None
        self.simulator = None
        self.simulator_env = None
        self._reset_seed = None

        # used when `.render_mode` is set to 'human'
        self.screen = None

    def _generate_scenario(self) -> Simulator:
        """Generate a scenario in the simulator."""

        raise NotImplementedError

    @contextlib.contextmanager
    def _use_reset_seed_for_scenario_generation(self, seed: int | None):
        """Apply the reset seed while generating a scenario.

        Args:
            seed: seed passed to `.reset(...)`.
        """

        seed_attribute_mode = (
            self.scenario_seed_mode == ScenarioGenSeedMode.RESET_SEED_ATTRIBUTE
        )
        legacy_rng_mode = (
            self.scenario_seed_mode == ScenarioGenSeedMode.LEGACY_MODULE_RNGS
        )

        # pass seed to scenario managers with explicit seed support
        self._reset_seed = seed if seed_attribute_mode else None

        # prepare seed for scenario managers using legacy module-level RNGs
        legacy_seed = seed if legacy_rng_mode else None

        random_state, numpy_random_state = None, None
        try:
            if legacy_seed is not None:
                random_state = random.getstate()
                numpy_random_state = np.random.get_state()

                numpy_seed = np.random.default_rng(legacy_seed).integers(
                    0, 2**32, dtype=np.uint32
                )
                random.seed(legacy_seed)
                np.random.seed(numpy_seed)

            yield

        finally:
            # restore global RNG state and clear the reset seed
            if random_state is not None and numpy_random_state is not None:
                random.setstate(random_state)
                np.random.set_state(numpy_random_state)

            self._reset_seed = None

    def _evolve_simulation(self, evolve_period: int | float) -> None:
        """Step forward in the simulation."""

        self.simulator.evolve(evolve_period)

    def _aircraft_curr_sector(self, callsign: str) -> str:
        """Get the current sector an aircraft is located in.

        Args:
            callsign: the identifier of the aircraft

        Returns:
            the name of the current sector that the aircraft located.
        """

        airspace_sectors = list(self.simulator_env.airspace.sectors.keys())
        if len(airspace_sectors) == 1:
            _sectors = [
                self.active_airspace_sector,
                SECTOR_BACKGROUND,
            ]

            # below should also be correct.
            # _sectors = airspace_sectors.copy() + [SECTOR_BACKGROUND,]
        else:  # > 1 sectors
            _sectors = airspace_sectors

        _sector = self.simulator_env.aircraft[callsign].current_sector
        if _sector not in _sectors:
            raise ValueError(
                f"{callsign} current sector {_sector} is not set to one "
                f"of the following: {_sectors}."
            )

        return self.simulator_env.aircraft[callsign].current_sector

    def _ignore_aircraft(self, callsign: str) -> tuple[bool, bool]:
        """Retrieve the list of aircraft to ignore."""

        # ignore aircraft if it is a background traffic or if it is yet
        # to incomm into the sector and its distance to the entry fix is
        # greater than a set threshold.

        simulator_env = self.simulator_env
        airspace = simulator_env.airspace
        _entry_coord = aircraft_entry_coordination(
            callsign, simulator_env, self.active_airspace_sector
        )

        bkgnd = self._is_background_traffic(callsign, _entry_coord)
        if bkgnd:
            # aircraft is a background traffic, so it should be ignored.
            # no need to check for the second condition (distance to entry).
            ignore_aircraft = True
        else:
            # aircraft is not a background traffic. now, check if it is
            # yet to incomm/enter the sector and if the distance to the
            # entry fix is greater than a set threshold. if true, then
            # the aircraft should be ignored.

            in_sector = self._aircraft_in_sector(callsign)
            curr_sector = simulator_env.aircraft[callsign].current_sector
            if curr_sector == self.active_airspace_sector and in_sector:
                # likely the start (or the first few steps) of the scenario
                # and the aircraft was spawn within the sector.
                # sometimes, the aircraft's current sector is updated
                # (in the simulator) to the `active_airspace_sector` just
                # before it enters the sector.
                ignore_aircraft = False

            elif _entry_coord is not None:
                entry_fix_pos = airspace.fixes.places[_entry_coord.fix]
                aircraft_pos = simulator_env.aircraft[callsign].pos2d()
                _dist = aircraft_pos.distance(entry_fix_pos)
                if _dist > DISTANCE_TO_ENTRY_THRESHOLD:
                    ignore_aircraft = True
                else:
                    ignore_aircraft = False

            else:
                # aircraft does not have entry coordination into the
                # the given sector and it's current_sector is not set
                # to the given sector. Therefore, ignore the aircraft
                msg = (
                    "invalid state for aircraft {0}. it does not have any "
                    "entry coordination into the subject sector {1} and "
                    "its current sector is set as {2}. it should be a "
                    "background aircraft."
                )
                raise ValueError(
                    msg.format(
                        callsign, self.active_airspace_sector, curr_sector
                    )
                )
                # ignore_aircraft = True # use this or raise ValueError

        return ignore_aircraft, bkgnd

    def _is_background_traffic(
        self, callsign: str, entry_coord: None | Coordination
    ) -> bool:
        """Check if an aircraft is a background traffic."""

        # False by default, as is the case for artificial sectors
        # (i.e., X, Y and I sectors; and also springfield sector).

        return False

    def _get_retracking_list(self) -> list[str]:
        """Check if any archived aircraft needs to be reactivated.

        Such reactivation will only occur with aircraft that was archived due
        to incorrect navigation out of the sector (i.e., position status set
        to OUT_SECTOR). Depending on the geometry of the sector (e.g.,
        X sector, or Y sector), a situation might occur where such archived
        aircraft navigates back into the sector. In such a case, the affected
        aircraft needs to be identified (as they'll be placed back into the
        active aircraft tracker `.ac_tracker` in the step function.

        NOTE: the above situation is not always the case. Also, it applies to
        only aircraft with position status set to SECTOR at the
        point when they were archived. Also, such re-entering of the sector
        due to its geometry will only be applicable to lateral sector
        boundaries. An aircraft with an out-of-sector status due to being
        outside the allowed flight level range of the sector will not
        be able to re-enter the sector without it being controlled and
        re-coordinated back in by another agent (in the new sector that the
        aircraft entered when it left the current sector).

        Returns:
            list, of identified aircraft to reactivate. or an empty list if
            there are no aircraft that meet the above condition.
        """

        ret = []
        for callsign, ac_data in self.ac_archived.items():
            # ignore aircraft if last position status was not out-of-sector
            if ac_data.pos_status != PositionStatus.OUT_SECTOR:
                continue

            # ignore aircraft if it no longer exist in the simulator
            if callsign not in self.simulator_env.aircraft.keys():
                continue

            ac = self.simulator_env.aircraft[callsign]
            airspace = self.simulator_env.airspace
            # check whether the aircraft is in the sector
            _sector = airspace.sectors[self.active_airspace_sector]
            if _sector.contains(ac.pos3d()) is True:
                # the aircraft is back in the sector
                ret.append(callsign)

        return ret

    @property
    def out_sector_control(self) -> bool:
        """Get the flag which determine out of sector control."""
        return self._out_sector_control

    @out_sector_control.setter
    def out_sector_control(self, value: bool) -> None:
        """Set the flag which determine out of sector control."""

        assert isinstance(value, bool)
        self._out_sector_control = value

        # update position status filter for states
        if self._out_sector_control:
            self._pos_status_filter_for_states = [
                PositionStatus.BEFORE_ENTRY,
                PositionStatus.IN_SECTOR,
                PositionStatus.OUT_SECTOR,
            ]
        else:
            self._pos_status_filter_for_states = [
                PositionStatus.BEFORE_ENTRY,
                PositionStatus.IN_SECTOR,
            ]

    def get_forward_fixes_info(self) -> ForwardFixesInfo:
        """Get the forward fixes configuration."""
        return self.forward_fixes_info

    def get_simulator(self) -> Simulator:
        """Get the current simulator."""
        return self.simulator

    def get_manager(self) -> SimulatorEnvManager:
        """Get the current simulator environment manager."""
        return self.manager

    def get_simulator_env(self) -> SimulatorEnv:
        """Get the current simulator environment."""
        return self.simulator_env

    def get_active_airspace_sector(self) -> str:
        """Get the active airspace sector in use."""

        return self.active_airspace_sector

    def get_rollout_predictor(self) -> Predictor:
        """Get the predictor used for aircraft future rollout prediction."""
        return self.rollout_predictor

    def get_action_parser(self) -> ActionParser:
        """Get the action parser instance that manages the action space."""
        return self.action_p

    def get_traffic_monitor(self) -> TrafficMonitor:
        """Get the traffic monitor."""
        return self.traffic_monitor

    def get_tracked_aircraft_data_previous(
        self, callsign: None | str = None, copy_data: bool = False
    ) -> dict[str, ACStateTracker] | ACStateTracker | None:
        """Returns aircraft tracked data for the previous step.

        Retrieves either the tracked data of all active aircraft, or the
        tracked data of a specific aircraft as indicated by the `callsign`
        argument.

        Args:
            callsign (str): the callsign to retrieved tracked data.
                Optional, if set to `None`, the tracked data for all
                active aircraft is returned.
                Defaults to `None`.
            copy_data (bool): defines whether a copy of the data should
                be returned.
                Defaults to `False`.

        Returns
            dict of tracked information for all aircraft or tracked
            information for a specific aircraft or `None`.
            if `callsign` is set, the method returns the tracked data for
            the specific aircraft to which the callsign belongs. if the set
            callsign is invalid (i.e, not in the tracker `dict`), then `None`
            is returned.

            if `callsign` is set to None, then a `dict` containing the
            tracked data for all aircraft is returned.
        """

        if callsign is None:
            ret = self.ac_tracker_prev_step
        else:
            ret = self.ac_tracker_prev_step.get(callsign, None)

        if copy_data and ret is not None:
            ret = copy.deepcopy(ret)

        return ret

    def get_tracked_aircraft_data(
        self, callsign: None | str = None, copy_data: bool = False
    ) -> dict[str, ACStateTracker] | ACStateTracker | None:
        """Returns aircraft tracked data.

        Retrieves either the tracked data of all active aircraft, or the
        tracked data of a specific aircraft as indicated by the `callsign`
        argument.

        Args:
            callsign (str): the callsign to retrieved tracked data.
                Optional, if set to `None`, the tracked data for all
                active aircraft is returned.
                Defaults to `None`.
            copy_data (bool): defines whether a copy of the data should
                be returned.
                Defaults to `False`.

        Returns
            dict of tracked information for all aircraft or tracked
            information for a specific aircraft or `None`.
            if `callsign` is set, the method returns the tracked data for
            the specific aircraft to which the callsign belongs. if the set
            callsign is invalid (i.e, not in the tracker `dict`), then `None`
            is returned.

            if `callsign` is set to None, then a `dict` containing the
            tracked data for all aircraft is returned.
        """

        if callsign is None:
            ret = self.ac_tracker
        else:
            ret = self.ac_tracker.get(callsign, None)

        if copy_data and ret is not None:
            ret = copy.deepcopy(ret)

        return ret

    def compute_success_metric(
        self, callsign: str, previous_value: int
    ) -> SuccessMetric:
        """Metric to judge whether an aircraft exit the sector correctly.

        Args:
            callsign: defines the aircraft's identifier.
            previous_value: defines the value of the success metric at the
                previous time step.

        Returns:
            success metric value defined as a tenary integer:
             1 => PASS: if the aircraft correctly exited the sector
            -1 => FAIL: if the aircraft incorrectly exited the sector:
                  either via an excursion (incorrect exit position) or
                  the correct exit position but at a wrong flight level.
             0 => PENDING: if the aircraft is still in the sector
        """

        # aircraft has been deleted from the simulator
        if callsign not in self.simulator_env.aircraft.keys():
            return previous_value

        # aircraft is no longer being tracked due to either its completed
        # navigation through the sector or incorrect sector exit where the
        # out_sector_control flag in config.airspace_config is set to `False`
        if callsign not in self.ac_tracker.keys():
            return previous_value

        aircraft = self.simulator_env.aircraft[callsign]
        # tracked aircraft state: current and previous timesteps
        _state = self.ac_tracker[callsign]
        # if tracked information about the aircraft did not exist
        # in the previous step, then use the current step's state
        _prev_state = self.ac_tracker_prev_step.get(callsign, _state)

        if (
            _state.pos_status == PositionStatus.EXIT_REACHED
            and _prev_state.pos_status == PositionStatus.IN_SECTOR
        ):
            exit_coordination = _state.exit_coords[self.active_airspace_sector]

            if aircraft.fl == exit_coordination.fl:
                metric = SuccessMetric.PASS
            else:
                metric = SuccessMetric.FAIL

        elif (
            _state.pos_status == PositionStatus.OUT_SECTOR
            and _prev_state.pos_status == PositionStatus.IN_SECTOR
        ):
            # aircraft incorrectly exited the airspace (exercusion)
            metric = SuccessMetric.FAIL

        else:
            # the aircraft is in one of the ff scenarios. it is:
            # - yet to incomm into the airspace
            # - still in the sector
            # - was previously assessed as a pass for correct exit
            # - was previously assessed as a fail for incorrect exit
            metric = previous_value

        return metric

    def actions_simulator_to_gym(
        self, actions_st: dict[str, SimAction]
    ) -> tuple[ActionType, dict[str, int], dict[str, tuple[int, str]]]:
        """Helper method to convert simulator actions to gym actions

        Args:
            actions_st: defines the actions to convert to gym actions. a dict,
                with each key-value pair representing a callsign and a
                corresponding simulator action as the value.
        Returns
            three-element tuple:
            - gym action as `int` for centralized setup and `dict` for
              decentralized setup.
            - the reformatted action specific to each aircraft.
            - the action information specific to each aircraft.
        """

        action_p = self.action_p

        actions_rf_int = {}
        actions_info = {}
        for callsign, action_st in actions_st.items():
            action_tuple = action_p.convert_simulator_action_to_gym_action(
                action_st, self
            )
            # action_tuple contains (action_idx, action_str) or None
            actions_info[callsign] = action_tuple

            if action_tuple is None:
                actions_rf_int[callsign] = None
            else:
                actions_rf_int[callsign] = action_tuple[0]

        # returns a single integer value for centralized
        # setup and a dict for decentralized setup
        gym_action_int = action_p.reverse_action_formatter(
            actions_rf_int, self.selected_aircraft_prev_step
        )

        return gym_action_int, actions_rf_int, actions_info

    def actions_gym_to_simulator(
        self, action: ActionType
    ) -> tuple[dict[str, int], dict[str, SimAction]]:
        """Helper method to convert gym env actions to simulator actions

        Args:
            action: defines the gym action to convert to the
                simulator action(s). `int` for centralized setup and `dict`
                for decentralized setup.
        Returns
            two-element tuple:
            - `dict` with each key-value pair representing a callsign and the
              corresponding action (using the integer representation of
              actions per aircraft).
            - `dict` with each key-value pair that represents callsign and a
              corresponding simulator action.
        """

        action_p = self.action_p
        actions_rf_int = action_p.action_formatter(
            action, self.selected_aircraft
        )

        actions_st = {}
        for callsign, action_rf_int in actions_rf_int.items():
            if callsign is None:  # NOOP action (action 0)
                actions_st[callsign] = None
            elif DUMMY_CALLSIGN_PREFIX in callsign:
                actions_st[callsign] = None
            elif (
                self.ac_tracker[callsign].pos_status
                == PositionStatus.BEFORE_ENTRY
            ):
                # aircraft are not controllable before incomm (when they
                # get into the sector/airspace). so, while pre-incomm
                # aircraft are exposed to the agent via the state
                # observation, actions for the aircraft are ignored until
                # incomm.
                actions_st[callsign] = None
            else:
                action_st = action_p.convert_gym_action_to_simulator_action(
                    callsign, action_rf_int, self
                )
                actions_st[callsign] = action_st

        return actions_rf_int, actions_st

    def update_selected_aircraft_prev_step(self, value: list[str]) -> None:
        """Set the list of selected aircraft's callsigns in the previous step"""
        self.selected_aircraft_prev_step.clear()
        for callsign in value:
            self.selected_aircraft_prev_step.append(callsign)

    def update_ac_tracker_prev_step(
        self,
        tracked_data: dict[str, ACStateTracker],
        exclude: list[str] | None = None,
    ) -> None:
        """Set previous step aircraft state tracker to the current data."""

        # self.ac_tracker_prev_step = copy.deepcopy(self.ac_tracker)

        if len(self.ac_tracker) == 0:
            self.ac_tracker_prev_step.clear()

        else:
            keys_1 = set(self.ac_tracker.keys())
            keys_2 = set(self.ac_tracker_prev_step.keys())
            to_del = list(keys_2.difference(keys_1))
            to_add = list(keys_1.difference(keys_2))
            for callsign in to_del:
                del self.ac_tracker_prev_step[callsign]
            for callsign in to_add:
                self.ac_tracker_prev_step[callsign] = ACStateTracker()

            callsign = list(keys_1)[0]
            attributes = list(vars(self.ac_tracker[callsign]).keys())
            if exclude is not None:
                for attribute in exclude:
                    attributes.remove(attribute)

            for callsign, state_tracker in self.ac_tracker.items():
                for attribute in attributes:
                    attr = getattr(state_tracker, attribute)
                    attr_ps = getattr(
                        self.ac_tracker_prev_step[callsign], attribute
                    )
                    if isinstance(attr, Pos2D):
                        # memory management: reuse the same Pos2D object.
                        if attr_ps is None:
                            attr_ps = Pos2D(0.0, 0.0)
                        attr_ps.lat = attr.lat
                        attr_ps.lon = attr.lon
                        setattr(
                            self.ac_tracker_prev_step[callsign],
                            attribute,
                            attr_ps,
                        )
                    else:
                        setattr(
                            self.ac_tracker_prev_step[callsign],
                            attribute,
                            attr,
                        )

        return

    def _set_static_data(self, callsign: str) -> None:
        """Get/compute a one-off data to track for an aircraft.

        The obtained data is computed once, specifically for new aircraft that
        is introduced to the airspace (at the instance of its first appearance
        in the airspace). The data remains unchanged during the course of the
        aircraft being tracked in `self.ac_tracker`. Thus, the name 'static'
        data.

        Args:
            callsign: defines the callsign of the aircraft
        """

        aircraft = self.simulator_env.aircraft[callsign]
        airspace = self.simulator_env.airspace

        # static (one-off) data to track

        ## entry and exit coordinations
        entry_coords = {}
        exit_coords = {}
        _sectors = list(self.simulator_env.airspace.sectors.keys())
        for sector_name in _sectors:
            entry_coords[sector_name] = aircraft_entry_coordination(
                callsign, self.simulator_env, sector_name
            )
            exit_coords[sector_name] = aircraft_exit_coordination(
                callsign, self.simulator_env, sector_name
            )

        self.ac_tracker[callsign].entry_coords = entry_coords
        self.ac_tracker[callsign].exit_coords = exit_coords

        # entry and exit position. entry and exit fixes in the coordination
        # data should be sufficient. however, for the aritificial sectors
        # the entry/exit fixes in the coordination data refers to fixes
        # outside the sector. so, try computing the positions (without
        # using coordination data) and if it fails, gracefully revert back
        # to the fixes in the entry and exit coordination data
        try:
            entry_pos = airspace.get_entry_point(
                aircraft, self.active_airspace_sector
            )
        except:
            entry_fix = entry_coords[self.active_airspace_sector].fix
            entry_pos = airspace.fixes.places[entry_fix]

        try:
            exit_pos = airspace.get_exit_point(
                aircraft, self.active_airspace_sector
            )
        except:
            exit_fix = exit_coords[self.active_airspace_sector].fix
            exit_pos = airspace.fixes.places[exit_fix]

        exit_window = airspace.get_exit_window(
            aircraft, self.exit_window_width, self.active_airspace_sector
        )
        exit_window = (
            Pos2D.from_array(exit_window[0]),
            Pos2D.from_array(exit_window[1]),
        )

        ## should we use entry and exit fix position instead?
        self.ac_tracker[callsign].sector_entry_pos = entry_pos
        self.ac_tracker[callsign].sector_exit_pos = exit_pos
        self.ac_tracker[callsign].sector_exit_window = exit_window
        self.ac_tracker[callsign].sector_entry_timestep = self.timestep

        return

    def _initialize_dynamic_data(self, callsign: str) -> None:
        """Initialize data store for an aircraft tracked dynamic data.

        Initialization only done once, when the aircraft first appear in the
        airspace in the simulation.

        Args:
            callsign: defines the callsign of the aircraft.
        """

        # there are some dynamic data store that needs to be initialised for
        # the aircraft before they are updated per step `.step(...)`

        aircraft = self.simulator_env.aircraft[callsign]

        ## step counter
        self.ac_tracker[callsign].step_counter = -1

        ## note, this initialization assumes that the new aircraft has not
        ## yet incommed. however, some scenarios exist when the aircraft
        ## is initialized within the sector.
        ## the correct position status, incomm and outcomm status will be
        ## computed and set when the dynamic data is updated a few lines
        ## below (i.e. in the for loop where the aircraft tracker dynamic
        ## data is updated).
        self.ac_tracker[callsign].pos_status = PositionStatus.BEFORE_ENTRY
        self.ac_tracker[callsign].incomm_status = False
        self.ac_tracker[callsign].outcomm_status = False
        self.ac_tracker[callsign].dist_to_sector_entry = None
        self.ac_tracker[callsign].dist_away_from_sector_exit = None
        # at initialization we don't yet know if the aircraft exited
        # the sector incorrectly. aircraft is likely entering the
        # sector or in-sector rather than incorrectly exiting it.
        self.ac_tracker[callsign].dist_away_from_incorrect_sector_exit = None
        self.ac_tracker[callsign].incorrect_sector_exit_pos = None

        ## this is updated in safety reward function(s)
        self.ac_tracker[callsign].safety_debug = {}

        ## instantiate steps since action
        self.ac_tracker[callsign].steps_since_action = STEPS_SINCE_ACTION_MAX

        ## initialize position at last route direct
        _route = aircraft.flight_plan.route
        if _route.filed == _route.current:
            # final check
            prev_fix_fr, next_fix_fr = prev_next_fixes(
                callsign, self.simulator_env, use_filed_route=True
            )
            if prev_fix_fr == next_fix_fr:
                # the previous and next fixes are set to the same value (this
                # might occur when the aircraft is spawned before the location
                # of the first fix in its route). assume that its current
                # position is the last route direct location.
                self.ac_tracker[
                    callsign
                ].pos_at_last_route_direct = aircraft.pos2d()

            else:
                # else, initialise it as the previous fix position.
                _pos = self.simulator_env.airspace.fixes.places[prev_fix_fr]
                self.ac_tracker[callsign].pos_at_last_route_direct = _pos
        else:
            # this is more likely to occur in real world sectors.
            # assume there was a route direct at the spawn of the aircraft or
            # a route direct from a previous sector into the current sector.
            # therefore, use the aircraft's current position since
            # the aircraft was just spawned or just started being tracked.
            self.ac_tracker[
                callsign
            ].pos_at_last_route_direct = aircraft.pos2d()

        return

    def _update_dynamic_data(
        self,
        callsign: str,
        callsign_chosen: str | None = None,
        aircraft_action: int | None = None,
        aircraft_sim_action: SimAction | None = None,
    ) -> None:
        """Get/compute tracked dynamic data for an aircraft

        The data in the tracker (i.e., `self.ac_tracker` dict) is continually
        updated at each timestep to obtain the current state of the aircraft.

        Args:
            callsign: defines the callsign/identifier of the aircraft.
            callsign_chosen: defines the aircraft identifier for which an
                action was taken.
            aircraft_action: the action taken for the aircraft. if set to
                `None`, then no action was taken (ACTION_NOOP).
            aircraft_sim_action: the representation of the aircraft's action
                in the simulator's action format.
        """

        assert aircraft_action is not None

        aircraft = self.simulator_env.aircraft[callsign]

        ## position at the last route direct.
        ## note, update this first because `centreline_info_cr` depends
        ## on the updated value for accurate calculation
        if (
            aircraft_sim_action is not None
            and aircraft_sim_action.kind == "route_direct_to"
        ):
            # cache the position where the route direct action was issued.
            # this is used alongside with the aircraft's current route, which
            # is updated by the simulator for route direct action.

            # use the previous step position due to the fact the simulator has
            # already been evolved/updated before the call to this method.
            pos = self.ac_tracker_prev_step[callsign].position.location
            self.ac_tracker[
                callsign
            ].pos_at_last_route_direct = Pos2D.from_array(pos)

        ## distance from the route's centreline (aka centreline distance)
        ## do a computation for both filed route and current route.
        centreline_info_fr = get_centreline_distance(
            aircraft.pos2d(),
            aircraft.flight_plan.route.filed,
            self.simulator_env.airspace,
        )
        self.ac_tracker[callsign].centreline_info_fr = centreline_info_fr

        _route = aircraft.flight_plan.route
        if _route.filed == _route.current:
            # save computation time.
            self.ac_tracker[callsign].centreline_info_cr = centreline_info_fr
        else:
            centreline_info_cr = get_centreline_distance(
                aircraft.pos2d(),
                aircraft.flight_plan.route.current,
                self.simulator_env.airspace,
                self.ac_tracker[callsign].pos_at_last_route_direct,
            )
            self.ac_tracker[callsign].centreline_info_cr = centreline_info_cr

        ## current sector
        curr_sector = self._aircraft_curr_sector(callsign)
        self.ac_tracker[callsign].curr_sector = curr_sector

        ## look ahead into the future (rollout) based on aircraft state
        if self.ac_tracker[callsign].future_trajectory is not None:
            # memory management.
            self.ac_tracker[callsign].future_trajectory.clear()
        self.ac_tracker[callsign].future_trajectory = predict_trajectory(
            aircraft,
            self.rollout_predictor,
            duration=FUTURE_TRAJ_DURATION,
            curr_time=self.simulator_env.time,
        )

        ## previous and next fixes:
        ## based on filed route (fr) and current route (cr)
        prev_fix_fr, next_fix_fr = prev_next_fixes(
            callsign, self.simulator_env, use_filed_route=True
        )
        self.ac_tracker[callsign].previous_fix_fr = prev_fix_fr
        self.ac_tracker[callsign].next_fix_fr = next_fix_fr

        prev_fix_cr, next_fix_cr = prev_next_fixes(
            callsign, self.simulator_env, use_filed_route=False
        )
        self.ac_tracker[callsign].previous_fix_cr = prev_fix_cr
        self.ac_tracker[callsign].next_fix_cr = next_fix_cr

        ## flight level, heading, position, lateral speed, vertical speed
        self.ac_tracker[callsign].flight_level = aircraft.fl
        self.ac_tracker[
            callsign
        ].selected_flight_level = get_aircraft_selected_flight_level(aircraft)
        self.ac_tracker[callsign].heading = aircraft.heading
        self.ac_tracker[
            callsign
        ].selected_heading = get_aircraft_selected_heading(aircraft)
        self.ac_tracker[callsign].position = aircraft.pos2d()
        self.ac_tracker[callsign].speed_tas = aircraft.speed_tas
        self.ac_tracker[callsign].speed_ground = aircraft.ground_speed
        self.ac_tracker[
            callsign
        ].selected_speed_cas = get_aircraft_selected_cas(aircraft)
        self.ac_tracker[callsign].vertical_speed = aircraft.vertical_speed

        ## distance to exit position and distance to exit flight level
        ret = basic_distances(
            aircraft,
            self.simulator_env.airspace,
            self.ac_tracker[callsign].sector_exit_pos,
            self.ac_tracker[callsign]
            .exit_coords[self.active_airspace_sector]
            .fl,
            self.ac_tracker[callsign].pos_at_last_route_direct,
        )
        self.ac_tracker[callsign].linear_dist_to_exit = ret[0]
        self.ac_tracker[callsign].track_dist_to_exit_fr = ret[1]
        self.ac_tracker[callsign].track_dist_to_exit_cr = ret[2]
        self.ac_tracker[callsign].dist_to_target_fl = ret[3]

        ## sector boundary position closest to the aircraft
        ## within the aircraft's 360 degrees periphery.
        ret = nearest_360_boundary_position(
            aircraft,
            self.simulator_env.airspace.sectors[self.active_airspace_sector],
        )
        self.ac_tracker[callsign].nearest_360_boundary_pos = ret[0]
        self.ac_tracker[callsign].nearest_360_boundary_dist = ret[1]
        self.ac_tracker[callsign].nearest_360_boundary_bear = ret[2]

        ## sector boundary position closest to the aircraft forward trajectory
        ## note, requires the updated `next_fix_cr` from above. hence, this
        ## needs to be placed after `next_fix_cr` has been updated above.
        ret = nearest_forward_boundary_position(
            aircraft,
            self.simulator_env.airspace.sectors[self.active_airspace_sector],
            self.simulator_env.airspace.fixes.places[
                self.ac_tracker[callsign].next_fix_cr
            ],
            self.simulator_env.airspace.fixes.places[
                self.ac_tracker[callsign]
                .exit_coords[self.active_airspace_sector]
                .fix
            ],
            self.ac_tracker[callsign].track_dist_to_exit_cr,
        )
        self.ac_tracker[callsign].nearest_forward_boundary_pos = ret[0]
        self.ac_tracker[callsign].nearest_forward_boundary_dist = ret[1]

        ## steps, incomm, outcomm, pos status (in or out sector)
        self.ac_tracker[callsign].step_counter += 1
        _info = self.check_pos_information(
            callsign,
            self.ac_tracker[callsign].pos_status,
            self.ac_tracker[callsign].incomm_status,
            self.ac_tracker[callsign].outcomm_status,
            self.ac_tracker[callsign].incorrect_sector_exit_pos,
        )
        self.ac_tracker[callsign].pos_status = _info.position_status
        self.ac_tracker[callsign].incomm_status = _info.incomm_status
        self.ac_tracker[callsign].outcomm_status = _info.outcomm_status
        self.ac_tracker[
            callsign
        ].dist_to_sector_entry = _info.dist_to_sector_entry
        self.ac_tracker[
            callsign
        ].dist_away_from_sector_exit = _info.dist_away_from_sector_exit
        ### extra data to track: only set when aircraft is out of the
        ### sector (i.e., `.ac_tracker.pos_status` set to OUT_SECTOR)
        self.ac_tracker[
            callsign
        ].dist_away_from_incorrect_sector_exit = (
            _info.dist_away_from_incorrect_sector_exit
        )
        self.ac_tracker[
            callsign
        ].incorrect_sector_exit_pos = _info.incorrect_exit_position

        if callsign_chosen == callsign:
            self.ac_tracker[callsign].steps_since_action = 0
        elif (
            self.ac_tracker[callsign].steps_since_action
            < STEPS_SINCE_ACTION_MAX
        ):
            self.ac_tracker[callsign].steps_since_action += 1

        ### aircraft action
        ### note, in centralized setup, this is the reformatted action
        ### (i.e., action_rf_int`), which are the actions specific to
        ### the aircraft. decentralized setup do not require the reformatting
        ### as each aircraft is an agent with its own set of action.
        self.ac_tracker[callsign].action = aircraft_action

        return

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[ObsType, InfoType]:
        """Environment reset.

        Reset the environment to an initial state, randomising the scenario.

        Args:
            seed (int, optional): seed for random number generator.
                Defaults to None.
            options (dict, optional): additional configuration for the reset.
                Defaults to None.

        Returns:
            tuple, containing two elements, the state representation and
            the log information at the initial state.

            if `.config.view_config["type"]` is set as "centralized",
            then the state representation is a `numpy.ndarray` and the log
            information is a `dict`. If there are no aircraft in the airspace
            at the initial state, the state is represented using a zero
            vector.

            if `.config.view_config["type"]` is set as "decentralized",
            then the state representation is a `dict` with each element
            specific to an aircraft (aircraft callsign `str` as the key and
            `numpy.ndarray as the corresponding value). The log information
            is a `dict` with each element specific to an aircraft (the key
            is the aircraft callsign `str` and the value is a `dict`). If
            there are no aircraft in the airspace at the initial state, the
            state and log dictionaries are empty.
        """
        super(BaseEnv, self).reset(seed=seed, options=options)

        self.timestep = 0

        self.ac_success_metric.clear()
        self.ac_archived.clear()
        self.ac_background_traffic.clear()
        self.ac_tracker.clear()
        self.ac_tracker_prev_step.clear()
        self.selected_aircraft.clear()
        self.selected_aircraft_prev_step.clear()
        self._current_time_step_outcomm_buffer.clear()
        self._logged_actions.clear()
        self._episode_actions.clear()

        # reset the traffic monitor
        self.traffic_monitor.reset()

        # generate the scenario and instantiate simulator
        with self._use_reset_seed_for_scenario_generation(seed):
            self.simulator = self._generate_scenario()
        self.manager = self.simulator.manager
        self.simulator_env = self.manager.environment

        # reset the radar screen
        if self.radar is not None:
            self.radar.reset()

        return self._finish_reset()

    def _finish_reset(self) -> tuple[ObsType, InfoType]:
        # build aircraft callsign list, set transfer status,
        # initial state(s) and info
        _state = {}
        info = {}
        for callsign in self.simulator_env.aircraft.keys():
            ignore_aircraft, _background = self._ignore_aircraft(callsign)
            if ignore_aircraft:
                # before skipping, log aircraft if it is a background traffic
                if _background and callsign not in self.ac_background_traffic:
                    self.ac_background_traffic.append(callsign)
                continue

            # sometimes, an aircraft that was just spawned in a scenario
            # (either in the sector or outside the sector) has its speed
            # attribute (speed_tas) set to `None`. use the trajectory
            # predictor to infer its speed.
            if (
                self.simulator_env.aircraft[callsign].speed_tas is None
                or self.simulator_env.aircraft[callsign].speed_tas == 0.0
                or self.simulator_env.aircraft[callsign].ground_speed is None
                or self.simulator_env.aircraft[callsign].ground_speed == 0.0
            ):
                _speed_tas, _speed_gs = infer_aircraft_speed(
                    self.simulator_env.aircraft[callsign],
                    self.rollout_predictor,
                )
                self.simulator_env.aircraft[callsign].speed_tas = _speed_tas
                self.simulator_env.aircraft[callsign].ground_speed = _speed_gs

            # initialize success metric for the aircraft
            self.ac_success_metric[callsign] = SuccessMetric.PENDING

            # initialize a tracker for the aircraft
            self.ac_tracker[callsign] = ACStateTracker()

            # static (one-off) data in the tracker
            self._set_static_data(callsign)

            # some of the dynamic data store in the tracker
            # require initialization. initialize them.
            self._initialize_dynamic_data(callsign)

            # update aircraft tracker: dynamic data to continually track
            # at reset, no action has been taken for any aircraft. so,
            # set the action to no operation (noop, which is 0)
            self._update_dynamic_data(
                callsign, aircraft_action=ACTION_NOOP, aircraft_sim_action=None
            )

        # update the traffic monitor
        self.traffic_monitor.update(self)

        # now generate the state/obs after setting up `.ac_tracker`
        for callsign in self.ac_tracker.keys():
            # generate the new state(s)/observation(s)
            # only generate states for aircraft that are controllable
            # and are in the sector.
            # note, uncontrollable aircraft could still be encoded in
            # the state but only as a neighbouring aircraft of a
            # represented aircraft.
            _statuses = self._pos_status_filter_for_states
            if (
                self.simulator_env.aircraft[callsign].controllable is True
                and self.ac_tracker[callsign].pos_status in _statuses
            ):
                _state[callsign] = self.state_encoder.repr(
                    self,
                    callsign,
                )
            del _statuses

            info[callsign] = {}

        # state
        state, self.selected_aircraft = self.state_formatter(_state)
        self._prev_timestep_obs = state
        self.update_selected_aircraft_prev_step(self.selected_aircraft)

        # output a png of the radar screen at this step
        if self.render_mode == "human":
            self.render()

        info["simulator_environment"] = self.simulator_env
        return state, info

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, RewardType, DoneType, TruncatedType, InfoType]:
        """Move simulation forward one step.

        Move the simulation forward one step, performing actions and compute
        reward/next observation.

        Args:
            action: action(s) for the aircraft.

        Returns:
            A tuple containing next state, reward, done, truncated, info

            Note: if the value of the "type" key in `state_repr` config is
            "centralized", then next state is `numpy.ndarray`, reward is a
            float, done is a bool, truncated is a bool and info is a dict.

            Otherwise, if "decentralized", then each item in the returned
            tuple is a dict, with key-value pairs as an aircraft callsign and
            the corresponding value for the aircraft. For example, next state
            is a dict with each key-value pair as aircraft callsign and
            `numpy.ndarray`.
        """

        self._logged_actions.clear()

        self._episode_actions.append(action)

        # convert actions from `int` (gym based) to simulator actions
        # also filter out dummy callsign for actions to send to the
        # simulator
        actions_rf_int, actions_st = self.actions_gym_to_simulator(action)
        to_send = []
        for callsign in actions_st.keys():
            if (
                callsign is not None
                and DUMMY_CALLSIGN_PREFIX not in callsign
                and actions_st[callsign] is not None
            ):
                action_st = actions_st[callsign]
                action_rf_int = actions_rf_int[callsign]

                to_send.append(action_st)

                outcomm_actions = self.action_p.get_outcomm_actions()
                assert len(outcomm_actions) in [0, 1]
                if (
                    len(outcomm_actions) == 1
                    and action_rf_int == outcomm_actions[0]
                ):
                    self._current_time_step_outcomm_buffer.append(callsign)

        # send actions to the simulator
        if len(to_send) > 0:
            self.send_actions_to_simulator(to_send)

            self._logged_actions.extend(
                [action for action in to_send if action is not None]
            )

        # get action(s) from default outcomm policy if it is being employed.
        # and send it to the simulator
        if self.use_default_outcomm_policy is True:
            default_outcomm_policy_actions = default_outcomm_policy(
                self,
                actions_rf_int,
                DISTANCE_TO_EXIT_THRESHOLD,
            )
            if len(default_outcomm_policy_actions) > 0:
                to_send_outcomm = list(default_outcomm_policy_actions.values())
                self.send_actions_to_simulator(to_send_outcomm)

                self._logged_actions.extend(
                    [action for action in to_send_outcomm if action is not None]
                )

                self._current_time_step_outcomm_buffer.extend(
                    list(default_outcomm_policy_actions.keys())
                )

        # store the previous state of tracked data before they are updated
        # (in `self.step_fn(...)`) after the simulation is evolved via a call
        # to evolve simulation (update implicity through `self.manager`)
        self.update_ac_tracker_prev_step(
            self.ac_tracker,
            exclude=[
                "future_trajectory",
                "extra_future_trajectory",
                "safety_debug",
            ],
        )
        self.update_selected_aircraft_prev_step(self.selected_aircraft)

        # evolve/step simulation
        self._evolve_simulation(self.scenario_sec_per_step)

        # update steps counter
        self.timestep += 1

        # now call step function specific to centralized or decentralized
        # update aircraft tracker, compute reward and generate the next state
        ret_values = self.step_fn(actions_rf_int, actions_st, action)

        # reset the current time step outcomm buffer
        self._current_time_step_outcomm_buffer.clear()

        # output a png of the radar screen at this step
        if self.render_mode == "human":
            self.render()

        # save the simulation logs to disk if the episode has ended
        # (i.e., done or truncated set to `True`) and "save_simulation"
        # flag in `self.config.simulation_log_config` is set to `True`.
        if (
            ret_values[2] or ret_values[3]
        ) and self.config.simulation_log_config["save_simulation"]:
            self.save_simulation_logs()

        return ret_values

    def _state_formatter_centralized(
        self, _state: dict[str, NDArray[numpy.float32]]
    ) -> tuple[NDArray[numpy.float32], list[str]]:
        """Generate the final state representation for centralized setup.

        This works by:
        1. selecting all aircraft or a subset (if the number of aircraft in
           the scenario at the current timestep is greater than the
           configured view size).
        2. concatenate representation for each selected aircraft into a
           single representation. note that if the number of aircraft in the
           scenario at the currnet timestep is less than the configured view
           size, then a zero-padding is added to this final representation.

        Args
            _state: defines the per aircraft representations, stored in a
                where the key is the callsign for a specific aircraft and the
                value is the representation vector for the aircraft.

        Returns:
            two-element tuple:
            - the final state representation for the current time step.
            - the list of selected aircraft callsigns that was used to
              generate the final state representation.
        """

        # concatenate states from all aircraft into a single vector
        n_ac = self.config.view_config["centralized_params"][
            "num_sampled_aircraft"
        ]
        strategy = self.config.view_config["centralized_params"][
            "sample_strategy"
        ]

        if (
            self.timestep == 0
            and strategy == CentralizedSampler.RANDOM_EPISODAL
        ):
            # the environment was just reset (a new episode). so, reset the
            # centralized sample episodal mask
            num_sampled_aircraft = self.config.view_config[
                "centralized_params"
            ]["num_sampled_aircraft"]
            self._mask_centralized_random_episodal = np.random.permutation(
                num_sampled_aircraft
            )

        _callsigns = list(_state.keys())

        # sort callsigns based on the time step the
        # respective aircraft entered the airspace/sector.
        # note: when > 1 aircraft have the same entry timestep, such ties
        # need to be broken, use entry and exit flight levels and fixes
        def lambda_fn(cs):
            _sector = self.active_airspace_sector
            _en_coord = self.ac_tracker[cs].entry_coords[_sector]
            _ex_coord = self.ac_tracker[cs].exit_coords[_sector]
            return (
                self.ac_tracker[cs].sector_entry_timestep,
                _en_coord.fl,
                _ex_coord.fl,
                _en_coord.fix,
                _ex_coord.fix,
            )

        _callsigns = sorted(
            _callsigns,
            key=lambda_fn,
        )

        if strategy == CentralizedSampler.COMBINED:
            # a combination of all strategies. not yet implemented.
            pass

        if strategy == CentralizedSampler.EARLIEST:
            # nothing to do here. callsigns are already sorted according to
            # the time of each aircraft entry to the sector (earliest entry
            # first and latest entry last)
            pass

        elif strategy == CentralizedSampler.LATEST:
            # reverse the sorted list. now latest sector entry first, and the
            # earliest sector entry stored in the last position of the list)
            _callsigns = _callsigns[::-1]

        elif strategy == CentralizedSampler.RANDOM_STEP:
            # generate new mask every step
            _mask = np.random.permutation(len(_callsigns))
            _callsigns = [_callsigns[idx] for idx in _mask]

        elif strategy == CentralizedSampler.RANDOM_EPISODAL:
            num_active_aircraft = len(_callsigns)
            _mask_all = self._mask_centralized_random_episodal
            # filter mask to only include positions of active aircraft
            _mask = [idx for idx in _mask_all if idx < num_active_aircraft]
            _callsigns = [_callsigns[idx] for idx in _mask]

        if n_ac <= len(_callsigns):
            _callsigns = _callsigns[0:n_ac]

            _callsigns_dummy = []
            _ac_dummy = []
        else:
            _callsigns_dummy = []
            _ac_dummy = []

            for idx in range(n_ac - len(_callsigns)):
                _callsigns_dummy.append(
                    "{0}{1}".format(DUMMY_CALLSIGN_PREFIX, idx)
                )
                _ac_dummy.append(
                    np.zeros(self.state_encoder.low.shape, dtype=np.float32)
                )
        _state = [_state[cs] for cs in _callsigns] + _ac_dummy
        _callsigns += _callsigns_dummy
        return np.concatenate(_state, dtype=np.float32), _callsigns

    def _step_centralized(
        self,
        actions_rf_int: dict[str, int],
        actions_st: dict[str, SimAction],
        original_action: int,
    ) -> tuple[
        NDArray[numpy.float32], float, bool, bool, dict[str, typing.Any]
    ]:
        """Move simulation forward one step.

        Move the simulation forward one step, performing actions and compute
        reward/next observation.

        Args:
            actions_rf_int: the reformatted action specific to the aircraft.
                a one element `dict` in the centralized setup.
            actions_st: the simulator action for the aircraft. a one element
                `dict` in the centralized setup.
            original_action: the original gymnasium action taken by the
                external agent.

        Returns:
            five-element tuple:
            - next state
            - reward
            - done
            - truncated
            - info
        """

        # a single element in the `actions_rf_int` and `actions_st`
        # dicts for the centralized set up
        callsign_chosen = list(actions_st.keys())[0]
        action_st = list(actions_st.values())[0]
        action_rf = list(actions_rf_int.values())[0]
        action = original_action

        # preparation for the block of code below
        s1 = set(self.simulator_env.aircraft.keys())
        s2 = set(self.ac_tracker.keys())
        s3 = set(list(s2) + list(self.ac_archived.keys()))

        # check if new aircraft have entered the airspace
        # since the last call to move the simulator forward.
        # if yes, add them to relevant tracker (e.g., update
        # transferred, sector_entry, and sector_exit dict.
        new_callsigns = list(s1.difference(s3))

        # some archived aircraft may need to be re-tracked. get them.
        new_callsigns += self._get_retracking_list()

        for callsign in new_callsigns:
            ignore_aircraft, _background = self._ignore_aircraft(callsign)
            if ignore_aircraft:
                # before skipping, log aircraft if it is a background traffic
                if _background and callsign not in self.ac_background_traffic:
                    self.ac_background_traffic.append(callsign)
                continue

            # some times, an aircraft that was just spawned in a scenario
            # (either in the sector or outside the sector) has its speed
            # attribute (speed_tas) set to `None`. use the trajectory
            # predictor to infer its speed.
            if (
                self.simulator_env.aircraft[callsign].speed_tas is None
                or self.simulator_env.aircraft[callsign].speed_tas == 0.0
                or self.simulator_env.aircraft[callsign].ground_speed is None
                or self.simulator_env.aircraft[callsign].ground_speed == 0.0
            ):
                _speed_tas, _speed_gs = infer_aircraft_speed(
                    self.simulator_env.aircraft[callsign],
                    self.rollout_predictor,
                )
                self.simulator_env.aircraft[callsign].speed_tas = _speed_tas
                self.simulator_env.aircraft[callsign].ground_speed = _speed_gs

            # initialize success metric for the aircraft
            self.ac_success_metric[callsign] = SuccessMetric.PENDING

            # initialize a tracker for the aircraft
            self.ac_tracker[callsign] = ACStateTracker()

            # static (one-off) data in the tracker
            self._set_static_data(callsign)

            # some of the dynamic data store in the tracker
            # require initialization. initialize them.
            self._initialize_dynamic_data(callsign)

        # unsure if this is a simulator bug or feature
        # check if an existing aircraft (being tracked) have been removed
        # from the airspace by the simulator as a result of the completion
        # of flight through the sector/airspace (hence considered as
        # transferred; the simulator sometimes delete it).
        # if yes, move them from the tracker to archive.
        callsigns_to_move = s2.difference(s1)
        for callsign in callsigns_to_move:
            self.ac_archived[callsign] = self.ac_tracker[callsign]
            del self.ac_tracker[callsign]
        # take note of these callsigns, to be used later.
        callsigns_deleted = callsigns_to_move

        # action map for all aircraft: used for reward computation.
        # initialize map. all set to 0 (no action, aka noop or no_operation).
        actions_in = {}  # action represented as `int` from the agent.
        actions_rf_in = {}  # action represented as `int` reformatted.
        actions_st = {}  # `int` action represented in `simulator` format.
        for callsign in self.ac_tracker.keys():
            actions_in[callsign] = ACTION_NOOP
            actions_rf_in[callsign] = ACTION_NOOP
            actions_st[callsign] = None
        # set the action for the aircraft for which an action was taken
        if (
            callsign_chosen is not None
            and DUMMY_CALLSIGN_PREFIX not in callsign_chosen
        ):
            actions_in[callsign_chosen] = action
            actions_rf_in[callsign_chosen] = action_rf
            actions_st[callsign_chosen] = action_st

        # update aircraft tracker: dynamic data to continually track
        for callsign in self.ac_tracker.keys():
            self._update_dynamic_data(
                callsign,
                callsign_chosen=callsign_chosen,
                aircraft_action=actions_rf_in[callsign],
                aircraft_sim_action=actions_st[callsign],
            )

        ###### useful note:
        ###### after the above update of the dynamic data, `.ac_tracker`
        ###### now holds tracks the state after the simulator evolve step.
        ###### hence, it's no longer the same as the data used in generating
        ###### the state (now referred to as the previous state) for which
        ###### an agent produced the current action being assessed.
        ###### rather, `.ac_tracker` will be used for reward evaluation and
        ###### for generating the next state. for any computation related
        ###### to the previous state, (including where a reward function
        ###### require info about the previous state), use
        ###### `.ac_tracker_prev_step`

        # update the traffic monitor
        self.traffic_monitor.update(self)

        # update the success metrics dictionary
        for callsign in self.ac_success_metric.keys():
            prev_step_sm = self.ac_success_metric[callsign]
            self.ac_success_metric[callsign] = self.compute_success_metric(
                callsign, prev_step_sm
            )

        # now compute the reward, log info, and generate state representation
        # per aircraft.
        # this loop is separated from the previous loop as some reward
        # functions require the dynamic data for all aircraft to have been
        # pre-calculated. e.g., safety.
        _state = {}
        # stores reward for all callsign in `.ac_tracker`
        _rewards = {}
        # callsigns used to compute the final reward returned from the env
        _final_reward_callsigns = []
        # eventually reside in `info` at the end of the function
        ac_info = {}
        info = {}

        for callsign in self.ac_tracker.keys():
            ac_info[callsign] = {}

            # success metric
            ac_info[callsign]["success_metric"] = self.ac_success_metric[
                callsign
            ]

            # position, incomm and outcomm status
            ac_info[callsign]["pos_status"] = self.ac_tracker[
                callsign
            ].pos_status.name
            ac_info[callsign]["incomm_status"] = self.ac_tracker[
                callsign
            ].incomm_status
            ac_info[callsign]["outcomm_status"] = self.ac_tracker[
                callsign
            ].outcomm_status

            # rewards
            _rewards[callsign] = self.compute_reward(
                callsign, actions_rf_in[callsign]
            )

            if callsign in self.ac_tracker_prev_step.keys():
                # previous step (p_) outcomm and position status
                p_outcomm = self.ac_tracker_prev_step[callsign].outcomm_status
                p_pos = self.ac_tracker_prev_step[callsign].pos_status
            else:
                # aircraft was just added, after the simulator was
                # updated/evolved in this current call to `step(...)`.
                # therefore, it's not available in the tracker computed
                # in the previous call time step (`.ac_tracker_prev_step`).

                # so, use the current/next outcomm status and position status,
                # which should be set to `False` and `BEFORE_ENTRY`
                # respectively when the dynamic data was initialized for the
                # new aircraft in an earlier portion of this mdethod.

                # assumed previous step (p_) outcomm and position status
                p_outcomm = self.ac_tracker[callsign].outcomm_status
                p_pos = self.ac_tracker[callsign].pos_status

            if p_pos == PositionStatus.BEFORE_ENTRY:
                # aircraft is yet to incomm. therefore, mask all rewards
                # as zeros (regardless of compute rewards). this is because
                # the aircraft is not yet in the controlled sector.
                # TODO: implement this in an efficient way to avoid
                # computation of rewards if the aircraft has not yet
                # entered the controlled airspace. e.g., this if statement can
                # be placed before a call to the reward function computation
                for key in _rewards[callsign].keys():
                    _rewards[callsign][key] = 0.0
                _final_reward_callsigns.append(callsign)

            elif (
                p_pos
                in [PositionStatus.OUT_SECTOR, PositionStatus.EXIT_REACHED]
                or p_outcomm is True
            ):
                # aircraft that has outcommed, or incorrect navigated out of
                # sector or reached the sector exit it's still tracked for
                # a number of timestep (i.e., in `.ac_tracker`) even though
                # they are not made visible in the state representation
                # (except as a neighbour of a visible/active aircraft).
                # however, their rewards should be zeroed out since the
                # external agent can no longer issue actions to them.
                #
                # note, if `self._out_sector_control` is `True`, then aircraft
                # with position status OUT_SECTOR is included in
                # the final reward computation as the aircraft can still be
                # controlled by the external agent.
                if (
                    p_pos == PositionStatus.OUT_SECTOR
                    and self._out_sector_control is True
                ):
                    _final_reward_callsigns.append(callsign)
                else:
                    for key in _rewards[callsign].keys():
                        _rewards[callsign][key] = 0.0
                    # do not include the aircraft in `_final_reward_callsigns`

            else:
                # aircraft that hasn't outcommed and position status
                # set to in sector PositionStatus.IN_SECTOR
                _final_reward_callsigns.append(callsign)

            ac_info[callsign]["rewards"] = _rewards[callsign]
            ac_info[callsign]["total_reward"] = np.sum(
                list(_rewards[callsign].values())
            )

            # log action in info
            ac_info[callsign]["action_int"] = actions_in[callsign]
            ac_info[callsign]["action_reformat_int"] = actions_rf_in[callsign]
            ac_info[callsign]["action_simulator"] = "{0}".format(
                actions_st[callsign]
            )

            # log diagnostics
            ac_info[callsign]["diagnostics"] = self._diagnostics(callsign)

            # generate the new state(s)/observation(s)
            # only generate states for aircraft that are: controllable, not
            # yet outcommed (transfer status in `.ac_tracker` set to False),
            # and position status as (in sector or pre-incomm)
            # note, uncontrollable aircraft or those transferred out
            # of the sector could still be encoded in the state but only
            # as a neighborouring aircraft of a represented aircraft.
            #
            # therefore, this excludes an aircraft that has outcommed (i.e.,
            # `ac_tracker.outcomm_status` is set to True; whether it's still
            # in the sector or not), or navigated out of the sector
            # (i.e., `ac_tracker.pos_status` is set to OUT_SECTOR) or reached
            # its defined sector exit point (i.e., `.ac_tracker.pos_status`
            # is set to EXIT_REACHED).
            #
            # note, if '._out_sector_control' is `True`, then aircraft with
            # position status OUT_SECTOR are not excluded in the state
            # representation.
            #
            # the exclusion also means that actions cannot be issued to
            # such aircraft (i.e., aircraft in `_state` determines the list
            # of aircraft actions can be issued.
            #
            # note: see __init__ where the filter is defined

            _statuses = self._pos_status_filter_for_states
            if (
                self.simulator_env.aircraft[callsign].controllable is True
                and self.ac_tracker[callsign].outcomm_status is False
                and self.ac_tracker[callsign].pos_status in _statuses
            ):
                _state[callsign] = self.state_encoder.repr(
                    self,
                    callsign,
                )
            del _statuses

        # next state
        next_state, self.selected_aircraft = self.state_formatter(_state)

        # concatenate (previous) action to the (next) state if the
        # `concat_prev_action` flag is set in `self.config.view_config`
        next_state = self.maybe_concat_action(
            next_state, original_action, self.action_p.get_total_num_actions()
        )

        # log more information in outer (main info dict)
        info["default_outcomm_policy"] = (
            self._current_time_step_outcomm_buffer.copy()
        )
        info["simulator_time"] = self.simulator_env.datetime.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        info["action"] = original_action
        info["obs"] = self._prev_timestep_obs
        info["next_obs"] = next_state
        self._prev_timestep_obs = next_state

        # finalize info dict and compute total reward
        # if len(self.ac_tracker) > 0:
        if len(_final_reward_callsigns) > 0:
            reward = np.mean(
                # [ac_info[cs]["total_reward"] for cs in self.ac_tracker.keys()]
                [ac_info[cs]["total_reward"] for cs in _final_reward_callsigns]
            )
            # set 1.0 as upper bound. this is the best performance that can
            # be achieved in a single step except when aircraft transfers
            # out of sector where it receives an additional bonus reward
            self._prev_timestep_reward = min(reward, 1.0)
        else:
            # pad reward when there is no aircraft in the airspace
            # use the reward from the previous time step
            # reward = self._prev_timestep_reward
            reward = 0.0  # TODO: consider removing or keeping
        info["total_step_reward"] = reward

        # check for done/truncated flag
        done = truncated = False
        if self.timestep >= self.maxstep:
            done = truncated = True

        # mop up.
        # check for aircraft that the agent can no longer control.
        # such aircraft are removed from being actively tracked (in
        # `ac_tracker`) and represented in states/observations. the
        # aircraft will be moved from `.ac_tracker` to `.ac_archived` dict
        # if it meets any of the following conditions:
        # (i) the aircraft has outcommed (i.e., its transfer status is
        #     set to True). the outcomm instruction is given before the
        #     aircraft leaves the sector.
        # (ii) the aircraft was not outcommed but it navigated through its
        #      defined *exit fix/window*. such aircraft is still considered
        #      to have exited the sector successfully.
        # (iii) the aircraft navigates *out of the sector* without going
        #       through the defined exit fix/window in its route. this is a
        #       negative situation and agents should optimize their policy
        #       to avoid it.
        #       NOTE: depending on the geometry of the sector/airspace, an
        #       aircraft that goes out of the sector re-enter the sector
        #       even though no control was done on it (left to navigate
        #       based on its last set headings). Even in this case, assume
        #       that the aircraft still cannot be controlled. Once archived,
        #       do no re-activate the tracker. TODO: this assumption can be
        #       revisited in the future and changed (to let agent control
        #       the aircraft if it re-enters the sector) if this is possible
        #       within real-world ATC formulation.
        #       NOTE 2: if `self._out_sector_control` is `True`, do NOT apply
        #       threshold to check when to remove the aircraft from being
        #       actively tracked. Rather, keep aircraft in the active tracker.

        callsigns_to_move = []
        for callsign in self.ac_tracker.keys():
            # next/current (c_) outcomm and position status
            c_outcomm = self.ac_tracker[callsign].outcomm_status
            c_pos = self.ac_tracker[callsign].pos_status
            if (
                c_outcomm is True
                or c_pos == PositionStatus.EXIT_REACHED
                or c_pos == PositionStatus.OUT_SECTOR
            ):
                # do not stopped an out of sector from being tracked (even
                # after distance threshold is reached) if the out of sector
                # control flag is set.
                if (
                    c_pos == PositionStatus.OUT_SECTOR
                    and self._out_sector_control is True
                ):
                    continue

                _d1 = self.ac_tracker[
                    callsign
                ].dist_away_from_incorrect_sector_exit
                _d2 = self.ac_tracker[
                    callsign
                ].dist_away_from_incorrect_sector_exit
                if (
                    _d1 > DISTANCE_AWAY_FROM_EXIT_THRESHOLD
                    or _d2 > DISTANCE_AWAY_FROM_INCORRECT_EXIT_THRESHOLD
                ):
                    callsigns_to_move.append(callsign)

        for callsign in callsigns_to_move:
            self.ac_archived[callsign] = self.ac_tracker[callsign]
            del self.ac_tracker[callsign]

        # final steps: deal with the simulator bug/feature where
        # aircraft are deleted from the airspace (sometimes before
        # they are transferred.
        for callsign in callsigns_deleted:
            ac_info[callsign] = None
            # to consider:
            # should such aircraft be included in `self.ac_archived`?
            # using previous step tracked data in `self.ac_tracker_prev_step`
            # this would mean the archived data for such aircraft would be
            # one step behind other aircraft archived in the current timestep
            # (i.e., as other aircraft use data in `self.ac_tracker`)

        info["aircraft_info"] = ac_info
        info["simulator_environment"] = self.simulator_env
        # also add the success metric to the info dict at a global level
        info["success_metric"] = self.ac_success_metric.copy()

        return next_state, reward, done, truncated, info

    def _state_formatter_decentralized(
        self, _state: dict[str, NDArray[numpy.float32]]
    ) -> dict[str, NDArray[numpy.float32]]:
        """Generate the final state representation for decentralized setup.

        note that the final state representation for the decentralized setup
        is the same `_state` argument passed into the function.

        Args
            _state: defines the per aircraft representations, stored in a
                where the key is the callsign for a specific aircraft and the
                value is the representation vector for the aircraft.

        Returns:
            two-element tuple:
            - the final state representation for the current time step.
            - the list of selected aircraft callsigns that was used to
              generate the final state representation.
        """
        return _state, list(_state.keys())

    def _step_decentralized(
        self,
        actions_rf_int: dict[str, int],
        actions_st: dict[str, SimAction],
        original_actions: dict[str, int],
    ) -> tuple[
        dict[str, NDArray[numpy.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, typing.Any]],
    ]:
        """Move simulation forward one step.

        Move the simulation forward one step, performing actions and compute
        reward/next observation.

        Args:
            actions_rf_int: key specified the callsign of an aircraft and
                value is the action taken for the aircraft.
            actions_st: key specified the callsign of an aircraft and
                value is the simulator action taken for the aircraft.
            original_actions: key specified the callsign of an aircraft and
                value is the action taken for the aircraft.

        Returns:
            A tuple, with elements next state, reward, done, truncated, and
            info. Unlike `_step_centralized`, each element is a `dict`, with
            the key defined as the callsign for the aircraft/agent and value
            as the corresponding data for the aircraft/agent. For example,
            next state is a dict with key of type `str` (the callsign) and
            the value of type `numpy.ndarray`, the state for the aircraft.
        """

        actions = actions_rf_int  # same as original_actions for decentralized

        # preparation for the block of code below
        s1 = set(self.simulator_env.aircraft.keys())
        s2 = set(self.ac_tracker.keys())
        s3 = set(list(s2) + list(self.ac_archived.keys()))

        # check if new aircraft have entered the airspace
        # since the last call to move the simulator forward.
        # if yes, add them to relevant tracker (e.g., set
        # it in action dict 0 (no action), and then update
        # transferred, sector_entry, and sector_exit dict.
        new_callsigns = list(s1.difference(s3))

        # some archived aircraft may need to be re-tracked. get them.
        new_callsigns += self._get_retracking_list()

        for callsign in new_callsigns:
            ignore_aircraft, _background = self._ignore_aircraft(callsign)
            if ignore_aircraft:
                # before skipping, log aircraft if it is a background traffic
                if _background and callsign not in self.ac_background_traffic:
                    self.ac_background_traffic.append(callsign)
                continue

            # sometimes, an aircraft that was just spawned in a scenario
            # (either in the sector or outside the sector) has its speed
            # attribute (speed_tas) set to `None`. use the trajectory
            # predictor to infer its speed.
            if (
                self.simulator_env.aircraft[callsign].speed_tas is None
                or self.simulator_env.aircraft[callsign].speed_tas == 0.0
                or self.simulator_env.aircraft[callsign].ground_speed is None
                or self.simulator_env.aircraft[callsign].ground_speed == 0.0
            ):
                _speed_tas, _speed_gs = infer_aircraft_speed(
                    self.simulator_env.aircraft[callsign],
                    self.rollout_predictor,
                )
                self.simulator_env.aircraft[callsign].speed_tas = _speed_tas
                self.simulator_env.aircraft[callsign].ground_speed = _speed_gs

            # initialize success metric for the aircraft
            self.ac_success_metric[callsign] = SuccessMetric.PENDING

            # initialize a tracker for the aircraft
            self.ac_tracker[callsign] = ACStateTracker()

            # static (one-off) data in the tracker
            self._set_static_data(callsign)

            # some of the dynamic data store in the tracker
            # require initialization. initialize them.
            self._initialize_dynamic_data(callsign)

            # initialise actions for new aircraft
            actions[callsign] = 0  # set to no action
            actions_st[callsign] = None  # set to no action

        # unsure if this is a simulator bug or feature
        # check if an existing aircraft (being tracked) have been removed
        # from the airspace by the simulator as a result of the completion
        # of flight through the sector/airspace (hence considered as
        # transferred; the simulator sometimes delete it).
        # if yes, move them from the tracker to archive.
        callsigns_to_move = s2.difference(s1)
        for callsign in callsigns_to_move:
            self.ac_archived[callsign] = self.ac_tracker[callsign]
            del self.ac_tracker[callsign]
        # take note of these callsigns, to be used later.
        callsigns_deleted = callsigns_to_move

        # update aircraft tracker: dynamic data to continually track
        for callsign in self.ac_tracker.keys():
            self._update_dynamic_data(
                callsign,
                aircraft_action=actions.get(callsign, ACTION_NOOP),
                aircraft_sim_action=actions_st.get(callsign, None),
            )

        ###### useful note:
        ###### after the above update of the dynamic data, `.ac_tracker`
        ###### now holds tracks the state after the simulator evolve step.
        ###### hence, it's no longer the same as the data used in generating
        ###### the state (now referred to as the previous state) for which
        ###### an agent produced the current action being assessed.
        ###### rather, `.ac_tracker` will be used for reward evaluation and
        ###### for generating the next state. for any computation related
        ###### to the previous state, (including where a reward function
        ###### require info about the previous state), use
        ###### `.ac_tracker_prev_step`

        # update the traffic monitor
        self.traffic_monitor.update(self)

        # update the success metrics dictionary
        for callsign in self.ac_success_metric.keys():
            prev_step_sm = self.ac_success_metric[callsign]
            self.ac_success_metric[callsign] = self.compute_success_metric(
                callsign, prev_step_sm
            )

        # now compute the reward, log info, and generate state representation
        # per aircraft.
        # this loop is separated from the previous loop as some reward
        # functions require the dynamic data for all aircraft to have been
        # pre-calculated. e.g., safety.
        _state = {}
        # stores reward for all callsign in `.ac_tracker`
        _rewards = {}
        # callsigns that will be included in the reward dict returned by env
        _final_reward_callsigns = []
        info = {}
        # callsigns that will be included in the done dict returned by env
        _final_dones = {}

        for callsign in self.ac_tracker.keys():
            info[callsign] = {}

            # success metric
            info[callsign]["success_metric"] = self.ac_success_metric[callsign]

            # position, incomm and outcomm status
            info[callsign]["pos_status"] = self.ac_tracker[
                callsign
            ].pos_status.name
            info[callsign]["incomm_status"] = self.ac_tracker[
                callsign
            ].incomm_status
            info[callsign]["outcomm_status"] = self.ac_tracker[
                callsign
            ].outcomm_status

            # rewards
            # if this is a new aircraft, there is no previous
            # action. hence, assume a NOOP action.
            act = actions.get(callsign, ACTION_NOOP)
            _rewards[callsign] = self.compute_reward(callsign, act)

            if callsign in self.ac_tracker_prev_step.keys():
                # previous step (p_) outcomm and position status
                p_outcomm = self.ac_tracker_prev_step[callsign].outcomm_status
                p_pos = self.ac_tracker_prev_step[callsign].pos_status
            else:
                # aircraft was just added, after the simulator was
                # updated/evolved in this current call to `step(...)`.
                # therefore, it's not available in the tracker computed
                # in the previous call time step (`.ac_tracker_prev_step`).

                # so, use the current outcomm status and position status,
                # which should be set to `False` and `BEFORE_ENTRY`
                # respectively when the dynamic data was initialized for the
                # new aircraft in an earlier portion of this mdethod.

                # assumed previous step (p_) outcomm and position status
                p_outcomm = self.ac_tracker[callsign].outcomm_status
                p_pos = self.ac_tracker[callsign].pos_status

            if p_pos == PositionStatus.BEFORE_ENTRY:
                # aircraft is yet to incomm. therefore, mask all rewards
                # as zeros (regardless of compute rewards). this is because
                # the aircraft is not yet in the controlled sector.
                # TODO: implement this in an efficient way to avoid
                # computation of rewards if aircraft has not entered
                # the controlled airspace. e.g., this if system can
                # be placed before a call to the reward function computation
                for key in _rewards[callsign].keys():
                    _rewards[callsign][key] = 0.0
                _final_reward_callsigns.append(callsign)

            elif (
                p_pos
                in [PositionStatus.OUT_SECTOR, PositionStatus.EXIT_REACHED]
                or p_outcomm is True
            ):
                # aircraft that has outcommed, or incorrect navigated out of
                # sector or reached the sector exit it's still tracked for
                # a number of timestep (i.e., in `.ac_tracker`) even though
                # they are not made visible in the state representation
                # (except as a neighbour of a visible/active aircraft).
                # however, their rewards should be zeroed out since the
                # external agent can no longer issue actions to them.
                #
                # note, if `self._out_sector_control` is `True`, then aircraft
                # with position status OUT_SECTOR is included in the final
                # reward computation as the aircraft can still be controlled
                # by the external agent.
                if (
                    p_pos == PositionStatus.OUT_SECTOR
                    and self._out_sector_control is True
                ):
                    _final_reward_callsigns.append(callsign)
                else:
                    for key in _rewards[callsign].keys():
                        _rewards[callsign][key] = 0.0
                    # do not include the aircraft in `_final_reward_callsigns`

            else:
                # aircraft that hasn't outcommed and position status
                # set to in sector `IN_SECTOR`
                _final_reward_callsigns.append(callsign)

            info[callsign]["rewards"] = _rewards[callsign]
            info[callsign]["total_reward"] = np.sum(
                list(_rewards[callsign].values())
            )

            # compute final dones for aircraft
            # if callsign in actions.keys():
            if callsign in _final_reward_callsigns:
                c_outcomm = self.ac_tracker[callsign].outcomm_status
                c_pos = self.ac_tracker[callsign].pos_status
                if c_pos == PositionStatus.EXIT_REACHED or c_outcomm is True:
                    _final_dones[callsign] = True
                elif (
                    c_pos == PositionStatus.OUT_SECTOR
                    and self._out_sector_control is False
                ):
                    _final_dones[callsign] = True
                else:
                    _final_dones[callsign] = False

            # log action in info
            # if this is a new aircraft or an outcommed aircraft, there is
            # no previous action. hence, assume a NOOP action.
            if actions.get(callsign, None) is None:
                info[callsign]["action_int"] = ACTION_NOOP
                info[callsign]["action_simulator"] = None
            else:
                info[callsign]["action_int"] = actions[callsign]
                info[callsign]["action_simulator"] = "{0}".format(
                    actions_st[callsign]
                )

            # log diagnostics
            info[callsign]["diagnostics"] = self._diagnostics(callsign)

            # generate the new state(s)/observation(s)
            # only generate states for aircraft that are: controllable, not
            # yet outcommed (transfer status in `.ac_tracker set to True),
            # and position status as (in sector or pre-incomm)
            # note, uncontrollable aircraft or those transferred out
            # of the sector could still be encoded in the state but only
            # as a neighborouring aircraft of a represented aircraft.
            #
            # therefore, this excludes an aircraft that has outcommed (i.e.,
            # `ac_tracker.outcomm_status` is set to True; whether it's still
            # in the sector or not), or navigated out of the sector
            # (i.e., `ac_tracker.pos_status` is set to OUT_SECTOR) or reached
            # its defined sector exit point (i.e., `.ac_tracker.pos_status`
            # is set to EXIT_REACHED).
            #
            # note, if '._out_sector_control' is `True`, then aircraft with
            # position status OUT_SECTOR are not excluded in the
            # state representation.
            #
            # the exclusion also means that actions cannot be issued to
            # such aircraft (i.e., aircraft in `_state` determines the list
            # of aircraft actions can be issued.
            #
            # note: see __init__ where the filter is defined

            _statuses = self._pos_status_filter_for_states
            if (
                self.simulator_env.aircraft[callsign].controllable is True
                and self.ac_tracker[callsign].outcomm_status is False
                and self.ac_tracker[callsign].pos_status in _statuses
            ):
                _state[callsign] = self.state_encoder.repr(
                    self,
                    callsign,
                )
            del _statuses

        # next state
        next_state, self.selected_aircraft = self.state_formatter(_state)

        # concatenate (previous) action to the (next) state if the
        # `concat_prev_action` flag is set in `self.config.view_config`
        next_state = {
            callsign: self.maybe_concat_action(
                state_vec,
                # aircraft with no action (new or outcommed)
                # should be passed as noop action.
                actions.get(callsign, ACTION_NOOP),
                self.action_p.get_total_num_actions(),
            )
            for callsign, state_vec in next_state.items()
        }

        # log more information
        _callsigns = set(next_state.keys()).intersection(
            set(self._prev_timestep_obs.keys())
        )

        for callsign in _callsigns:
            sim_time = self.simulator_env.datetime
            info[callsign]["simulator_time"] = sim_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            info[callsign]["obs"] = self._prev_timestep_obs[callsign]
            info[callsign]["next_obs"] = next_state[callsign]

        self._prev_step_obs = next_state

        # get the total reward for each aircraft
        reward = {
            # callsign: ac_data["total_reward"]
            # for callsign, ac_data in info.items()
            callsign: info[callsign]["total_reward"]
            for callsign in _final_reward_callsigns
        }

        # check for done/truncated flag
        if self.timestep >= self.maxstep:
            done = {callsign: True for callsign in _final_dones.keys()}
            truncated = {callsign: True for callsign in _final_dones.keys()}
        else:
            done = _final_dones
            truncated = {callsign: False for callsign in _final_dones.keys()}

        # mop up.
        # check for aircraft that the agent can no longer control.
        # such aircraft are removed from being actively tracked (in
        # `ac_tracker`) and represented in states/observations. the
        # aircraft will be moved from `.ac_tracker` to `.ac_archived` dict
        # if it meets any of the following conditions:
        # (i) the aircraft has outcommed (i.e., its transfer status is
        #     set to True). the outcomm instruction is given before the
        #     aircraft leaves the sector.
        # (ii) the aircraft was not outcommed but it navigated through its
        #      defined *exit fix/window*. such aircraft is still considered
        #      to have exited the sector successfully.
        # (iii) the aircraft navigates *out of the sector* without going
        #       through the defined exit fix/window in its route. this is a
        #       negative situation and agents should optimize their policy
        #       to avoid it.
        #       NOTE: depending on the geometry of the sector/airspace, an
        #       aircraft that goes out of the sector re-enter the sector
        #       even though no control was done on it (left to navigate
        #       based on its last set headings). Even in this case, assume
        #       that the aircraft still cannot be controlled. Once archived,
        #       do no re-activate the tracker. TODO: this assumption can be
        #       revisited in the future and changed (to let agent control
        #       the aircraft if it re-enters the sector) if this is possible
        #       within real-world ATC formulation.
        #       NOTE 2: if `self._out_sector_control` is `True`, do NOT apply
        #       threshold to check when to remove the aircraft from being
        #       actively tracked. Rather, keep aircraft in the active tracker.

        callsigns_to_move = []
        for callsign in self.ac_tracker.keys():
            # next/current (c_) outcomm and position status
            c_outcomm = self.ac_tracker[callsign].outcomm_status
            c_pos = self.ac_tracker[callsign].pos_status
            if (
                c_outcomm is True
                or c_pos == PositionStatus.EXIT_REACHED
                or c_pos == PositionStatus.OUT_SECTOR
            ):
                # do not stopped an out of sector from being tracked (even
                # after distance threshold is reached) if the out of sector
                # control flag is set.
                if (
                    c_pos == PositionStatus.OUT_SECTOR
                    and self._out_sector_control is True
                ):
                    continue

                _d1 = self.ac_tracker[callsign].dist_away_from_sector_exit
                _d2 = self.ac_tracker[
                    callsign
                ].dist_away_from_incorrect_sector_exit
                if (
                    _d1 > DISTANCE_AWAY_FROM_EXIT_THRESHOLD
                    or _d2 > DISTANCE_AWAY_FROM_INCORRECT_EXIT_THRESHOLD
                ):
                    callsigns_to_move.append(callsign)

        for callsign in callsigns_to_move:
            self.ac_archived[callsign] = self.ac_tracker[callsign]
            del self.ac_tracker[callsign]

        # final steps: deal with the simulator bug/feature where
        # aircraft are deleted from the airspace (sometimes before
        # it they are transferred.
        for callsign in callsigns_deleted:
            reward[callsign] = 0.0
            done[callsign] = True
            truncated[callsign] = True
            info[callsign] = None
            # to consider:
            # should such aircraft be included in `self.ac_archived`?
            # using previous step tracked data in `self.ac_tracker_prev_step`
            # this would mean the archived data for such aircraft would be
            # one step behind other aircraft archived in the current timestep
            # (i.e., as other aircraft use data in `self.ac_tracker`)

        info["simulator_environment"] = self.simulator_env
        return next_state, reward, done, truncated, info

    def _aircraft_in_sector(self, callsign: str) -> bool:
        """Check whether an aircraft is currently in the defined sector."""

        ac = self.simulator_env.aircraft[callsign]
        airspace = self.simulator_env.airspace

        # note, using only the set sector for the env instance.
        # i.e., `self.active_airspace_sector`
        # assumption: the sector is defined in the airspace
        _sector = airspace.sectors[self.active_airspace_sector]
        in_sector = _sector.contains(ac.pos3d())

        if not in_sector:
            # if there are conditional volumes in the sector, then
            # check them as well.
            _volumes = _sector.get_conditional_volumes_for_aircraft(ac)
            for _volume in _volumes.values():
                if _volume.contains(ac.pos3d()):
                    in_sector = True
                    break

        return in_sector

    def check_pos_information(
        self,
        callsign: str,
        prev_position_status: int,
        prev_incomm_status: bool,
        prev_outcomm_status: bool,
        prev_incorrect_exit_position: None | Pos2D,
    ) -> ACPositionInfo:
        """Check the status of an aircraft in relation to the sector.

        Checks include: the position status, whether incommed, outcommed,
        distance to sector entry (pre-incomm), distance away from sector
        exit (after outcomm and exit from the sector).

        Args:
            callsign: the name of the aircraft.
            prev_position_status: the previous step position status
            prev_incomm_status: the previous step incomm status
            prev_outcomm_status: the previous step outcomm status
            prev_incorrect_exit_position: the previous step tracker of the
                position where an aircraft incorrectly exited the ssector. If
                set to None, it means that the aircraft has not incorrectly
                exited the sector at an earlier time step.

        Returns:
            the position information of the aircraft which contains the
            following information:
            - position status
            - incomm status
            - outcomm status
            - distance to sector entry (> 0 pre-incomm, set to 0.0 afterwards)
            - distance away from sector exit (> 0 only after exiting the
              sector via the correct exit position on its route. set to 0.0
              otherwise.).
            - distance away from incorrect sector exit (> 0 only after exiting
              the sector via an incorrect position on the sector
              boundary (not defined on its route). set to 0.0 otherwise.).
            - incorrect exit position: only updated once, which happens when
              an aircraft that was in sector at the previous time step
              navigates out of the sector (at the current time step) through
              an incorrect 'exit position' (the sector boundary). at other
              times, it maintains what was set in the previous step, i.e.,
              `None` (if there was previously no incorrect sector exit) or
              a position in the sector (if there was an incorrect sector exit
              at an earlier time step).
        """

        ac = self.simulator_env.aircraft[callsign]
        airspace = self.simulator_env.airspace

        # note, using only the set sector for the env instance.
        # i.e., `self.active_airspace_sector`
        # assumption: the sector is defined in the airspace
        in_sector = self._aircraft_in_sector(callsign)

        position_status = None
        incomm_status = None
        outcomm_status = None
        distance_to_sector_entry = None
        distance_away_from_sector_exit = None
        # extra data to track: due to incorrect sector exit
        # i.e., when the position status is OUT_SECTOR
        distance_away_from_incorrect_exit = None
        ## the approximate position (lat/lon) of the sector boundary from
        ## which the aircraft exits
        incorrect_exit_position = None

        if in_sector is False:
            if prev_position_status == PositionStatus.BEFORE_ENTRY:
                # position status
                position_status = PositionStatus.BEFORE_ENTRY

                # incomm status
                incomm_status = False
                entry_pos = self.ac_tracker[callsign].sector_entry_pos
                distance_to_sector_entry = ac.distance(entry_pos)

                # outcomm status
                # there should be no outcomm when the aircraft hasn't incommed
                assert callsign not in self._current_time_step_outcomm_buffer
                outcomm_status = False
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to `OUT_SECTOR`)
                distance_away_from_incorrect_exit = 0.0
                incorrect_exit_position = prev_incorrect_exit_position

            elif prev_position_status == PositionStatus.IN_SECTOR:
                # position status
                position_status = PositionStatus.OUT_SECTOR

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                # was an outcomm clearance issued (in the current time step),
                # given that the aircraft is now out of sector?
                if callsign in self._current_time_step_outcomm_buffer:
                    outcomm_status = True
                else:
                    outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to `OUT_SECTOR`).
                # aircraft just left the sector, so set the distance to 0.0
                distance_away_from_incorrect_exit = 0.0

                # the aircraft was in the sector at the previous time step,
                # but it is now out of sector (in this step). use its
                # current position as an approximation of the sector
                # boundary from whence it incorrectly exited the sector.
                # this check holds true only once for each aircraft that
                # goes out of the sector incorrectly, to correctly store the
                # exit point.
                # NOTE, this ONLY accounts for lateral boundaries,
                # without considering flight levels outside the sector's
                # flight level range (an out of sector aircraft can still
                # be within the lateral sector bounds, but outside the allowed
                # flight level range)
                incorrect_exit_position = ac.pos2d()

            elif prev_position_status == PositionStatus.OUT_SECTOR:
                # position status
                position_status = PositionStatus.OUT_SECTOR

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                # the aircraft is no longer controllable. so, the outcomm
                # status remains the same as the previous time step.
                outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0
                # however, if default env outcomm policy is active, it could
                # stil outcomm out-of-sector aircraft (this ensures that the
                # external agent receives punishment for out-of-sector
                # behaviour before it outcomms the aircraft). therefore,
                # check whether the out-of-sector aircraft has been outcommed
                # by the defaul env outcomm policy and override the set
                # outcomm status.
                if (
                    self.use_default_outcomm_policy
                    and callsign in self._current_time_step_outcomm_buffer
                ):
                    outcomm_status = True

                # out of sector status (only active when the position status
                # is set to `OUT_SECTOR`)
                _p = prev_incorrect_exit_position
                distance_away_from_incorrect_exit = ac.distance(_p)
                incorrect_exit_position = prev_incorrect_exit_position

            elif prev_position_status == PositionStatus.EXIT_REACHED:
                # position status
                position_status = PositionStatus.EXIT_REACHED

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                # the aircraft is no longer controllable. so, the outcomm
                # status remains the same as the previous time step.
                outcomm_status = prev_outcomm_status
                exit_pos = self.ac_tracker[callsign].sector_exit_pos
                distance_away_from_sector_exit = ac.distance(exit_pos)

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                distance_away_from_incorrect_exit = 0.0
                assert prev_incorrect_exit_position is None
                incorrect_exit_position = prev_incorrect_exit_position

            else:
                raise ValueError("Invalid condition reached.")

        else:  # `in_sector` is True
            exit_reached = at_exit_window(
                ac,
                self.ac_tracker[callsign].track_dist_to_exit_cr,  # filed route?
                self.ac_tracker[callsign].sector_exit_pos,
                self.ac_tracker[callsign].sector_exit_window,
            )

            if prev_position_status == PositionStatus.BEFORE_ENTRY:
                # aircraft has just incommed, now that it's inside the sector.
                # position status
                position_status = PositionStatus.IN_SECTOR

                # incomm status
                assert prev_incomm_status is False
                incomm_status = True
                distance_to_sector_entry = 0.0

                # outcomm status
                # aircraft is just being incommed at the current time step.
                # so, there shouldn't be an outcomm clearance for it.
                assert callsign not in self._current_time_step_outcomm_buffer
                outcomm_status = False
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                distance_away_from_incorrect_exit = 0.0
                try:
                    assert prev_incorrect_exit_position is None
                    incorrect_exit_position = prev_incorrect_exit_position
                except:
                    incorrect_exit_position = None

            elif exit_reached is True and prev_incomm_status is True:
                # position status
                position_status = PositionStatus.EXIT_REACHED

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                if callsign in self._current_time_step_outcomm_buffer:
                    outcomm_status = True
                else:
                    outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                distance_away_from_incorrect_exit = 0.0
                try:
                    assert prev_incorrect_exit_position is None
                    incorrect_exit_position = prev_incorrect_exit_position
                except:
                    incorrect_exit_position = None

            elif prev_position_status == PositionStatus.EXIT_REACHED:
                # position status
                position_status = PositionStatus.EXIT_REACHED

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0  # 0 until outside sector

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                distance_away_from_incorrect_exit = 0.0
                try:
                    assert prev_incorrect_exit_position is None
                    incorrect_exit_position = prev_incorrect_exit_position
                except:
                    incorrect_exit_position = None

            elif prev_position_status == PositionStatus.IN_SECTOR:
                # position status
                position_status = PositionStatus.IN_SECTOR

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                if callsign in self._current_time_step_outcomm_buffer:
                    outcomm_status = True
                else:
                    outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                distance_away_from_incorrect_exit = 0.0
                try:
                    assert prev_incorrect_exit_position is None
                    incorrect_exit_position = prev_incorrect_exit_position
                except:
                    incorrect_exit_position = None

            elif prev_position_status == PositionStatus.OUT_SECTOR:
                # special case: aircraft exited the sector incorrectly
                # at a previous step. however, due to the geometry of the
                # sector/airspace, the out-of-sector aircraft is now back
                # into the sector. example of a sector/airspace where this
                # can occur is x sector due to its geometry.
                #
                # two logic/options that can be implemented
                # option 1: assume that the aircraft cannot be re-coordinated
                # back into the sector. so, the aircraft is continually
                # tracked as out-of-sector (inactive in the airspace)
                # option 2: assume that the aircraft was re-coordinated back
                # into the sector (another incomm) and continue tracking
                # the aircraft as an active aircraft in the sector.
                #
                # for the below implementation, option 2 is chosen.

                # position status
                # position_status = PositionStatus.OUT_SECTOR # option 1
                position_status = PositionStatus.IN_SECTOR

                # incomm status
                assert prev_incomm_status is True
                incomm_status = prev_incomm_status
                distance_to_sector_entry = 0.0

                # outcomm status
                outcomm_status = prev_outcomm_status
                distance_away_from_sector_exit = 0.0

                # out of sector status (only active when the position status
                # is set to OUT_SECTOR)
                ## option 1
                ##_p = prev_incorrect_exit_position
                ##distance_away_from_incorrect_exit = ac.distance(_p)
                ##incorrect_exit_position = prev_incorrect_exit_position

                ## option 2
                distance_away_from_incorrect_exit = 0.0
                assert prev_incorrect_exit_position is not None
                incorrect_exit_position = None
            else:
                raise ValueError("Invalid condition reached.")

        return ACPositionInfo(
            position_status,
            incomm_status,
            outcomm_status,
            distance_to_sector_entry,
            distance_away_from_sector_exit,
            distance_away_from_incorrect_exit,
            incorrect_exit_position,
        )

    def compute_reward(self, callsign: str, action: int) -> dict[str, float]:
        """Compute reward given the current state and action.

        Args:
            callsign: the identifier/name of the aircraft.
            action: the action taken by the agent.

        Returns:
            `dict` of the individual reward components for the aircraft.
        """
        # TODO need to fix symmetric for reward functions that is based
        # on aircraft pairs (e.g., safety_simple_avoidance)

        simulator_env = self.simulator_env

        # compute the reward given the current simulator state
        rewards = {}
        ac = simulator_env.aircraft[callsign]
        if ac.controllable:
            for str_fn, coeff in zip(
                self.config.reward_config["fns"],
                self.config.reward_config["coeffs"],
            ):
                fn = registry_reward_fn[str_fn]
                rewards[str_fn] = coeff * fn(self, callsign, action)
        return rewards

    def send_actions_to_simulator(self, actions_st: list[SimAction]) -> None:
        """Send the simulator action(s) to the simulator.

        Args:
            actions_st: stores the action of each aircraft to be sent to the
                simulator. each dict item has its key as an aircraft callsign
                and value as the simulator action for the corresponding
                aircraft.
        """

        # send action to the simulator
        to_send = [action.data() for action in actions_st if action is not None]

        self.simulator.action(to_send)

    def enable_auto_route_following(self) -> None:
        """Enable automatic route following for aircraft."""

        for callsign in self.simulator_env.aircraft:
            self.simulator_env.aircraft[callsign].on_route = True

    def disable_auto_route_following(self) -> None:
        """Disable automatic route following for aircraft."""

        for callsign in self.simulator_env.aircraft:
            self.simulator_env.aircraft[callsign].on_route = False

    def set_radar(self) -> None:
        """Configure the radar for the simulator visualisation."""

        centre = self.config.airspace_config.get("origin", None)
        assert centre is not None
        lat, lon = centre  # latitude and longitude coordinate
        centre = Pos2D(lat, lon)

        aspect_ratio = self.config.radar_config["aspect_ratio"]
        scale = self.config.radar_config["scale"]
        render_fixes = self.config.radar_config.get("render_fixes", False)
        render_routes = self.config.radar_config.get("render_routes", False)
        show_spines = self.config.radar_config.get("show_spines", False)
        display_units = self.config.radar_config.get("display_units", "lonlat")
        scaled = self.config.radar_config.get("scaled", False)
        display_actions = self.config.radar_config.get("display_actions", False)

        self.radar = Radar(
            centre,
            scale,
            aspect_ratio,
            render_fixes=render_fixes,
            render_routes=render_routes,
            sector_name=self.active_airspace_sector,
            show_spines=show_spines,
            display_units=display_units,
            scaled=scaled,
            display_actions=display_actions,
        )

        if not os.path.exists(self.config.radar_config["render_dir"]):
            os.makedirs(self.config.radar_config["render_dir"])

    def get_radar_figure(
        self,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        """Get the figure and axes used to plot the simulation radar display."""

        if self.radar is None:
            self.set_radar()

        figure = self.radar.get_figure()
        ax = figure.get_axes()[0]
        return figure, ax

    def get_render_mode(self) -> str:
        """Retrieve the render mode of the simulator.

        Returns:
            render_mode (str or None): the render mode set. It contains either
                None or one of the following: 'human', 'rgb_array', 'file'.
        """

        return self.render_mode

    def set_render_mode(self, render_mode: str) -> None:
        """Set the render mode of the simulator.

        Args:
            render_mode (str or None): the render mode to set. It can only be
                set to None or one of the following:
                'human', 'rgb_array', 'file'.
        """

        if render_mode is None or render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
        else:
            _msg = (
                "`render_mode` can only be set to `None` or one of the"
                "following: {0}"
            )
            raise ValueError(_msg.format(self.metadata["render_modes"]))

    def render(self) -> None | NDArray[numpy.float32]:
        """Render a frame and save to disk the current simulator state.

        Render a frame based on the current state of simulator and
        save to disk. The rendered frame is an image that is generated using
        the defined radar.
        """

        def _mpl_to_rgb_array(figure, image_format):
            import io

            from PIL import Image

            buffer = io.BytesIO()
            figure.savefig(buffer, format=image_format)
            buffer.seek(0)

            # shape: (height, width, channels)
            image = np.asarray(Image.open(buffer))
            buffer.close()

            return image

        if self.render_mode is None:
            gym.logger.warn(
                "`.render()` called with `render_mode` set to None "
                "(i.e., no render mode was specified). `render_mode` "
                "can be specified as an argument at the initalization of "
                "the environment. It can be set to one of the following: "
                "{0}".format(self.metadata["render_modes"])
            )
            return None

        _msg = (
            "`render_mode` must be set to one of the following before"
            "`render(...`) is called. {}."
        )
        _options = self.metadata["render_modes"]
        assert self.render_mode is not None, _msg.format(_options)

        # for visualisation, output the current simulator state as an image.
        simulator_env = self.simulator_env
        if self.radar is None:
            self.set_radar()

        if self.render_mode == "file":
            filename = os.path.join(
                self.config.radar_config["render_dir"],
                "{0}_{1:03d}.png".format(
                    self.config.radar_config["prefix"], self.timestep
                ),
            )

            self.radar.auto_display = False

            # draw frame
            figure, ax = self.radar.draw(simulator_env, self._logged_actions)

            # save the frame to disk.
            self.radar.save(filename)

            return None

        elif self.render_mode == "human":
            self.radar.auto_display = True

            # draw frame
            figure, ax = self.radar.draw(simulator_env, self._logged_actions)

            # do nothing as the matplotlib handles drawing the radar
            # on screen when `self.radar.draw(...)` is called.

            return None

        elif self.render_mode == "rgb_array":
            self.radar.auto_display = False

            # draw frame
            figure, ax = self.radar.draw(simulator_env, self._logged_actions)

            # save rgb image to buffer and return
            # shape: (height, width, channels)
            return _mpl_to_rgb_array(figure, "png")
        return None

    def render_w_overlay_trajectory(
        self, traj_dict: None | dict[str, list[Pos4D]] = None
    ) -> None:
        """Render a frame and save to disk the current simulator state.

        Render a frame based on the current state of simulator and
        render the alternate trajectory of aircraft received from the external
        caller, and save the generated frame to disk.
        The rendered frame is an image that is generated using
        the defined radar.

        Note, this method is for debugging purpose and should be only used when
        all associated gymnasium wrappers have been removed/stripped and the
        environment object can be accessed directly.

        Also note, only "file" render mode is currently supported.
        """

        msg_ = "This method presently support only render_mode set to 'file'"
        assert self.render_mode in [
            "file",
        ], msg_

        # for visualisation, output the current simulator state as an image.
        simulator_env = self.simulator_env

        if self.radar is None:
            self.set_radar()

        filename = os.path.join(
            self.config.radar_config["render_dir"],
            "{0}_{1:03d}.png".format(
                self.config.radar_config["prefix"], self.timestep
            ),
        )

        # rendered frame is saved to disk by default in the simulator
        self.radar.draw(simulator_env)
        if traj_dict is not None:
            for callsign, trajectory in traj_dict.items():
                self.radar.draw_trajectory(trajectory)

        self.radar.save(os.path.splitext(filename)[0])

    def close(self) -> None:
        if self.radar is not None:
            del self.radar

    def _full_diagnostics(self, callsign: str) -> dict[str, typing.Any]:
        """Return a complete report about an aircraft.

        Args:
            callsign: the identifier of the aircraft.

        Returns:
            dict, containing the diagnostic information which includes:

            - filed route (list)
            - current route (list)
            - next fix (str)
            - position (3d: current lat, long, flight level)
            - sector entry flight level
            - sector exit flight level
            - route following behaviour
            - current heading
            - speed (true air speed)
            - ... and more (from `Aircraft.data()`

            Other data include:
            - tracked state of the aircraft currently in the airspace
        """
        ret = self.simulator_env.aircraft[callsign].data()

        ret["tracked_state"] = {}
        d = asdict(copy.deepcopy(self.ac_tracker[callsign]))
        for k, v in d.items():
            if isinstance(v, Pos2D):
                ret["tracked_state"][k] = v.__str__()
            else:
                ret["tracked_state"][k] = v

        return ret

    def _minimal_diagnostics(self, callsign: str) -> dict[str, typing.Any]:
        """Return a minimal report about an aircraft.

        Args:
            callsign: the identifier of the aircraft.

        Returns:
            dict, containing the diagnostic information which includes:

            - filed route (list)
            - current route (list)
            - next fix (str)
            - position (3d: current lat, long, flight level)
            - sector entry flight level
            - sector exit flight level
            - route following behaviour
            - current heading
            - speed (true air speed)

            Other data include:
            - tracked state of the aircraft currently in the airspace
        """
        aircraft = self.simulator_env.aircraft[callsign]

        ret = {
            "route_filed": aircraft.flight_plan.route.filed,
            "route_current": aircraft.flight_plan.route.current,
            "previous_fix_fr": self.ac_tracker[callsign].previous_fix_fr,
            "next_fix_fr": self.ac_tracker[callsign].next_fix_fr,
            "previous_fix_cr": self.ac_tracker[callsign].previous_fix_cr,
            "next_fix_cr": self.ac_tracker[callsign].next_fix_cr,
            "pos": "{0}".format(aircraft.pos3d()),
            "entry_fl": self.ac_tracker[callsign]
            .entry_coords[self.active_airspace_sector]
            .fl,
            "exit_fl": self.ac_tracker[callsign]
            .exit_coords[self.active_airspace_sector]
            .fl,
            "current_fl": aircraft.fl,
            "selected_fl": aircraft.selected_fl,
            "route_following": aircraft.on_route,
            "heading": aircraft.heading,
            "speed_tas": aircraft.speed_tas,
            "controllable": aircraft.controllable,
        }

        ret["tracked_state"] = {}
        d = asdict(self.ac_tracker[callsign])
        for k, v in d.items():
            if isinstance(v, (Pos2D, Pos3D, Pos4D)):
                ret["tracked_state"][k] = v.__str__()
            elif k == "sector_exit_window":
                ret["tracked_state"][k] = [pos.__str__() for pos in v]
            elif k == "future_trajectory":
                # ret["tracked_state"][k] = [pos4d.__str__() for pos4d in v]
                # skip it for now. # NOTE
                continue
            elif k in ["entry_coord", "exit_coord"]:
                ret["tracked_state"][k] = {
                    sector_name: coord.data() if coord is not None else None
                    for sector_name, coord in v.items()
                }
            else:
                ret["tracked_state"][k] = v

        return ret

    def save_simulation_logs(self) -> None:
        """Save the logs of the simulation.

        Save the logs of the simulation based on the history of the events
        and aircraft trajectory in the simulation. The operation is based on
        the underlying save operation defined in simulator which saves the logs
        to disk as csv and parquet files. The logs are saved to the simulator's
        log directory.

        The saved files can be analyzed and used to replay the simulation
        (example, replay via the HMI).
        """

        if self.simulator is not None:
            self.simulator.save()
        else:
            raise ValueError(
                "`.simulator` should be first initialised (by calling "
                "`.reset(...)` before a simulation can be saved."
            )

    def debug_get_actions_history(self) -> list[ActionType]:
        """Returns the history of actions taken within an episode."""

        return self._episode_actions

    def debug_get_actions_for_aircraft(
        self,
        callsign: str,
    ) -> dict[int, str] | None:
        """Get the available actions for a given aircraft.

        A useful debugging method

        Args:
            callsign: defines the identifier of the aircraft

        Returns:
            the available actions for the aircraft.
            note, in centralized setup, if the aircraft was not selected in
            the generation of state for the current time step, then `None`
            is returned. this does not apply to decentralized setup.
        """
        try:
            aircraft_idx = self.selected_aircraft.index(callsign)
        except:
            return None

        if self.config.view_config["type"] == ViewType.CENTRALIZED:
            action_map = self.action_p.action_formatter_map
            num_actions = self.action_p.get_num_actions_per_aircraft(
                exclude_noop_action=True
            )
            m = {
                i: (aircraft_idx * num_actions) + i
                for i in range(1, num_actions + 1)
            }

            aircraft_action_map = {}
            for action_int, action_str in action_map.items():
                if action_int == ACTION_NOOP:
                    # noop action stays the same in centralized setup
                    reformatted_action_int = action_int
                else:
                    reformatted_action_int = m[action_int]
                aircraft_action_map[reformatted_action_int] = action_str
        else:
            aircraft_action_map = copy.deepcopy(
                self.action_p.action_formatter_map
            )

        return aircraft_action_map

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

        raise NotImplementedError
