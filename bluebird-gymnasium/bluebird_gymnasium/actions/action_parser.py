from __future__ import annotations

from bluebird_dt.core import Action, Pos4D

from bluebird_gymnasium.actions import (
    ACTION_NOOP,
    DEFAULT_RELATIVE_CLIMB_DESCENT,
    DEFAULT_RELATIVE_HEADING,
    DEFAULT_RELATIVE_SPEED,
    DEFAULT_ROUTE_DIRECT,
    DEFAULT_ROUTE_PARALLEL,
    NUM_NOOP_ACTIONS,
    registry_actions,
)

from bluebird_gymnasium.utils.simulator_utils import get_aircraft_selected_cas

import typing

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs import ActionConfig
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import ForwardFixesInfo, Number


def validate_action_list_route_direct(
    action_name: str, data_list: list[Number], route_direct_max_fixes: int
) -> tuple[bool, None | str]:
    """Validate the input values in a route direct action list.

    A validation specific to only route direct actions.

    Args:
        action_name: defines the name of the action.
        data_list: defines the list of route direct values to validate.

    Returns:
        a two-element tuple:
        - validation status
        - message. None if validation status is True. Otherwise, a message
          string which describes the reason for the failed validation.
    """

    # check 1: specific to route direct. check that the highest number of
    # forward fix is less than or equal to route_direct_max_fixes
    valid_route_direct_config = (
        min(data_list) <= route_direct_max_fixes
        and max(data_list) <= route_direct_max_fixes
    )
    if not valid_route_direct_config:
        msg = (
            f"{action_name} in action_config: the values in the list "
            f"should be within the range [1, {route_direct_max_fixes}]."
        )
        return False, msg

    return True, None


def validate_action_list(
    action_name: str, data_list: list[Number]
) -> tuple[bool, None | str]:
    """Validate the input values in an action list.

    A general validation for different action types.

    Args:
        action_name: defines the name of the action.
        data_list: defines the list of action values to validate.

    Returns:
        a two-element tuple:
        - validation status
        - message. None if validation status is True. Otherwise, a message
          string which describes the reason for the failed validation.
    """

    # check 1: the items in the list should be integers or floats
    positive_num_status = [
        (isinstance(v, int) or isinstance(v, float)) and (v > 0)
        for v in data_list
    ]
    positive_num_status = all(positive_num_status)
    if not positive_num_status:
        msg = (
            f"{action_name} in action_config: the list should contain only "
            "integers or floats, and the numbers should be greater than zero."
        )
        return False, msg

    # check 2: there should be no duplicates in the list
    unique_items_list = set(data_list)
    if len(unique_items_list) != len(data_list):
        msg = (
            f"{action_name} in action_config: the list should contain only "
            "unique numbers."
        )
        return False, msg

    return True, None


class ActionParser:
    supported_actions: list[str] = [
        # noop
        "action_noop",
        # heading
        "simple_heading_left",
        "simple_heading_right",
        "simple_heading_route_parallel",
        "simple_heading_maintain_current",
        # climb/descend
        "simple_fl_climb",
        "simple_fl_descent",
        "simple_fl_intermediate",
        "simple_fl_exit",
        # speed
        "simple_speed_increase",
        "simple_speed_decrease",
        "simple_speed_maintain_current",
        "simple_speed_choose_own",
        # route direct
        "simple_route_direct",
        # outcomm
        "simple_outcomm",
    ]

    def __init__(
        self,
        action_config: ActionConfig,
        forward_fixes_info: ForwardFixesInfo,
        num_sampled_aircraft: int | None = None,
    ):
        # No operation (NOOP) action (1 type): do nothing operation
        self._action_formatter_map = {ACTION_NOOP: "action_noop"}

        # define number of actions for a single aircraft
        num_actions = 0

        for action_name, param in action_config.items():
            if isinstance(param, list):
                # validate the list: throws an exception if incorrect
                passed, msg = validate_action_list(action_name, param)
                if not passed:
                    raise ValueError(msg)

                # extra check, specific to route direct and route parallel
                # actions: throws an exception if incorrect.
                if (
                    action_name == "simple_route_direct"
                    or action_name == "simple_heading_route_parallel"
                ):
                    passed, msg = validate_action_list_route_direct(
                        action_name, param, forward_fixes_info.num_fixes
                    )
                    if not passed:
                        raise ValueError(msg)

                for v in param:
                    _name = "{0}__{1}".format(action_name, v)
                    self._action_formatter_map[num_actions + 1] = _name
                    num_actions += 1

            elif isinstance(param, bool) and param is True:
                _to_check_heading = [
                    "simple_heading_left",
                    "simple_heading_right",
                ]
                _to_check_climb_descent = [
                    "simple_fl_climb",
                    "simple_fl_descent",
                ]
                _to_check_speed = [
                    "simple_speed_increase",
                    "simple_speed_decrease",
                ]
                _to_check_route_direct = [
                    "simple_route_direct",
                ]
                _to_check_route_parallel = [
                    "simple_heading_route_parallel",
                ]

                if action_name in _to_check_heading:
                    # use a default value since a list of parameter values
                    # was not specified
                    _name = "{0}__{1}".format(
                        action_name, DEFAULT_RELATIVE_HEADING
                    )
                    self._action_formatter_map[num_actions + 1] = _name

                elif action_name in _to_check_climb_descent:
                    # use a default value since a list of parameter values
                    # was not specified
                    _name = "{0}__{1}".format(
                        action_name, DEFAULT_RELATIVE_CLIMB_DESCENT
                    )
                    self._action_formatter_map[num_actions + 1] = _name

                elif action_name in _to_check_speed:
                    # use a default value since a list of parameter values
                    # was not specified
                    _name = "{0}__{1}".format(
                        action_name, DEFAULT_RELATIVE_SPEED
                    )
                    self._action_formatter_map[num_actions + 1] = _name

                elif action_name in _to_check_route_direct:
                    # use a default value since a list of parameter values
                    # was not specified
                    _name = "{0}__{1}".format(action_name, DEFAULT_ROUTE_DIRECT)
                    self._action_formatter_map[num_actions + 1] = _name

                elif action_name in _to_check_route_parallel:
                    # use a default value since a list of parameter values
                    # was not specified
                    _name = "{0}__{1}".format(
                        action_name, DEFAULT_ROUTE_PARALLEL
                    )
                    self._action_formatter_map[num_actions + 1] = _name

                else:
                    self._action_formatter_map[num_actions + 1] = action_name

                num_actions += 1

            else:
                if param is not False:
                    _msg = (
                        "Incorrect parameter value `{0}` set for `{1}` in "
                        "action config. Values can be set as either `bool` or "
                        "`list`."
                    )
                    raise ValueError(_msg.format(param, action_name))

        self.num_actions_per_aircraft = NUM_NOOP_ACTIONS + num_actions

        if isinstance(num_sampled_aircraft, int) and num_sampled_aircraft > 0:
            # centralized (single agent) action set up.
            # one global noop action rather than each aircraft
            # assigned a noop action
            self.total_num_actions = NUM_NOOP_ACTIONS + (
                num_sampled_aircraft
                * num_actions  # note, multiplication without noop actions
            )

            ## (re-)format int actions received in step to the specific
            ## aircraft with the specific int action for that aircraft
            self.action_formatter = self._action_formatter_centralized
            self.reverse_action_formatter = (
                self._action_formatter_centralized_reversed
            )

        elif num_sampled_aircraft is None:
            # decentralized (multi-agent) action set up.
            self.total_num_actions = self.num_actions_per_aircraft

            ## dict of ints action formatter for decentralized set up
            self.action_formatter = self._action_formatter_decentralized
            self.reverse_action_formatter = (
                self._action_formatter_decentralized_reversed
            )
        else:
            _msg = "Incorrect parameter value for `num_sampled_aircraft`. "
            "It should be set to `None` (for decentralized agent/action set "
            "up) or an `int` > 0 (for centralized agent/action set up)."
            raise ValueError(_msg)

        # define buffers to hold the different categories of `int` actions
        ## relative actions. one or more `int` actions if set
        self.actions_heading_left = []
        self.actions_heading_right = []
        self.actions_fl_climb = []
        self.actions_fl_descent = []
        self.actions_speed_incr = []
        self.actions_speed_decr = []
        self.actions_route_direct = []
        self.actions_heading_parallel = []

        ## absolute actions. only one `int` action if set
        self.actions_noop = []
        self.actions_heading_maintain = []
        self.actions_fl_intermediate = []
        self.actions_fl_exit = []
        self.actions_speed_maintain = []
        self.actions_speed_choose_own = []
        self.actions_outcomm = []

        self.set_action_category_buffers()

    def set_action_category_buffers(self):
        for action_int, action_str in self._action_formatter_map.items():
            action_name = action_str.split("__")[0]

            if action_name == "action_noop":
                self.actions_noop.append(action_int)

            elif action_name == "simple_heading_left":
                self.actions_heading_left.append(action_int)

            elif action_name == "simple_heading_right":
                self.actions_heading_right.append(action_int)

            elif action_name == "simple_heading_route_parallel":
                self.actions_heading_parallel.append(action_int)

            elif action_name == "simple_heading_maintain_current":
                self.actions_heading_maintain.append(action_int)

            elif action_name == "simple_fl_climb":
                self.actions_fl_climb.append(action_int)

            elif action_name == "simple_fl_descent":
                self.actions_fl_descent.append(action_int)

            elif action_name == "simple_fl_intermediate":
                self.actions_fl_intermediate.append(action_int)

            elif action_name == "simple_fl_exit":
                self.actions_fl_exit.append(action_int)

            elif action_name == "simple_speed_increase":
                self.actions_speed_incr.append(action_int)

            elif action_name == "simple_speed_decrease":
                self.actions_speed_decr.append(action_int)

            elif action_name == "simple_speed_maintain_current":
                self.actions_speed_maintain.append(action_int)

            elif action_name == "simple_speed_choose_own":
                self.actions_speed_choose_own.append(action_int)

            elif action_name == "simple_route_direct":
                self.actions_route_direct.append(action_int)

            elif action_name == "simple_outcomm":
                self.actions_outcomm.append(action_int)

            else:
                raise ValueError(
                    f"Invalid action '{action_str}'. Supported actions: \n"
                    f"{self.supported_actions}"
                )

    def get_num_actions_per_aircraft(
        self, exclude_noop_action: bool = True
    ) -> int:
        if exclude_noop_action:
            # should be minus 1
            return self.num_actions_per_aircraft - NUM_NOOP_ACTIONS
        else:
            return self.num_actions_per_aircraft

    def get_total_num_actions(self) -> int:
        return self.total_num_actions

    def _action_formatter_centralized(
        self, action: int, sampled_aircraft: list[str]
    ) -> dict[str, int]:
        """Format integer action to the integer action for a specific aircraft

        This is necessary for centralized (single-agent) setup because the
        gym environment exposes the actions to the agent based on the total
        number of actions per aircraft multiplied by the number of selected
        (sampled aircraft).

        For example, if there are 3 actions are
        available to each aircraft and there are 5 sampled aircraft, then
        the action space for the centralized setup would be 16
        (3 actions per aircraft x 5 aircraft) + 1 no-op/dummy action.
        Hence [0, 1, .... 15]. Note, no-op action is assigned as integer
        action 0 by default.

        When a non-zero action is selected, the aircraft to which the action
        should be applied, and the specific action (of the three actions)
        needs to be computed. This helper method does described task.


        Args:
            action: the action selected from the gym enviornment.
            sampled_aircraft: the list of sampled/selected aircraft at the
                current time step.

        Returns:
            dict, with a single key-value pair. the key is the callsign of
            the aircraft to which the action should be applied, and the value
            is the specific (formatted) integer action from the set of action
            available to the aircraft (i.e. using `._action_formatter_map`)
        """

        # final action formatting based on the action types
        # in the action_config that is set to True

        if action != ACTION_NOOP:
            # note, aircraft actions are 1,2,...exclude action 0 (noop action)

            if action > (self.total_num_actions - 1):
                _msg = (
                    "Invalid action {0}. Valid actions are integers "
                    "between [0, {1}]. Please check `env.action_space.n`."
                )
                raise ValueError(
                    _msg.format(action, self.total_num_actions - 1)
                )

            num_actions_per_aircraft = self.get_num_actions_per_aircraft(True)
            # compute the aircraft to which the action is assigned
            ## reformat (rf) action value to make it easier
            ## to compute which aircraft to assign action
            action_rf = ((action - 1) % num_actions_per_aircraft) + 1
            aircraft_idx = (action - 1) // num_actions_per_aircraft
            callsign_chosen = sampled_aircraft[aircraft_idx]

        else:
            callsign_chosen = None
            action_rf = action  # i.e., 0

        return {callsign_chosen: action_rf}

    def _action_formatter_decentralized(
        self, action: dict[str, int], sampled_aircraft: None | list[str] = None
    ) -> dict[str, int]:
        """Helper method for action formatting.

        In the decentralized (multi-agent) setup, nothing needs to be done
        as each agent/aircraft has its own action space which is number of
        actions available to it. the action argument (which is a `dict`) is
        already in the correct format.

        Args:
            action: defines each key-value as an aircraft's callsign and
                integer action for the aircraft.
            sampled_aircraft (list): the list of sampled/selected aircraft
                at the current time step. Defaults to None in decentralized
                setup as it is not needed.

        Returns:
            `dict`, the action argument passed. it is left unchanged as no
            action formatting is required in the decentralized setup.
        """

        # final action formatting based on the action types
        # in the action_config that is set to True

        # note, `sampled_aircraft` is not used here. it was added to
        # retain compatibility with the `_action_formatter_centralized`
        # method signature as both methods are used in envs.

        if not isinstance(action, dict):
            _msg = "`action` should be of type {0}, not {1}."
            raise ValueError(_msg.format(type({}), type(action)))

        for callsign, act in action.items():
            if act > (self.total_num_actions - 1):
                _msg = (
                    "Invalid action {0}. Valid actions are integers "
                    "between [0, {1}]. Please check `env.action_space.n`."
                )
                raise ValueError(_msg.format(act, self.total_num_actions - 1))

        return action

    def convert_gym_action_to_simulator_action(
        self, callsign: str, action_int: int, gym_env: BaseEnv
    ) -> Action:
        """Convert gym action (`int`) to a simulator action

        Args:
            callsign: defines the identifier of the aircraft.
            action_int: defines the action to convert to simulator action.
            gym_env: defines the gym environment.

        Returns:
            the simulator action.
        """

        ac_tracker = gym_env.get_tracked_aircraft_data()
        active_airspace_sector = gym_env.get_active_airspace_sector()
        simulator_env = gym_env.get_simulator_env()

        if action_int == ACTION_NOOP:
            # no action taken
            a = None

        else:
            action_str = self._action_formatter_map[action_int]
            # action_str is in one of two formats:
            # `<action_name>` or `<action_name>__<action_value>`
            ret = action_str.split("__")
            _fn = registry_actions[ret[0]]

            value = int(ret[1]) if len(ret) == 2 else None
            a = _fn(callsign, gym_env, value)

        return a

    # getter: NOOP action
    def get_noop_actions(self) -> list[int]:
        """Get the no operation (NOOP) action."""
        return self.actions_noop.copy()

    # getter: heading actions
    def get_heading_left_actions(self) -> list[int]:
        """Get the set left turn heading actions."""
        return self.actions_heading_left.copy()

    def get_heading_right_actions(self) -> list[int]:
        """Get the set right turn heading actions."""
        return self.actions_heading_right.copy()

    def get_heading_route_parallel_actions(self) -> list[int]:
        """Get the fly parallel to route heading actions."""
        return self.actions_heading_parallel.copy()

    def get_heading_maintain_current_actions(self) -> list[int]:
        """Get the maintain current heading actions."""
        return self.actions_heading_maintain.copy()

    def get_relative_heading_actions(self) -> list[int]:
        """Get all relative heading actions."""
        return (
            self.get_heading_left_actions()
            + self.get_heading_right_actions()
            + self.get_heading_route_parallel_actions()
        )

    def get_absolute_heading_actions(self) -> list[int]:
        """Get all absolute heading actions."""
        return self.get_heading_maintain_current_actions()

    def get_heading_actions(self) -> list[int]:
        """Get all heading (absolute and relative) actions."""
        return (
            self.get_relative_heading_actions()
            + self.get_absolute_heading_actions()
        )

    # getter: climb/descent actions
    def get_fl_climb_actions(self) -> list[int]:
        """Get the climb actions."""
        return self.actions_fl_climb.copy()

    def get_fl_descent_actions(self) -> list[int]:
        """Get the descent actions."""
        return self.actions_fl_descent.copy()

    def get_fl_intermediate_actions(self) -> list[int]:
        """Get the climb/descent to intermediate flight level actions."""
        return self.actions_fl_intermediate.copy()

    def get_fl_exit_actions(self) -> list[int]:
        """Get the climb/descent to exit flight level actions."""
        return self.actions_fl_exit.copy()

    def get_relative_fl_actions(self) -> list[int]:
        """Get all relative flight level change actions (climb and descent)."""
        return self.get_fl_climb_actions() + self.get_fl_descent_actions()

    def get_absolute_fl_actions(self) -> list[int]:
        """Get all absolute flight level change actions (climb and descent)."""
        return self.get_fl_intermediate_actions() + self.get_fl_exit_actions()

    def get_fl_actions(self) -> list[int]:
        """Get all flight level change actions (absolute and relative)."""
        return self.get_relative_fl_actions() + self.get_absolute_fl_actions()

    # getter: speed actions
    def get_speed_increase_actions(self) -> list[int]:
        """Get the speed increase actions."""
        return self.actions_speed_incr.copy()

    def get_speed_decrease_actions(self) -> list[int]:
        """Get the speed decrease actions."""
        return self.actions_speed_decr.copy()

    def get_speed_maintain_current_actions(self) -> list[int]:
        """Get the maintain current speed actions."""
        return self.actions_speed_maintain.copy()

    def get_speed_choose_actions(self) -> list[int]:
        """Get the choose speed actions."""
        return self.actions_speed_choose_own.copy()

    def get_relative_speed_actions(self) -> list[int]:
        """Get all relative speed (increase and decrease) actions."""
        return (
            self.get_speed_increase_actions()
            + self.get_speed_decrease_actions()
        )

    def get_absolute_speed_actions(self) -> list[int]:
        """Get all absolute speed (increase and decrease) actions."""
        return (
            self.get_speed_maintain_current_actions()
            + self.get_speed_choose_actions()
        )

    def get_speed_actions(self) -> list[int]:
        """Get all speed (absolute and relative) actions."""
        return (
            self.get_relative_speed_actions()
            + self.get_absolute_speed_actions()
        )

    # getter: route direct actions
    def get_route_direct_actions(self) -> list[int]:
        """Get all route direct actions."""
        return self.actions_route_direct.copy()

    # getter: outcomm actions
    def get_outcomm_actions(self) -> list[int]:
        """Get all outcomm actions."""
        return self.actions_outcomm.copy()

    @property
    def action_formatter_map(self) -> dict[int, str]:
        return self._action_formatter_map

    def convert_simulator_action_to_gym_action(
        self, action_st: Action, gym_env: BaseEnv
    ) -> tuple[int, str] | None:
        """Convert simulator action to an integer (`int`) representation.

        This method is the reverse of `convert_gym_action_to_simulator_action`.
        It takes a simulator action and converts it to an integer number based
        on the `.action_formatter_map` dict.

        If the simulator action `action_st` is not supported in an instance
        of an environment (i.e., not specified in the action_config or
        set to False), then `None` is returned.

        Purpose: this is useful when there are existing replay logs from
        simulator that can be used for imitation learning or offline RL.

        Args:
            action_st: defines the simulator
                action to convert to integer action.
            gym_env: defines the gym environment

        Returns:
            tuple of two elements or `None`.
            the tuple contains the action id (`int`) and the string identifier
            of the action specified in `.action_formatter_map`.
            if `None`, is returned it inidicates that the action is not
            supported in the current instance of the environment.
        """

        callsign = action_st.callsign

        tracked_data = gym_env.get_tracked_aircraft_data(
            callsign, copy_data=True
        )
        simulator_env = gym_env.get_simulator_env()
        manager = gym_env.get_manager()
        active_airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

        if (
            action_st.kind == "change_heading_by"
            or action_st.kind == "change_heading_to"
        ):
            # threshold to decide if a heading is parallel to a route segment
            RP_THRESH = 4.0
            # threshold to decide if a heading is a relative left or right turn
            RT_THRESH = 3.0

            tracked_data_p = gym_env.get_tracked_aircraft_data_previous(
                callsign
            )
            if action_st.kind == "change_heading_by":
                heading_diff = action_st.value
                heading_diff = int(round(heading_diff, 0))
                curr_selected_heading = tracked_data.selected_heading

            else:
                # difference betweeen current and previous selected headings
                curr_selected_heading = tracked_data.selected_heading
                if curr_selected_heading != action_st.value:
                    raise ValueError("Inconsistent data: selected heading")

                heading_diff = (
                    curr_selected_heading - tracked_data_p.selected_heading
                )
                heading_diff = int(round(heading_diff, 0))

            ####### first, check if action is a relative turn action
            if heading_diff < 0:
                # get relative left headings actions and their values
                _actions = [
                    (_int_action, self._action_formatter_map[_int_action])
                    for _int_action in self.get_heading_left_actions()
                ]
                relative_headings_values = [
                    -int(x[1].split("__")[1]) for x in _actions
                ]
                relative_headings_actions = _actions

            elif heading_diff > 0:
                # get relative right headings actions and their values
                _actions = [
                    (_int_action, self._action_formatter_map[_int_action])
                    for _int_action in self.get_heading_right_actions()
                ]
                relative_headings_values = [
                    int(x[1].split("__")[1]) for x in _actions
                ]
                relative_headings_actions = _actions

            else:
                # equivalent to maintain present heading. hence set the
                # relative headings actions/values to empty lists.
                relative_headings_actions = []
                relative_headings_values = []

            found_idx = None
            for idx, _value in enumerate(relative_headings_values):
                _diff = abs(abs(heading_diff) - abs(_value))
                if _diff <= RT_THRESH:
                    found_idx = idx
                    break

            if heading_diff == 0:
                modified_action_st = Action(
                    callsign=callsign, kind="maintain_current_heading", value=0
                )
                _action = self.convert_simulator_action_to_gym_action(
                    modified_action_st, gym_env
                )

            elif found_idx is not None:
                # a relative (left or right) turn action
                _action = relative_headings_actions[found_idx]

            ####### if not, check if it is route parallel action
            else:
                _actions = [
                    (_int_action, self._action_formatter_map[_int_action])
                    for _int_action in self.get_heading_route_parallel_actions()
                ]
                route_parallel_values = [
                    int(x[1].split("__")[1]) for x in _actions
                ]
                route_parallel_actions = _actions

                from bluebird_gymnasium.actions.simple.heading import (
                    get_forward_segment_angle,
                )

                forward_fixes_info = gym_env.get_forward_fixes_info()
                angles = [
                    get_forward_segment_angle(
                        simulator_env.aircraft[callsign],
                        simulator_env.airspace,
                        tracked_data,
                        _value,
                        forward_fixes_info.use_filed_route,
                    )
                    for _value in route_parallel_values
                ]

                found_idx = None
                for idx, angle in enumerate(angles):
                    diff = abs(curr_selected_heading - angle)
                    if diff == 0 or diff <= RP_THRESH:
                        found_idx = idx
                        break

                if found_idx is not None:
                    # fly parallel to a route segment which has a similar
                    # angle to the aircraft's selected heading.
                    _action = route_parallel_actions[idx]

                ####### not a relative nor route parallel action. raise error.
                else:
                    prev_selected_heading = tracked_data_p.selected_heading
                    raise ValueError(
                        f"Could not parse heading action {action_st}.\n"
                        "It did not fit into any of the configured relative "
                        "or route parallel heading actions.\n"
                        f"previous selected heading: {prev_selected_heading}\n"
                        f"relative: {relative_headings_values}\n"
                        f"route parallel: {route_parallel_values} => {angles}"
                    )

        elif action_st.kind == "maintain_current_heading":
            _int_actions = self.get_heading_maintain_current_actions()
            _actions = {
                _int_action: self._action_formatter_map[_int_action]
                for _int_action in _int_actions
                if self._action_formatter_map[_int_action]
                == "simple_heading_maintain_current"
            }
            assert len(_actions) == 1
            _actions = list(_actions.items())
            _action = _actions[0]

        elif action_st.kind == "change_flight_level_by":
            if action_st.value > 0.0:
                _int_actions = self.get_fl_climb_actions()
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())
                relative_fls = [float(x[1].split("__")[1]) for x in _actions]

            else:
                _int_actions = self.get_fl_descent_actions()
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())
                relative_fls = [
                    -float(x[1].split("__")[1]) for x in _actions
                ]  # make value negative.

            assert float(action_st.value) in relative_fls
            _idx = relative_fls.index(float(action_st.value))
            _action = _actions[_idx]

        elif (
            action_st.kind == "change_flight_level_to"
            or action_st.kind == "descend_now,level_by_fix"
        ):
            # note: "change_flight_level_to" is used by climb or descend
            # clearances (to the exit or intermediate flight level). however,
            # "descend_now,level_by_fix" is strictly used by descend
            # clearance (to the exit/intermediate flight level) only.

            exit_fl = tracked_data.exit_coords[active_airspace_sector].fl
            exit_fix_name = tracked_data.exit_coords[active_airspace_sector].fix

            action_fl_value = None
            if action_st.kind == "descend_now,level_by_fix":
                action_fl_value = action_st.value[0]
                action_fix_name = action_st.value[1]

                msg = (
                    f"In bluebird_gymnasium, {action_st.kind} is only supported when the "
                    f"fix is the exit fix. aircraft {action_st.callsign} has "
                    f"clearance {action_st.__str__()} but its exit fix is "
                    f"{exit_fix_name}"
                )
                # assert action_fix_name == exit_fix_name, msg
                if action_fix_name != exit_fix_name:
                    print(msg)
                    msg2 = (
                        "To keep the parser from crashing, the fix will be "
                        "set as the exit fix if its a change to exit fl."
                    )
                    print(msg2)
            else:
                action_fl_value = action_st.value
                action_fix_name = None

            if action_fl_value == exit_fl:
                _int_actions = self.get_fl_exit_actions()
                assert len(_int_actions) == 1
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())

            else:
                _int_actions = self.get_fl_intermediate_actions()
                assert len(_int_actions) == 1
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())

            _action = _actions[0]

        elif action_st.kind == "change_cas_to":
            # get selected cas
            selected_cas = get_aircraft_selected_cas(aircraft)

            relative_cas_speed = float(action_st.value - selected_cas)

            if action_st.value is None:
                # fly own speed action
                pass

            elif relative_cas_speed > 0.0:
                # increase speed action
                _int_actions = self.get_speed_increase_actions()
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())
                relative_cas_speeds = [
                    float(x[1].split("__")[1]) for x in _actions
                ]

            elif relative_cas_speed == 0.0:
                # maintain current speed action
                raise NotImplementedError

            else:
                # decrease current speed action
                _int_actions = self.get_speed_decrease_actions()
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())
                relative_cas_speeds = [
                    -float(x[1].split("__")[1]) for x in _actions
                ]  # make value negative.

            assert relative_cas_speed in relative_cas_speeds
            _idx = relative_cas_speeds.index(relative_cas_speed)
            _action = _actions[_idx]

        elif action_st.kind == "route_direct_to":
            forward_fixes_info = gym_env.get_forward_fixes_info()
            if forward_fixes_info.use_filed_route:
                route = aircraft.flight_plan.route.filed
                next_fix = tracked_data.next_fix_fr
            else:
                route = aircraft.flight_plan.route.current
                next_fix = tracked_data.next_fix_cr

            next_fix_idx = route.index(next_fix)
            selected_fix = action_st.value
            selected_fix_idx = route.index(selected_fix)

            assert selected_fix_idx >= next_fix_idx, (
                f"route_direct_to: {action_st.value} should be a future fix"
                f" and not a previous fix for aircraft {action_st.callsign}",
            )

            # if selected fix is next fix, then the result below is set to 1
            # if selected fix is the fix after next, then result is 2...
            relative_future_fix_num = (selected_fix_idx - next_fix_idx) + 1

            _int_actions = self.get_route_direct_actions()
            _actions = {
                _int_action: self._action_formatter_map[_int_action]
                for _int_action in _int_actions
            }
            _actions = list(_actions.items())
            relative_future_fix_nums = [
                int(x[1].split("__")[1]) for x in _actions
            ]
            _idx = relative_future_fix_nums.index(relative_future_fix_num)
            _action = _actions[_idx]

        elif action_st.kind == "outcomm":
            if gym_env.action_config["simple_outcomm"] is True:
                _int_actions = self.get_outcomm_actions()
                assert len(_int_actions) == 1
                _actions = {
                    _int_action: self._action_formatter_map[_int_action]
                    for _int_action in _int_actions
                }
                _actions = list(_actions.items())
                _action = _actions[0]
            else:
                # outcomm clearance was issued for an aircraft in log files but
                # the outcomm action is disabled in current env instance.
                _action = None

        else:
            _action = None

        return _action

    def _action_formatter_centralized_reversed(
        self, actions_rf_dict: dict[str, int], sampled_aircraft: list[str]
    ) -> int | None:
        """Reverse of the `_action_formatter_centralized(...)` method.

        Converts an action dict (should contain a single key-value pair) into
        integer representation which captures all sampled aircraft actions.
        Please see the docstring of `_action_formatter_centralized` for more
        information.

        Args:
            action_rf_dict: defines a dict with a single key-value pair.
                the key is the callsign of the aircraft to which the action
                should be applied, and the value is the specific (formatted)
                integer action from the set of action available to the
                aircraft (i.e. using `._action_formatter_map`)
            sampled_aircraft: the list of sampled/selected aircraft at the
                current time step

        Returns:
            action, `int` (based on the centralized setup action space for the
            gym environment) or `None` if the value in `action_rf_dict` is
            `None` (which indicates that the action issued for the aircraft is
            not supported in the current environment instance).

        """

        num_actions_per_aircraft = self.get_num_actions_per_aircraft(True)
        assert len(actions_rf_dict) == 1
        callsign_chosen, action_rf = list(actions_rf_dict.items())[0]

        if callsign_chosen not in sampled_aircraft:
            # the issued action is for an aircraft that is not selected for
            # the current time step (i.e., `gym_env.selected_aircraft`).
            # hence ignore it.
            gym_action_int = None
        elif action_rf is None:
            # this indiciates that the action is not supported
            # in the current env instannce. see the docstring of
            # `.convert_simulator_action_to_gym_action(...)` for more details
            gym_action_int = None
        else:
            aircraft_idx = sampled_aircraft.index(callsign_chosen)
            gym_action_int = (
                aircraft_idx * num_actions_per_aircraft
            ) + action_rf

        return gym_action_int

    def _action_formatter_decentralized_reversed(
        self,
        actions_dict: dict[str, int],
        sampled_aircraft: list[str] | None = None,
    ) -> dict[str, int]:
        """Reverse of the `_action_formatter_decentralized(...)` method

        No reverse formatting needs to be done for the decentralized setup as
        the actions from the gym env are never formatted as they're already in
        the correct format.

        Args:
            action_dict (dict): defines each key-value as an aircraft's
                callsign and integer action for the aircraft.
            sampled_aircraft (list): the list of sampled/selected aircraft
                at the current time step. Defaults to None in decentralized
                setup as it is not needed.

        Returns:
            `dict`, the actions_int argument passed. it is left unchanged as no
            reverse action formatting is required in the decentralized setup.
        """

        # do nothing.
        # decentralized setup does not use action formatting.
        # see `._action_formatter_decentralized(...)`
        return actions_int
