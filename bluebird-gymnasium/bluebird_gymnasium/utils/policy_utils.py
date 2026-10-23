from __future__ import annotations

import typing

from bluebird_dt.core import Action

from bluebird_gymnasium.actions import ACTION_NOOP
from bluebird_gymnasium.utils.geo_utils import angle_diff, left_right_check
from bluebird_gymnasium.utils.simulator_utils import aircraft_exit_coordination
from bluebird_gymnasium.utils.types import PositionStatus, TurnDirection

if typing.TYPE_CHECKING:
    from bluebird_dt.core.environment import Environment as SimulatorEnv

    from bluebird_gymnasium.envs.base import BaseEnv


def _get_relevant_data(
    ac_tracked_state: dict[str, typing.Any],
    callsign: str,
    simulator_env: SimulatorEnv,
    airspace_sector: str,
) -> tuple[float, str]:
    """Get the exit flight level and next sector of an aircraft."""

    if ac_tracked_state is None:
        exit_coord = aircraft_exit_coordination(callsign, simulator_env, airspace_sector)
        exit_fl = exit_coord.fl
        next_sector = exit_coord.to_sector

    else:
        exit_coord = ac_tracked_state.exit_coords[airspace_sector]
        exit_fl = exit_coord.fl
        next_sector = exit_coord.to_sector

    return exit_fl, next_sector


def default_outcomm_policy_simple(
    gym_env: BaseEnv,
    external_agent_actions: dict[str, int],  # noqa: ARG001
    exit_distance_threshold: float = 10.0,
    angle_threshold: float = 5.0,
    single_action: bool = True,
    agent: str = "Outcomm Agent",
) -> dict[str, Action]:
    """A simple policy for outcomming aircraft.

    The policy scans through all aircraft in a sector/airspace and issues
    an outcomm clearance for an aircraft if any of the defined conditions
    are met.
    condition 1:
        - if an aircraft is < 10 nautical miles from the exit fix/window,
        - and its cleared flight level is set to the exit flight level,
        - and its heading the same as the bearing of the fix before its exit
        - and its exit fix ( +/- 5 degrees tolerance),
        - then issue an outcomm clearance to the aircraft.
    condition 2:
        - if an aircraft is < 1 nautical mile from the exit fix/window,
        - then issue an outcomm clearance to the aircraft (regardless of
        - its cleared flight level and heading).
        - then issue an outcomm clearance to the aircraft.
    condition 3:
        - if an aircraft is about to navigate out of the sector (i.e.,
          incorrect; not via its exit fix) and out of sector control is False,
        - then issue an outcomm clearance to the aircraft.
    condition 4 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the minimum allowable flight level in the sector),
          and out of sector control is False,
        - then issue an outcomm clearance to the aircraft.
    condition 5 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the maximum allowable flight level in the sector),
          and out of sector control is False,
        - then issue an outcomm clearance to the aircraft.

    Args:
        gym_env: specifies an instance of a gymnasium environment.
        external_agent_actions: defines the external agent's actions. each
            key-value specifies the action taken for an aircraft (i.e., the
            key is the callsign and the value is the given action).
        exit_distance_threshold: defines the maximum distance (in nautical
            miles) from an aircraft's sector exit position in which an outcomm
            action can be initiated.
            Optional. Defaults to 10.0 nautical miles.
        angle_threshold: defines the maximum lateral deviation (in degrees)
            of the an aircraft's heading relative to the target heading
            required to exit the sector (the target heading is the bearing of
            the exit position from the fix before the position.)
            Optional. Defaults to 5.0 degrees.
        single_action: defines whether or not to return a single action if
            there are multiple aircraft eligible for an outcomm action. if set
            to `True`, and there multiple eligible aircraft, then priority is
            given to the aircraft that requires an immediate outcomm clearance
            (e.g., an aircraft out of sector requires an immediate clearance
            in comparison to an aircraft that is 8 nautical miles away from
            its defined route sector exit).
            Optional. Defaults to True.
        agent: defines the name of the outcomm agent.

    Returns:
        dict, each key-value pair defines an aircraft's callsign and
        corresponding outcomm action (in the simulator's action format).
        The returned dict will be empty if there are no aircraft that satisfy
        the defined outcomm conditions.
    """

    # constants
    OUTCOMM_PRIORITY_LOW = 1
    OUTCOMM_PRIORITY_MEDIUM = 2
    OUTCOMM_PRIORITY_HIGH = 3

    # get the simulator
    simulator_env = gym_env.get_simulator_env()

    # get the tracked data for all active aircraft
    all_tracked_aircraft = gym_env.get_tracked_aircraft_data()

    # get active airspace sector
    airspace_sector = gym_env.get_active_airspace_sector()

    actions = {}
    outcomm_priorities = {}
    for callsign, ac_tracked_data in all_tracked_aircraft.items():
        # get the aircraft
        aircraft = simulator_env.aircraft[callsign]

        # exit flight level
        exit_fl, next_sector = _get_relevant_data(
            ac_tracked_data,
            callsign,
            simulator_env,
            airspace_sector,
        )

        # instantiate the outcomm action
        action = Action(
            callsign,
            kind="outcomm",
            value=next_sector,
            agent=agent,
        )

        if ac_tracked_data.pos_status == PositionStatus.BEFORE_ENTRY:
            # nothing to do here as aircraft has not yet incommed
            continue

        if ac_tracked_data.outcomm_status:
            # if the aircraft has already outcommed (from a previous timestep)
            # then there's nothing more to be done.
            continue

        if ac_tracked_data.pos_status in [
            PositionStatus.OUT_SECTOR,
            PositionStatus.EXIT_REACHED,
        ]:
            # sanity check
            _msg = (
                "The code should never arrive at this condition as this "
                "policy should have outcommed aircraft in this status except "
                "it is limited by a per step single action."
            )
            raise AssertionError(_msg)

        # another sanity check
        assert ac_tracked_data.pos_status == PositionStatus.IN_SECTOR

        # compute the bearing of the previous fix to the next fix
        prev_fix = ac_tracked_data.previous_fix_fr
        next_fix = ac_tracked_data.next_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]
        next_fix_pos = simulator_env.airspace.fixes.places[next_fix]
        pf_nf_bearing = prev_fix_pos.bearing_to(next_fix_pos)

        # condition 1
        cond_1_1 = ac_tracked_data.track_dist_to_exit_cr < exit_distance_threshold
        cond_1_2 = aircraft.selected_fl == exit_fl
        cond_1_3 = abs(aircraft.heading - pf_nf_bearing) < angle_threshold

        # condition 2
        FINAL_THRESHOLD = 3.0
        cond_2_1 = ac_tracked_data.track_dist_to_exit_cr < FINAL_THRESHOLD

        # condition 3
        cond_3_1 = ac_tracked_data.pos_status == PositionStatus.IN_SECTOR
        cond_3_2 = ac_tracked_data.nearest_360_boundary_dist < 1.0
        angle_diff_ac_nb = angle_diff(
            aircraft.heading,
            ac_tracked_data.nearest_360_boundary_bear,
        )
        # < 90 degrees means that the aircraft is flying towards the boundary.
        # =90 degrees means that the aircraft is flying parallel to boundary.
        # > 90 degrees means that the aircraft is flying away from the
        # boundary. in both scenario, indicates that won't exit the sector
        # unless another clearance is issued to change the heading towards
        # the sector boundary.
        cond_3_3 = angle_diff_ac_nb < 90.0

        # to be used by conditions 4 and 5
        # the threshold to outcomm the aircraft before it incorrectly exits
        # the sector vertically.
        INCORRECT_SECTOR_EXIT_FL_THRESHOLD = 5.0

        ## find the volume of the sector that the aircraft is located in
        ## the aircraft should be in one of them because we
        ## have previously ascertained that the aircraft is in the sector
        found_idx = -1
        airspace = simulator_env.airspace
        for i, volume in enumerate(airspace.sectors[airspace_sector].volumes):
            if volume.contains(aircraft):
                found_idx = i
                break
        min_fl = airspace.sectors[airspace_sector].volumes[found_idx].min_fl
        max_fl = airspace.sectors[airspace_sector].volumes[found_idx].max_fl

        # condition 4
        # checks the minimum allowable flight level
        cond_4_1 = aircraft.selected_fl < min_fl
        cond_4_2 = abs(aircraft.fl - min_fl) < INCORRECT_SECTOR_EXIT_FL_THRESHOLD

        # condition 5
        # checks the maximum allowable flight level
        cond_5_1 = aircraft.selected_fl > max_fl
        cond_5_2 = abs(aircraft.fl - max_fl) < INCORRECT_SECTOR_EXIT_FL_THRESHOLD

        if cond_1_1 and cond_1_2 and cond_1_3:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_LOW
        elif cond_2_1:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_MEDIUM
        elif (cond_3_1 and cond_3_2 and cond_3_3) or (cond_4_1 and cond_4_2) or (cond_5_1 and cond_5_2):
            if not gym_env.out_sector_control:
                # issue an outcomm clearance
                actions[callsign] = action
                outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH
        else:
            pass

    if single_action and len(actions) > 1:
        priorities = list(outcomm_priorities.items())
        # sort in descending order of priorities (highest first), using the
        # priority level. the tie between two (or more) aircraft with the same
        # priority level are arbitrarily broken by the sorting algorithm.
        #
        # NOTE: in the future, it might be useful to consider how such ties
        # are broken, taking into the account the properties of the
        # sector/airspace. for example, if two aircraft go out of sector, and
        # one of the aircraft enters a military zone, while the other enters
        # a non-military zone, priority may be given to the aircraft in a
        # military zone.
        # alternative approach: a high level priority could be assigned to the
        # military zone out of sector scenario.

        priorities = sorted(priorities, key=lambda x: x[1], reverse=True)
        selected_callsign = priorities[0][0]

        actions = {selected_callsign: actions[selected_callsign]}

    return actions


def default_outcomm_policy_lenient(
    gym_env: BaseEnv,
    external_agent_actions: dict[str, int],
    exit_distance_threshold: float = 10.0,
    angle_threshold: float = 5.0,
    single_action: bool = True,
    agent: str = "Outcomm Agent",
) -> dict[str, Action]:
    """A simple policy for outcomming aircraft.

    This policy is *lenient* to the external agent (e.g., rl policy) because
    when it encounters a scenario where an aircraft is about to incorrectly
    exit a sector (out-of-sector), it saves the external agent from being
    punished (for letting the aircraft out of the sector) by outcomming the
    aircraft before it incorrectly exits the sector.

    The policy scans through all aircraft in a sector/airspace and issues
    an outcomm clearance for an aircraft if any of the defined conditions
    are met.
    condition 1:
        - if an aircraft is < 10 nautical miles from the exit fix/window,
        - and its cleared flight level is set to the exit flight level,
        - and its heading the same as the bearing of the fix before its exit
        - and its exit fix ( +/- 5 degrees tolerance),
        - then issue an outcomm clearance to the aircraft.
    condition 2:
        - if an aircraft is < 1 nautical mile from the exit fix/window,
        - then issue an outcomm clearance to the aircraft (regardless of
        - its cleared flight level and heading).
        - then issue an outcomm clearance to the aircraft.
    condition 3 (lateral out-of-sector check):
        - if an aircraft is about to navigate laterally out of the sector
          (i.e., incorrect; not via its exit fix), and out of sector control
          is False,
        - then issue an outcomm clearance to the aircraft.
    condition 4 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the minimum allowable flight level in the sector),
          and out of sector control is False,
        - then issue an outcomm clearance to the aircraft.
    condition 5 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the maximum allowable flight level in the sector),
          and out of sector control is False,
        - then issue an outcomm clearance to the aircraft.

    Args:
        gym_env: specifies an instance of a gymnasium environment.
        external_agent_actions: defines the external agent's actions. each
            key-value specifies the action taken for an aircraft (i.e., the
            key is the callsign and the value is the given action).
        exit_distance_threshold: defines the maximum distance (in nautical
            miles) from an aircraft's sector exit position in which an outcomm
            action can be initiated.
            Optional. Defaults to 10.0 nautical miles.
        angle_threshold: defines the maximum lateral deviation (in degrees)
            of the an aircraft's heading relative to the target heading
            required to exit the sector (the target heading is the bearing of
            the exit position from the fix before the position.)
            Optional. Defaults to 5.0 degrees.
        single_action: defines whether or not to return a single action if
            there are multiple aircraft eligible for an outcomm action. if set
            to `True`, and there multiple eligible aircraft, then priority is
            given to the aircraft that requires an immediate outcomm clearance
            (e.g., an aircraft out of sector requires an immediate clearance
            in comparison to an aircraft that is 8 nautical miles away from
            its defined route sector exit).
            Optional. Defaults to True.
        agent: defines the name of the outcomm agent.

    Returns:
        dict, each key-value pair defines an aircraft's callsign and
        corresponding outcomm action (in the simulator's action format).
        The returned dict will be empty if there are no aircraft that satisfy
        the defined outcomm conditions.
    """

    # constants
    OUTCOMM_PRIORITY_LOW = 1
    OUTCOMM_PRIORITY_MEDIUM = 2
    OUTCOMM_PRIORITY_HIGH = 3
    OUTCOMM_PRIORITY_VERY_HIGH = 4

    # get the simulator_env
    simulator_env = gym_env.get_simulator_env()

    # get the tracked data for all active aircraft
    all_tracked_aircraft = gym_env.get_tracked_aircraft_data()

    # get active airspace sector
    airspace_sector = gym_env.get_active_airspace_sector()

    actions = {}
    outcomm_priorities = {}
    for callsign, ac_tracked_data in all_tracked_aircraft.items():
        # get the aircraft
        aircraft = simulator_env.aircraft[callsign]

        # exit flight level
        exit_fl, next_sector = _get_relevant_data(
            ac_tracked_data,
            callsign,
            simulator_env,
            airspace_sector,
        )

        # instantiate the outcomm action
        action = Action(
            callsign,
            kind="outcomm",
            value=next_sector,
            agent=agent,
        )

        if ac_tracked_data.outcomm_status:
            # if the aircraft has already outcommed (from a previous timestep)
            # then there's nothing more to be done.
            continue

        if ac_tracked_data.pos_status == PositionStatus.BEFORE_ENTRY:
            # nothing to do here as aircraft has not yet incommed
            continue

        if ac_tracked_data.pos_status in [
            PositionStatus.OUT_SECTOR,
            PositionStatus.EXIT_REACHED,
        ]:
            # the aircraft has either correctly reached the sector exit
            # (and exited or about to)
            # OR the aircraft has incorrectly exited the sector (or about to)
            # via a point in the boundary that is not within its exit window
            # (i.e. OUT_SECTOR).
            # however, it hasn't been outcommed (it should have been outcommed
            # earlier. if `single_action` is set to True, the aircraft may
            # have been delayed by other aircraft that needed outcomm with a
            # higher priority.
            # outcomm the aircraft immediately.
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_VERY_HIGH
            continue

        # another sanity check
        assert ac_tracked_data.pos_status == PositionStatus.IN_SECTOR

        # compute the bearing of the previous fix to the next fix
        prev_fix = ac_tracked_data.previous_fix_fr
        next_fix = ac_tracked_data.next_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]
        next_fix_pos = simulator_env.airspace.fixes.places[next_fix]
        pf_nf_bearing = prev_fix_pos.bearing_to(next_fix_pos)

        # condition 1
        cond_1_1 = ac_tracked_data.track_dist_to_exit_cr < exit_distance_threshold
        cond_1_2 = aircraft.selected_fl == exit_fl
        cond_1_3 = abs(aircraft.heading - pf_nf_bearing) < angle_threshold
        # cond_1_4 implicitly assumed as the only valid position status at
        # this point is: IN_SECTOR

        # condition 2
        # final threshold to outcomm aircraft if exit flight levels haven't
        # been met before the aircraft exits through its correct exit window.
        FINAL_EXIT_DISTANCE_THRESHOLD = 3.0
        cond_2_1 = ac_tracked_data.track_dist_to_exit_cr < FINAL_EXIT_DISTANCE_THRESHOLD
        # cond_2_4 implicitly assumed as the only valid position status at
        # this point is: IN_SECTOR

        # condition 3
        # threshold to outcomm aircraft before it incorrectly exits the
        # sector (not via its defined exit position). the threshold is the
        # distance to the closest incorrect exit position (at a point on the
        # sector boundary)
        INCORRECT_SECTOR_EXIT_DISTANCE_THRESHOLD = 1.0
        cond_3_1 = ac_tracked_data.pos_status == PositionStatus.IN_SECTOR
        cond_3_2 = ac_tracked_data.nearest_360_boundary_dist < INCORRECT_SECTOR_EXIT_DISTANCE_THRESHOLD
        angle_diff_ac_nb = angle_diff(
            aircraft.heading,
            ac_tracked_data.nearest_360_boundary_bear,
        )
        # < 90 degrees means that the aircraft is flying towards the boundary.
        # =90 degrees means that the aircraft is flying parallel to boundary.
        # > 90 degrees means that the aircraft is flying away from the
        # boundary. The 2nd and 3rd scenarios indicate that the aircraft won't
        # exit the sector unless another clearance is issued to change the
        # heading towards the sector boundary.
        cond_3_3 = angle_diff_ac_nb < 90.0

        # to be used by conditions 4 and 5
        # the threshold to outcomm the aircraft before it incorrectlys exits
        # the sector vertically.
        INCORRECT_SECTOR_EXIT_FL_THRESHOLD = 5.0

        ## find the volume of the sector that the aircraft is located in
        ## the aircraft should be in one of them because we
        ## have previously ascertained that the aircraft is in the sector
        found_idx = -1
        airspace = simulator_env.airspace
        for i, volume in enumerate(airspace.sectors[airspace_sector].volumes):
            if volume.contains(aircraft):
                found_idx = i
                break
        min_fl = airspace.sectors[airspace_sector].volumes[found_idx].min_fl
        max_fl = airspace.sectors[airspace_sector].volumes[found_idx].max_fl

        # condition 4
        # checks the minimum allowable flight level
        cond_4_1 = aircraft.selected_fl < min_fl
        cond_4_2 = abs(aircraft.fl - min_fl) < INCORRECT_SECTOR_EXIT_FL_THRESHOLD

        # condition 5
        # checks the maximum allowable flight level
        cond_5_1 = aircraft.selected_fl > max_fl
        cond_5_2 = abs(aircraft.fl - max_fl) < INCORRECT_SECTOR_EXIT_FL_THRESHOLD

        if cond_1_1 and cond_1_2 and cond_1_3:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_LOW

        elif cond_2_1:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_MEDIUM

        elif cond_3_1 and cond_3_2 and cond_3_3:
            # check if external agent has issued a heading action to resolve
            # the issue (to avoid incorrectly exiting sector lateral boundary).
            ext_action = external_agent_actions.get(callsign, ACTION_NOOP)
            turn_dir = left_right_check(aircraft.heading, ac_tracked_data.nearest_360_boundary_bear)

            # get categories of heading actions
            action_parser = gym_env.get_action_parser()
            actions_decr_hd = action_parser.get_heading_left_actions()
            actions_incr_hd = action_parser.get_heading_right_actions()
            actions_set_hd = action_parser.get_absolute_heading_actions()

            if (
                (turn_dir == TurnDirection.LEFT and ext_action in (actions_incr_hd + actions_set_hd))
                or (turn_dir == TurnDirection.RIGHT and ext_action in (actions_decr_hd + actions_set_hd))
                or (
                    turn_dir == TurnDirection.NO_TURN
                    and ext_action in (actions_decr_hd + actions_incr_hd + actions_set_hd)
                )
            ):
                pass
            else:
                # external agent's action does not resolve the issue.
                # issue an outcomm clearance
                if not gym_env.out_sector_control:
                    actions[callsign] = action
                    outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH

        elif cond_4_1 and cond_4_2:
            # check if external agent has issued a vertical action to resolve
            # the issue (to avoid incorrectly exiting the sector's lower
            # vertical boundary)

            # get categories of flight level actions
            action_parser = gym_env.get_action_parser()
            actions_incr_fl = action_parser.get_fl_climb_actions()
            actions_set_fl = action_parser.get_absolute_fl_actions()

            ext_action = external_agent_actions.get(callsign, ACTION_NOOP)
            if ext_action in (actions_incr_fl + actions_set_fl):
                pass
            else:
                if not gym_env.out_sector_control:
                    # issue an outcomm clearance
                    actions[callsign] = action
                    outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH

        elif cond_5_1 and cond_5_2:
            # check if external agent has issued a vertical action to resolve
            # the issue (to avoid incorrectly exiting the sector's upper
            # vertical boundary)

            # get categories of flight level actions
            action_parser = gym_env.get_action_parser()
            actions_decr_fl = action_parser.get_fl_descent_actions()
            actions_set_fl = action_parser.get_absolute_fl_actions()

            if ext_action in (actions_decr_fl + actions_set_fl):
                pass
            else:
                if not gym_env.out_sector_control:
                    # issue an outcomm clearance
                    actions[callsign] = action
                    outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH
        else:
            pass

    if single_action and len(actions) > 1:
        priorities = list(outcomm_priorities.items())
        # sort in descending order of priorities (highest first), using the
        # priority level. the tie between two (or more) aircraft with the same
        # priority level are arbitrarily broken by the sorting algorithm.
        #
        # NOTE: in the future, it might be useful to consider how such ties
        # are broken, taking into the account the properties of the
        # sector/airspace. for example, if two aircraft go out of sector, and
        # one of the aircraft enters a military zone, while the other enters
        # a non-military zone, priority may be given to the aircraft in a
        # military zone.
        # alternative approach: a high level priority could be assigned to the
        # military zone out of sector scenario.

        priorities = sorted(priorities, key=lambda x: x[1], reverse=True)
        selected_callsign = priorities[0][0]

        actions = {selected_callsign: actions[selected_callsign]}

    return actions


def default_outcomm_policy(
    gym_env: BaseEnv,
    external_agent_actions: dict[str, int],  # noqa: ARG001
    exit_distance_threshold: float = 10.0,
    angle_threshold: float = 5.0,
    single_action: bool = True,
    agent: str = "Outcomm Agent",
) -> dict[str, Action]:
    """A simple policy for outcomming aircraft.

    This policy is *strict* to the external agent (e.g., rl policy) because
    when it encounters a scenario where an aircraft is about to incorrectly
    exit a sector (out-of-sector), it lets the aircraft first go out of the
    sector (thus allowing for the agent to be punished if the external agent
    does nothing) before it outcomms the aircraft.

    The policy scans through all aircraft in a sector/airspace and issues
    an outcomm clearance for an aircraft if any of the defined conditions
    are met.
    condition 1:
        - if an aircraft is < 10 nautical miles from the exit fix/window,
        - and its cleared flight level is set to the exit flight level,
        - and its heading the same as the bearing of the fix before its exit
        - and its exit fix ( +/- 5 degrees tolerance),
        - then issue an outcomm clearance to the aircraft.
    condition 2:
        - if an aircraft is < 1 nautical mile from the exit fix/window,
        - then issue an outcomm clearance to the aircraft (regardless of
        - its cleared flight level and heading).
        - then issue an outcomm clearance to the aircraft.
    condition 3 (lateral out-of-sector check):
        - if an aircraft is about to navigate laterally out of the sector
          (i.e., incorrect; not via its exit fix),
        - then issue an outcomm clearance to the aircraft.
    condition 4 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the minimum allowable flight level in the sector),
        - then issue an outcomm clearance to the aircraft.
    condition 5 (vertical out-of-sector check):
        - if an aircraft is about to navigate vertically out of the sector
          (i.e., outside the maximum allowable flight level in the sector),
        - then issue an outcomm clearance to the aircraft.

        gym_env: specifies an instance of a gymnasium environment.
        external_agent_actions: defines the external agent's actions. each
            key-value specifies the action taken for an aircraft (i.e., the
            key is the callsign and the value is the given action).
        exit_distance_threshold: defines the maximum distance (in nautical
            miles) from an aircraft's sector exit position in which an outcomm
            action can be initiated.
            Optional. Defaults to 10.0 nautical miles.
        angle_threshold: defines the maximum lateral deviation (in degrees)
            of the an aircraft's heading relative to the target heading
            required to exit the sector (the target heading is the bearing of
            the exit position from the fix before the position.)
            Optional. Defaults to 5.0 degrees.
        single_action: defines whether or not to return a single action if
            there are multiple aircraft eligible for an outcomm action. if set
            to `True`, and there multiple eligible aircraft, then priority is
            given to the aircraft that requires an immediate outcomm clearance
            (e.g., an aircraft out of sector requires an immediate clearance
            in comparison to an aircraft that is 8 nautical miles away from
            its defined route sector exit).
            Optional. Defaults to True.
        agent: defines the name of the outcomm agent.

    Returns:
        dict, each key-value pair defines an aircraft's callsign and
        corresponding outcomm action (in the simulator's action format).
        The returned dict will be empty if there are no aircraft that satisfy
        the defined outcomm conditions.
    """

    # constants
    OUTCOMM_PRIORITY_LOW = 1
    OUTCOMM_PRIORITY_MEDIUM = 2
    OUTCOMM_PRIORITY_HIGH = 3

    # get the simulator_env
    simulator_env = gym_env.get_simulator_env()

    # get the tracked data for all active aircraft
    all_tracked_aircraft = gym_env.get_tracked_aircraft_data()

    # get active airspace sector
    airspace_sector = gym_env.get_active_airspace_sector()

    actions = {}
    outcomm_priorities = {}
    for callsign, ac_tracked_data in all_tracked_aircraft.items():
        # get the aircraft
        aircraft = simulator_env.aircraft[callsign]

        # exit flight level
        exit_fl, next_sector = _get_relevant_data(
            ac_tracked_data,
            callsign,
            simulator_env,
            airspace_sector,
        )

        # instantiate the outcomm action
        action = Action(
            callsign,
            kind="outcomm",
            value=next_sector,
            agent=agent,
        )

        if ac_tracked_data.outcomm_status:
            # if the aircraft has already outcommed (from a previous timestep)
            # then there's nothing more to be done.
            continue

        if ac_tracked_data.pos_status == PositionStatus.BEFORE_ENTRY:
            # nothing to do here as aircraft has not yet incommed
            continue

        if ac_tracked_data.pos_status == PositionStatus.EXIT_REACHED:
            # aircraft should have been outcommed earlier. if `single_action`
            # is set to True, the aircraft may have been delayed by other
            # aircraft that needed outcomm with a higher priority.
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH
            continue

        if ac_tracked_data.pos_status == PositionStatus.OUT_SECTOR:
            # the aircraft has incorrectly exited the sector (and the
            # the external agent has been punished in the previous time step)
            # now, outcomm the aircraft immediately.

            if not gym_env.out_sector_control:
                actions[callsign] = action
                outcomm_priorities[callsign] = OUTCOMM_PRIORITY_HIGH
            continue

        # another sanity check
        assert ac_tracked_data.pos_status == PositionStatus.IN_SECTOR

        # compute the bearing of the previous fix to the next fix
        prev_fix = ac_tracked_data.previous_fix_fr
        next_fix = ac_tracked_data.next_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]
        next_fix_pos = simulator_env.airspace.fixes.places[next_fix]
        pf_nf_bearing = prev_fix_pos.bearing_to(next_fix_pos)

        # condition 1
        cond_1_1 = ac_tracked_data.track_dist_to_exit_cr < exit_distance_threshold
        cond_1_2 = aircraft.selected_fl == exit_fl
        cond_1_3 = abs(aircraft.heading - pf_nf_bearing) < angle_threshold

        # condition 2
        # final threshold to outcomm aircraft if exit flight level haven't
        # been met before the aircraft exits through its correct exit window.
        FINAL_EXIT_DISTANCE_THRESHOLD = 3.0
        cond_2_1 = ac_tracked_data.track_dist_to_exit_cr < FINAL_EXIT_DISTANCE_THRESHOLD
        cond_2_2 = ac_tracked_data.pos_status == PositionStatus.IN_SECTOR

        if cond_1_1 and cond_1_2 and cond_1_3:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_LOW
        elif cond_2_1 and cond_2_2:
            # issue an outcomm clearance
            actions[callsign] = action
            outcomm_priorities[callsign] = OUTCOMM_PRIORITY_MEDIUM
        else:
            pass

    if single_action and len(actions) > 1:
        priorities = list(outcomm_priorities.items())
        # sort in descending order of priorities (highest first), using the
        # priority level. the tie between two (or more) aircraft with the same
        # priority level are arbitrarily broken by the sorting algorithm.
        #
        # NOTE: in the future, it might be useful to consider how such ties
        # are broken, taking into the account the properties of the
        # sector/airspace. for example, if two aircraft go out of sector, and
        # one of the aircraft enters a military zone, while the other enters
        # a non-military zone, priority may be given to the aircraft in a
        # military zone.
        # alternative approach: a high level priority could be assigned to the
        # military zone out of sector scenario.

        priorities = sorted(priorities, key=lambda x: x[1], reverse=True)
        selected_callsign = priorities[0][0]

        actions = {selected_callsign: actions[selected_callsign]}

    return actions
