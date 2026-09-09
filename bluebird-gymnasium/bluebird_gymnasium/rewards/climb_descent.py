from __future__ import annotations

import typing

import numpy as np

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.constants import DIST_EXIT_LEVEL_THRESHOLD
from bluebird_gymnasium.utils.interaction_utils import (
    get_optimal_unblocked_flight_level,
)
from bluebird_gymnasium.utils.simulator_utils import (
    aircraft_entry_coordination,
    aircraft_exit_coordination,
    basic_distances,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.core.environment import Environment as SimulatorEnv
    from bluebird_dt.core.pos2d import Pos2D

    from bluebird_gymnasium.utils.types import ACStateTracker, Number


# useful constants
_FL_SCALER = 10.0
_MAX_PENALTY = 10


def normalized_exponential(x: Number, k: int = 2, reverse: bool = True) -> float:
    """Compute a scaled exponential value normalized between 0 and 1.

    if reverse=False:
        ramps from r(-1)=0 to r(0)=1. Best for progress toward a goal.
    if reverse=True:
        ramps from r(0)=1 to r(1)=0. Best for distance/error penalties.

    Args:
        x: defines the input value
        k: steepness factor. Higher values create a sharper ramp.
        reverse: flips the direction of the exponential curve.

    Returns:
        the normalized value, within range [0.0, 1.0]
    """

    if k < 1 or k > 10:
        raise ValueError("k should be in range [1, 10]")

    denominator = 1 - np.exp(-k)
    if reverse:  # noqa: SIM108
        # range [0, 1] -> reward [1, 0]
        output = (np.exp(-k * x) - np.exp(-k)) / denominator
    else:
        # range [-1, 0] -> reward [0, 1]
        output = (np.exp(k * x) - np.exp(-k)) / denominator

    return np.clip(output, 0.0, 1.0)


def _get_relevant_data(
    ac_tracked_state: ACStateTracker,
    callsign: str,
    simulator_env: SimulatorEnv,
    airspace_sector: str,
) -> tuple[float, float, Pos2D]:
    if ac_tracked_state is None:
        entry_fl = aircraft_entry_coordination(callsign, simulator_env, airspace_sector).fl
        exit_fl = aircraft_exit_coordination(callsign, simulator_env, airspace_sector).fl

        aircraft = simulator_env.aircraft[callsign]
        exit_pos2d = simulator_env.airspace.get_exit_point(aircraft)
    else:
        entry_fl = ac_tracked_state.entry_coords[airspace_sector].fl
        exit_fl = ac_tracked_state.exit_coords[airspace_sector].fl
        exit_fix_name = ac_tracked_state.exit_coords[airspace_sector].fix
        exit_pos2d = simulator_env.airspace.fixes.places[exit_fix_name]

    return entry_fl, exit_fl, exit_pos2d


def overflier_const(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for overflier aircraft (i.e., same entry and exit flight levels).

    Note: Constant function.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: 0.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, _ = ret

    if entry_fl == exit_fl:
        if ac.selected_fl == exit_fl:
            return 1.0
        return 0.0
    return 0.0


def overflier_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for overflier aircraft (i.e., same entry and exit flight levels).

    Note: Linear function.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0.0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, _ = ret

    if entry_fl == exit_fl:
        fl_diff = abs(ac.selected_fl - exit_fl) / _FL_SCALER
        reward = -1.0 * fl_diff
    else:
        # not an overflier scenario: return the maximum reward for this case
        reward = 0.0

    return float(reward)


def overflier_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for overflier aircraft (i.e., same entry and exit flight levels).

    Note: Quadratic function.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0.0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, _ = ret

    if entry_fl == exit_fl:
        fl_diff = abs(ac.selected_fl - exit_fl) / _FL_SCALER
        reward = -1.0 * fl_diff**2
    else:
        # not an overflier scenario: return the maximum reward for this case
        reward = 0.0

    return float(reward)


def overflier_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for overflier aircraft (i.e., same entry and exit flight levels).

    Note: Exponential function.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -5.0 to 1.0).
    """

    reward = 0.0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, _ = ret

    if entry_fl == exit_fl:
        # aircraft should not climb or descend when it should overfly.
        # penalise if it changes its exit flight level which should be
        # its current flight level.
        alpha = -5.0 if ac.selected_fl != exit_fl else np.exp(-1.0 * abs(ac.fl - exit_fl) / _FL_SCALER)

        reward = alpha * 1.0

    else:
        # not an overflier scenario: return the minimum reward for this case
        reward = 0.0

    return float(reward)


def climb_target_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft ascending to the correct exit flight level.

    Note: Linear function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl >= exit_fl:
        # not an ascent (climb) scenario, so, give the maximum reward
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        if distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.

            if ac.selected_fl == exit_fl:  # noqa: SIM108
                # soft penalty as correct exit flight level chosen
                reward = min(abs(ac.fl - exit_fl), _MAX_PENALTY)
            else:
                # otherwise, issue a hard penalty
                reward = _MAX_PENALTY
        else:
            # ascent: climb as early as possible
            reward = abs(ac.selected_fl - exit_fl)

        reward = -1.0 * reward

    return float(reward)


def descent_target_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft descending to the correct exit flight level.

    Note: Linear function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl <= exit_fl:
        # not an descend scenario, so, give the maximum reward
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        if distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.

            if ac.selected_fl == exit_fl:  # noqa: SIM108
                # soft penalty as correct exit flight level chosen
                reward = min(abs(ac.fl - exit_fl) / _FL_SCALER, _MAX_PENALTY)
            else:
                # otherwise, issue a hard penalty
                reward = _MAX_PENALTY
        else:
            # descent: descend as late as possible

            distance_diff = distance_to_exit - distance_to_level
            distance_diff = distance_diff - DIST_EXIT_LEVEL_THRESHOLD

            alpha = np.exp(-max(distance_diff, 0.0))

            entry_reward = abs(ac.selected_fl - entry_fl) / _FL_SCALER
            exit_reward = abs(ac.selected_fl - exit_fl) / _FL_SCALER

            reward = ((1.0 - alpha) * entry_reward) + (alpha * exit_reward)

        reward = -1.0 * reward

    return float(reward)


def descent_target_linear_shaped(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft descending to the correct exit flight level.

    Note: Linear function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl <= exit_fl:
        # not an descend scenario, so, give the maximum reward
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        if distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.

            if ac.selected_fl == exit_fl:  # noqa: SIM108
                # soft penalty as correct exit flight level chosen
                reward = min(abs(ac.fl - exit_fl), _MAX_PENALTY)
            else:
                # otherwise, issue a hard penalty
                reward = _MAX_PENALTY
        else:
            # descent: descend as late as possible

            distance_diff = distance_to_exit - distance_to_level
            distance_diff = distance_diff - DIST_EXIT_LEVEL_THRESHOLD

            if distance_diff > 0.0:
                if ac.selected_fl == entry_fl:
                    if ac.fl == entry_fl:
                        reward = 0.0
                    elif (ac.fl > entry_fl and ac.vertical_speed < 0) or (ac.fl < entry_fl and ac.vertical_speed > 0):
                        reward = 0.2
                    else:
                        # assert False, "never get here"
                        return 1.0
                else:
                    reward = 0.9
            else:
                reward = 0.0 if ac.selected_fl == exit_fl else 0.8

        reward = -1.0 * reward

    return float(reward)


def climb_target_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft ascending to the correct exit flight level.

    Note: Quadratic function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl >= exit_fl:
        # not an ascent (climb) scenario, so, give the maximum reward
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        if distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.
            fl_diff = abs(ac.fl - exit_fl) / _FL_SCALER

        else:
            # ascent: climb as early as possible
            fl_diff = abs(ac.selected_fl - exit_fl) / _FL_SCALER

        reward = -1.0 * fl_diff**2

    return float(reward)


def descent_target_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft descending to the correct exit flight level.

    Note: Quadratic function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl <= exit_fl:
        # not an descend scenario, so, give the maximum reward
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        if distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.

            if ac.selected_fl == exit_fl:
                # soft penalty as correct exit flight level chosen
                # give a small bonus for correctly choosing
                # the exit (target) flight level
                small_bonus = 5.0
                reward = (abs(ac.fl - exit_fl) / _FL_SCALER) ** 2
                reward -= small_bonus  # note, subtract in this case
                reward = min(reward, 0.0)  # clip reward [-inf, 0.0]
            else:
                # otherwise, issue a hard penalty
                reward = (abs(ac.fl - exit_fl) / _FL_SCALER) ** 2
        else:
            # descent: descend as late as possible

            distance_diff = distance_to_exit - distance_to_level
            distance_diff = distance_diff - DIST_EXIT_LEVEL_THRESHOLD

            alpha = np.exp(-max(distance_diff, 0.0))

            entry_reward = (abs(ac.selected_fl - entry_fl) / _FL_SCALER) ** 2
            exit_reward = (abs(ac.selected_fl - exit_fl) / _FL_SCALER) ** 2

            reward = ((1.0 - alpha) * entry_reward) + (alpha * exit_reward)

        reward = -1.0 * reward

    return float(reward)


def climb_target_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft ascending to the correct exit flight level.

    Note: Exponential function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -5.0 to 1.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl >= exit_fl:
        # not an ascent (climb) scenario, so, give no reward
        # of 0.0, which is the minimum in this formulation.
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        _scaler = abs(exit_fl - entry_fl)
        if ac.selected_fl < entry_fl:
            # aircraft should not descend when it should climb. penalise.
            reward = -5.0

        else:
            # ascent: climb as early as possible
            alpha = normalized_exponential(
                x=abs(ac.selected_fl - exit_fl) / _scaler,
                k=2,
                reverse=True,
            )
            reward = alpha * 1.0

            # penalty 1: penalty for late climb or climb beyond exit fl
            # range: [-2.0, 0.0]
            if ac.selected_fl < exit_fl:
                # get the intermediate descent fl at the current time step.
                tracked_aircraft = gym_env.get_tracked_aircraft_data()
                rollout_predictor = gym_env.get_rollout_predictor()
                traffic_monitor = gym_env.get_traffic_monitor()
                intermediate_fl = get_optimal_unblocked_flight_level(
                    callsign,
                    _sector,
                    tracked_aircraft,
                    simulator_env,
                    rollout_predictor,
                    traffic_monitor.get_relevant_traffic(callsign),
                )
                penalty_1 = -2.0 if ac.selected_fl < intermediate_fl else 0.0
            else:
                penalty_1 = 0.0

            # penalty 2: penalty for climb higher than exit flight level
            # range: [-5.0, 0.0]
            penalty_2 = -5.0 if ac.selected_fl > exit_fl else 0.0

            # penalty 3: failure to complete exit fl climb before sector exit
            # range: [-0.5, 0.0]
            penalty_3 = -0.5 if distance_to_exit < distance_to_level else 0.0

            reward = reward + penalty_1 + penalty_2 + penalty_3

    return float(reward)


def descent_target_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for the aircraft descending to the correct exit flight level.

    Note: Exponential function

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -5.0 to 1.0).
    """

    reward = 0

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]
    _sector = gym_env.get_active_airspace_sector()

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # get relevant aircraft data
    ret = _get_relevant_data(ac_tracked_state, callsign, simulator_env, _sector)
    entry_fl, exit_fl, exit_pos2d = ret

    if entry_fl <= exit_fl:
        # not an descent scenario, so, give no reward
        # of 0.0, which is the minimum in this formulation.
        reward = 0.0

    else:
        # get the (lateral) track distance to exit position
        # and the (lateral) travel distance to the exit flight level
        if ac_tracked_state is None:
            ret = basic_distances(ac, simulator_env.airspace, exit_pos2d, exit_fl)
            ret = ret[2:]
        else:
            ret = (
                ac_tracked_state.track_dist_to_exit_cr,
                ac_tracked_state.dist_to_target_fl,
            )
        distance_to_exit, distance_to_level = ret

        _scaler = abs(exit_fl - entry_fl)
        if ac.selected_fl > entry_fl:
            # aircraft should not climb when it should descend. penalise.
            reward = -5.0

        elif distance_to_exit < distance_to_level:
            # penalise: aircraft will not make the exit flight level in time.
            reward = normalized_exponential(
                x=abs(ac.selected_fl - exit_fl) / _scaler,
                k=2,
                reverse=True,
            )
            small_penalty = -0.5
            reward = reward + small_penalty
            reward = min(reward, 1.0)  # clip reward to [-inf, 1.0]
        else:
            # descent: descend as late as possible

            distance_diff = distance_to_exit - distance_to_level
            distance_diff = distance_diff - DIST_EXIT_LEVEL_THRESHOLD

            alpha = np.exp(-max(distance_diff, 0.0))
            entry_reward = normalized_exponential(
                x=abs(ac.selected_fl - entry_fl) / _scaler,
                k=2,
                reverse=True,
            )
            exit_reward = normalized_exponential(
                x=abs(ac.selected_fl - exit_fl) / _scaler,
                k=2,
                reverse=True,
            )
            reward = ((1.0 - alpha) * entry_reward) + (alpha * exit_reward)

            # penalty 1: penalise early descent.
            # range: [-2.0, 0.0]
            if ac.selected_fl < entry_fl and distance_diff > 0:
                # aircraft should not descend early. however, check if early
                # descent is due to conflict. if yes, give a small penalty.
                # otherwise, give a hard penalty.

                # get the intermediate descent fl at the current time step.
                traffic_monitor = gym_env.get_traffic_monitor()
                intermediate_fl = get_optimal_unblocked_flight_level(
                    callsign,
                    _sector,
                    gym_env.get_tracked_aircraft_data(),
                    simulator_env,
                    gym_env.get_rollout_predictor(),
                    traffic_monitor.get_relevant_traffic(callsign),
                )
                if intermediate_fl > exit_fl:  # noqa: SIM108
                    # exit fl is not available due to conflict.
                    penalty_1 = -1.0  # small penalty
                else:
                    # full penalty
                    penalty_1 = -2.0

            # elif ac.selected_fl >= entry_fl and distance_diff <= 0:
            #    # aircraft should be descending
            #    penalty_1 = -2.0

            else:
                penalty_1 = 0.0

            # penalty 2: descent lower than the exit flight level
            # range: [-5.0, 0.0]
            penalty_2 = -5.0 if ac.selected_fl < exit_fl else 0.0

            # penalty 3: failure to complete exit fl descent before sector exit
            # range: [-0.5, 0.0]
            penalty_3 = -0.5 if distance_to_exit < distance_to_level else 0.0

            reward = reward + penalty_1 + penalty_2 + penalty_3

    return float(reward)
