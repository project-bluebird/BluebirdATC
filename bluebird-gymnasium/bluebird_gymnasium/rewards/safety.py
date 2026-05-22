import numpy as np
from bluebird_dt.core import Pos4D

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.constants import FUTURE_TRAJ_DURATION
from bluebird_gymnasium.utils.simulator_utils import predict_trajectory
from bluebird_gymnasium.utils.types import MinAircraftSeparation


def _ensured_lateral_separation(aircraft_1_cps: list[Pos4D], aircraft_2_cps: list[Pos4D]) -> tuple[bool, list[int]]:
    """Check lateral separation assurance between aircraft trajectory pair.

    Checks based on the list of control points (trajectory) provided
    for each aircraft

    Args:
        aircraft_1_cps: the first aircraft control points.
            control points.
        aircraft_2_cps: the second aircraft control points.

    Returns:
        tuple, the first element a bool (stores `True` if separation is
        ensured, `False otherwise), and the second element a list of
        measured distance between each pair of control points from both
        aircraft.
    """

    ensured_separation = True
    cp_dist_buffer = []

    # check if both aircraft have already passed each other.
    # if yes, distance will grow farther apart as time progress.
    distance_t0 = aircraft_1_cps[0].distance(aircraft_2_cps[0])
    distance_t1 = aircraft_1_cps[1].distance(aircraft_2_cps[1])
    if distance_t1 >= distance_t0 and distance_t0 >= MinAircraftSeparation.LATERAL:
        cp_dist_buffer = [distance_t0, distance_t1]
        return ensured_separation, cp_dist_buffer

    for ac_cp, other_ac_cp in zip(aircraft_1_cps, aircraft_2_cps, strict=True):
        cp_dist = ac_cp.distance(other_ac_cp)
        cp_dist_buffer.append(cp_dist)

        if cp_dist < MinAircraftSeparation.LATERAL:
            # loss of separation
            ensured_separation = False
            break

    return ensured_separation, cp_dist_buffer


def safety_simple_avoidance_nvl(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for avoidance of aircraft loss of separation.

    Note: safety violation is not logged, hence the `nvl` suffix (no violation
    log).

    Note: this is a reactive reward function as it only considers the current
    lateral distance between aircraft pair. it does not consider future
    lateral distance between aircraft pair.

    Note: this function does not take vertical separation into account.
    therefore, for aircraft pair in different flight levels, if their
    lateral distance is less than the defined threshold, they're
    considered as violating safety.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()

    ac = simulator_env.aircraft[callsign]
    reward = 0
    for other_callsign in simulator_env.aircraft:
        if callsign != other_callsign:
            other_ac = simulator_env.aircraft[other_callsign]
            dist = ac.pos2d().distance(other_ac.pos2d())
            if dist < 7:
                reward -= 1 / (dist + 0.01)
    return reward


def safety_simple_avoidance_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for avoidance of aircraft loss of separation.

    This includes a forward rollout check into the future for loss of
    separation rather than the reactive style of only check for loss
    of separation based on the aircraft current position.

    Note: this function caters only for loss of lateral separation and
    loss of separation based current flight level (without considering
    climb or descent situations).

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    # function constant(s)
    SCALER = 10
    DISTANCE_THRESH = 50.0

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # note: potentially remove this in the future
    _safety_info = {}

    aircraft = simulator_env.aircraft[callsign]

    # future trajectory of the aircraft based on current state (look-ahead)
    if hasattr(gym_env, "ac_tracker"):
        ac_cps = gym_env.ac_tracker[callsign].future_trajectory
        callsigns = list(gym_env.ac_tracker.keys())
    else:
        ac_cps = predict_trajectory(
            aircraft,
            gym_env.predictor,
            duration=FUTURE_TRAJ_DURATION,
            curr_time=simulator_env.time,
        )
        # use all aircraft: not optimal, and hasn't yet been tested.
        callsigns = list(simulator_env.aircraft.keys())

    _rewards = []
    callsigns.remove(callsign)

    for other_callsign in callsigns:
        other_aircraft = simulator_env.aircraft[other_callsign]
        buffer = {}

        ## get future trajectory of the other aircraft
        ## optimize: this could be optimized with a lambda fn
        ## so that `if` is not checked in every loop iteration.
        other_ac_tracked_state = gym_env.get_tracked_aircraft_data(other_callsign)
        if other_ac_tracked_state is not None:
            other_ac_cps = other_ac_tracked_state.future_trajectory
        else:
            other_ac_cps = predict_trajectory(
                other_aircraft,
                gym_env.rollout_predictor,
                duration=FUTURE_TRAJ_DURATION,
                curr_time=simulator_env.time,
            )

        ## check that aircraft are vertically separated
        vsep = abs(aircraft.fl - other_aircraft.fl) >= MinAircraftSeparation.VERTICAL
        buffer["ensured_vertical_sep"] = vsep

        ## now check for lateral separation into the future based on rollout
        lsep, dist_buffer = _ensured_lateral_separation(ac_cps, other_ac_cps)
        buffer["future_traj_dist"] = dist_buffer
        buffer["ensured_lateral_sep"] = lsep

        if vsep or lsep:
            _rewards.append(0.0)

        else:
            # aircraft pair will lose lateral separation in the future if
            # they both continue on their current trajectory. compute
            # the penalty based on distance of their current position.
            curr_dist = aircraft.pos2d().distance(other_aircraft.pos2d())
            _rewards.append(-SCALER * np.exp(-curr_dist / DISTANCE_THRESH))

            # logging.
            # if current distance has already lost separation,
            # log it as a violation of technical safety.
            if curr_dist < MinAircraftSeparation.LATERAL:
                buffer["separation_loss"] = ac_tracked_state.sector_entry_timestep + ac_tracked_state.step_counter

            else:
                buffer["separation_loss"] = None

        _safety_info[other_callsign] = buffer

    if ac_tracked_state is not None:
        ac_tracked_state.safety_debug = _safety_info

    return 0.0 if len(_rewards) == 0 else float(np.sum(_rewards))
