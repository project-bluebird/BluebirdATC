from typing import Any

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.rewards.lateral_termination_check import (
    lateral_termination_check_mac,
    lateral_termination_check_sac,
)
from bluebird_gymnasium.utils.types import PositionStatus


def custom_reward_fn(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Template reward function. Implement your code here.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (-inf to +inf, depending on the values of
            the coefficients used in the computation).
    """

    # implement reward function here.
    return 0.0


def route_progress_terminal_reward(gym_env: BaseEnv, callsign: str) -> float:
    """Reward progress toward exit and penalize stalling at episode end.

    This reward is intended for the PPO curriculum notebooks where the agent
    was observed to loiter near its entry fix. It combines three signals:

    - a positive terminal bonus for correctly reaching the exit window
    - a per-step progress reward for reducing route-track distance to exit
    - a timeout penalty if the episode ends with the aircraft still far away

    Returns:
        float: shaped reward that encourages forward progress through sector.
    """

    reward = 0.0

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if ac_tracked_state is None:
        return reward

    if ac_tracked_state.pos_status == PositionStatus.EXIT_REACHED:
        reward += 20.0

    if (
        prev_ac_tracked_state is not None
        and ac_tracked_state.track_dist_to_exit_cr is not None
        and prev_ac_tracked_state.track_dist_to_exit_cr is not None
    ):
        curr_dist = float(ac_tracked_state.track_dist_to_exit_cr)
        prev_dist = float(prev_ac_tracked_state.track_dist_to_exit_cr)
        distance_improvement = prev_dist - curr_dist

        # Positive when the aircraft gets closer to its exit, negative when it
        # drifts away or loiters. Clip to keep this component bounded.
        reward += max(min(distance_improvement / 5.0, 2.0), -2.0)

    if (
        gym_env.timestep >= gym_env.maxstep
        and ac_tracked_state.track_dist_to_exit_cr is not None
        and ac_tracked_state.pos_status != PositionStatus.EXIT_REACHED
    ):
        timeout_distance = float(ac_tracked_state.track_dist_to_exit_cr)
        reward -= min(timeout_distance / 20.0, 10.0)

    return float(reward)


def lateral_termination_check_sac_env(
    gym_env: BaseEnv, callsign: str, action: int, **kwargs: dict[str, dict[str, Any]]
) -> float:
    """Notebook-safe wrapper for the SAC lateral termination reward.

    The base reward registry invokes reward functions with only
    `(gym_env, callsign, action)`. This wrapper adapts the original helper,
    which expects explicit `timestep` and `maxstep` arguments.
    """

    return float(
        lateral_termination_check_sac(
            gym_env=gym_env,
            callsign=callsign,
            action=action,
            timestep=gym_env.timestep,
            maxstep=gym_env.maxstep,
            **kwargs,
        )
    )


def lateral_termination_check_mac_env(
    gym_env: BaseEnv, callsign: str, action: int, **kwargs: dict[str, dict[str, Any]]
) -> float:
    """Notebook-safe wrapper for the MAC lateral termination reward."""

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    transferred = ac_tracked_state is not None and ac_tracked_state.pos_status == PositionStatus.EXIT_REACHED

    return float(
        lateral_termination_check_mac(
            gym_env=gym_env,
            callsign=callsign,
            action=action,
            timestep=gym_env.timestep,
            maxstep=gym_env.maxstep,
            transferred=transferred,
            **kwargs,
        )
    )


def anti_loiter_route_rejoin_reward(gym_env: BaseEnv, callsign: str, action: int) -> float:
    """Reward progress and route rejoin, penalize stalling near the entry.

    This is used by the PPO curriculum notebook to address a specific failure
    mode seen in multi-aircraft stages: orbiting near the start fix to avoid
    future conflict instead of making progress, taking an avoidance action,
    then returning to the route.
    """

    simulator_env = gym_env.get_simulator_env()
    aircraft = simulator_env.aircraft[callsign]

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if ac_tracked_state is None or prev_ac_tracked_state is None:
        return 0.0

    curr_exit_dist = ac_tracked_state.track_dist_to_exit_cr
    prev_exit_dist = prev_ac_tracked_state.track_dist_to_exit_cr

    if curr_exit_dist is None or prev_exit_dist is None:
        return 0.0

    reward = 0.0

    progress_nm = float(prev_exit_dist - curr_exit_dist)

    curr_centre_dist = None
    prev_centre_dist = None
    if ac_tracked_state.centreline_info_fr is not None:
        curr_centre_dist = float(ac_tracked_state.centreline_info_fr[0])
    if prev_ac_tracked_state.centreline_info_fr is not None:
        prev_centre_dist = float(prev_ac_tracked_state.centreline_info_fr[0])

    if progress_nm > 0.15:
        reward += min(progress_nm / 2.0, 1.0)
    elif float(curr_exit_dist) > 20.0:
        reward -= 0.35

    if curr_centre_dist is not None and prev_centre_dist is not None:
        centreline_improvement = prev_centre_dist - curr_centre_dist
        if centreline_improvement > 0.1:
            reward += min(centreline_improvement / 2.0, 0.75)
        elif curr_centre_dist > 8.0 and action == 0:
            reward -= 0.25

        if curr_centre_dist < 3.0 and progress_nm > 0.15:
            reward += 0.4

    if aircraft.on_route and progress_nm > 0.15:
        reward += 0.25

    return float(reward)
