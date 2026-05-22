from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.types import PositionStatus


def position_status_const(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Computes the reward for an aircraft's current position status.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        a float (-1.0 to 0.0), the computed reward.
    """

    # this could also be view as the tracked data for the next time step,
    # as the action from the external agent has been applied and the simulator
    # has been evolved to get the next state.
    ac_tracked_data_curr = gym_env.get_tracked_aircraft_data(callsign)

    if ac_tracked_data_curr.pos_status == PositionStatus.OUT_SECTOR:
        return -1.0
    return 0.0
