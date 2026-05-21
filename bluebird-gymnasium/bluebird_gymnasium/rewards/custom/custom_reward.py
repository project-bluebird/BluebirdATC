from bluebird_gymnasium.envs.base import BaseEnv


def custom_reward_fn(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:
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
    reward = 0.0

    return reward
