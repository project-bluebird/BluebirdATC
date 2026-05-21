from bluebird_gymnasium.envs.base import BaseEnv


def lateral_termination_check_sac(
    gym_env: BaseEnv,
    callsign: str,
    action: int,
    timestep: int,
    maxstep: int,
    **kwargs,
) -> float:
    """Reward for aircraft's termination from the sector

    Single aircraft (sac) only.

    Computed based on the aircraft's distance from the exit fix given maximum
    timesteps in the sector has been reached.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.
        timestep: the current time step of training.
        maxstep: the maximum training step.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()

    reward = 0
    if timestep >= maxstep:
        ac = simulator_env.aircraft[callsign]
        exit_loc_ac = simulator_env.airspace.route_exit_fix(ac.flight_plan.route)
        exit_dist = ac.pos2d().distance(exit_loc_ac)
        reward = -1.0 * exit_dist
    return reward


def lateral_termination_check_mac(
    gym_env: BaseEnv,
    callsign: str,
    action: int,
    timestep: int,
    maxstep: int,
    transferred: bool,
    **kwargs,
) -> float:
    """Reward for aircraft termination from the sector.

    Multiple aircraft (mac).

    Computed based on the aircraft's distance from the exit fix given maximum
    timesteps in the sector has been reached.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.
        timestep: the current time step in the episode.
        maxstep: the maximum number of steps in an episode.
        transferred: flag to signify whether the aircraft has been
            transferred out of the sector. if True, the aircraft
            has successful navigated through the sector and exited.
            otherwise, false.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()

    reward = 0
    if timestep >= maxstep:
        for callsign in simulator_env.aircraft:
            if not transferred:
                ac = simulator_env.aircraft[callsign]
                exit_loc_ac = simulator_env.airspace.route_exit_fix(ac.flight_plan.route)
                exit_dist = ac.pos2d().distance(exit_loc_ac)
                reward -= exit_dist
    return reward
