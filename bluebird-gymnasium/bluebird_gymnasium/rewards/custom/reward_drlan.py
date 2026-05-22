"""Reward function specification for the paper:

Improving Autonomous Separation Assurance through Distributed Reinforcement
Learning with Attention Networks

url: https://arxiv.org/abs/2308.04958

"""

from bluebird_dt.utility.convert import (
    FT_TO_FL,  # feet to flight level
    FT_TO_NMI,  # feet to nautical miles
)

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.types import InteractionRelevance

DEFAULT_DX_NMAC = 500 * FT_TO_NMI
DEFAULT_DZ_NMAC = 100 * FT_TO_FL
DEFAULT_D_MAX = 3280 * FT_TO_NMI

DEFAULT_CHI = 1e-1
DEFAULT_DELTA = 1e-4
DEFAULT_EPSILON = 1e-3
DEFAULT_LAMBDA = 1e-2
DEFAULT_OMEGA = 1e-3


def reward_drlan(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float: # noqa: ARG001, ANN003
    """Reward function implementation for IASA_DRLAN.

    Improving Autonomous Separation Assurance through Distributed Reinforcement
    Learning with Attention Networks

    url: https://arxiv.org/abs/2308.04958

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (-inf to +inf, depending on the values of
            the coefficients used in the computation).
    """

    coeff_chi = DEFAULT_CHI
    coeff_delta = DEFAULT_DELTA
    coeff_epsilon = DEFAULT_EPSILON
    coeff_lambda = DEFAULT_LAMBDA
    coeff_omega = DEFAULT_OMEGA

    action_parser = gym_env.get_action_parser()

    # compute r_st_ht
    r_st_ht = 0.0
    interactions = gym_env.get_traffic_monitor().get_relevant_traffic(
        callsign,
        relevance=InteractionRelevance.LEVEL_1,  # primary interactions
    )

    if len(interactions) == 0:
        r_st_ht = 0.0

    else:
        # get information of the closest aircraft to the current aircraft
        closest_interaction = interactions[0]
        distance_aircraft_other_aircraft = closest_interaction.dist_ac_other
        fl_diff_aircraft_other_aircraft = closest_interaction.fl_diff_ac_other

        if distance_aircraft_other_aircraft < DEFAULT_DX_NMAC and fl_diff_aircraft_other_aircraft < DEFAULT_DZ_NMAC:
            r_st_ht = -1.0

        elif distance_aircraft_other_aircraft >= DEFAULT_DX_NMAC and distance_aircraft_other_aircraft < DEFAULT_D_MAX:
            r_st_ht = -coeff_chi + (coeff_delta * distance_aircraft_other_aircraft)

        else:
            r_st_ht = 0.0

    # compute r_at
    r_at = 0.0
    if action in action_parser.get_noop_actions():
        r_at = 0.0
    elif action in action_parser.get_relative_speed_actions():
        r_at = -coeff_epsilon
    elif action in action_parser.get_relative_fl_actions():
        r_at = -coeff_lambda
    else:
        msg = "`iasa_drlan` only support set up where relative speed,relative flight level actions/clearances are used."
        raise ValueError(msg)

    return r_st_ht + r_at - coeff_omega
