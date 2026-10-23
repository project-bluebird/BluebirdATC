from __future__ import annotations

import typing

import numpy as np

from bluebird_gymnasium.utils.geo_utils import project_x_from_range_to_range
from bluebird_gymnasium.utils.interaction_utils import (
    main_or_proxy_intersection_location,
    top_of_ascent_before_intersection,
    top_of_descent_after_intersection,
)
from bluebird_gymnasium.utils.types import PositionStatus

if typing.TYPE_CHECKING:
    from bluebird_dt.core.environment import Environment as SimulatorEnv

    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import (
        ACStateTracker,
        InteractionInfo,
        Number,
    )


def get_overlap_coeff(
    callsign: str,
    other_callsign: str,
    simulator_env: SimulatorEnv,
    sector: str,
    tracked_data: dict[str, ACStateTracker],
    interaction: InteractionInfo | None = None,
) -> Number:
    """Get the coefficient for conflict resolution penalty based on interaction

    The scaling coefficient is determined based on whether the conflict can be
    resolved by only lateral resolution techniques (or resolved using either
    lateral or vertical techniques).

    Note the returned coefficient based on the requirements of both aircraft.
    - Both aircraft need to climb
        - vertical resolution is possible
            - vertical resolution is being actioned: 0.0
            - vertical resolution is not being actioned: 0.5
        - vertical resolution is not possible: 1.0
    - Both aircraft need to descend
        - one intersection point and both aircraft top of descent occur
          after the intersection: 0.0
        - anything else: 1.0
    - One aircraft needs to descend while the other overfly:
        - one intersection point and the descending aircraft top of descent
          occurs after the intersection: 0.0
        - anything else: 1.0
    - Anything else (climb-overflier, descend-overflier, climb-descend,...)
        - vertical resolution is not possible: 1.0

    Args:
        callsign: defines the identifier of the aircraft in the simulation.
        other_callsign: defines the identifier of the other aircraft.
        simulator_env: defines the underlying simulator environment.
        sector: defines the name of the sector that the aircraft is located.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        interaction: defines the interaction details between both aircraft.

    Returns:
        the coefficient which takes the value 0.0, 0.5 or 1.0
    """
    ac_fl = float(simulator_env.aircraft[callsign].fl)
    ac_exit_fl = float(tracked_data[callsign].exit_coords[sector].fl)
    ac_selected_fl = float(simulator_env.aircraft[callsign].selected_fl)

    other_ac_fl = float(simulator_env.aircraft[other_callsign].fl)
    other_ac_exit_fl = tracked_data[other_callsign].exit_coords[sector].fl
    other_ac_exit_fl = float(other_ac_exit_fl)
    other_ac_selected_fl = simulator_env.aircraft[other_callsign].selected_fl
    other_ac_selected_fl = float(other_ac_selected_fl)

    # Note:
    # low coefficient which translates to reduced punishment for failing to
    # resolve the conflict. this is because the conflict can be resolved using
    # an alternative step climb/descend approach. that is, climb the aircraft
    # with the highest exit flight level to its exit flight level, and
    # then subsequently climb the other aircraft to its exit flight level.
    coeff = None  # initialisation

    if (ac_fl < ac_exit_fl) and (other_ac_fl < other_ac_exit_fl):
        # both aircraft need to climb to reach their exit flight levels.
        if (ac_fl < other_ac_fl and ac_exit_fl < other_ac_exit_fl) or (
            other_ac_fl < ac_fl and other_ac_exit_fl < ac_exit_fl
        ):
            # vertical conflict resolution is possible

            # now check if vertical resolution is being actioned.
            if ac_fl < other_ac_fl and ac_exit_fl < other_ac_selected_fl:
                # other aircraft is the higher aircraft and conflict is being
                # resolved vertically
                coeff = 0.0
            elif other_ac_fl < ac_fl and other_ac_exit_fl < ac_selected_fl:
                # aircraft is the higher aircraft and conflict is being
                # resolved vertically
                coeff = 0.0
            else:
                coeff = 0.5
        else:
            # vertical conflict resolution is *not* possible.
            # the aircraft at the lower current flight level has an exit
            # flight level that is higher than the exit flight level of the
            # other aircraft. so, lateral deconfliction might be required.

            # final check:
            # check whether top of ascent for both aircraft can be achieved
            # before intersection point. if yes lateral deconfliction is not
            # required. otherwise, it is required.

            _intersection = main_or_proxy_intersection_location(callsign, interaction)
            ac_status = top_of_ascent_before_intersection(
                callsign,
                simulator_env,
                tracked_data[callsign].dist_to_target_fl,
                _intersection,
                tracked_data[callsign],
                sector,
                interaction_distance=interaction.lateral_dist_thresh_sv,
            )
            other_ac_status = top_of_ascent_before_intersection(
                other_callsign,
                simulator_env,
                tracked_data[callsign].dist_to_target_fl,
                _intersection,
                tracked_data[other_callsign],
                sector,
                interaction_distance=interaction.lateral_dist_thresh_sv,
            )
            coeff = 0.0 if ac_status and other_ac_status else 1.0

    elif (ac_fl > ac_exit_fl) and (other_ac_fl > other_ac_exit_fl):
        # both aircraft need to descend to reach their exit flight levels.
        _location = (main_or_proxy_intersection_location(callsign, interaction)).location

        ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[callsign],
            simulator_env.airspace,
            tracked_data[callsign].dist_to_target_fl,
            _location,
            tracked_data[callsign].sector_exit_pos,
            tracked_data[callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        other_ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[other_callsign],
            simulator_env.airspace,
            tracked_data[other_callsign].dist_to_target_fl,
            _location,
            tracked_data[other_callsign].sector_exit_pos,
            tracked_data[other_callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        coeff = 0.0 if ac_status and other_ac_status else 1.0

    elif (ac_fl > ac_exit_fl) and (other_ac_fl == other_ac_exit_fl):
        # aircraft: descend, other aircraft: overfly
        _location = (main_or_proxy_intersection_location(callsign, interaction)).location

        ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[callsign],
            simulator_env.airspace,
            tracked_data[callsign].dist_to_target_fl,
            _location,
            tracked_data[callsign].sector_exit_pos,
            tracked_data[callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        coeff = 0.0 if ac_status else 1.0

    elif (ac_fl == ac_exit_fl) and (other_ac_fl > other_ac_exit_fl):
        # aircraft: overfly, other aircraft: descend
        _location = (main_or_proxy_intersection_location(callsign, interaction)).location

        other_ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[other_callsign],
            simulator_env.airspace,
            tracked_data[other_callsign].dist_to_target_fl,
            _location,
            tracked_data[other_callsign].sector_exit_pos,
            tracked_data[other_callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        coeff = 0.0 if other_ac_status else 1.0

    elif (ac_fl < ac_exit_fl) and (other_ac_fl == other_ac_exit_fl):
        # aircraft: climb, other aircraft: overfly
        _intersection = main_or_proxy_intersection_location(callsign, interaction)

        ac_status = top_of_ascent_before_intersection(
            callsign,
            simulator_env,
            tracked_data[callsign].dist_to_target_fl,
            _intersection,
            tracked_data[callsign],
            sector,
            interaction_distance=interaction.lateral_dist_thresh_sv,
        )
        coeff = 0.0 if ac_status else 1.0

    elif (ac_fl == ac_exit_fl) and (other_ac_fl < other_ac_exit_fl):
        # aircraft: overfly, other aircraft: climb
        _intersection = main_or_proxy_intersection_location(callsign, interaction)

        other_ac_status = top_of_ascent_before_intersection(
            other_callsign,
            simulator_env,
            tracked_data[other_callsign].dist_to_target_fl,
            _intersection,
            tracked_data[other_callsign],
            sector,
            interaction_distance=interaction.lateral_dist_thresh_sv,
        )
        coeff = 0.0 if other_ac_status else 1.0

    elif (ac_fl < ac_exit_fl) and (other_ac_fl > other_ac_exit_fl):
        # aircraft: climb, other aircraft: descend
        _location = (main_or_proxy_intersection_location(callsign, interaction)).location

        other_ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[other_callsign],
            simulator_env.airspace,
            tracked_data[other_callsign].dist_to_target_fl,
            _location,
            tracked_data[other_callsign].sector_exit_pos,
            tracked_data[other_callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        coeff = 0.0 if other_ac_status else 1.0

    elif (ac_fl > ac_exit_fl) and (other_ac_fl < other_ac_exit_fl):
        # aircraft: descend, other aircraft: climb
        _location = (main_or_proxy_intersection_location(callsign, interaction)).location

        ac_status = top_of_descent_after_intersection(
            simulator_env.aircraft[callsign],
            simulator_env.airspace,
            tracked_data[callsign].dist_to_target_fl,
            _location,
            tracked_data[callsign].sector_exit_pos,
            tracked_data[callsign].pos_at_last_route_direct,
            # uncertainty is already added to distance to target fl
            uncertainty_distance=0,
        )
        coeff = 0.0 if ac_status else 1.0

    else:
        # default case: lateral deconfliction is required.
        # overflier_overflier
        coeff = 1.0

    return coeff


def conflict_resolution_tanh(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for resolving conflict through heading deconfliction strategy.

    Note: when there are no conflict to resolve, a 0.0 reward is returned.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    sector = gym_env.get_active_airspace_sector()
    tracked_data = gym_env.get_tracked_aircraft_data(copy_data=False)

    # get the aircraft's interactions with other active aircraft being tracked
    interactions = gym_env.get_traffic_monitor().get_relevant_traffic(callsign)

    _rewards = []
    if len(interactions) > 0:
        # get aircraft centreline info
        _centre_info = tracked_data[callsign].centreline_info_cr
        ac_centre_dist_cr = _centre_info[0] * _centre_info[1]

        for interaction in interactions:
            other_callsign = interaction.other_callsign
            # lateral separation penalty: as aircraft become laterally
            # separated up to 5.0nm if both aircraft are in the sector or
            # 2.5 if only the subject aircraft is in sector (i.e., the other
            # aircraft is not in the sector).
            # the penalty reduces from -1.0 to 0.0
            pair_distance = interaction.centreline_dist_diff_cr
            if tracked_data[other_callsign].pos_status == PositionStatus.IN_SECTOR:
                _diff = pair_distance - 2.5
            else:
                # pair_distance only increases as the subject aircraft moves
                # away from the its centreline as the other aircraft is not
                # yet in route. compensate for this.
                _diff = (pair_distance - 2.5) * 2

            # range: [-1.0, 0.0]
            penalty_1 = (np.tanh(_diff) - 1.0) / 2.0
            penalty_1 = round(penalty_1, 2)

            # fairness penalty: ensure that the current aircraft is
            # contributing to the lateral separation. without this, the
            # policy could learn to only take on aircraft off route to
            # achieve the separation objective.
            # range: [-1.0, 0.0]
            clipped_dist = float(np.clip(abs(ac_centre_dist_cr), 0.0, 2.5))
            penalty_2 = project_x_from_range_to_range(
                # clipped_dist, (0.0, 2.5), (-2.0, 1.0)
                clipped_dist,
                (0.0, 2.5),
                (-1.0, 0.0),
            )

            # get overlap coefficient
            # range: [0, 1.0]
            coeff_1 = get_overlap_coeff(
                callsign,
                interaction.other_callsign,
                simulator_env,
                sector,
                tracked_data,
                interaction,
            )

            reward = (penalty_1 * coeff_1) + (penalty_2 * coeff_1)

            _rewards.append(float(reward / 2.0))

    return 0.0 if len(_rewards) == 0 else float(np.sum(_rewards))


def conflict_resolution_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for resolving conflict through heading deconfliction strategy.

    Note: when there are no conflict to resolve, a 0.0 reward is returned.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    sector = gym_env.get_active_airspace_sector()
    tracked_data = gym_env.get_tracked_aircraft_data(copy_data=False)

    # get the aircraft's interactions with other active aircraft being tracked
    interactions = gym_env.get_traffic_monitor().get_relevant_traffic(callsign)

    _rewards = []
    if len(interactions) > 0:
        # get aircraft's current route centreline info
        _centre_info = tracked_data[callsign].centreline_info_cr
        ac_centre_dist_cr = _centre_info[0] * _centre_info[1]

        for interaction in interactions:
            other_callsign = interaction.other_callsign
            # lateral separation penalty: as aircraft become laterally
            # separated up to 5.0nm, the penalty reduces from -1.0 to 0.0
            pair_distance = interaction.centreline_dist_diff_cr
            if tracked_data[other_callsign].pos_status == PositionStatus.IN_SECTOR:
                _diff = pair_distance
            else:
                # pair_distance only increases as the subject aircraft moves
                # away from the its centreline as the other aircraft is not
                # yet in route. compensate for this.
                _diff = pair_distance * 2

            penalty_1 = -np.exp(-0.15 * (pair_distance**2))
            penalty_1 = round(penalty_1, 2)

            # fairness penalty: ensure that the current aircraft is
            # contributing to the lateral separation. without this, the
            # policy could learn to only take on aircraft off route to
            # achieve the separation objective. the penalty reduces from
            # -1.0 to 0.0
            clipped_dist = float(np.clip(abs(ac_centre_dist_cr), 0.0, 2.5))
            penalty_2 = clipped_dist - 2.5
            penalty_2 = penalty_2 / 2.5  # scale to range [-1.0, 0]

            # get overlap coefficient
            coeff_1 = get_overlap_coeff(
                callsign,
                interaction.other_callsign,
                simulator_env,
                sector,
                tracked_data,
                interaction,
            )

            reward = (penalty_1 * coeff_1) + (penalty_2 * coeff_1)
            _rewards.append(float(reward / 2.0))

    return 0.0 if len(_rewards) == 0 else float(np.sum(_rewards))
