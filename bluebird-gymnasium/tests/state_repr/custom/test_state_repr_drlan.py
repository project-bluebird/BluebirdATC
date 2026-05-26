from __future__ import annotations

import typing

import numpy as np
import pytest

from bluebird_gymnasium.state_repr.custom.state_repr_drlan import DrlanRepresentation, DrlanRepresentationRaw

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_init_exception(gym_env: BaseEnv, knn: int):

    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(exclude_noop_action=False)
    DrlanRepresentation(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )
    DrlanRepresentationRaw(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_drlan_repr(gym_env: BaseEnv, knn: int):

    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(exclude_noop_action=False)
    state_repr = DrlanRepresentation(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )

    simulator_env = gym_env.get_simulator_env()
    callsign = next(iter(simulator_env.aircraft.keys()))
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_drlan_repr_raw(gym_env: BaseEnv, knn: int):
    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(exclude_noop_action=False)
    state_repr = DrlanRepresentationRaw(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )

    simulator_env = gym_env.get_simulator_env()
    callsign = next(iter(simulator_env.aircraft.keys()))
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)
