import numpy as np
import pytest

from bluebird_gymnasium.state_repr.custom.state_repr_drlan import DrlanRepresentation
from bluebird_gymnasium.state_repr.custom.state_repr_drlan import DrlanRepresentationRaw


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_init_exception(gym_env, knn):

    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(
        exclude_noop_action=False
    )
    _ = DrlanRepresentation(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )
    _ = DrlanRepresentationRaw(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )

@pytest.mark.parametrize("knn", [0, 2, 4])
def test_drlan_repr(gym_env, knn):

    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(
        exclude_noop_action=False
    )
    state_repr = DrlanRepresentation(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)

@pytest.mark.parametrize("knn", [0, 2, 4])
def test_drlan_repr_raw(gym_env, knn):
    action_parser = gym_env.get_action_parser()
    num_actions_per_aircraft = action_parser.get_num_actions_per_aircraft(
        exclude_noop_action=False
    )
    state_repr = DrlanRepresentationRaw(
        knn=knn,
        num_forward_fixes=3,
        use_filed_route=True,
        num_actions=num_actions_per_aircraft,
    )

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)
