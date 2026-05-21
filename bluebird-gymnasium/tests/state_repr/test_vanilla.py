import numpy as np
import pytest

from bluebird_gymnasium.state_repr.vanilla import VanillaRepresentation
from bluebird_gymnasium.state_repr.vanilla import VanillaRepresentationRaw


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_init_exception(gym_env, knn):
    VanillaRepresentation(knn=knn)
    VanillaRepresentationRaw(knn=knn)


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_vanilla_repr(gym_env, knn):
    state_repr = VanillaRepresentation(knn=knn)

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_vanilla_repr_raw(gym_env, knn):
    state_repr = VanillaRepresentationRaw(knn=knn)

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)
