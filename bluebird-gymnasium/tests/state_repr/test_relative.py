import numpy as np
import pytest

from bluebird_gymnasium.state_repr.relative import RelativeRepresentation
from bluebird_gymnasium.state_repr.relative import RelativeRepresentationRaw


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_init_exception(gym_env, knn):
    _ = RelativeRepresentation(knn=knn)
    _ = RelativeRepresentationRaw(knn=knn)


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_relative_repr(gym_env, knn):
    state_repr = RelativeRepresentation(knn=knn)

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)


@pytest.mark.parametrize("knn", [0, 2, 4])
def test_relative_repr_raw(gym_env, knn):
    state_repr = RelativeRepresentationRaw(knn=knn)

    simulator_env = gym_env.get_simulator_env()
    callsign = list(simulator_env.aircraft.keys())[0]
    state = state_repr.repr(gym_env, callsign)

    assert isinstance(state, np.ndarray)
    assert state.shape == state_repr.low.shape
    assert np.all(state >= state_repr.low)
    assert np.all(state <= state_repr.high)
