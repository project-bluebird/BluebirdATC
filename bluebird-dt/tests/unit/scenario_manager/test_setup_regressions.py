import pytest

from bluebird_dt.airspace_generator.airspace_loader import AirspaceLoader
from bluebird_dt.predictor import SimplePredictor
from bluebird_dt.scenario_manager import Custom, Infinite, Regular, TwoAircraft
from bluebird_dt.simulator import Simulator


@pytest.mark.parametrize("start_time", [0, 60, 61])
@pytest.mark.parametrize("dt", [1, 4])
@pytest.mark.parametrize("total_time", [10, 12])
def test_regular_initial_advance(start_time: float, dt: float, total_time: float):
    sim = Regular.setup(
        "I-Sector",
        total_time=total_time,
        num_aircraft=1,
        start_time=start_time,
        random_seed=1,
        predictor=SimplePredictor(dt=dt, fix_proximity_threshold=2),
        save_log_to_file=False,
        save_csv=False,
        autosave_interval=None,
    )
    # One aircraft enters halfway through total_time. The initial
    # advance must pass that event, without counting the start timestamp twice.
    expected_advance = (6 if total_time == 10 else 12) if dt == 1 else 8
    assert sim.manager.environment.time == start_time + expected_advance
    assert len(sim.manager.environment.aircraft) == 1


@pytest.mark.parametrize("category", ["Two Aircraft", "Regular", "Custom", "Infinite"])
def test_factory_preserves_simulator_subclass(category: str):
    class DerivedSimulator(Simulator):
        pass

    sim = DerivedSimulator.from_category(
        category,
        "I-Sector",
        random_seed=1,
        save_log_to_file=False,
        save_csv=False,
        autosave_interval=None,
    )
    assert type(sim) is DerivedSimulator


def test_custom_accepts_populated_event_handler():
    airspace, routes, _ = AirspaceLoader.load("I-Sector")
    handler = Custom(1, airspace, routes, random_seed=1).create_event_handler()
    manager = Custom(0, airspace, routes).create_env_manager(event_handler=handler)
    assert manager.event_handler is handler
    assert len(manager.environment.aircraft) == 1


def test_custom_rejects_empty_supplied_handler():
    airspace, routes, _ = AirspaceLoader.load("I-Sector")
    manager = Custom(0, airspace, routes)
    with pytest.raises(RuntimeError, match="No initial or custom aircraft"):
        manager.create_env_manager(event_handler=manager.create_event_handler())


@pytest.mark.parametrize("manager_type", [TwoAircraft, Infinite])
@pytest.mark.parametrize("dt", [1, 4])
def test_other_managers_initialise_with_predictor_timestep(manager_type: type[TwoAircraft | Infinite], dt: float):
    sim = manager_type.setup(
        "I-Sector",
        random_seed=1,
        predictor=SimplePredictor(dt=dt, fix_proximity_threshold=2),
        save_log_to_file=False,
        save_csv=False,
        autosave_interval=None,
    )
    assert sim.manager.environment.time == (6 if dt == 1 else 8)
    assert len(sim.manager.environment.aircraft) > 0
