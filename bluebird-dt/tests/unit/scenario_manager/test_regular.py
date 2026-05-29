from bluebird_dt.scenario_manager import Regular
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.simulator import Simulator

def test_to_simulator(generate_i):
    """
    Test the Regular.to_simulator() method creates the required components and sets the appropriate variables correctly.
    """
        
    airspace, routes = generate_i
    num_aircraft = 10

    simulator = Regular(
        total_time=100,
        num_aircraft=num_aircraft,
        airspace=airspace,
        routes=routes,
        start_time=12).to_simulator(scenario_name="test-scenario")

    assert isinstance(simulator.manager, EnvironmentManager)
    assert isinstance(simulator.scenario_manager, Regular)
    assert simulator.projection_centre is None
    assert isinstance(simulator, Simulator)

    assert simulator.scenario_name == "test-scenario"
    assert simulator.category is None

    radar_events_df = simulator.manager.event_handler.radar_df
    ac_internal_events_df = simulator.manager.event_handler.aircraft_internals_df
    coord_df = simulator.manager.event_handler.coordination_df

    # test number of events matches number of aircraft
    assert len(radar_events_df) == num_aircraft
    assert len(ac_internal_events_df) == num_aircraft

    # test number of coordination is double the number of aircraft (exit/entry per aircraft)
    assert len(coord_df) == num_aircraft * 2

    # check start time
    assert simulator.manager.environment.start_time == 12
