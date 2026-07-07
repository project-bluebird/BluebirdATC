import pytest
from bluebird_dt.events.event_handler import EventHandler
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
    ac_internal_events_df = simulator.manager.event_handler.ac_internals_df
    coord_df = simulator.manager.event_handler.coordination_df

    # test number of events matches number of aircraft
    assert len(radar_events_df) == num_aircraft
    assert len(ac_internal_events_df) == num_aircraft

    # test number of coordination is double the number of aircraft (exit/entry per aircraft)
    assert len(coord_df) == num_aircraft * 2

    # check start time
    assert simulator.manager.environment.start_time == 12

@pytest.mark.parametrize(
        "scenario_name", 
        ("I-Sector","Y-Sector", "X-Sector", "Xplus-Sector", "Springfield")
)
def test_sim_from_category(scenario_name):
    """
    Test that we can instantiate the simulator using "from_category"
    """
    s = Simulator.from_category("Regular", scenario_name)
    assert isinstance(s, Simulator)

def test_init_exceptions(generate_i):
    """
    Test ValueError is raised when class is initiated with invalid inputs.
    """

    airspace, routes = generate_i

    with pytest.raises(ValueError):
        Regular(total_time=0, num_aircraft=1, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Regular(total_time=1, num_aircraft=0, airspace=airspace, routes=routes)

@pytest.mark.parametrize(
    "airspace_routes",
    [
        "generate_i",
        "generate_x",
        "generate_y",
        "generate_thunderdome",
    ],
)
def test_all_airspaces(airspace_routes, request):
    """
    Test generator works for all available Airspaces.

    For each num_aircraft and Airspace, check:
       - the correct number of Aircraft is generated
       - the entry/exit Coordinations FLs are within the Airspace limits at entry/exit
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    sector_name = list(airspace.sectors.keys())[0]
    volume = airspace.sectors[sector_name].volumes[0]

    for num_aircraft in [1, 2, 5, 10, 100]:
        em = Regular(total_time=10, num_aircraft = num_aircraft, airspace=airspace, routes=routes).create_env_manager()
        radar_df = em.event_handler.radar_df
        coord_df = em.event_handler.coordination_df
        assert len(radar_df) == num_aircraft
        assert len(coord_df) == 2 * num_aircraft

        for _, row in radar_df.iterrows():
            entry_coord = coord_df[(coord_df.callsign == row.callsign) & (coord_df.to_sector == sector_name)].squeeze()
            exit_coord = coord_df[(coord_df.callsign == row.callsign) & (coord_df.from_sector == sector_name)].squeeze()
            assert volume.min_fl <= entry_coord.fl <= volume.max_fl
            assert volume.min_fl <= exit_coord.fl <= volume.max_fl


def test_repeat(generate_i):
    """
    Check generator works when called multiple times.
    """

    airspace, routes = generate_i

    for i in range(5):
        em = Regular(total_time=10, num_aircraft=10, airspace=airspace, routes=routes).create_env_manager()
        assert len(em.event_handler.radar_df) == 10


def test_create_event_handler(generate_i):
    """
    Test that an event handler can be created
    """
    airspace, routes = generate_i
    scenario = Regular(10, 10, airspace, routes)
    eh = scenario.create_event_handler()
    assert isinstance(eh, EventHandler)

def test_create_env_manager(generate_i):
    """
    Test that environment manager can be created
    """
    airspace, routes = generate_i
    scenario = Regular(10, 10, airspace, routes)

    em = scenario.create_env_manager()

    assert em is not None
    assert em.environment is not None

    assert em.environment.airspace is airspace
    assert em.event_handler is not None

    assert em.environment.wind_field is None
    assert em.environment.forecast_wind_field is None
