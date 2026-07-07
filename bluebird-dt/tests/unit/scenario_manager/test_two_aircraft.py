from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.simulator import Simulator
import pytest

from bluebird_dt.core.airspace import Airspace
from bluebird_dt.core.route import Route
from bluebird_dt.events import EventHandler
from bluebird_dt.scenario_manager import TwoAircraft


def test_init_exceptions(generate_i):
    """
    TwoAircraft must be initiated with a positive time.
    If speed_range is provided, it must be a list of length 2.
    """

    airspace, routes = generate_i

    with pytest.raises(TypeError):
        TwoAircraft()

    with pytest.raises(ValueError):
        TwoAircraft(total_time=0, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        TwoAircraft(total_time=-1, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        TwoAircraft(total_time=1, speed_range=[100, 200, 500], airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        TwoAircraft(total_time=1, speed_range=[], airspace=airspace, routes=routes)
    

@pytest.mark.parametrize(
    "airspace_routes",
    [
        "generate_i",
        "generate_x",
        "generate_y",
    ],
)
def test_all_airspaces(airspace_routes, request):
    """
    Test generator works for all available Airspaces.

    For each Airspace, check:
       - the correct number of Aircraft is generated
       - the two Aircraft are travelling in opposite direction on the same Route
       - the entry/exit Coordinations FLs are within the Airspace limits at entry/exit
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    sector_name = list(airspace.sectors.keys())[0]
    em = TwoAircraft(total_time=100, scenario_type="random", airspace=airspace, routes=routes).create_env_manager()

    flight_plans = em.event_handler.flight_plan_df

    assert len(em.event_handler.radar_df) == 2
    ac1_flight_plan = flight_plans.iloc[0]
    ac2_flight_plan = flight_plans.iloc[1]

    assert ac1_flight_plan.route_filed == ac2_flight_plan.route_filed[::-1]
    volume = airspace.sectors[sector_name].volumes[0]
    for fp in [ac1_flight_plan, ac2_flight_plan]:
        assert volume.min_fl <= em.environment.entry_coordination(sector_name, fp.callsign).fl <= volume.max_fl
        assert volume.min_fl <= em.environment.exit_coordination(sector_name, fp.callsign).fl <= volume.max_fl


def test_constrained_airspace(generate_i_fl300_to_fl310):
    """
    Check all works as expected in lowest possible airspace (height of 10FL).
    In this case there is only one possible solution for climber/descender scenarios.
    """

    airspace, routes = generate_i_fl300_to_fl310
    sector_name = list(airspace.sectors.keys())[0]
    gen_climber = TwoAircraft(total_time=100, scenario_type="climber", airspace=airspace, routes=routes)

    # climber goes from 300 - 310 and overflier is at 310
    em = gen_climber.create_env_manager()

    assert em.environment.entry_coordination(sector_name, "AIR0").fl == 310
    assert em.environment.entry_coordination(sector_name, "AIR1").fl == 300
    assert em.environment.exit_coordination(sector_name, "AIR1").fl == 310

    # descender goes from 310 - 310 and overflier is at 300
    gen_descender = TwoAircraft(total_time=100, scenario_type="descender", airspace=airspace, routes=routes)
    em = gen_descender.create_env_manager()

    assert em.environment.entry_coordination(sector_name, "AIR0").fl == 300
    assert em.environment.entry_coordination(sector_name, "AIR1").fl == 310
    assert em.environment.exit_coordination(sector_name, "AIR1").fl == 300


def test_repeat(generate_i):
    """
    Check generator works when called multiple times.
    """

    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, airspace=airspace, routes=routes)

    for _i in range(5):
        env_manager = gen.create_env_manager()

        radar_events_df = env_manager.event_handler.radar_df
        ac_internal_events_df = env_manager.event_handler.ac_internals_df
        coord_df = env_manager.event_handler.coordination_df

        # test number of events matches number of aircraft
        assert len(radar_events_df) == 2
        assert len(ac_internal_events_df) == 2

        # test number of coordination is double the number of aircraft (exit/entry per aircraft)
        assert len(coord_df) == 2 * 2


def test_overflier(generate_i):
    """
    Check overflier behaviour is as expected.
    """

    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, scenario_type="overflier", airspace=airspace, routes=routes)
    sector_name = list(airspace.sectors.keys())[0]
    em = gen.create_env_manager()

    # check both aircraft are overfliers on the same FL
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        == em.environment.exit_coordination(sector_name, "AIR0").fl
    )
    assert (
        em.environment.entry_coordination(sector_name, "AIR1").fl
        == em.environment.exit_coordination(sector_name, "AIR1").fl
    )
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        == em.environment.entry_coordination(sector_name, "AIR1").fl
    )


def test_descender(generate_i):
    """
    Check descender-overflier behaviour is as expected.
    """

    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, scenario_type="descender", airspace=airspace, routes=routes)
    sector_name = list(airspace.sectors.keys())[0]
    em = gen.create_env_manager()

    # check first aircraft is an overflier && second aircraft is a descender
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        == em.environment.exit_coordination(sector_name, "AIR0").fl
    )
    assert (
        em.environment.entry_coordination(sector_name, "AIR1").fl
        > em.environment.exit_coordination(sector_name, "AIR1").fl
    )

    # check the two aircraft paths cross
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        < em.environment.entry_coordination(sector_name, "AIR1").fl
    )
    assert (
        em.environment.exit_coordination(sector_name, "AIR0").fl
        >= em.environment.exit_coordination(sector_name, "AIR1").fl
    )


def test_climber(generate_i: tuple[Airspace, list[Route]]):
    """
    Check climber-overflier behaviour is as expected.
    """

    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, scenario_type="climber", airspace=airspace, routes=routes)
    sector_name = next(iter(airspace.sectors.keys()))
    em = gen.create_env_manager()

    # check first aircraft is an overflier && second aircraft is a climber
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        == em.environment.exit_coordination(sector_name, "AIR0").fl
    )
    assert (
        em.environment.entry_coordination(sector_name, "AIR1").fl
        < em.environment.exit_coordination(sector_name, "AIR1").fl
    )

    # check the two aircraft paths cross
    assert (
        em.environment.entry_coordination(sector_name, "AIR0").fl
        > em.environment.entry_coordination(sector_name, "AIR1").fl
    )
    assert (
        em.environment.exit_coordination(sector_name, "AIR0").fl
        <= em.environment.exit_coordination(sector_name, "AIR1").fl
    )


def test_speed_range(generate_i):
    """
    If speed_range is provided, check the Aircraft speed is within this range.
    """

    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, speed_range=[100, 200], scenario_type="overflier", airspace=airspace, routes=routes)
    em = gen.create_env_manager()

    air0 = em.environment.aircraft["AIR0"]
    air1 = em.environment.aircraft["AIR0"]

    assert air0.speed_tas is not None
    assert air1.speed_tas is not None

    assert air0.speed_tas >= 100
    assert air0.speed_tas <= 200
    assert air1.speed_tas >= 100
    assert air1.speed_tas <= 200


def test_invalid_scenario_type(generate_i):
    """
    Check generate_aircraft_events raises Value Error if passed an invalid scenario type.
    """

    airspace, routes = generate_i

    with pytest.raises(ValueError):
        gen = TwoAircraft(total_time=100, scenario_type="fake scenario type", airspace=airspace, routes=routes)

def test_create_event_handler(generate_i):
    airspace, routes = generate_i
    gen = TwoAircraft(total_time=100, airspace=airspace, routes=routes)
    eh =gen.create_event_handler()
    assert isinstance(eh, EventHandler)

def test_to_simulator(generate_i):
    """
    Test the TwoAircraft.to_simulator() method creates the required components and sets the appropriate variables correctly.
    """
    
    airspace, routes = generate_i
    sector_name = next(iter(airspace.sectors.keys()))
    
    simulator = TwoAircraft(total_time=100,
        airspace=airspace,
        routes=routes, 
        scenario_type='climber',
        start_time=12).to_simulator(scenario_name="test-scenario")

    assert isinstance(simulator.manager, EnvironmentManager)
    assert isinstance(simulator.scenario_manager, TwoAircraft)
    assert simulator.projection_centre is None
    assert isinstance(simulator, Simulator)

    assert simulator.scenario_name == "test-scenario"
    assert simulator.category is None

    # check climber type
    # check first aircraft is an overflier && second aircraft is a climber
    assert (
        simulator.manager.environment.entry_coordination(sector_name, "AIR0").fl
        == simulator.manager.environment.exit_coordination(sector_name, "AIR0").fl
    )
    assert (
        simulator.manager.environment.entry_coordination(sector_name, "AIR1").fl
        < simulator.manager.environment.exit_coordination(sector_name, "AIR1").fl
    )

    # check the two aircraft paths cross
    assert (
        simulator.manager.environment.entry_coordination(sector_name, "AIR0").fl
        > simulator.manager.environment.entry_coordination(sector_name, "AIR1").fl
    )
    assert (
        simulator.manager.environment.exit_coordination(sector_name, "AIR0").fl
        <= simulator.manager.environment.exit_coordination(sector_name, "AIR1").fl
    )

    # check start time
    assert simulator.manager.environment.start_time == 12
    
    # Check returned simulator works as expected
    simulator.evolve(6)
    assert len(simulator.manager.environment.aircraft) is not None

@pytest.mark.parametrize(
        "scenario_name", 
        ("I-Sector","Y-Sector", "X-Sector", "Xplus-Sector", "Springfield")
)        
def test_sim_from_category(scenario_name):
    """
    Test that we can instantiate the simulator using "from_category"
    """
    s = Simulator.from_category("Two Aircraft", scenario_name)
    assert isinstance(s, Simulator)
