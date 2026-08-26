import numpy as np
import pytest

from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.core.airspace import Airspace
from bluebird_dt.core.route import Route
from bluebird_dt.events import EventHandler
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.scenario_manager.infinite import Infinite, check_safe_to_spawn
from bluebird_dt.simulator import Simulator


def test_init_exceptions(generate_i):
    """
    Test various misuses of the constructor
    """

    airspace, routes = generate_i

    # needs at least airspace and routes as constructor args
    with pytest.raises(TypeError):
        Infinite()

    with pytest.raises(ValueError):
        Infinite(airspace=airspace, routes=routes, initial_spawn_rate=-0.4)

    with pytest.raises(ValueError):
        Infinite(airspace=airspace, routes=routes, initial_spawn_rate=0.1, max_spawn_rate=0.05)
    

@pytest.mark.parametrize(
    "airspace_routes, num_starter_aircraft",
    [
        ("generate_i", 2),
        ("generate_x", 4),
        ("generate_y", 3),
    ],
)
def test_all_airspaces(airspace_routes, num_starter_aircraft, request):
    """
    Test generator works for all available Airspaces.

    For each Airspace, check:
       - the correct number of Aircraft is generated
       - the two Aircraft are travelling in opposite direction on the same Route
       - the entry/exit Coordinations FLs are within the Airspace limits at entry/exit
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    sector_name = list(airspace.sectors.keys())[0]
    em = Infinite(airspace=airspace, routes=routes, num_starter_aircraft=num_starter_aircraft).create_env_manager()

    flight_plans = em.event_handler.flight_df

    assert len(em.event_handler.radar_df) == num_starter_aircraft
    # evolve the simulation a few steps to allow all starter aircraft to spawn
    for _ in range(num_starter_aircraft):
        em.evolve(6)
    volume = airspace.sectors[sector_name].volumes[0]
    for i in range(num_starter_aircraft):
        flight_plan = flight_plans.iloc[i]
        assert volume.min_fl <= em.environment.entry_coordination(sector_name, flight_plan.callsign).fl <= volume.max_fl

@pytest.mark.parametrize(
    "initial_spawn_rate, spawn_rate_increment, spawn_rate_increase_interval, max_spawn_rate",
    [
        (0.01, 0.01, 6, 0.1),
        (0.02, 0.03, 12, 0.3),
        (0.05, 0.1, 24, 0.08),
    ],
)
def test_spawn_frequency_ramp_up(generate_x, initial_spawn_rate, spawn_rate_increment, spawn_rate_increase_interval, max_spawn_rate):
    """
    The spawn frequency should increment by a specified amount after a specified
    interval, until it reaches a maximum.  Check that
    - it starts off at the correct value
    - it never exceeds the maximum
    - it increments as expected
    """
    airspace, routes = generate_x
    sim = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=initial_spawn_rate, 
        spawn_rate_increment=spawn_rate_increment, 
        spawn_rate_increase_interval=spawn_rate_increase_interval, 
        max_spawn_rate=max_spawn_rate
    ).to_simulator()
    assert sim.scenario_manager.current_spawn_rate == initial_spawn_rate
    
    for i in range(10):
        sim.evolve(6)
        num_increments = i*6 // spawn_rate_increase_interval
        expected_spawn_rate = min(
            max_spawn_rate,
            initial_spawn_rate + num_increments*spawn_rate_increment
        )
        assert np.allclose(sim.scenario_manager.current_spawn_rate, expected_spawn_rate, atol=0.001)

def test_higher_spawn_rate_gives_more_aircraft(generate_x):
    """
    Aircraft spawning is stochastic, so hard to predict the actual amount,
    but if we have a much higher spawn rate we should have more aircraft in the
    environment after a number of steps.
    """

    airspace, routes = generate_x
    sim1 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.05, 
    ).to_simulator() 
    airspace, routes = generate_x
    sim2 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.2,
        max_spawn_rate=0.3,
    ).to_simulator()
    for _ in range(20):
        sim1.evolve(6)
        sim2.evolve(6)
    assert len(sim2.manager.environment.aircraft) > len(sim1.manager.environment.aircraft)

def test_dont_set_random_seed(generate_x):
    """
    If we don't set the random seed, we should get different results every run.
    """
    airspace, routes = generate_x
    sim1 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.1, 
    ).to_simulator() 
    # another instance, all settings the same, no random seed set
    sim2 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.1,
    ).to_simulator()
    for _ in range(5):
        sim1.evolve(6)
        sim2.evolve(6)
    for k, v in sim1.manager.environment.aircraft.items():
        if not k in sim2.manager.environment.aircraft:
            continue
        assert sim2.manager.environment.aircraft[k].data() != v.data()

def test_set_random_seed(generate_x):
    """
    If we do set the random seed, we should get identical results every run.
    """
    airspace, routes = generate_x
    sim1 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.1,
        random_seed=1234, 
    ).to_simulator() 
    # another identical instance, including same random seed
    sim2 = Infinite(
        airspace=airspace, 
        routes=routes, 
        initial_spawn_rate=0.1,
        random_seed=1234
    ).to_simulator()
    # evolve both simulators 100 steps of 6s.
    for _ in range(100):
        sim1.evolve(6)
        sim2.evolve(6)
    for k, v in sim1.manager.environment.aircraft.items():
        assert sim2.manager.environment.aircraft[k].data() == v.data()

@pytest.mark.parametrize(
    "speed_range",
    [
        None, [500,600], [300,800]
    ],
)
def test_set_speed_range(speed_range):
    """
    Test that we can set the speed range of generated aircraft via 'setup'
    """
    sim = Infinite.setup(
        "X-Sector",
        initial_spawn_rate=0.1,
        speed_range=speed_range
    )
 
    for _ in range(50):
        sim.evolve(6)
        for ac in sim.manager.environment.aircraft.values():
            if not speed_range:
                # default range is 350-450
                assert 350. < ac.speed_tas < 450.
            else:
                assert speed_range[0] < ac.speed_tas < speed_range[1]


def test_sometimes_generate_more_than_one_aircraft():
    """
    Test that with a higher spawn rate, we sometimes generate >1 aircraft per tick.
    """
    sim = Infinite.setup(
        "X-Sector",
        initial_spawn_rate=0.25,
        max_spawn_rate=0.5,
    )
    # move forward a couple of ticks, so as not to consider starter aircraft
    for _ in range(3):
        sim.evolve(6)
    # now count the number of new aircraft per tick
    max_num_new_aircraft = 0
    current_num_aircraft = len(sim.manager.environment.aircraft)
    for _ in range(10):
        sim.evolve(6)
        num_new_aircraft = len(sim.manager.environment.aircraft) - current_num_aircraft
        max_num_new_aircraft = max(max_num_new_aircraft, num_new_aircraft)
        current_num_aircraft = len(sim.manager.environment.aircraft)
    assert max_num_new_aircraft > 1

@pytest.mark.parametrize(
    "min_spawn_distance",
    [
        5., 10., 20.
    ],
)
def test_check_safe_to_spawn(generate_simple_environment, min_spawn_distance):
    """
    Test the function that vetoes aircraft from spawning too close to others.
    Simple environment fixture has two aircraft:
    AIR0: lat,lon = (-1.0,0.0), FL=150
    AIR1: lat,lon = (1.0,0.0), FL=200
    """
    env = generate_simple_environment
    # generate 1000 random candidate aircraft between (-1,-1) and (1,1)
    for _ in range(100):
        lat = np.random.random()*2 - 1
        lon = np.random.random()*2 - 1
        possible_fls = list(range(140,220,10))
        fl = float(np.random.choice(possible_fls))
        sfl = int(np.random.choice(possible_fls))
        heading = np.random.randint(0,360)
        a = Aircraft(callsign="TEST",lat=lat,lon=lon,fl=fl,heading=heading, flight_plan=None)
        a.selected_instructions.fl=sfl

        expected_safe = True
        for existing_ac in env.aircraft.values():
            if existing_ac.current_sector == "background":
                continue
            distance = a.pos2d().distance(existing_ac.pos2d())
            same_fl = a.fl == existing_ac.fl
            overlapping_fl = (
                (min(a.fl, a.selected_fl) <= max(existing_ac.fl, existing_ac.selected_fl)) and \
                (max(a.fl, a.selected_fl) >= min(existing_ac.fl, existing_ac.selected_fl))
            )
            if (distance < 60.) and same_fl:
                expected_safe = False
                break
            if (distance < min_spawn_distance) and overlapping_fl:
                expected_safe = False
                break
        assert expected_safe == check_safe_to_spawn(a, env, min_spawn_distance)


@pytest.mark.parametrize(
        "scenario_name", 
        ("I-Sector","Y-Sector", "X-Sector", "Xplus-Sector", "Springfield")
)        
def test_sim_from_category(scenario_name):
    """
    Test that we can instantiate the simulator using "from_category"
    """
    s = Simulator.from_category("Infinite", scenario_name)
    assert isinstance(s, Simulator)