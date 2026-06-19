from bluebird_dt.core import Action, Pos2D
from bluebird_dt.scenario_manager import SpringfieldScenarioManager
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.simulator import Simulator

def test_incomm():
    """
    Test that aircraft have their current_sector updated when they enter,
    and that the `on_route` flag gets set to False
    """
    sim = SpringfieldScenarioManager(scenario_name="testScenario").to_simulator(
        log_filename="test",
        autosave=False,
    )
    # aircraft AIR01 starts off outside the sector
    assert "AIR01" in sim.manager.environment.aircraft
    assert sim.manager.environment.aircraft["AIR01"].current_sector == "background"
    assert sim.manager.environment.aircraft["AIR01"].on_route == True
    # roll the simulation forwards 3 mins
    for _ in range(30):
        sim.evolve(6)
    # AIR01 should now be in the sector, with on_route flag set to False
    assert sim.manager.environment.aircraft["AIR01"].current_sector == "SPRINGFIELD"
    assert sim.manager.environment.aircraft["AIR01"].on_route == False

def test_outcomm():
    """
    Test that aircraft are outcommed when leaving the sector via the correct boundary edge
    """
    sim = SpringfieldScenarioManager(scenario_name="testScenario").to_simulator(
        log_filename="test",
        autosave=False,
    )
    # AIR01 is going from TERMI in the NE to LEGGO in the south.
    # If, when it gets close to SIMPS, it changes heading to 180,
    # it should exit through the correct edge of the sector.
    sim.evolve(500)
    fix_pos = sim.manager.environment.airspace.fixes.places["SIMPS"]
    while True:
        sim.evolve(10)
        aircraft = sim.manager.environment.aircraft["AIR01"]
        
        if Pos2D(aircraft.lat, aircraft.lon).distance(fix_pos) < 10.0:
            break
    action = Action(callsign="AIR01", kind="change_heading_to", value=180., agent="test", sector="SPRINGFIELD")
    sim.manager.receive_actions([action])
    for _ in range(100):
        sim.evolve(10)
    # by now, should have been outcommed
    assert sim.manager.environment.aircraft["AIR01"].current_sector == "background"
    # should still have "on_route" flag set to False
    assert sim.manager.environment.aircraft["AIR01"].on_route == False


def test_landing():
    """
    Test that aircraft with final fix at EGRR are removed from the simulation 
    when they get close to it.
    """
    sim = SpringfieldScenarioManager(scenario_name="test-airport-scenario").to_simulator(
        log_filename="test",
        autosave=False,
    )
    sim.evolve(800)
    # AIR03 has a route starting at CAKES and ending at EGRR
    # when we get to STEPP, set heading towards EGGR
    fix_pos = sim.manager.environment.airspace.fixes.places["STEPP"]
    while True:
        sim.evolve(10)
        aircraft = sim.manager.environment.aircraft["AIR03"]
        if Pos2D(aircraft.lat, aircraft.lon).distance(fix_pos) < 10.0:
            break
    fix_pos = sim.manager.environment.airspace.fixes.places["EGRR"]
    heading_target = Pos2D(aircraft.lat, aircraft.lon).bearing_to(fix_pos)
    action = Action(callsign="AIR03", kind="change_heading_to", value=heading_target, agent="test", sector="SPRINGFIELD")
    sim.manager.receive_actions([action])
    for _ in range(20):
        sim.evolve(10)
    # set the heading again in case the turn model means we would miss EGRR 
    heading_target = Pos2D(aircraft.lat, aircraft.lon).bearing_to(fix_pos)
    action = Action(callsign="AIR03", kind="change_heading_to", value=heading_target, agent="test", sector="SPRINGFIELD")
    sim.manager.receive_actions([action])
    for _ in range(60):
        sim.evolve(10)
    # by this point, AIR03 should have been removed from the environment
    print(Pos2D(aircraft.lat, aircraft.lon).distance(fix_pos))
    assert "AIR03" not in sim.manager.environment.aircraft

def test_to_simulator():
    """
    Test the SpringfieldScenarioManager.to_simulator() method creates the required components and sets the appropriate variables correctly.
    """
    
    simulator = SpringfieldScenarioManager(scenario_name="example-scenario",).to_simulator(
        log_filename="test",
        autosave=False,
    )

    assert isinstance(simulator.manager, EnvironmentManager)
    assert isinstance(simulator.scenario_manager, SpringfieldScenarioManager)
    assert isinstance(simulator.projection_centre, tuple)
    assert isinstance(simulator, Simulator)

    assert simulator.scenario_name == "example-scenario"
    assert simulator.category is None

    assert simulator.manager.event_logger.log_name == "test"
    assert simulator.autosave is False

    # Check returned simulator works as expected
    simulator.evolve(6)
    assert len(simulator.manager.environment.aircraft) is not None
