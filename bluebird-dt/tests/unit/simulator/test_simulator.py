import copy
import gc
from datetime import date, datetime, timedelta

from pydantic import ValidationError
import pandas as pd
import pytest
import weakref

from bluebird_dt.core import Action, Coordination
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager, SpringfieldScenarioManagerConfig
from bluebird_dt.simulator import Simulator
from bluebird_dt.simulator.simconfig import SaveConfig
from bluebird_dt.utility import convert
from bluebird_dt.simulator import Simulator
from typing import Any


def test_springfield_simulation():
    """
    Test the Springfield simulation loads as expected
    """

    sf_sim = Simulator.from_category(category="Springfield", scenario_name="testScenario")

    # evolve 1 timestep
    sf_sim.evolve(6)

    # Test that demo loads
    airspace = sf_sim.manager.environment.airspace
    sector_list = list(airspace.sectors)

    assert sorted(sector_list) == sorted(
        [
            "SPRINGFIELD"
        ]
    )

@pytest.mark.parametrize(
    ("category", "scenario_name"),
    [
        ("Artificial", "I-Sector Two Aircraft"),
        ("Springfield", "testScenario"),
    ],
)
def test_log_filename(category: str, scenario_name: str):
    """
    Test the log_filename and save_log_to_file parameters work as expected for each Simulator category
    """
    datetime_format = "%Y_%m_%d__%H_%M_%S"

    # check that if save_log_to_file is True but the log_filename is None then
    # the first part is {scenario_category}_{scenario_name} and the datetime matches today's date
    sim = Simulator.from_category(category=category, scenario_name=scenario_name, log_filename=None, save_log_to_file=True)
    sanitised_scenario_name = scenario_name.replace(":", "_")
    assert sim.manager.event_logger.log_name[:-21] == f"{category}_{sanitised_scenario_name}"
    assert datetime.strptime(sim.manager.event_logger.log_name[-20:], datetime_format).date() == date.today()

    test_log_filename = "test_filename"
    # if a filename is set then check that this is used
    sim = Simulator.from_category(category=category, scenario_name=scenario_name, log_filename=test_log_filename)
    assert sim.manager.event_logger.log_name == test_log_filename


# @pytest.mark.parametrize(
#     ("category", "scenario_name"),
#     [
#         ("Springfield", "testScenario"),
#     ],
# )
# def test_route_following_initialised(category: str, scenario_name: str):
#     """
#     Test that route_following has been initialised, by checking that not every aircraft has a next fix of 0
#     """
#     # check that if save_log_to_file is True but the log_filename is None then
#     # the first part is {scenario_category}_{scenario_name} and the datetime matches today's date
#     sim = Simulator(category, scenario_name, save_log_to_file=False)

#     # next_fix_index defaults to 1 when aircraft created and can be set to zero by route_direct clearances at initiation
#     # In the demo scenario, the maximum next fix index should be greater than 1 for a number of aircraft if
#     # route progression has been initialised
#     max_next_fix_index = max(
#         [a.next_fix_index for a in sim.manager.environment.aircraft.values() if a.next_fix_index is not None]
#     )
#     assert max_next_fix_index > 1


@pytest.mark.parametrize(
    ("category", "scenario_name", "scenario_config_type"),
    [
        ("Springfield", "testScenario", SpringfieldScenarioManagerConfig),
    ],
)
def test_sim_config(category: str, scenario_name:str, scenario_config_type: type[SpringfieldScenarioManager]):
    sim = Simulator.from_category(category=category, scenario_name=scenario_name, autosave=False)
    assert isinstance(sim.config(), SaveConfig)
    assert isinstance(sim.config().scenario, scenario_config_type)

@pytest.mark.parametrize(
    "the_datetime, expected_number_of_new_coordinations",
    [
        ( datetime(2920, 7, 7, 1, 12, 0, 0), 1 ),
        ( str(datetime(1947, 7, 7, 1, 12, 0, 0)) ,1 ),
        ( "2029-07-01 12:30:44", 1 ),
        ( None, 1 ),
        ( "Nonsense", 0 )
    ],
)
def test_coord_with_various_datetimes(the_datetime: Any, expected_number_of_new_coordinations: int):
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    env = sim.manager.environment
    callsign = list(env.aircraft.keys())[0]

    try:
        input_coord = Coordination(
            callsign="AIR01",
            from_sector="DUMMY",
            to_sector="FICTIONAL",
            fl=320,
            fix="DUMMY",
            direction="Horizontal",
            level_by=False,
            secondary_coord_conditions="stuff",
            the_datetime=the_datetime,
        )
        sim.request_coordination("Accept", input_coord)
    except ValidationError:
        assert expected_number_of_new_coordinations == 0
        return

    # Calling evolve and save will test serialisation (will throw if there are failures in the file writing)
    sim.evolve(2)
    sim.save()

    coords = env.coordinations.get(callsign,"DUMMY", "FICTIONAL")
    assert len(coords) == expected_number_of_new_coordinations
    for coord in coords:
        assert str(coord.datetime) == str(input_coord.datetime)


def test_from_category_with_unknown_category():
    """
    Verify that an erroneous category gets rejected by from_category.
    """
    with pytest.raises(ValueError):
        Simulator.from_category(category="Unknown", scenario_name="nope")


def test_evolve_with_invalid_delta():
    """
    Create a sim and check that calling evolve with an invalid (zero) delta is rejected.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)
    with pytest.raises(ValueError):
        sim.evolve(0)


def test_action_invalid_payload_returns_false():
    """
    Create a sim and check that adding an invalid action to the queue is rejected.

    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)
    assert sim.action([{"callsign": "AIR01"}]) is False


def test_save_autosave_skips_if_interval_not_elapsed():
    """
    Create a sim and check that save() fails when called before interval has elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)
    sim.last_save_time = datetime.now()
    sim.save_interval = timedelta(minutes=10)
    assert sim.save(autosave=True) is False

def test_simulator_gets_garbage_collected():
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)
    sim_ref = weakref.ref(sim)

    # Remove the last strong reference
    del sim

    # Force garbage collection
    gc.collect()

    assert sim_ref() is None

