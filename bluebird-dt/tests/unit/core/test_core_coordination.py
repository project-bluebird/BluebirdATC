import copy
from datetime import datetime
import io
import json
import pandas as pd
from pydantic import ValidationError
import pytest
import random
import tarfile
from typing import Any

from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.core.coordination import Coordination, CoordinationsManager
from bluebird_dt.utility.logging_utils import (save_df_to_parquet_tar)

from conftest import make_random_coordination

@pytest.mark.parametrize(
    "callsign, from_sector, to_sector, fl, fix, direction, level_by, level_by_details, secondary_coord_conditions, the_datetime",
    [
        (
            "AIR0",
            None,
            "sector_x",
            -1,
            "OCK",
            "Up",
            True,
            {"OCK": 290.0},
            "DCT SFT",
            str(datetime(2910, 7, 7, 1, 12, 0, 0)),
        ),
        (
            "AIR0",
            None,
            "sector_x",
            270,
            "OCK",
            "sideways",
            True,
            {"OCK": 290.0},
            "DCT SFT",
            str(datetime(2910, 7, 7, 1, 12, 0, 0)),
        ),
        (
            "AIR0",
            None,
            "sector_x",
            270,
            "OCK",
            "Up",
            False,
            {"OCK": 290.0},
            "DCT SFT",
            str(datetime(2910, 7, 7, 1, 12, 0, 0)),
        ),
        ("AIR0", None, "sector_x", 270, "OCK", "Up", True, None, "DCT SFT", str(datetime(2910, 7, 7, 1, 12, 0, 0))),
    ],
)
def test_init_exceptions(
    callsign,
    from_sector,
    to_sector,
    fl,
    fix,
    direction,
    level_by,
    level_by_details,
    secondary_coord_conditions,
    the_datetime,
):
    """
    Test exceptions are raised when invalid arguments are used
    """

    with pytest.raises(ValueError):
        Coordination(
            callsign=callsign,
            from_sector=from_sector,
            to_sector=to_sector,
            fl=fl,
            fix=fix,
            direction=direction,
            level_by=level_by,
            level_by_details=level_by_details,
            secondary_coord_conditions=secondary_coord_conditions,
            the_datetime=the_datetime,
        )

def test_coordination_from_json(example_coordination_json: str):
    """Test Coordination from_json works as expected."""

    coordination = Coordination.from_json(example_coordination_json)

    assert isinstance(coordination, Coordination)
    assert coordination.fl == pytest.approx(200)

    null_coordination = Coordination.from_json("null")

    assert null_coordination is None

def test_coordinations_same_excluding_datetime(example_coordination_json: str):
    """
    Test that coord comparison method works (datetime can be different)
    """
    coordination = Coordination.from_json(example_coordination_json)
    clone = copy.deepcopy(coordination)
    clone.datetime = datetime.now()
    assert coordination.same_excluding_datetime(clone) == True

    clone.callsign = "BLAH"
    assert coordination.same_excluding_datetime(clone) == False

def test_coordinations_manager_init(example_coordination_json: str):
    """
    Test coord manager init
    """
    coordination = Coordination.from_json(example_coordination_json)
    manager = CoordinationsManager([coordination])
    assert coordination.same_excluding_datetime(manager.values()[0])

def test_coordinations_manager_from_json(example_coordination_dict_json: str):
    """
    Test coord manager can be initialised from json
    """
    manager = CoordinationsManager.from_json(example_coordination_dict_json)
    assert manager.values()[0].callsign == "AIR0"

def test_coord_manager_add_get_data_json_remove(example_coordination_json: str):
    """
    Test coord manager add() get data() to_json() and remove() methods.
    """
    coordination = Coordination.from_json(example_coordination_json)
    callsign = "AIR0"
    manager = CoordinationsManager()
    manager.add(coordination)
    assert (callsign, None) in manager.coords

    latest_coord = manager.get(callsign, None)
    assert latest_coord[0] == coordination

    data = manager.data()
    assert data[0]["callsign"] == callsign

    mock_json_data = json.dumps(data, indent=4)
    assert mock_json_data == manager.to_json()

    manager.remove(callsign, None)
    assert (callsign, None) not in manager.coords

def test_coord_manager_contains_excluding_times(example_coordination_json: str):
    """
    Test that the environment gets logged when a simulation is saved by checking the relevant generated files.
    """
    coordination = Coordination.from_json(example_coordination_json)
    clone = copy.deepcopy(coordination)
    clone.datetime = datetime.now()

    manager = CoordinationsManager()
    manager.add(coordination)
    assert manager.contains_excluding_times(clone)

    clone.callsign = "BLAH"
    assert manager.contains_excluding_times(clone) == False

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
def test_writing_coord_to_file(the_datetime: Any, expected_number_of_new_coordinations: int):
    """
    Test that we can write Coordination objects to file with various coord datetime input formats
    """
    try:
        crd = make_random_coordination(the_datetime=the_datetime)
    except ValidationError:
        assert expected_number_of_new_coordinations == 0
        return

    # Now build a list of coordinations in the same way the event logger does
    coords = []
    # We need a timestamped coord to be added first to make pandas coerce datetimes to Timestamps
    # Then any unconverted coordination times will fail
    crd_with_timestamp = make_random_coordination(the_datetime=datetime.now())
    for crd in [ crd_with_timestamp, crd ]:
        coord_data = crd.data()
        coord_data["datetime"] = crd.datetime
        coords.append(coord_data)

    # Build a dataframe, try to write it to file and clean up afterwards
    coord_df = pd.DataFrame.from_records(coords)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        save_df_to_parquet_tar(coord_df, tar, "coordination")


def test__aircraft_with_coordinations(generate_i):
    """
    Test the function that creates aircraft and entry and exit coordinations.
    """
    airspace, routes = generate_i
    sector_name = "sector_i"
    for i, route in enumerate(routes):
        callsign = f"AIR-0{i}"
        # spawn aircraft on first fix on the route
        first_fix_pos = airspace.fixes.places[route.filed[0]]
        next_fix_pos = airspace.fixes.places[route.filed[1]]
        heading = first_fix_pos.bearing_to(next_fix_pos)
        fl = random.randint(200,350)
        pos = first_fix_pos.pos3d(fl)
        speed = random.randint(300,500)
        entry_fl = random.randint(200,350)
        exit_fl = random.randint(200,350)
        aircraft, coord_entry, coord_exit = CoordinationsManager.aircraft_with_coordinations(
            callsign=callsign,
            pos=pos,
            heading=heading,
            speed=speed,
            route=route,
            sector_name=sector_name,
            entry_fl=entry_fl,
            exit_fl=exit_fl,
            airspace=airspace
        )
        assert isinstance(aircraft, Aircraft)
        assert aircraft.callsign == callsign
        assert aircraft.lat == pos.lat
        assert aircraft.lon == pos.lon
        assert aircraft.heading == heading
        assert aircraft.speed_tas == speed
        assert aircraft.selected_instructions.cas == speed
        assert aircraft.flight_plan.route == route
        assert isinstance(coord_entry, Coordination)
        assert coord_entry.fl == entry_fl
        assert coord_entry.from_sector == "background"
        assert coord_entry.to_sector == sector_name
        assert isinstance(coord_exit, Coordination)
        assert coord_exit.fl == exit_fl
        assert coord_exit.from_sector == sector_name
        assert coord_exit.to_sector == "background"


def test_find_entry_exit_fixes(generate_i):
    """
    Test the function that returns fixes nearest the entry and exit of 
    a route through a sector.
    """
    airspace, routes = generate_i
    # first route goes from "FIRE" to "SPIRIT", should enter sector at "EARTH"
    # and exit at "AIR"
    entry_fix, exit_fix = CoordinationsManager.find_entry_exit_fixes(airspace, routes[0], "sector_i")
    assert entry_fix == "EARTH"
    assert exit_fix == "AIR"
    # second route should be the reverse
    entry_fix, exit_fix = CoordinationsManager.find_entry_exit_fixes(airspace, routes[1], "sector_i")
    assert entry_fix == "AIR"
    assert exit_fix == "EARTH"