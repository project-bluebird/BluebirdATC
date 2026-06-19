import copy
from datetime import datetime, timezone
import logging
import io
import tarfile
import warnings

import pandas as pd
import pytest

from bluebird_dt.core import Action, Aircraft, Airspace, FlightPlan, Route
from bluebird_dt.events import EventHandler
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import RouteFollowPredictor, SimplePredictor
from bluebird_dt.scenario_manager import Tactical
from bluebird_dt.core import Coordination
from bluebird_dt.utility import constants

def test_bandboxing(generate_two_sector: tuple[Airspace, list[Route]]) -> None:
    """
    Sector band-boxing changes:
    - airspace
    - current sector of aircraft in environment
    """

    # Airspace and EventHandler, Predictor & EM
    airspace, _ = generate_two_sector

    event_handler = EventHandler()

    # add aircraft in the centre of sector_i1
    centre_i1 = airspace.sectors["sector_i1"].centre
    event_handler.add_aircraft(
        the_datetime=datetime(2020, 5, 10),
        aircraft=Aircraft(
            callsign="AIR0",
            lat=centre_i1.lat,
            lon=centre_i1.lon,
            fl=centre_i1.fl,
            heading=180,
            flight_plan=None,
            current_sector="sector_i1",
        ),
    )

    # add aircraft in the centre of sector_i2
    centre_i2 = airspace.sectors["sector_i2"].centre
    event_handler.add_aircraft(
        the_datetime=datetime(2020, 5, 10),
        aircraft=Aircraft(
            callsign="AIR1",
            lat=centre_i2.lat,
            lon=centre_i2.lon,
            fl=centre_i2.fl,
            heading=180,
            flight_plan=None,
            current_sector="sector_i2",
        ),
    )

    em = EnvironmentManager(
        airspace=airspace, event_handler=event_handler, predictor=SimplePredictor(dt=1, fix_proximity_threshold=2)
    )

    # initialise to put aircraft into environment
    em.initialise_env_with_event_handler()

    # check both aircraft have the correct current sectors
    # and are contained in the correct sectors
    aircraft = em.environment.aircraft
    assert aircraft["AIR0"].current_sector == "sector_i1"
    assert aircraft["AIR1"].current_sector == "sector_i2"
    assert airspace.sectors["sector_i1"].contains(aircraft["AIR0"].pos3d())
    assert airspace.sectors["sector_i2"].contains(aircraft["AIR1"].pos3d())

    # check we have 2 sectors
    assert len(em.environment.airspace.sectors) == 2

    # bandbox sectors
    em.bandbox_sectors({"combined": ["sector_i1", "sector_i2"]})

    # check sectors after bandboxing
    assert "combined" in em.environment.airspace.sectors
    assert len(em.environment.airspace.sectors) == 1

    # check aircraft current sectors after bandboxing
    assert aircraft["AIR0"].current_sector == "combined"
    assert aircraft["AIR1"].current_sector == "combined"


@pytest.mark.parametrize("airspace_routes", ["generate_i", "generate_x", "generate_y"])
def test_receive_and_process_actions(
    airspace_routes: str, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Warnings should be given if Action can't be issued
    (Aircraft doesn't exist in the Environment or is uncontrollable),
    All received Actions are logged, keeping track of whether the action was issued or not.
    Tests behaviour in multiple Airspaces.
    """

    # Airspace and EventHandler
    airspace, routes = request.getfixturevalue(airspace_routes)
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager()

    actions = [Action("AIR0", "change_heading_by", 5)]

    # setup for the Predictors
    predictor_dt = 1.0
    fix_proximity_threshold = 2.0  # NMI

    # example invalid Action to issue as AIR3 isn't in the environment
    # invalid_actions = [Action("AIR3", "change_heading_by", 5)]

    # AIR3 is not in the scenario at all and so a warning should be issued
    # with caplog.at_level(logging.WARN):
    #     em.receive_actions(invalid_actions)
    #     em.process_actions()
    #     assert "AIR3" in caplog.text

    assert len(em._actions_to_issue) == 0

    # example valid Action to issue
    actions = [Action("AIR0", "change_heading_by", 5)]

    # there is no warning, store as `action_to_apply`
    current_log_length = len(caplog.record_tuples)
    with caplog.at_level(logging.WARN):
        em.receive_actions(actions)
        assert len(caplog.record_tuples) == current_log_length

    assert len(em._actions_to_issue) == 1

    # the Action won't get issued until em.evolve() is called
    assert len(em._actions_to_issue) == 1

    # evolve - Action should now be recorded as issued (at previous Environment time)
    em.evolve(5)
    assert len(em.environment.aircraft["AIR0"].pilot.action_queue) == 0

    # # EM with RouteFollowPredictor --------------------------------------------------------
    predictor = RouteFollowPredictor(predictor_dt, fix_proximity_threshold, airspace.fixes)
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager(predictor)
    em.evolve(5)

    # Actions can be received but warning is given due to predictor RouteFollowPredictor
    with caplog.at_level(logging.WARN):
        em.receive_actions(actions)
        assert "The predictor RouteFollowPredictor does not respond to Actions..." in caplog.text


@pytest.mark.parametrize("airspace_routes", ["generate_i", "generate_x", "generate_y"])
def test_controllable_ignore_actions(
    airspace_routes: str, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test whether uncontrollable aircraft ignore actions.
    """

    # Airspace and EventHandler
    airspace, routes = request.getfixturevalue(airspace_routes)
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager()

    # example Action to issue
    actions = [Action("AIR0", "change_heading_by", 95)]

    em.evolve(5)

    # If aircraft is uncontrollable, actions are ignored by pilots
    em.environment.aircraft["AIR0"].controllable = False

    length_log = len(caplog.record_tuples)
    with caplog.at_level(logging.DEBUG):
        em.receive_actions(actions)
        em.process_actions()
        assert length_log + 2 == len(caplog.record_tuples)

    # Set flag to True and issue a new action
    em.environment.aircraft["AIR0"].controllable = True
    new_action = [Action("AIR0", "change_heading_to", 200)]
    em.receive_actions(new_action)
    em.evolve(5)

    assert em.environment.aircraft["AIR0"].selected_instructions.heading == 200


@pytest.mark.parametrize("predictor_dt", [2, 5, 10])
@pytest.mark.parametrize("predictor_type", ["SimplePredictor", "RouteFollowPredictor"])
def test_evolve_fail(predictor_dt: int, predictor_type: str, generate_x: tuple[Airspace, list[Route]]) -> None:
    """
    Test that EM fails if the step_time provided when calling evolve()
    is not compatible with the Predictor dt.
    """

    # Airspace, AddAircraft Events
    airspace, routes = generate_x

    # Predictor
    if predictor_type == "SimplePredictor":
        predictor = SimplePredictor(predictor_dt, 2.0, airspace.fixes)
    elif predictor_type == "RouteFollowPredictor":
        predictor = RouteFollowPredictor(predictor_dt, 2.0, airspace.fixes)
    else:
        raise ValueError(f"Invalid predictor_type given: {predictor_type}")

    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager(predictor)

    step_time = 0
    with pytest.raises(ValueError, match=r"not compatible with the Predictor\.dt"):
        em.evolve(step_time)

    step_time = 1
    with pytest.raises(ValueError, match=r"not compatible with the Predictor\.dt"):
        em.evolve(step_time)

    step_time = predictor_dt + 1
    with pytest.raises(ValueError, match=r"not compatible with the Predictor\.dt"):
        em.evolve(step_time)

    step_time = 2 * predictor_dt - 1
    with pytest.raises(ValueError, match=r"not compatible with the Predictor\.dt"):
        em.evolve(step_time)


def test_remove_aircraft_outside_airspace_penumbra(generate_i: tuple[Airspace, list[Route]]) -> None:
    """
    Simulated aircraft should be removed if:
    - they are outside airspace penumbra
    In any other circumstances, they stay in the environment
    """

    # Airspace and EventHandler, Predictor & EM
    airspace, routes = generate_i
    predictor = RouteFollowPredictor(1.0, 2.0, fixes=airspace.fixes)
    route = routes[0]

    start_pos = airspace.fixes.places[route.filed[0]]
    centre_pos = airspace.fixes.places[route.filed[2]]
    final_fix_pos = airspace.fixes.places[route.filed[4]]
    heading = start_pos.bearing_to(centre_pos)
    fl = 200

    event_handler = EventHandler()
    # 1. create aircraft within airspace
    # - won't get removed since within airspace
    event_handler.add_aircraft(
        the_datetime=datetime(2020, 2, 1, 11, 45, 23),
        aircraft=Aircraft(
            centre_pos.lat,
            centre_pos.lon,
            fl,
            heading,
            FlightPlan(route=route),
            callsign="AIR0",
        ),
    )

    em = EnvironmentManager(
        airspace,
        event_handler,
        predictor,
        penumbra_lat=30.0,
        penumbra_fl=10,  # large enough include all fixes
    )

    em.initialise_env_with_event_handler()

    em.remove_aircraft_outside_airspace_penumbra()
    # aircraft should still be in environment since it's within the airspace
    assert "AIR0" in em.environment.aircraft

    # 2. move aircraft just outside the airspace
    aircraft = em.environment.aircraft["AIR0"]
    aircraft.lat = final_fix_pos.lat
    aircraft.lon = final_fix_pos.lon

    # check aircraft outside sector
    assert not airspace.sectors["sector_i"].contains(aircraft.pos3d())

    # should not be removed as within penumbra
    em.remove_aircraft_outside_airspace_penumbra()
    assert "AIR0" in em.environment.aircraft

    # 3. move aircraft outside lateral penumbra
    aircraft = em.environment.aircraft["AIR0"]
    aircraft.lat = final_fix_pos.lat + 1.0
    aircraft.lon = final_fix_pos.lon

    # aircraft should be removed as it's outside the lateral penumbra
    assert "AIR0" in em.environment.aircraft
    em.remove_aircraft_outside_airspace_penumbra()
    assert "AIR0" not in em.environment.aircraft


def test_evolve(generate_i: tuple[Airspace, list[Route]]) -> None:
    """
    Check that evolve adds new aircraft to the environment at the correct time
    and alters the position of aircraft already in the environment
    """

    # Airspace, AddAircraft Events and Predictor
    airspace, routes = generate_i

    # SIMULATED AIRCRAFT x3
    em = Tactical(3, airspace=airspace, routes=routes, initialise_with_event_handler=False).create_env_manager()

    # change start_times - set by callsign to avoid dependence on row order
    radar_df = em.event_handler.radar_df
    ac_internal_df = em.event_handler.aircraft_internals_df
    new_start_times = {
        "AIR0": datetime(1970, 1, 1, 0, 0, 0),
        "AIR1": datetime(1970, 1, 1, 0, 0, 6),
        "AIR2": datetime(1970, 1, 1, 0, 0, 12),
    }
    radar_idx = radar_df.index.to_numpy()
    for i, callsign in enumerate(radar_df["callsign"]):
        radar_idx[i] = new_start_times[callsign]
    ac_internal_idx = ac_internal_df.index.to_numpy()
    for i, callsign in enumerate(ac_internal_df["callsign"]):
        ac_internal_idx[i] = new_start_times[callsign]

    em.initialise_env_with_event_handler()

    # AIR0 should be the only aircraft at the start of the scenario
    assert len(em.environment.aircraft) == 1
    assert next(iter(em.environment.aircraft)) == "AIR0"

    # evolving by 2 seconds should move AIR0 but no other aircraft should appear
    ac_0 = em.environment.aircraft["AIR0"]
    lat_0 = ac_0.lat
    lon_0 = ac_0.lon
    em.evolve(2.0)
    assert len(em.environment.aircraft) == 1
    assert next(iter(em.environment.aircraft)) == "AIR0"
    assert ac_0.lat != lat_0 or ac_0.lon != lon_0

    # at 6 seconds env time, AIR1 is in the sim (evolving 2 + 4 = 6)
    em.evolve(4.0)
    assert len(em.environment.aircraft) == 2

    # at 9 seconds env time, there are no new aircraft but AIR0 and AIR1 have moved
    lat_0 = ac_0.lat
    lon_0 = ac_0.lon
    ac_1 = em.environment.aircraft["AIR1"]
    lat_1 = ac_1.lat
    lon_1 = ac_1.lon
    em.evolve(3.0)
    assert len(em.environment.aircraft) == 2
    assert ac_0.lat != lat_0 or ac_0.lon != lon_0
    assert ac_1.lat != lat_1 or ac_1.lon != lon_1

    # at 12 seconds env time, AIR3 is also in the sim
    em.evolve(3.0)
    assert len(em.environment.aircraft) == 3


def test_rewind_to_time(generate_i: tuple[Airspace, list[Route]]) -> None:
    """
    Check that rewind takes environment back to specific time and that the rollout
    is identical the second time if replaying only
    """

    # Airspace, AddAircraft Events and Predictor
    airspace, routes = generate_i

    # SIMULATED AIRCRAFT x5
    em = Tactical(5, airspace=airspace, routes=routes).create_env_manager()

    all_actions = {
        0: ("AIR0", "change_heading_to", 175),
        6: ("AIR1", "change_flight_level_to", 210),
        12: ("AIR2", "change_cas_to", 290),
        18: ("AIR3", "outcomm", "background"),
        24: ("AIR4", "change_mach_to", 0.81),
    }

    end_time = 60
    time_step = 6

    # evolve for 60 seconds with preset actions
    while em.environment.time < end_time:
        if em.environment.time in all_actions:
            callsign, kind, value = all_actions[int(em.environment.time)]
            action = Action(
                callsign=callsign, kind=kind, value=value, sector=em.environment.aircraft[callsign].current_sector
            )
            em.receive_actions([action])
        em.evolve(time_step)

    # To test as replay, set all aircraft logs such that aircraft.simulated = False
    for ac_attribute_dict in em.event_logger.aircraft_internals_log:
        ac_attribute_dict["simulated"] = False

    # also set for the "previous_ac_internals" which is used to identify updated attributes
    for ac_attribute_dict in em.event_logger._previous_ac_internals:
        ac_attribute_dict["simulated"] = False

    # make a copy of the event log for comparison later
    orig_event_logger = copy.deepcopy(em.event_logger)

    # rewind to a specific time
    rewind_time = 12
    em.rewind_to_time(pd.to_datetime(rewind_time, unit="s"))

    # run to the end again
    while em.environment.time < end_time:
        em.evolve(time_step)

    # assert logs are identical
    assert em.event_logger == orig_event_logger

def _make_basic_em(airspace, *, predictor_dt=1.0, fix_thresh=2.0, penumbra_lat=5.0, penumbra_fl=10):
    """
    Helper to create a minimal EnvironmentManager with an empty EventHandler.
    """
    event_handler = EventHandler()
    predictor = SimplePredictor(dt=predictor_dt, fix_proximity_threshold=fix_thresh, fixes=airspace.fixes)
    return EnvironmentManager(
        airspace=airspace,
        event_handler=event_handler,
        predictor=predictor,
        penumbra_lat=penumbra_lat,
        penumbra_fl=penumbra_fl,
    )

def test_request_coord(generate_i):
    """
    Check that:
    - Reject/Offer/CounterOffer add to coord_requests
    - Accept adds to environment.coordinations and removes any matching request
    - RemoveRequest removes request
    """
    airspace, routes = generate_i
    em = Tactical(1, airspace=airspace, routes=routes).create_env_manager()
    coord = Coordination(callsign="AIR0", from_sector="sector_i", to_sector="sector_i", fl=300, fix="Fix", direction="Up")

    # Offer should add to requests
    em.request_coord("Offer", coord)
    assert len(em.coord_requests) == 1
    assert em.coord_requests[0].status == "Offer"

    # Accept should add to environment coordinations and remove request
    em.request_coord("Accept", coord)
    assert len(em.coord_requests) == 0
    assert any(c.callsign == "AIR0" for c in em.environment.coordinations.values())

    # Reject
    em.request_coord("Reject", coord)
    assert len(em.coord_requests) == 1
    
    # Remove
    em.request_coord("RemoveRequest", coord)
    assert len(em.coord_requests) == 0


def test_coord_requests(generate_i):
    """
    Check that coord_requests property returns list view of internal dict values.
    """
    airspace, _ = generate_i
    em = _make_basic_em(airspace)
    first_callsign = "AIR0"
    second_callsign = "AIR1"

    first_coord = Coordination(callsign=first_callsign, from_sector="S1", to_sector="S2", fl=300, fix="Dummy Fix", direction="Up")
    second_coord = Coordination(callsign=second_callsign, from_sector="S1", to_sector="S3", fl=300, fix="Dummy Fix", direction="Up")

    em.request_coord("Offer", first_coord)
    em.request_coord("CounterOffer", second_coord)

    assert {request.coord.callsign for request in em.coord_requests} == { first_callsign, second_callsign }


def test_get_coord_requests(generate_i):
    """
    Check that filtering by callsign/to_sector/from_sector works independently and in combination.
    """
    airspace, _ = generate_i
    em = _make_basic_em(airspace)
    first_callsign = "AIR0"
    second_callsign = "AIR1"

    first_coord = Coordination(callsign=first_callsign, from_sector="S1", to_sector="S2", fl=300, fix="Dummy Fix", direction="Up")
    second_coord = Coordination(callsign=second_callsign, from_sector="S1", to_sector="S2", fl=300, fix="Dummy Fix", direction="Up")
    third_coord = Coordination(callsign=first_callsign, from_sector="S9", to_sector="S2", fl=300, fix="Dummy Fix", direction="Up")

    em.request_coord("Offer", first_coord)
    em.request_coord("Offer", second_coord)
    em.request_coord("Offer", third_coord)

    assert len(em.get_coord_requests()) == 3
    assert {r.coord.callsign for r in em.get_coord_requests(callsign=first_callsign)} == {first_callsign}
    assert len(em.get_coord_requests(to_sector="S2")) == 3
    assert len(em.get_coord_requests(from_sector="S1")) == 2
    assert len(em.get_coord_requests(callsign=first_callsign, from_sector="S1")) == 1


def test_square_penumbra_limits(generate_i):
    """
    Check that square_penumbra_limits expands min/max lat/lon by penumbra_lat/60 degrees.
    """
    airspace, _ = generate_i
    penumbra_lat_under_test = 30.0
    em = _make_basic_em(airspace, penumbra_lat=penumbra_lat_under_test)
    min_lat, max_lat, min_lon, max_lon = em.square_penumbra_limits()

    # Compare against raw boundary + fixes with same formula (penumbra includes named fixes)
    all_pts = list(em.environment.airspace.boundary().boundary_vertices) + list(em.environment.airspace.fixes.places.values())
    raw_min_lat = min(p.lat for p in all_pts)
    raw_max_lat = max(p.lat for p in all_pts)
    raw_min_lon = min(p.lon for p in all_pts)
    raw_max_lon = max(p.lon for p in all_pts)

    assert min_lat == pytest.approx(raw_min_lat - penumbra_lat_under_test / constants.NM_PER_DEGREE)
    assert max_lat == pytest.approx(raw_max_lat + penumbra_lat_under_test / constants.NM_PER_DEGREE)
    assert min_lon == pytest.approx(raw_min_lon - penumbra_lat_under_test / constants.NM_PER_DEGREE)
    assert max_lon == pytest.approx(raw_max_lon + penumbra_lat_under_test / constants.NM_PER_DEGREE)


def test_get_sector_airspace(generate_two_sector):
    """
    - sector_names=None returns "all individual sectors"
    - sector_names="name" returns a 1-sector airspace
    - local_fixes=True returns a Fixes object limited to within penumbra square
    """
    airspace, _ = generate_two_sector
    em = _make_basic_em(airspace, penumbra_lat=5.0)

    # Check all sectors in airspace returned
    asp_all = em.get_sector_airspace(None, local_fixes=False)
    assert set(asp_all.sectors.keys()) == set(em.environment.airspace._individual_sectors.keys())

    # Single sector by name
    one = em.get_sector_airspace("sector_i1", local_fixes=False)
    assert list(one.sectors.keys()) == ["sector_i1"]

    # Local fixes subset (should be <= all fixes)
    one_local = em.get_sector_airspace("sector_i1", local_fixes=True)
    assert len(one_local.fixes.places) <= len(em.environment.airspace.fixes.places)


def test_finished(generate_i):
    """
    Test that finished() returns correct answers.  Result should be True iff:
    - env time is after last radar_df timestamp AND
    - no aircraft remain
    """
    airspace, routes = generate_i
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager()

    # Check that we're not already finished
    assert em.finished() is False

    # Remove aircraft and jump time beyond last event
    em.environment.aircraft = {}
    last_event_ts = em.event_handler.radar_df.index.max().replace(tzinfo=timezone.utc).timestamp()
    em.environment.time = last_event_ts + 10
    assert em.finished() is True

def test_assign_aircraft_to_bandboxed_sector(generate_two_sector):
    """
    Check that assign_aircraft_to_bandboxed_sector changes the current_sector for aircraft
    currently in the individual sectors.
    """
    airspace, _ = generate_two_sector
    event_handler = EventHandler()

    # Put one aircraft per individual sector
    first_centre = airspace.sectors["sector_i1"].centre
    second_centre = airspace.sectors["sector_i2"].centre

    event_handler.add_aircraft(
        the_datetime=datetime(2020, 5, 10),
        aircraft=Aircraft(
            callsign="AIR0",
            lat=first_centre.lat,
            lon=first_centre.lon,
            fl=first_centre.fl,
            heading=180,
            flight_plan=None,
            current_sector="sector_i1",
        ),
    )
    event_handler.add_aircraft(
        the_datetime=datetime(2020, 5, 10),
        aircraft=Aircraft(
            callsign="AIR1",
            lat=second_centre.lat,
            lon=second_centre.lon,
            fl=second_centre.fl,
            heading=180,
            flight_plan=None,
            current_sector="sector_i2",
        ),
    )


    em = EnvironmentManager(
        airspace=airspace,
        event_handler=event_handler,
        predictor=SimplePredictor(dt=1, fix_proximity_threshold=2, fixes=airspace.fixes),
    )
    em.initialise_env_with_event_handler()

    em.assign_aircraft_to_bandboxed_sector({"combined": ["sector_i1", "sector_i2"]})

    assert em.environment.aircraft["AIR0"].current_sector == "combined"
    assert em.environment.aircraft["AIR1"].current_sector == "combined"


def test_assign_aircraft_to_individual_sector(generate_two_sector):
    """
    Check that assign_aircraft_to_individual_sector reassigns aircraft from a bandboxed sector
    back into constituent individual sectors.
    """
    airspace, routes = generate_two_sector
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager()

    em.initialise_env_with_event_handler()
    em.bandbox_sectors({"combined": ["sector_i1", "sector_i2"]})

    # Force both aircraft "in" combined so they are eligible for split assignment
    for ac in em.environment.aircraft.values():
        ac._current_sector = "combined"

    em.assign_aircraft_to_individual_sector("combined")

    # They should now be assigned to one of the constituent sectors
    for ac in em.environment.aircraft.values():
        assert ac.current_sector in {"sector_i1", "sector_i2"}


def test_split_sector(generate_two_sector):
    """
    Test that split_sector does the following:
    - reassign aircraft out of the bandboxed sector
    - update airspace sectors to include the original individuals
    """
    airspace, routes = generate_two_sector
    em = Tactical(2, airspace=airspace, routes=routes).create_env_manager()

    em.initialise_env_with_event_handler()
    em.bandbox_sectors({"combined": ["sector_i1", "sector_i2"]})
    assert set(em.environment.airspace.sectors.keys()) == {"combined"}

    # Force both aircraft to be marked as in the combined sector
    for ac in em.environment.aircraft.values():
        ac._current_sector = "combined"

    em.split_sector("combined")

    # Airspace should now contain individual sectors again
    assert "sector_i1" in em.environment.airspace.sectors
    assert "sector_i2" in em.environment.airspace.sectors

def test_get_assigned_bay(generate_two_sector):
    """
    Test that get_assigned_bay returns:
    - INCOMM if sector_name == current_sector
    - OUTCOMM if sector_name == previous_sector
    - PENDING if sector_name == next sector
    - None otherwise
    """
    airspace, routes = generate_two_sector
    em = Tactical(1, airspace=airspace, routes=routes).create_env_manager()
    em.initialise_env_with_event_handler()

    callsign = next(iter(em.environment.aircraft))
    ac = em.environment.aircraft[callsign]

    # Ensure previous/current are set in a predictable way
    first_sector = "sector_i1"
    second_sector = "sector_i2"
    ac._current_sector = second_sector

    assert em.get_assigned_bay(second_sector, callsign) == "INCOMM"
    ac._previous_sector = first_sector
    assert em.get_assigned_bay(first_sector, callsign) == "OUTCOMM"
    assert em.get_assigned_bay("some_other_sector", callsign) is None

    # Give the ac an exit coordination so that we can test appearance of a strip in the next sector's pending bay
    exit_coordination = Coordination(
        callsign=callsign,
        from_sector=second_sector,
        to_sector="sector_i3",
        fl=320,
        fix="TABBY",
        direction="Up"
    )
    em.environment.coordinations.add(exit_coordination)
    next_sec = em.environment.next_sector_of_aircraft(callsign)
    assert em.get_assigned_bay(next_sec, callsign) == "PENDING"


def test_get_bays_name(generate_i):
    """
    Test that get_bays_name returns the correct bay names
    """
    airspace, _ = generate_i
    em = _make_basic_em(airspace)
    assert em.get_bays_names() == ["PENDING", "INCOMM", "OUTCOMM"]


def test_replace_all_aircraft(generate_i, monkeypatch):
    """
    Test that replace_all_aircraft swaps env aircraft dict and calls event_handler.reset_events().
    """
    airspace, routes = generate_i
    em = Tactical(1, airspace=airspace, routes=routes).create_env_manager()
    em.initialise_env_with_event_handler()

    called = {"reset": 0}

    def _dummy_reset_events():
        called["reset"] += 1

    monkeypatch.setattr(em.event_handler, "reset_events", _dummy_reset_events)

    empty_aircraft = {}
    em.replace_all_aircraft(empty_aircraft)

    assert em.environment.aircraft is empty_aircraft
    assert called["reset"] == 1

def test_replace_environment(generate_i):
    """
    Test that replace_environment should:
    - set environment
    - reset events
    - update penumbra params
    - update predictor.fixes
    - set reload_environment True
    """
    airspace, routes = generate_i
    em = Tactical(1, airspace=airspace, routes=routes).create_env_manager()
    em.initialise_env_with_event_handler()

    # Create a fresh environment with same airspace but no aircraft
    new_env = em.typeof_environment()(
        time=123.0,
        airspace=airspace,
        aircraft={},
        coordinations=[],
    )

    em.replace_environment(new_env, airspace_settings={"penumbra_fl": 77, "penumbra_lat": 11.0})

    assert em.environment is new_env
    assert em.penumbra_fl == 77
    assert em.penumbra_lat == 11.0
    assert em.predictor.fixes is new_env.airspace.fixes
    assert em.reload_environment is True

def test_save_logs(generate_i):
    """
    Test that write_logs_to_buffer returns a BytesIO containing a tar archive (by default).
    """
    airspace, routes = generate_i
    em = Tactical(1, airspace=airspace, routes=routes).create_env_manager()
    em.initialise_env_with_event_handler()
    em.evolve(2.0)

    buf = em.write_logs_to_buffer(sim_config=em.config(), save_csv=True)
    assert isinstance(buf, io.BytesIO)

    buf.seek(0)
    # Validate it is a tar archive and contains at least something
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        members = tf.getmembers()
        assert len(members) > 0


def test_initialise_with_event_handler(generate_i):
    """
    Test that initialise_env_with_event_handler does the following:
    - jump env time to first event time by default
    - populate aircraft that appear at that time
    - log environment (radar log) at least once when log=True
    """
    airspace, routes = generate_i
    em = Tactical(3, airspace=airspace, routes=routes, initialise_with_event_handler=False).create_env_manager()

    first_event_time = em.event_handler.radar_df.index.min()
    em.initialise_env_with_event_handler(jump_to_first_event=True, log=True)

    assert em.environment.datetime == first_event_time
    # At least 1 aircraft should exist at the first timestamp
    assert len(em.environment.aircraft) >= 1
    # Logging should have occurred
    assert em.event_logger.radar_log is not None
