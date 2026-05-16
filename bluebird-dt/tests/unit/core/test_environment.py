from collections import defaultdict

import pytest
from .conftest import make_random_aircraft

from bluebird_dt.core import Environment
from bluebird_dt.core.coordination import Coordination


def test_init_exceptions(generate_simple_environment):
    """
    Environment must be initiated with non-negative time.
    """
    env = generate_simple_environment
    with pytest.raises(ValueError):
        Environment(-1, env.airspace, env.aircraft)


def test_controllable_aircraft(generate_simple_environment):
    """
    Test all/only controllable Aircraft in the Environment are returned.
    """

    env = generate_simple_environment
    env.aircraft = {}  # make it an empty environment
    # there are no Aircraft --> nothing to control
    assert len(env.controllable_aircraft()) == 0

    # instantiate 10 Aircraft --> all are controllable
    aircraft = {f"AIR{i}": make_random_aircraft() for i in range(10)}
    for a in aircraft.values():
        a.controllable = True

    # add to environment and check all are controllable
    env.aircraft = aircraft
    assert env.controllable_aircraft() == [f"AIR{i}" for i in range(10)]

    # make 3 Aircraft uncontrollable
    uncontrollable_callsigns = [f"AIR{i}" for i in [3, 7, 9]]
    for callsign in uncontrollable_callsigns:
        env.aircraft[callsign].controllable = False
    controllable_callsigns = env.controllable_aircraft()
    assert len(controllable_callsigns) == 7
    for callsign in uncontrollable_callsigns:
        assert callsign not in controllable_callsigns


def test_not_controllable_aircraft(generate_simple_environment):
    """
    Test all/only non controllable Aircraft in the Environment are returned.
    """

    env = generate_simple_environment
    env.aircraft = {}  # make it an empty environment
    # there are no Aircraft --> nothing to control
    assert len(env.not_controllable_aircraft()) == 0

    # instantiate 10 not controllable Aircraft
    aircraft = {f"AIR{i}": make_random_aircraft() for i in range(10)}
    for a in aircraft.values():
        a.controllable = False

    # add to environment and check all are not controllable
    env.aircraft = aircraft
    assert env.not_controllable_aircraft() == [f"AIR{i}" for i in range(10)]

    # make 3 Aircraft controllable
    controllable_callsigns = [f"AIR{i}" for i in [3, 7, 9]]
    for callsign in controllable_callsigns:
        env.aircraft[callsign].controllable = True
    not_controllable_callsigns = env.not_controllable_aircraft()
    assert len(not_controllable_callsigns) == 7
    for callsign in controllable_callsigns:
        assert callsign not in not_controllable_callsigns


def test_simulated_aircraft(generate_simple_environment):
    """
    Test all/only simulated Aircraft in the Environment are returned.
    """

    env = generate_simple_environment
    env.aircraft = {}
    # there are no Aircraft
    assert len(env.simulated_aircraft()) == 0

    # instantiate 10 simulated Aircraft
    aircraft = {f"AIR{i}": make_random_aircraft() for i in range(10)}
    for a in aircraft.values():
        a.simulated = True

    # add to environment and check all are simulated
    env.aircraft = aircraft
    assert env.simulated_aircraft() == [f"AIR{i}" for i in range(10)]

    # make 3 Aircraft non simulated
    callsigns_to_change = [f"AIR{i}" for i in [3, 7, 9]]
    for callsign in callsigns_to_change:
        env.aircraft[callsign].simulated = False
    simulated_callsigns = env.simulated_aircraft()
    assert len(simulated_callsigns) == 7
    for callsign in callsigns_to_change:
        assert callsign not in simulated_callsigns

def test_replayed_aircraft(generate_simple_environment):
    """
    Test all/only replayed Aircraft in the Environment are returned.
    """

    env = generate_simple_environment
    env.aircraft = {}
    # there are no Aircraft
    assert len(env.replayed_aircraft()) == 0

    # instantiate 10 replayed Aircraft
    aircraft = {f"AIR{i}": make_random_aircraft() for i in range(10)}
    for a in aircraft.values():
        a.simulated = False

    # add to environment and check all are replayed
    env.aircraft = aircraft
    assert env.replayed_aircraft() == [f"AIR{i}" for i in range(10)]

    # make 3 Aircraft simulated
    callsigns_to_change = [f"AIR{i}" for i in [3, 7, 9]]
    for callsign in callsigns_to_change:
        env.aircraft[callsign].simulated = True
    replayed_callsigns = env.replayed_aircraft()
    assert len(replayed_callsigns) == 7
    for callsign in callsigns_to_change:
        assert callsign not in replayed_callsigns

def test_aircraft_in_sectors(generate_simple_environment):
    """
    Test querying the callsigns in a sector or sectors
    """

    env = generate_simple_environment

    # Test null case
    sector_names = [ None ]
    callsigns_in_sector = env.aircraft_in_sectors(sector_names)
    assert len(callsigns_in_sector) == 0

    # Test single sector as string
    sector_names = "sector_i"
    callsigns_in_sector = env.aircraft_in_sectors(sector_names)
    assert callsigns_in_sector == ["AIR1"]

    sector_names = [ "background" ]
    callsigns_in_sector = env.aircraft_in_sectors(sector_names)
    assert callsigns_in_sector == ["AIR0"]

    # Test multiple sectors as string arrays
    sector_names = [ "sector_i", "background" ]
    callsigns_in_sector = env.aircraft_in_sectors(sector_names)
    assert callsigns_in_sector == ["AIR0", "AIR1"]

def test_transform_coordination_to_bandboxed(generate_simple_environment, generate_two_sector):
    """
    Test transforming coordination sectors to bandboxed. Verify that:
    - null coordinations are transformed to background
    - coordinations to any constituent sectors get converted as coordinations to the combined sector
    """

    env = generate_simple_environment
    callsign = "AIR0"
    coordinations = [ crd for crd in env.coordinations.get(callsign) ]

    use_null_from_sector = coordinations[0]
    new_coordination = env.transform_coordination_to_bandboxed(use_null_from_sector)
    assert new_coordination.from_sector == "background"

    use_null_to_sector = coordinations[1]
    new_coordination = env.transform_coordination_to_bandboxed(use_null_to_sector)
    assert new_coordination.to_sector == "background"

    # Now set up bandboxing and test that the containing bandbox name is returned rather than individual sector
    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "combined"
    individual_sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: ["sector_i1", "sector_i2"]})
    # Add a coordination from one of the bandboxed sectors
    crd_to_add = Coordination(callsign=callsign, from_sector=individual_sector_name, to_sector="sector_i2", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_to_add)
    new_coordination = env.transform_coordination_to_bandboxed(crd_to_add)
    assert new_coordination.from_sector == combined_sector_name
    assert new_coordination.to_sector == combined_sector_name

def test_entry_coordination(generate_simple_environment, generate_two_sector):
    """
    Test getting entry coordination for callsign and sector pair.
    """
    env = generate_simple_environment
    callsign = "AIR0"
    sector_name = "sector_i"
    apply_bandboxing = True
    # Test that the matching mocked coordination (simple sector) is returned as the entry coord
    entry_crd = env.entry_coordination(sector_name, callsign, apply_bandboxing)
    assert entry_crd.to_sector == "sector_i"

    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "bandboxed"
    sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: ["sector_i1", "sector_i2"]})
    # Test None returned when there are no coordinations
    entry_crd = env.entry_coordination(sector_name, callsign, apply_bandboxing)
    assert entry_crd is None

    # Add a coordination to one of the bandboxed sectors
    crd_to_add = Coordination(callsign=callsign, from_sector=None, to_sector=sector_name, fl=300, fix="", direction="Up")
    env.coordinations.add(crd_to_add)

    # Test that with apply_bandboxing switched ON the returned to_sector is the combined sector name
    entry_crd = env.entry_coordination(sector_name, callsign, apply_bandboxing)
    assert entry_crd.to_sector == combined_sector_name
    assert entry_crd.from_sector == "background"
    assert entry_crd.fl == 300

    # Test that with apply_bandboxing switched OFF the returned to_sector is the individual sector name
    entry_crd = env.entry_coordination(sector_name, callsign, apply_bandboxing=False)
    assert entry_crd.to_sector == sector_name
    assert entry_crd.from_sector == "background"


def test_exit_coordination(generate_simple_environment, generate_two_sector):
    """
    Test retrieving the exit coordination for callsign and sector pair.
    """
    env = generate_simple_environment
    callsign = "AIR0"
    sector_name = "sector_i"
    apply_bandboxing = True
    # Test that the matching mocked coordination (simple sector) is returned as the exit coord
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing)
    assert exit_crd.to_sector == "background"
    assert exit_crd.from_sector == sector_name

    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "bandboxed"
    sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: ["sector_i1", "sector_i2"]})
    # Test None returned when there are no coordinations
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing)
    assert exit_crd is None

    # Add a coordination from one of the bandboxed sectors
    crd_to_add = Coordination(callsign=callsign, from_sector=sector_name, to_sector=None, fl=300, fix="", direction="Up")
    env.coordinations.add(crd_to_add)

    # Test that with apply bandboxing switched ON the returned from_sector is the combined sector name and to_sector is background
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing)
    assert exit_crd.from_sector == combined_sector_name
    assert exit_crd.to_sector == "background"
    assert exit_crd.fl == 300

    # Test that with apply bandboxing switched OFF the returned from_sector is the individual sector name and to_sector is background
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing=False)
    assert exit_crd.from_sector == sector_name
    assert exit_crd.to_sector == "background"


def test_next_sector_of_aircraft_from_sector(generate_simple_environment, generate_two_sector):
    """
    Test next_sector_of_aircraft_from_sector. 
    """

    # Build an environment and airspace with a bandbox
    env = generate_simple_environment
    callsign = "AIR0"
    sector_name = "sector_i"
    apply_bandboxing = True
    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "bandboxed"
    sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: ["sector_i1", "sector_i2"]})

    # Test the null case when there are no coordinations
    next_sector = env.next_sector_of_aircraft_from_sector(callsign, sector_name, apply_bandboxing=False)
    assert next_sector == None

    # Add coordinations into and out of the bandbox with one internal coordination
    crd_into_bandbox = Coordination(callsign=callsign, from_sector=None, to_sector="sector_i1", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_into_bandbox)
    crd_within_bandbox = Coordination(callsign=callsign, from_sector="sector_i1", to_sector="sector_i2", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_within_bandbox)
    crd_out_of_bandbox = Coordination(callsign=callsign, from_sector="sector_i2", to_sector="sector_i3", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_out_of_bandbox)

    # Call the function under test and compare with result of exit coordination
    # The results should differ because exit coordination simply looks up the coordination, the function under test recurses until it finds the next sector outside bandbox
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing)
    next_sector = env.next_sector_of_aircraft_from_sector(callsign, sector_name, apply_bandboxing)
    assert next_sector != exit_crd.to_sector
    assert next_sector != combined_sector_name
    assert next_sector == "background"

    # Now try with apply bandboxing off, we expect the nominal internal coordination to be returned.
    exit_crd = env.exit_coordination(sector_name, callsign, apply_bandboxing=False)
    next_sector = env.next_sector_of_aircraft_from_sector(callsign, sector_name, apply_bandboxing=False)
    assert next_sector != combined_sector_name
    assert next_sector == exit_crd.to_sector == "sector_i2"


def test_next_sector_of_aircraft(generate_simple_environment, generate_two_sector):
    """
    Test next_sector_of_aircraft (same as above but using current sector rather than specifying next sector). 
    """

    # Build an environment and airspace with a bandbox
    env = generate_simple_environment
    callsign = "AIR0"
    first_sector_name = "sector_i"
    apply_bandboxing = True
    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "bandboxed"
    first_sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: ["sector_i1", "sector_i2"]})
    aircraft = env.aircraft[callsign]

    # Test the null case when there are no coordinations
    next_sector = env.next_sector_of_aircraft(callsign, apply_bandboxing=False)
    assert next_sector == None

    # Add coordinations into and out of the bandbox with one internal coordination
    crd_into_bandbox = Coordination(callsign=callsign, from_sector=None, to_sector="sector_i1", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_into_bandbox)
    crd_within_bandbox = Coordination(callsign=callsign, from_sector="sector_i1", to_sector="sector_i2", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_within_bandbox)
    crd_out_of_bandbox = Coordination(callsign=callsign, from_sector="sector_i2", to_sector="sector_i3", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_out_of_bandbox)

    # Starting from background call the function under test and compare with result of exit coordination
    # The results should be the same with apply bandboxing on
    exit_crd = env.exit_coordination(aircraft.current_sector, callsign, apply_bandboxing)
    next_sector = env.next_sector_of_aircraft(callsign, apply_bandboxing)
    assert next_sector == exit_crd.to_sector == combined_sector_name

    exit_crd = env.exit_coordination(aircraft.current_sector, callsign, apply_bandboxing=False)
    next_sector = env.next_sector_of_aircraft(callsign, apply_bandboxing=False)
    assert next_sector == exit_crd.to_sector == first_sector_name

    # Set the current sector and check the coordinations
    aircraft.current_sector = "sector_i1"
    # Call the function under test and compare with result of exit coordination
    # The results should differ because exit coordination simply looks up the coordination, the function under test recurses until it finds the next sector outside bandbox
    exit_crd = env.exit_coordination(aircraft.current_sector, callsign, apply_bandboxing)
    next_sector = env.next_sector_of_aircraft(callsign, apply_bandboxing)
    assert next_sector != combined_sector_name
    assert next_sector == "background"

    # Now try with apply bandboxing off, we expect the nominal internal coordination to be returned.
    exit_crd = env.exit_coordination(aircraft.current_sector, callsign, apply_bandboxing=False)
    next_sector = env.next_sector_of_aircraft(callsign, apply_bandboxing=False)
    assert next_sector != combined_sector_name
    assert next_sector == exit_crd.to_sector == "sector_i2"


def test_remove_coordinations_within_sector(generate_simple_environment, generate_two_sector):
    """
    Test removal of coordinations to sectors within a bandboxed sector
    """
    env = generate_simple_environment
    callsign = "AIR0"
    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "bandboxed"
    first_sector_name = "sector_i1"
    env.airspace.bandbox_sectors({combined_sector_name: [first_sector_name, "sector_i2"]})

    # Clean out the old coordinations
    env.coordinations.remove(callsign, "sector_i")
    env.coordinations.remove(callsign, None)

    # Null case (when no coordinations in environment)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordinations_within_sector(first_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_before == num_coords_after

     # Add coordinations into and out of the bandbox with one internal coordination
    crd_into_bandbox = Coordination(callsign=callsign, from_sector=None, to_sector="sector_i1", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_into_bandbox)
    crd_within_bandbox = Coordination(callsign=callsign, from_sector="sector_i1", to_sector="sector_i2", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_within_bandbox)
    crd_out_of_bandbox = Coordination(callsign=callsign, from_sector="sector_i2", to_sector="sector_i3", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_out_of_bandbox)

    # Test that passing the name of an internal sector does nothing
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordinations_within_sector(first_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_before == num_coords_after

    # Test that passing the name of the bandbox sector removes the single internal coordination
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordinations_within_sector(combined_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1


def test_remove_coordination_to_sector(generate_simple_environment, generate_two_sector):
    """
    Test removal of coordinations (including where the to sector is a bandbox)
    """
    env = generate_simple_environment
    callsign = "AIR0"
    new_airspace, _ = generate_two_sector
    env.airspace = new_airspace
    combined_sector_name = "sectori1_sectori2"
    first_sector_name = "sector_i1"
    second_sector_name = "sector_i2"
    env.airspace.bandbox_sectors({combined_sector_name: [first_sector_name, second_sector_name]})

    # Clean out the old coordinations
    env.coordinations.remove(callsign, "sector_i")
    env.coordinations.remove(callsign, None)

    # Null case (when no coordinations in environment)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(first_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    print(num_coords_before)
    assert num_coords_before == num_coords_after

    # Add coordinations into and out of the bandbox with one internal coordination
    crd_into_bandbox = Coordination(callsign=callsign, from_sector=None, to_sector=first_sector_name, fl=300, fix="", direction="Up")
    env.coordinations.add(crd_into_bandbox)
    crd_within_bandbox = Coordination(callsign=callsign, from_sector=first_sector_name, to_sector=second_sector_name, fl=300, fix="", direction="Up")
    env.coordinations.add(crd_within_bandbox)
    crd_out_of_bandbox = Coordination(callsign=callsign, from_sector=second_sector_name, to_sector="sector_i3", fl=300, fix="", direction="Up")
    env.coordinations.add(crd_out_of_bandbox)

    # Test removing the coordination out of the bandbox.
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector("background", callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before -1

    # Test that passing a non existent from_sector does nothing.
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(first_sector_name, callsign, "NONSENSE")
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before

    # Test that deletion of the coordination into the bandbox succeeds using only the to_sector and callsign.
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(first_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1

    # Test that deletion of the internal coordination specifying the specific sub sector name succeeds.
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(second_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1

    # Add the original internal coordination back in and repeat the test this time additionally specifying the from_sector and check deletion occurs.
    env.coordinations.add(crd_within_bandbox)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(second_sector_name, callsign, first_sector_name)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1

    # Create a coordination into the bandbox name and check that calling the remove method with the bandbox name results in deletion of the coordination
    crd_into_bandbox_using_bandbox_name = Coordination(callsign=callsign, from_sector=None, to_sector=combined_sector_name, fl=300, fix="", direction="Up")
    env.coordinations.add(crd_into_bandbox_using_bandbox_name)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(combined_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1

    # Restore the original coordination into the bandbox then check removal occurs when specifying the bandbox name as the to-sector
    env.coordinations.add(crd_into_bandbox)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(combined_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before - 1

    # Restore the original internal coordination and test that calling the remove method passing the bandboxed name DOESN'T result in deletion of the coordination.
    env.coordinations.add(crd_within_bandbox)
    num_coords_before = len(env.coordinations.get(callsign))
    env.remove_coordination_to_sector(combined_sector_name, callsign)
    num_coords_after = len(env.coordinations.get(callsign))
    assert num_coords_after == num_coords_before
