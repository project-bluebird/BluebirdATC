"""Speed-table lookups and the cached Mach/CAS transition check of the LinearPredictor."""

import pytest

from bluebird_dt.core import Environment, FlightState
from bluebird_dt.predictor import LinearPredictor
from bluebird_dt.utility.convert import FL_TO_M, KT_TO_MPS, mach_cas_trans_altitude
from bluebird_dt.utility.performance import (
    STATE_LABELS,
    cas_and_mach_from_table,
    get_performance_table,
    lateral_speed_keys,
    lateral_speed_offset,
    rocd_from_table,
)


@pytest.mark.parametrize("state", list(FlightState))
def test_table_lookups_at_and_between_table_levels(state: FlightState):
    """At a table level the lookup returns that level's values; halfway between two levels, their average;
    beyond the table, the end values."""
    cas_key, mach_key = lateral_speed_keys(state)
    rocd_key = f"rocd_{STATE_LABELS[state]}"
    for table in get_performance_table(None).values():
        levels = table["flight_level"]
        cas = table[cas_key]
        mach = table.get(mach_key) if isinstance(table.get(mach_key), list) else None
        rocd = table.get(rocd_key) if isinstance(table.get(rocd_key), list) else None
        sign = -1.0 if rocd_key == "rocd_des" else 1.0
        for i in range(len(levels) - 1):
            lo, hi = levels[i], levels[i + 1]
            for fl, w in ((lo, 0.0), ((lo + hi) / 2, 0.5)):
                got_cas, got_mach = cas_and_mach_from_table(table, fl, state)
                if cas[i] is not None and cas[i + 1] is not None:
                    assert got_cas == pytest.approx((1 - w) * cas[i] + w * cas[i + 1], abs=1e-12)
                if mach is not None and mach[i] is not None and mach[i + 1] is not None:
                    assert got_mach == pytest.approx((1 - w) * mach[i] + w * mach[i + 1], abs=1e-12)
                if rocd is not None and rocd[i] is not None and rocd[i + 1] is not None and fl < levels[-1]:
                    got_rocd = rocd_from_table(table, None, None, fl, state)
                    assert got_rocd == pytest.approx(sign * ((1 - w) * rocd[i] + w * rocd[i + 1]), abs=1e-12)
        # clamped beyond the ends of the table
        assert cas_and_mach_from_table(table, levels[-1] + 50, state) == cas_and_mach_from_table(table, levels[-1], state)
        assert cas_and_mach_from_table(table, levels[0] - 50, state) == cas_and_mach_from_table(table, levels[0], state)


def _below_transition_reference(predictor: LinearPredictor, aircraft) -> bool:
    """The transition check computed directly from the tables, without any caching."""
    key = predictor._get_aircraft_lookup_key(aircraft.aircraft_type)
    table = predictor._get_performance_profile(key)
    uncertainty = predictor._get_performance_uncertainty(key)
    cas_key, mach_key = lateral_speed_keys(aircraft.flight_state)
    ranks = aircraft.percentile_rank_dict
    if aircraft.selected_instructions.cas is None:
        values = [v for v in table[cas_key] if v is not None]
        if not values:
            return False
        cas = max(values) + lateral_speed_offset(uncertainty, cas_key, ranks.get(cas_key))
    else:
        cas = aircraft.selected_instructions.cas
    if aircraft.selected_instructions.mach is None:
        values = [v for v in (table.get(mach_key) or []) if v is not None]
        if not values:
            return True
        mach = max(values) + lateral_speed_offset(uncertainty, mach_key, ranks.get(mach_key))
    else:
        mach = aircraft.selected_instructions.mach
    altitude = mach_cas_trans_altitude(mach, cas * KT_TO_MPS, delta_T=predictor.delta_T)
    return aircraft.fl * FL_TO_M - altitude < 0


@pytest.mark.parametrize("aircraft_type", ["A320", "B789", "C172", "A388"])
def test_cached_transition_check_matches_the_direct_computation(aircraft_type: str, generate_simple_environment: Environment):
    predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
    aircraft = generate_simple_environment.aircraft["AIR1"]
    aircraft.aircraft_type = aircraft_type
    for selected_fl in (100.0, 300.0, 450.0):  # descend / cruise / climb depending on fl
        for cas_pr in (None, 10.0, 90.0):
            for selected in ((None, None), (280.0, None), (None, 0.78), (290.0, 0.80)):
                aircraft.selected_fl = selected_fl
                aircraft.set_performance(cas_pr=cas_pr)
                aircraft.selected_instructions.cas, aircraft.selected_instructions.mach = selected
                for fl in range(50, 451, 25):
                    aircraft.fl = float(fl)
                    expected = _below_transition_reference(predictor, aircraft)
                    assert predictor.is_aircraft_below_transition(aircraft) is expected
                    assert predictor.is_aircraft_below_transition(aircraft) is expected  # cached path
