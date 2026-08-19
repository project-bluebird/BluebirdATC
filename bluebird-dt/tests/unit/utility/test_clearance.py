import logging
import pytest

from bluebird_dt.core import Action, Environment
from bluebird_dt.utility.clearance import (
    add_phraseology,
    beautify_callsign,
    spell_phonetically,
    text_phraseology,
    voice_phraseology,
    which_way,
)
from tests.unit.core.conftest import make_random_aircraft


@pytest.fixture
def env(generate_simple_environment: Environment) -> Environment:
    environment = generate_simple_environment

    # instantiate 10 Aircraft --> all are controllable
    aircraft = {f"AIR{i}": make_random_aircraft() for i in range(10)}
    for a in aircraft.values():
        a.controllable = True

    # add to environment and check all are controllable
    environment.aircraft = aircraft

    assert "AIR0" in environment.aircraft
    environment.aircraft["AIR0"].fl = 200.0
    environment.aircraft["AIR0"].heading = 0
    environment.aircraft["AIR0"].cleared_fl = 250.0
    environment.aircraft["AIR0"].cleared_instructions.on_route = True

    assert "AIR1" in environment.aircraft
    environment.aircraft["AIR1"].fl = 200.0
    environment.aircraft["AIR1"].cleared_fl = 150.0
    environment.aircraft["AIR1"].cleared_instructions.on_route = False

    return environment

def assert_action_voice_and_text(
        action: Action,
        expected_text_clearance: str,
        expected_text_response: str,
        expected_voice_clearance: str | None,
        expected_voice_response: str | None,
        env: Environment,
        ):
    text_representation = text_phraseology(action, env)
    assert text_representation.clearance == expected_text_clearance
    assert text_representation.pilot_response == expected_text_response

    voice_representation = voice_phraseology(action, env)
    if expected_voice_clearance is not None:
        assert voice_representation.clearance == expected_voice_clearance

    if expected_voice_response is not None:
        assert voice_representation.pilot_response == expected_voice_response

def test_missing_callsign(env: Environment, caplog: pytest.LogCaptureFixture):
    action = Action("MISS", "outcomm", None)

    with caplog.at_level(logging.DEBUG):
        _ = text_phraseology(action, env)
        assert ("bluebird_dt.logger", logging.WARNING, "Callsign MISS unavailable in environment.") in caplog.record_tuples


def test_maintain_current_heading_clearance(env: Environment):
    assert_action_voice_and_text(
            Action("AIR0", "maintain_current_heading", 0),
            "AIR0 continue present heading",
            "continue present heading AIR0",
            "alpha india romeo zero continue present heading",
            "continue present heading alpha india romeo zero",
            env,
            )


def test_climb_clearance(env: Environment):
    action = Action("AIR1", "change_flight_level_to", 300)
    assert env.aircraft["AIR1"].fl < 300

    assert_action_voice_and_text(
            action,
            "AIR1 climb flight level 300",
            "climb flight level 300 AIR1",
            "alpha india romeo wun climb flight level tree hundred",
            "climb flight level tree hundred alpha india romeo wun",
            env
            )


def test_climb_clearance_float_to_int(env: Environment):
    action = Action("AIR0", "change_flight_level_to", 210.0)
    assert env.aircraft["AIR0"].fl < 210.0

    assert_action_voice_and_text(
            action,
            "AIR0 climb flight level 210",
            "climb flight level 210 AIR0",
            "alpha india romeo zero climb flight level too wun zero",
            "climb flight level too wun zero alpha india romeo zero",
            env,
            )

def test_descend_clearance(env: Environment):
    action = Action("AIR0", "change_flight_level_to", 100)
    assert env.aircraft["AIR0"].fl > 100

    assert_action_voice_and_text(
            action,
            "AIR0 descend flight level 100",
            "descend flight level 100 AIR0",
            "alpha india romeo zero dee send flight level wun hundred",
            "dee send flight level wun hundred alpha india romeo zero",
            env,
            )

def test_route_direct_on_route_clearance(env: Environment):
    action = Action("AIR0", "route_direct_to", "ALPHA")
    assert env.aircraft["AIR0"].cleared_instructions.on_route

    assert_action_voice_and_text(
            action,
            "AIR0 route direct ALPHA",
            "route direct ALPHA AIR0",
            "alpha india romeo zero route direct ALPHA",
            "route direct ALPHA alpha india romeo zero",
            env,
            )

def test_route_direct_not_on_route_clearance(env: Environment):
    action = Action("AIR1", "route_direct_to", "BRAVO")
    assert not env.aircraft["AIR1"].cleared_instructions.on_route

    assert_action_voice_and_text(
            action,
            "AIR1 resume own navigation BRAVO",
            "resume own navigation BRAVO AIR1",
            "alpha india romeo wun resume own navigation BRAVO",
            "resume own navigation BRAVO alpha india romeo wun",
            env,
            )

def test_change_heading_clearance(env: Environment):
    action = Action("AIR0", "change_heading_to", 90)

    assert_action_voice_and_text(
            action,
            "AIR0 fly heading 090 degrees",
            "fly heading 090 degrees AIR0",
            "alpha india romeo zero turn right heading zero niner zero degrees",
            "turn right heading zero niner zero degrees alpha india romeo zero",
            env,
            )

def test_float_heading_clearance(env: Environment):
    action = Action("AIR0", "change_heading_to", 12.0)


    assert_action_voice_and_text(
            action,
            "AIR0 fly heading 012 degrees",
            "fly heading 012 degrees AIR0",
            "alpha india romeo zero turn right heading zero wun too",
            "turn right heading zero wun too alpha india romeo zero",
            env,
            )

def test_change_heading_by_left_clearance(env: Environment):
    action = Action("AIR0", "change_heading_by", -15)

    assert_action_voice_and_text(
            action,
            "AIR0 turn left 15 degrees",
            "turn left 15 degrees AIR0",
            "alpha india romeo zero turn left wun five degrees",
            "turn left wun five degrees alpha india romeo zero",
            env,
            )


def test_change_heading_by_right_clearance(env: Environment):
    action = Action("AIR0", "change_heading_by", 25)

    assert_action_voice_and_text(
            action,
            "AIR0 turn right 25 degrees",
            "turn right 25 degrees AIR0",
            "alpha india romeo zero turn right too five degrees",
            "turn right too five degrees alpha india romeo zero",
            env,
            )

def test_change_vertical_speed_climbing_clearance(env: Environment):
    action = Action("AIR0", "change_vertical_speed_to", 120)
    assert env.aircraft["AIR0"].cleared_fl > env.aircraft["AIR0"].fl

    assert_action_voice_and_text(
            action,
            "AIR0 rate of climb 120 feet per minute",
            "rate of climb 120 feet per minute AIR0",
            "alpha india romeo zero rate of climb 120 feet per minute",
            "rate of climb 120 feet per minute alpha india romeo zero",
            env,
            )

def test_change_vertical_speed_descending_clearance(env: Environment):
    action = Action("AIR1", "change_vertical_speed_to", 130)
    assert env.aircraft["AIR1"].cleared_fl < env.aircraft["AIR1"].fl

    assert_action_voice_and_text(
            action,
            "AIR1 rate of descent 130 feet per minute",
            "rate of descent 130 feet per minute AIR1",
            "alpha india romeo wun rate of dee send 130 feet per minute",
            "rate of dee send 130 feet per minute alpha india romeo wun",
            env,
            )

def test_change_cas_clearance(env: Environment):
    action = Action("AIR0", "change_cas_to", 300)

    assert_action_voice_and_text(
            action,
            "AIR0 fly speed 300 knots",
            "fly speed 300 knots AIR0",
            "alpha india romeo zero speed tree zero zero knots",
            "speed tree zero zero knots alpha india romeo zero",
            env,
            )


def test_spell_phonetically():
    """
    Verify that the phonetic alphabet is spelled out correctly
    """
    assert spell_phonetically("A1") == "alpha wun"
    assert spell_phonetically("B2") == "bravo too"
    assert spell_phonetically("PQ37") == "papa quebec tree seven"
    assert spell_phonetically("ZR900") == "zulu romeo niner zero zero"


def test_beautify_callsign():
    """
    Verify that callsigns are beautfied correctly. Test each beautification rule.
    """
    assert beautify_callsign("N12345") == "november tree four five"
    assert beautify_callsign("MAS3578") == "Malaysian tree five seven eight"
    assert beautify_callsign("AAL123") == "American wun too tree"
    assert beautify_callsign("ET6789") == "echo eight niner"
    assert beautify_callsign("ZZZ123") == "zulu zulu zulu wun too tree"


def test_which_way_left_and_right():
    """
    Test which way calculation for two angles
    """
    assert which_way(10, 350) == "left"
    assert which_way(350, 10) == "right"


def test_text_phraseology(env: Environment):
    """
    Create various actions and test the generated phraseology
    """
    action = Action("AIR0", "set_squawk", 1234)
    rep = text_phraseology(action, env)
    assert rep.clearance == "AIR0, set squawk 1234"
    assert rep.pilot_response == "Squawking 1234, AIR0"

    action = Action("AIR0", "squawk_ident", None)
    rep = text_phraseology(action, env)
    assert rep.clearance == "AIR0, squawk ident"
    assert rep.pilot_response == "Squawking ident, AIR0"

    action = Action("AIR0", "maintain_current_heading", 0)
    action = add_phraseology(action, env)
    assert action.text_representation is not None
    assert action.voice_representation is not None

    action = Action("AIR0", "set_squawk", "1234")
    text = text_phraseology(action, env)
    assert text.clearance == "AIR0, set squawk 1234"
    assert text.pilot_response == "Squawking 1234, AIR0"

    action = Action("AIR0", "squawk_ident", None)
    text = text_phraseology(action, env)
    assert text.clearance == "AIR0, squawk ident"
    assert text.pilot_response == "Squawking ident, AIR0"

    action = Action("AIR0", "change_heading_to_by_direction", (180, "left"))
    text = text_phraseology(action, env)
    assert text.clearance == "AIR0, turn left heading 180"
    assert text.pilot_response == "Turning left heading 180, AIR0"

    action = Action("AIR0", "message", "Hold position")
    text = text_phraseology(action, env)
    assert text.clearance == "Hold position"
    assert text.pilot_response == ""


def test_change_mach_clearance(env: Environment):
    action = Action("AIR0", "change_mach_to", 0.8)

    assert_action_voice_and_text(
            action,
            "AIR0 fly speed mach 0.8",
            "fly speed mach 0.8 AIR0",
            "alpha india romeo zero speed mach zero decimal eight",
            "speed mach zero decimal eight alpha india romeo zero",
            env,
            )

def test_outcomm_clearance(env: Environment):
    action = Action("AIR0", "outcomm", None)

    assert_action_voice_and_text(
            action,
            "AIR0 contact next frequency",
            "contact next frequency AIR0",
            None,
            None,
            env,
            )

def test_route_direct_waypoint_clearance(env: Environment):
    action = Action("AIR0", "route_direct_to", "LON")
    assert env.aircraft["AIR0"].cleared_instructions.on_route

    assert_action_voice_and_text(
            action,
            "AIR0 route direct London",
            "route direct London AIR0",
            "alpha india romeo zero route direct London",
            "route direct London alpha india romeo zero",
            env,
            )

def test_route_direct_multi_waypoint_clearance(env: Environment):
    action = Action("AIR0", "route_direct_to", ["ALPHA", "LON"])
    assert env.aircraft["AIR0"].cleared_instructions.on_route

    assert_action_voice_and_text(
            action,
            "AIR0 route direct [ALPHA, London]",
            "route direct [ALPHA, London] AIR0",
            "alpha india romeo zero route direct [ALPHA, London]",
            "route direct [ALPHA, London] alpha india romeo zero",
            env,
            )

def test_using_speed_limit(env: Environment):
    action = Action("AIR0", "using_speed_limit", True)

    assert_action_voice_and_text(
            action,
            "AIR0 obeying speed limit",
            "obeying speed limit AIR0",
            "alpha india romeo zero apply speed restrictions",
            "applying speed restrictions alpha india romeo zero",
            env
            )

def test_not_using_speed_limit(env: Environment):
    action = Action("AIR0", "using_speed_limit", False)

    assert_action_voice_and_text(
            action,
            "AIR0 no speed restrictions",
            "no speed restrictions AIR0",
            "alpha india romeo zero no a t c speed restrictions",
            "no speed restrictions alpha india romeo zero",
            env
            )

def test_expect_level_by_fix_climb_now(env: Environment):
    action = Action("AIR0", "expect_level_by_fix,climb",
                    (300, "OCK", 330)
                    )

    assert_action_voice_and_text(
            action,
            "AIR0 expect flight level 330 level by OCK climb now flight level 300",
            "expect flight level 330 level by OCK climb now flight level 300 AIR0",
            "alpha india romeo zero expect flight level tree tree zero level by OCK climb now flight level tree hundred",
            "expect flight level tree tree zero level by OCK climb now flight level tree hundred alpha india romeo zero",
            env,
            )

def test_expect_level_abeam_fix_climb_now(env: Environment):
    aircraft = env.aircraft.get("AIR0")
    assert aircraft is not None
    aircraft.on_route = False

    action = Action("AIR0", "expect_level_by_fix,climb", (300, "OCK", 330))

    assert_action_voice_and_text(
            action,
            "AIR0 expect flight level 330 level abeam OCK climb now flight level 300",
            "expect flight level 330 level abeam OCK climb now flight level 300 AIR0",
            "alpha india romeo zero expect flight level tree tree zero level abeam OCK climb now flight level tree hundred",
            "expect flight level tree tree zero level abeam OCK climb now flight level tree hundred alpha india romeo zero",
            env,
            )

def test_expect_level_by_fix_descend_now(env: Environment):
    action = Action("AIR0", "expect_level_by_fix,descend", (150, "OCK", 100))

    assert_action_voice_and_text(
            action,
            "AIR0 expect flight level 100 level by OCK descend now flight level 150",
            "expect flight level 100 level by OCK descend now flight level 150 AIR0",
            "alpha india romeo zero expect flight level wun hundred level by OCK descend now flight level wun five zero",
            "expect flight level wun hundred level by OCK descend now flight level wun five zero alpha india romeo zero",
            env,
            )


def test_expect_level_abeam_fix_descend_now(env: Environment):
    aircraft = env.aircraft.get("AIR0")
    assert aircraft is not None
    aircraft.on_route = False

    action = Action("AIR0", "expect_level_by_fix,descend", (150, "OCK", 100))

    assert_action_voice_and_text(
            action,
            "AIR0 expect flight level 100 level abeam OCK descend now flight level 150",
            "expect flight level 100 level abeam OCK descend now flight level 150 AIR0",
            "alpha india romeo zero expect flight level wun hundred level abeam OCK descend now flight level wun five zero",
            "expect flight level wun hundred level abeam OCK descend now flight level wun five zero alpha india romeo zero",
            env,
            )
