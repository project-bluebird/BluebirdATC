import random
import typing

import numpy as np
import pytest

from bluebird_gymnasium.envs import ViewType
from bluebird_gymnasium.envs.base import BaseEnv, ScenarioGenSeedMode
from bluebird_gymnasium.envs.infinite import CustomInfiniteEnv, InfiniteEnv
from bluebird_gymnasium.envs.sector_i import SectorIEnv
from bluebird_gymnasium.envs.sector_x import SectorXEnv
from bluebird_gymnasium.envs.sector_xplus import SectorXPlusEnv
from bluebird_gymnasium.envs.sector_y import SectorYEnv
from bluebird_gymnasium.envs.springfield import SpringfieldEnv

EnvCls: typing.TypeAlias = typing.Any

SECTOR_ENV_CLASSES = (SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv)
GENERATED_ENV_CLASSES = (
    InfiniteEnv,
    CustomInfiniteEnv,
    SectorIEnv,
    SectorXEnv,
    SectorXPlusEnv,
    SectorYEnv,
)


def _get_env_instance(env_cls: EnvCls):
    config = env_cls.get_default_env_config(ViewType.CENTRALIZED)

    if env_cls in SECTOR_ENV_CLASSES:
        config.scenario_config["args"]["num_aircraft"] = 4

    return env_cls(config=config)


def _initial_aircraft_signature(gym_env):
    """Build a stable snapshot of generated initial traffic."""

    def _coordination_signature(coordination):
        level_by_details = coordination.level_by_details
        if isinstance(level_by_details, dict):
            level_by_details = tuple(sorted(level_by_details.items()))

        coordination_time = coordination.datetime
        if coordination_time is not None:
            coordination_time = coordination_time.isoformat()

        return (
            coordination.from_sector,
            coordination.to_sector,
            coordination.fl,
            coordination.fix,
            coordination.direction,
            coordination.level_by,
            level_by_details,
            coordination.secondary_coord_conditions,
            coordination_time,
        )

    simulator_env = gym_env.get_simulator_env()

    return tuple(
        (
            callsign,
            aircraft.fl,
            aircraft.lat,
            aircraft.lon,
            aircraft.heading,
            aircraft.speed_tas,
            aircraft.random_seed,
            tuple(sorted(aircraft.percentile_rank_dict.items())),
            tuple(aircraft.flight_plan.route.filed),
            tuple(aircraft.flight_plan.route.current),
            aircraft.flight_plan.unexpanded_route,
            aircraft.flight_plan.requested_flight_level,
            aircraft.flight_plan.filed_true_airspeed,
            tuple(
                sorted(
                    _coordination_signature(coordination)
                    for coordination in (
                        simulator_env.coordinations.coords.values()
                    )
                    if coordination.callsign == callsign
                )
            ),
        )
        for callsign, aircraft in sorted(
            simulator_env.aircraft.items()
        )
    )


@pytest.mark.parametrize(
    (
        "seed_mode",
        "expected_reset_seed",
        "expects_legacy_rngs",
    ),
    [
        (ScenarioGenSeedMode.NONE, None, False),
        (ScenarioGenSeedMode.RESET_SEED_ATTRIBUTE, 42, False),
        (ScenarioGenSeedMode.LEGACY_MODULE_RNGS, None, True),
    ],
    ids=lambda param: param.name if isinstance(param, ScenarioGenSeedMode) else None,
)
def test_reset_seed_context_applies_seed_mode(
    seed_mode,
    expected_reset_seed,
    expects_legacy_rngs,
    monkeypatch,
):
    """Test how each seed mode applies reset seeds."""

    gym_env = object.__new__(BaseEnv)
    gym_env.scenario_seed_mode = seed_mode
    gym_env._reset_seed = None

    random_seeds = []
    numpy_seeds = []
    monkeypatch.setattr(random, "seed", random_seeds.append)
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)

    with gym_env._use_reset_seed_for_scenario_generation(42):
        assert gym_env._reset_seed == expected_reset_seed

    assert gym_env._reset_seed is None
    assert random_seeds == ([42] if expects_legacy_rngs else [])
    assert len(numpy_seeds) == (1 if expects_legacy_rngs else 0)


@pytest.mark.parametrize(
    "env_cls", GENERATED_ENV_CLASSES, ids=lambda env_cls: env_cls.__name__
)
def test_reset_seed_controls_generated_aircraft(env_cls: EnvCls):
    """Test that reset seeds are propagated to scenario generation."""

    gym_env = _get_env_instance(env_cls)

    gym_env.reset(seed=None)
    unseeded_signature = _initial_aircraft_signature(gym_env)

    gym_env.reset(seed=42)
    first_seed_signature = _initial_aircraft_signature(gym_env)

    gym_env.reset(seed=43)
    second_seed_signature = _initial_aircraft_signature(gym_env)

    gym_env.reset(seed=42)
    repeated_seed_signature = _initial_aircraft_signature(gym_env)

    assert repeated_seed_signature == first_seed_signature
    assert second_seed_signature != first_seed_signature
    assert unseeded_signature != first_seed_signature


def test_legacy_rng_seed_uses_full_reset_seed(monkeypatch):
    """Test that legacy RNG seeding uses the full reset seed."""

    gym_env = object.__new__(BaseEnv)
    gym_env.scenario_seed_mode = (
        ScenarioGenSeedMode.LEGACY_MODULE_RNGS
    )
    gym_env._reset_seed = None

    numpy_seeds = []
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)

    with gym_env._use_reset_seed_for_scenario_generation(42):
        assert gym_env._reset_seed is None

    with gym_env._use_reset_seed_for_scenario_generation(2**32 + 42):
        assert gym_env._reset_seed is None

    assert numpy_seeds[0] != numpy_seeds[1]


def test_reset_seed_context_clears_seed_after_error():
    """Test that the reset seed context cleans up after exceptions."""

    gym_env = object.__new__(BaseEnv)
    gym_env.scenario_seed_mode = (
        ScenarioGenSeedMode.RESET_SEED_ATTRIBUTE
    )
    gym_env._reset_seed = None

    random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    try:
        with pytest.raises(RuntimeError):
            with gym_env._use_reset_seed_for_scenario_generation(42):
                assert gym_env._reset_seed == 42
                raise RuntimeError

        assert gym_env._reset_seed is None

    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_random_state)


def test_reset_seed_does_not_control_springfield_aircraft():
    """Test that fixed-file scenarios are not changed by reset seeds."""

    gym_env = _get_env_instance(SpringfieldEnv)

    gym_env.reset(seed=42)
    first_seed_signature = _initial_aircraft_signature(gym_env)

    gym_env.reset(seed=43)
    second_seed_signature = _initial_aircraft_signature(gym_env)

    assert second_seed_signature == first_seed_signature
