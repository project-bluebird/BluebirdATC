# Migrating to sector-agnostic scenario managers

Scenario generation and airspace selection are now separate. This page records
changes for consumers upgrading from the interface before PR #71, including
Starling.

## Names and construction

| Previous interface | New interface |
| --- | --- |
| `bluebird_dt.scenario_manager.Tactical` | `bluebird_dt.scenario_manager.Custom` |
| `bluebird_dt.scenario_manager.tactical` | `bluebird_dt.scenario_manager.custom` |
| `TacticalScenarioManagerConfig` | `CustomScenarioManagerConfig` |
| Gymnasium scenario class `"tactical"` | `"custom"` |
| `Simulator.from_category("Artificial", "I-Sector Two Aircraft")` | `Simulator.from_category("Two Aircraft", "I-Sector")` |
| `typeof_eventlogger` | `typeof_event_logger` |
| Springfield `typeof_environmentmanager` | `typeof_environment_manager` |
| Springfield `typeof_eventhandler` | `typeof_event_handler` |

The same category/name split applies to the X and Y Two Aircraft scenarios.
Regular, Custom, Two Aircraft and Infinite accept airspace names from
`AirspaceLoader.list_airspaces()`. The Springfield category still selects recorded
Springfield scenarios by their existing names.

Pass optional setup arguments by keyword. The PR adds parameters, including
`random_seed`, to existing signatures; positional optional arguments need review.
Preserve downstream `typeof_aircraft`, `typeof_environment_manager`,
`typeof_event_handler`, `typeof_event_logger` and `typeof_simulator` hooks when
adapting dispatch code. Test the resulting object types, not just construction.

## Randomness and initialisation

Artificial managers use instance-local NumPy generators. Seed them with
`random_seed`; seeding Python's or NumPy's module-level RNG does not seed them.
For the I, X, Xplus and Y Gymnasium environments, a non-None `reset(seed=...)`
overrides `scenario_config["args"]["random_seed"]` for that reset. An unseeded
reset uses the configured scenario seed if present, otherwise an unseeded generator.
The configuration dictionary is not modified by the override.

Regular, Two Aircraft and Infinite initialise past the first aircraft event using
a relative duration. The usual six-second interval is increased when necessary
to satisfy the configured predictor timestep and its two-step minimum.

## Saved configurations and replay

The scenario configuration discriminator changes from `"tactical"` to `"custom"`.
Update downstream configuration unions and provide a read-time migration for old
saved configurations, including configurations nested inside replay records.
Renaming a Python import alone does not make those records readable. Keep original
log files unchanged and test representative old and new replay fixtures.

## Starling rollout

Starling currently pins both `bluebird-dt` and `bluebird-api` to an older Git
revision. Keep those pins until its imports, scenario dispatch, keyword arguments
and saved-configuration reader are migrated and tested against the candidate
Bluebird revision in an isolated environment. Update both pins together with the
Starling migration. This Bluebird change does not update Starling or its pins.

The inspected Starling integration points include `starling/simulator/simulator.py`,
`starling/utility/models.py`, scenario listing, and API/replay tests. Validate
artificial and Springfield loading, custom derived types, configuration round trips
and replay of existing logs before updating the pins.
