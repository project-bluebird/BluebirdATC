## Scenarios 
See the [scenario manager source code reference](../ource.md#scenario-manager).

### Two Aircraft

This scenario has two aircraft approaching one another from opposite sides of the sector.   Each aircraft can be a "climber", "descender" or "overflight".
See the [source code reference](../source.md#bluebird_dt.scenario_manager.TwoAircraft).

### Regular

The user can specify the total time and the number of aircraft for the scenario, and the aircraft will be emitted from route start points, quasi-regularly spaced out in time.
See the [source code reference](../source.md#bluebird_dt.scenario_manager.Regular).

### Custom

This is a more configurable option for generating simple custom scenarios.   The user can specify the total number of aircraft, the balance of climbers, descenders and overfliers, and the generator will spawn aircraft with randomly selected coordinations and speeds.  It is also possible for users to fully customize the scenario, adding aircraft with specified routes, positions, speeds, flight levels and coordinations, entering the airspace at specified times.
See the [source code reference](../source.md#bluebird_dt.scenario_manager.Custom).

### Infinite

Unlike the previous scenario generators that run for a specified length of time, the Infinite scenarios will go on indefinitely, with aircraft spawning stochastically at a given average frequency on randomly chosen routes.  Optionally, the user can ramp up the spawning frequency by a set interval after set periods of time, up to a specified maximum frequency.