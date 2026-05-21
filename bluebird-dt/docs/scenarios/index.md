# Scenarios 
Default scenarios are available via the Simulator class constructor:

```python
from bluebird_dt.simulator import Simulator

sim = Simulator.from_category("Artificial", "I-Sector Two Aircraft")
```

but more control is available by using a scenario manager as a simulator constructor.

### Two Aircraft

This scenario has two aircraft approaching one another from opposite sides of the sector.   Each aircraft can be a "climber", "descender" or "overflight".

It can easily be initialised using
```python
from bluebird_dt.simulator import Simulator

sim = Simulator.from_category("Artificial", "I-Sector Two Aircraft")
```

or, for more control over the scenario

```python
from bluebird_dt.scenario_manager import TwoAircraft
from bluebird_dt.airspace_generator import SectorX
airspace, routes = SectorX(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300),
                        alpha=52.5,
                    ).generate_airspace()


sim = TwoAircraft(
            airspace=airspace,
            routes=routes,
            total_time=1200,
            scenario_type="climber"
        ).to_simulator()
```

See the [source code reference](../source.md#bluebird_dt.scenario_manager.TwoAircraft) for documentation on all the parameters which could be passed in.

### Infinite

Unlike the other scenario generators that run for a specified length of time, the Infinite scenarios will go on indefinitely, with aircraft spawning stochastically at a given average frequency on randomly chosen routes.  Optionally, the user can ramp up the spawning frequency by a set interval after set periods of time, up to a specified maximum frequency.

It can easily be initialised using
```python
from bluebird_dt.simulator import Simulator

sim = Simulator.from_category("Infinite", "X-Sector")
```

or, for more control over the scenario

```python
from bluebird_dt.scenario_manager import Infinite
from bluebird_dt.airspace_generator import SectorX
airspace, routes = SectorX(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300),
                        alpha=52.5,
                    ).generate_airspace()


sim = Infinite(
            airspace=airspace,
            routes=routes,
            initial_spawn_rate = 0.005,
            max_spawn_rate = 0.2,
            spawn_rate_increment = 0.005,
            spawn_rate_increase_interval = 60
        ).to_simulator()
```

See the [source code reference](../source.md#bluebird_dt.scenario_manager.Infinite) for documentation on all the parameters which could be passed in.

### Springfield

Springfield is a realistic (though fictional) sector with a more complex route structure than X-Plus.
The Springfield scenario manager does not provide additional configuration options, therefore it is simplest to start it directly from the simulator.

```python
from bluebird_dt.scenario_manager import SpringfieldScenarioManager
from bluebird_dt.simulator import Simulator
import random

scenario_name = random.choice(SpringfieldScenarioManager.list_scenarios())
sim = Simulator.from_category("Springfield", scenario_name)

```

### Regular

The user can specify the total time and the number of aircraft for the scenario, and the aircraft will be emitted from route start points, quasi-regularly spaced out in time.

It can easily be initialised using

```python
from bluebird_dt.scenario_manager import Regular
from bluebird_dt.airspace_generator import SectorX
airspace, routes = SectorX(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300),
                        alpha=52.5,
                    ).generate_airspace()


sim = Regular(
            airspace=airspace,
            routes=routes,
            total_time=1200,
            num_aircraft=20,
        )
```

See the [source code reference](../source.md#bluebird_dt.scenario_manager.Regular) for documentation on all the parameters which could be passed in.

