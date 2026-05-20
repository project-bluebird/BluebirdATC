# The `bluebird_dt` digital twin.

The python package `bluebird_dt` encodes a digital twin of an airspace, including classes that represent:
* The geometry of the airspace - Sectors, Volumes, Airways, Fixes, ...
* Aircraft, with properties such as location, heading, flight level, ...
* Predictors, to model how the aircraft parameters evolve with the simulation.
* Action, defining the schema by which agents can interact with the simulation.
* Infrastructure classes such as Simulator, ScenarioManagers, EventHandlers, logger, to allow the user to define and run simulated ATC scenarios.

## Getting started

### Installation

`bluebird-dt` is available on pypi, therefore it can be installed using

```
pip install bluebird-dt
```

or, if using [UV](https://docs.astral.sh/uv/), you can add it to your environment using
```
uv add bluebird-dt
```

### Making an agent

To run your first simulation, run the following script which issues a single instruction to an aircraft. 

```
from bluebird_dt.core import Action
from bluebird_dt.simulator.simulator import Simulator

# Use Simulator
sim = Simulator.from_category("Artificial", "I-Sector Two Aircraft")

# Evolve for 60 seconds, in 6 second radar sweeps
for _ in range(0, 10):
    sim.evolve(6)

# List all the aircraft in the airspace
print(sim.manager.environment.aircraft)

# Issue an action to one of the aircraft
sim.manager.receive_actions(
        [
            Action("AIR0", "change_flight_level_to", 200)
            ]
        )
```

This example is very simple, various examples of using the `bluebird_dt` package can be found in the form of Jupyter notebooks in the [examples](https://github.com/project-bluebird/BluebirdATC/tree/main/bluebird-dt/examples) directory.

## Documentation

The online documentation for the `bluebird_dt` package can be found at in [https://docs.projectbluebird.ai](https://docs.projectbluebird.ai)

### Running the digital twin as a server.

A FastApi app is available as bluebird-api, allowing the simulation to be run as a server, with the user (or an agent) interacting via a REST API. For information on this, see [GitHub](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-api/README.md) or [Pypi](https://pypi.org/project/bluebird-api/).
