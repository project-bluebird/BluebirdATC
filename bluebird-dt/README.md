# The Digital Twin [![PyPI version](https://img.shields.io/pypi/v/bluebird-dt?logo=pypi&logoColor=white)](https://pypi.org/project/bluebird-dt/) <img src="../images/BBATC_logo.png" alt="BluebirdATC logo" align="right" height="160" />

`bluebird_dt` encodes a digital twin of an airspace, including classes that represent:
* The geometry of the airspace - Sectors, Volumes, Airways, Fixes, ...
* Aircraft, with properties such as location, heading, flight level, ...
* Predictors, to model how the aircraft parameters evolve with the simulation.
* Action, defining the schema by which agents can interact with the simulation.
* Infrastructure classes such as Simulator, ScenarioManagers, EventHandlers, logger, to allow the user to define and run simulated ATC scenarios.

## Getting started

### Installation

`bluebird-dt` is available on pypi, therefore it can be installed using

```bash
pip install bluebird-dt
```

or, if using [UV](https://docs.astral.sh/uv/), you can add it to your environment using
```bash
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

### Running the digital twin as a server.

A FastApi app is available as `bluebird-api`, allowing the simulation to be run as a server, with the user (or an agent) interacting via a REST API. For information on this, see [GitHub](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-api/README.md) or [Pypi](https://pypi.org/project/bluebird-api/).

## Documentation

The full documentation for the `bluebird-dt` package can be found at in [https://docs.projectbluebird.ai](https://docs.projectbluebird.ai)

<div align="center"><img src="../images/BB_logo.png" alt="ProjectBluebird"></div>

## Where logs are saved

When a simulation saves its logs (the per-run `.log` file and the replay `.tar.gz` archive), they are written to a per-user data directory rather than inside the installed package, so they are not lost when the virtual environment is rebuilt.

The location is resolved with [`platformdirs`](https://pypi.org/project/platformdirs/), a small cross-platform library that returns each operating system's conventional per-user data directory. The logs live under a `bluebird-scenario-logs/bluebird_dt` folder inside that directory:

| Platform | Default log location |
| --- | --- |
| Linux   | `~/.local/share/bluebird-scenario-logs/bluebird_dt` (or `$XDG_DATA_HOME/...`) |
| macOS   | `~/Library/Application Support/bluebird-scenario-logs/bluebird_dt` |
| Windows | `%LOCALAPPDATA%\bluebird-scenario-logs\bluebird_dt` |

To use a different location, set the `BLUEBIRD_LOG_DIR` environment variable before importing `bluebird_dt`; its value is used as the base directory instead of the platformdirs default.
