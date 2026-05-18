# The `bluebird_dt` digital twin.

The python package `bluebird_dt` encodes a digital twin of an airspace, including classes that represent:
* The geometry of the airspace - Sectors, Volumes, Airways, Fixes, ...
* Aircraft, with properties such as location, heading, flight level, ...
* Predictors, to model how the aircraft parameters evolve with the simulation.
* Action, defining the schema by which agents can interact with the simulation.
* Infrastructure classes such as Simulator, ScenarioManagers, EventHandlers, logger, to allow the user to define and run simulated ATC scenarios.


The user will typically instantiate a `Simulator`, providing a scenario category and a scenario name.   The Simulator instance can then `evolve` through finite time steps (typically 6 seconds), and at each step it can receive `Actions`, and provide an `Environment`, which contains the full instantaneous representation of the simulation state.
This basic loop is demonstrated in the example notebook `simulation.ipynb`.

## Quickstart

### Prerequisites

We recommend using the [`uv`](https://docs.astral.sh/uv/), package manager to install and run `bluebird_dt`.  Installation instructions for UV are available [here](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

Clone the BluebirdATC repository, and `cd` to this directory:
```
git clone https://github.com/project-bluebird/BluebirdATC
cd BluebirdATC/bluebird-dt
```

You can then install all the dependencies by running the command
```
uv sync
```
(though they will anyway be installed when needed if you do `uv run` before any other command, e.g. `uv run python`).

## Documentation

The online documentation for the `bluebird_dt` package can be found by running the command:
```
../scripts/docs-serve
```
from this directory, then navigating a web browser to `http://localhost:8010`.

### Notebooks

Various examples of using the `bluebird_dt` package can be found in the form of Jupyter notebooks in the [examples] directory.  To run these, type:
```
uv run jupyter notebook
```
then navigating a web browser to `http://localhost:8888` (if it doesn't open automatically).

### Running the digital twin as a server.

A FastApi app is included, allowing the simulation to be run as a server, with the user (or an agent) interacting via a REST API.   For information on this, see [here](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-api/README.md).
