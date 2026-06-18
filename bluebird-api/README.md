# The REST API for BluebirdATC [![PyPI version](https://img.shields.io/pypi/v/bluebird-api?logo=pypi&logoColor=white)](https://pypi.org/project/bluebird-api/) <img src="../images/BBATC_logo.png" alt="BluebirdATC logo" align="right" height="160" />


`bluebird-api` runs the BluebirdATC digital twin in a server process, such that the simulation will evolve at regular time intervals, and Agents and/or frontend visualization software can interact with it via HTTP requests.
In particular, users can:
* Query available scenario categories and scenarios.
* Load a selected scenario.
* Evolve the simulation by a specified time interval.
* Obtain the current state of the `Environment`.
* Submit `Actions` to individual aircraft.
* Save logfiles with data on all steps of the simulation.

## Getting started

The quickest way to start the server is with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uvx bluebird-api@latest
```

1. Open [http://localhost:8000](http://localhost:8000) — you should see `"Hello, BluebirdATC!"`.
1. Navigate to [http://localhost:8000/hmi](http://localhost:8000/hmi) to open the radar visualisation.
1. Select **Load new scenario** in the top left, choose a scenario type (eg. `Springfield`) and scenario (eg `test1`), and press **Load**.

> **Note: Running on a remote machine/cloud?**
Currently, the built version of the app is configured to look for the API running on `localhost`.  For deploying on remote machines, or a cloud service, it will be necessary to modify `src/api/config.ts` accordingly, and rebuild via `npm run build`.

## Using the API

Any language that can make HTTP requests can drive the simulation. The example below is a Python agent that directs each aircraft to its exit fix and cleared flight level as it enters the sector. Install the required packages first:

```bash
pip install bluebird-dt requests
```

```python
from bluebird_dt.core import Environment
import time, requests

callsigns_done = []

while True:
    response = requests.get("http://localhost:8000/environment")
    environment = Environment.from_json(response.text)

    actions_to_issue = []

    for aircraft in environment.aircraft.values():

        if aircraft.callsign in callsigns_done or aircraft.current_sector != "SPRINGFIELD":
            continue

        exit_coordination = environment.exit_coordination("SPRINGFIELD", aircraft.callsign)

        if exit_coordination is not None:
            actions_to_issue.extend([
                {
                    "callsign": aircraft.callsign,
                    "kind": "change_flight_level_to",
                    "value": exit_coordination.fl,
                    "sector": "SPRINGFIELD",
                    "agent": "agent"
                },
                {
                    "callsign": aircraft.callsign,
                    "kind": "route_direct_to",
                    "value": exit_coordination.fix,
                    "sector": "SPRINGFIELD",
                    "agent": "agent"
                }
            ])

        callsigns_done.append(aircraft.callsign)

    if len(actions_to_issue) > 0:
        requests.post("http://localhost:8000/actions", json=actions_to_issue)

    time.sleep(4)
```

A more complete multi-language example (including Julia) is in [NonPythonAgents.ipynb](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-api/examples/NonPythonAgents.ipynb).

## Documentation

Documentation of the endpoints of the API is available by running 

```bash
uvx bluebird-api@latest
```

and navigating to [http://localhost:8000/docs](http://localhost:8000/docs).

A json format of this API is also available in [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) which can be used to generate clients automatically using OpenAPI generators for the language you are using.

<div align="center"><img src="../images/BB_logo.png" alt="ProjectBluebird"></div>