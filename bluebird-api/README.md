# The REST API for BluebirdATC

It is possible to run the BluebirdATC digital twin in a server process, such that the simulation will evolve at regular time intervals, and Agents and/or frontend visualization software can interact with it via HTTP requests.
In particular, users can:
* Query available scenario categories and scenarios.
* Load a selected scenario.
* Evolve the simulation by a specified time interval.
* Obtain the current state of the `Environment`.
* Submit `Actions` to individual aircraft.
* Save logfiles with data on all steps of the simulation.

## Getting started

### Running the server

The simplest way to run the api is using uv [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/) and running 

```bash
uvx bluebird-api@latest
```

You should then be able to go to [http://localhost:8000](http://localhost:8000) in a web browser, and see the message "Hello, BluebirdATC!".

This package includes a prebuilt HMI available by navigating to [http://localhost:8000/hmi](http://localhost:8000/hmi).
Initially, no scenario would be loaded, therefore showing the Bluebird logo on the radar.
To load a scenario, the top left of the window select `Load new scenario`.
A window will appear in the middle of the screen, select `Springfield`, then `test1` and finally, `Load`.

### Using the API

Agents can interface with the simulator running behind a REST API, enabling its usage from any programming language.

The next script is an example of an agent in python, which requires the bluebird-dt and requests packages to be installed.

```bash
pip install bluebird-dt requests
```

It tells all aircraft, on incomm, to fly to their exit fix and climb directly to their exit flight level without ensuring safety or guaranteeing that aircraft will leave the sector.

```python
from bluebird_dt.core import Environment, Action
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
            actions_to_issue.extend(
                        [
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
                        ]
                    )

        callsigns_done.append(aircraft.callsign)

    if len(actions_to_issue) > 0:
        response = requests.post(
                "http://localhost:8000/actions",
                json=actions_to_issue
                )
    
    # Wait for the next tick
    time.sleep(4)
```

## Documentation

The documentation of the latest release is available at [https://docs.projectbluebird.ai](https://docs.projectbluebird.ai).

## OpenAPI

Documentation of the endpoints of the API is available by running 

```bash
uvx bluebird-api@latest
```

and navigating to [http://localhost:8000/docs](http://localhost:8000/docs).

A json format of this API is also available in [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) which can be used to generate clients automatically using OpenAPI generators for the language you are using.

## Frontend visualisation

The app also serves the frontend visualization (more details on that can be found [here](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-hmi/README.md)), at the URL [http://localhost:8000/hmi](http://localhost:8000/hmi).

## Julia example

A more complete example of how to use the API is available in [NonPythonAgents.ipynb](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-api/examples/NonPythonAgents.ipynb)
