# The REST API for BluebirdATC

It is possible to run the BluebirdATC digital twin in a server process, such that the simulation will evolve at regular time intervals, and Agents and/or frontend visualization software can interact with it via HTTP requests.
In particular, users can:
* Query available scenario categories and scenarios.
* Load a selected scenario.
* Evolve the simulation by a specified time interval.
* Obtain the current state of the `Environment`.
* Submit `Actions` to individual aircraft.
* Save logfiles with data on all steps of the simulation.

The simplest way to run the app is using uv [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/) and running 

```bash
uvx bluebird-api@latest
```

You should then be able to go to [http://localhost:8000](http://localhost:8000) in a web browser, and see the message "Hello, BluebirdATC!".

To see the full list and description of API endpoints, with the application running, go to [http://localhost:8000/docs](http://localhost:8000/docs).

## Simple agent interfacing though the REST API

Agents can interface with the simulator running as an API.
The next script is an example of an agent which tells all aircraft, on incomm, to fly to their exit fix and climb directly to their exit flight level without ensuring safety or garanteeing that aircraft will leave the sector.

```python
from bluebird_dt.core import Environment, Action
import time, requests, json

callsigns_done = []

while True:
    response = requests.get("http://localhost:8000/environment")
    environment = Environment.from_json(response.text)

    actions_to_issue: list[Action] = []

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
                data=json.dumps(actions_to_issue)
                )
    
    # Wait for the next tick
    time.sleep(4)

```

## Frontend visualisation

The app also serves the frontend visualization (more details on that can be found [here](https://github.com/project-bluebird/BluebirdATC/blob/main/bluebird-hmi/README.md)), at the URL [http://localhost:8000/hmi](http://localhost:8000/hmi).

## OpenAPI

Documentation of the endpoints of the API is available by running 

```bash
uvx bluebird-api@latest
```

and navigating to [http://localhost:8000/docs](http://localhost:8000/docs).
