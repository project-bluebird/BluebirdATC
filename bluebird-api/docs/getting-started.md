Bluebird API is a wrapper of the digital twin to run as a server process.
Agents, front end visualization and others can then interact with it via HTTP requests.

## Running the server

The simplest way to run the api is using uv [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/) and running 

```bash
uvx bluebird-api@latest
```

then navigating to [http://localhost:8000](http://localhost:8000) in a web browser, and see the message "Hello, BluebirdATC!".

This package includes a prebuilt HMI (i.e. web frontend) available by navigating to [http://localhost:8000/hmi](http://localhost:8000/hmi).
Initially, no scenario would be loaded, therefore showing the Bluebird logo on the radar.
To load a scenario, the top left of the window select `Load new scenario`.
A window will appear in the middle of the screen, select `Springfield`, then `test1` and finally, `Load`.

## Simple agent to interact with the API

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
    # Request and deserialize the environment from the server
    response = requests.get("http://localhost:8000/environment")
    environment = Environment.from_json(response.text)

    actions_to_issue = []

    # Iterate through all the aircraft in the airspace
    for aircraft in environment.aircraft.values():

        # If we have already issued the instructions to the aircraft, or it is
        # not yet incommed into the sector, we ignore this aircraft
        if aircraft.callsign in callsigns_done or aircraft.current_sector != "SPRINGFIELD":
            continue
        
        # Get the aircraft's exit coordination for the sector we are currently controlling.
        exit_coordination = environment.exit_coordination("SPRINGFIELD", aircraft.callsign)
        
        # If an exit coordination exist, issue two instructions.
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

        # Append the actions to be issued after the loop
        callsigns_done.append(aircraft.callsign)

    # If there are any action, send them to the 
    if len(actions_to_issue) > 0:
        response = requests.post(
                "http://localhost:8000/actions",
                json=actions_to_issue
                )
    
    # Wait for the next tick
    time.sleep(4)
```

## Making agents in other languages

Documentation of the endpoints of the API is available by running 

```bash
uvx bluebird-api@latest
```

and navigating to [http://localhost:8000/docs](http://localhost:8000/docs).

A json format of this API is also available in [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) which can be used to generate clients automatically using OpenAPI generators for the language you are using.

### Julia example

A more complete example of how to use the API is available in [NonPythonAgents.ipynb](examples/NonPythonAgents.ipynb)
