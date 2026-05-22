# Welcome to bluebird-dt

The `bluebird-dt` package is a Digital Twin of Air Traffic Control (ATC) scenarios.
The Digital Twin can be used as a sandbox for training and testing *AI Agents* to perform ATC tasks.

## Installation

`bluebird-dt` is available on pypi, therefore it can be installed using

```bash
pip install bluebird-dt
```

or, if using [UV](https://docs.astral.sh/uv/), you can add it to your environment using

```bash
uv add bluebird-dt
```

## Making an agent

To run your first simulation, run the following script which issues a single instruction to an aircraft. 

```python
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

This example is very simple, various examples of using the `bluebird_dt` package can be found in Jupyter notebooks, that can be viewed or downloaded via the *Examples* tab.
We recommend startin with the [introduction to the digital twin](../examples/Intro-Part-1/)
