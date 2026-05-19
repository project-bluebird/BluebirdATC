# BluebirdATC
A Digital Twin for use in ATC simulations, and a training environment for AI agents.

This repository contains the following packages:
 * `bluebird-dt` [![PyPI version](https://img.shields.io/pypi/v/bluebird-dt.svg)](https://pypi.org/project/bluebird-dt/) - the digital twin.  See [here](bluebird-dt/README.md) for more information.
 * `bluebird-api` [![PyPI version](https://img.shields.io/pypi/v/bluebird-api.svg)](https://pypi.org/project/bluebird-api/) - A REST api for the digital twin.  See [here](bluebird-api/README.md) for more information.
 * `bluebird-gymnasium` [![PyPI version](https://img.shields.io/pypi/v/bluebird-gymnasium.svg)](https://pypi.org/project/bluebird-gymnasium/) - a gym environment for AI agents.  See [here](bluebird-gymnasium/README.md) for more details. 
 * `bluebird-hmi` - an optional web-based visualisation package.  See [here](bluebird-hmi/README.md) for details.
  
## (AI)r traffic controller challenge
Information relating to the (AI)r traffic controller challenge can be found in the `competition` folder. See the `Competition-Intro.ipynb` notebook to get started with the competition specific setup. Detailed introduction and guide will be added close to the completion date.

## Running the digital twin

For quick start, please make sure uv is installed [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/) and run the following command in a terminal:

```bash
uvx bluebird-api@latest
```

then navigate to [http://localhost:8000/hmi/](http://localhost:8000/hmi/).

This site will open a radar HMI, initially with no scenario loaded.
To load a scenario, the top left of the window select `Load new scenario`.
A window will apear in the middle of the screen, select `Artificial`, then `I-Sector Two Aircraft` and finally, `Load`.

With the scenario loaded, the aircraft and sector should now be visible in the radar. Clicking the play icon in the top left of the screen will make the simulation evolve making the aircraft move.

## Quick start examples for developing agents

Examples for interfacing with the digital twin to make agents are available:
* [here](bluebird-gymnasium/README.md#getting-started) for using the gymnasium
* [here](bluebird-api/README.md#getting-started) for using the REST API from any language
* [here](bluebird-dt/README.md#getting-started) to directly interact with the digital twin

## Documentation

The documentation of the latest release is available at [https://docs.projectbluebird.ai](https://docs.projectbluebird.ai).

## Contributing

Please see the guidelines [here](CONTRIBUTING.md) if you would like to contribute to BluebirdATC.
