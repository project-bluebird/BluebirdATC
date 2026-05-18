# BluebirdATC
A Digital Twin for use in ATC simulations, and a training environment for AI agents.

This repository contains the following packages:
 * `bluebird-dt` - the digital twin.  See [here](bluebird-dt/README.md) for more information. [![PyPI version](https://img.shields.io/pypi/v/bluebird-dt.svg)](https://pypi.org/project/bluebird-dt/)
 * `bluebird-api` - A REST api for the digital twin.  See [here](bluebird-api/README.md) for more information. [![PyPI version](https://img.shields.io/pypi/v/bluebird-api.svg)](https://pypi.org/project/bluebird-api/)
 * `bluebird-gymnasium` - a gym environment for AI agents.  See [here](bluebird-gymnasium/README.md) for more details. [![PyPI version](https://img.shields.io/pypi/v/bluebird-gymnasium.svg)](https://pypi.org/project/bluebird-gymnasium/)
 * `bluebird-hmi` - an optional web-based visualisation package.  See [here](bluebird-hmi/README.md) for details.
  
## (AI)r traffic controller challenge
Information relating to the (AI)r traffic controller challenge can be found in the `competition` folder. See the `Competition-Intro.ipynb` notebook to get started with the competition specific setup. Detailed introduction and guide will be added close to the completion date.

## Getting started

For quick start, please make sure uv is installed [(installation guide)](https://docs.astral.sh/uv/getting-started/installation/) and run the following command in a terminal:

```bash
uvx bluebird-api@latest
```

then navigate to [http://localhost:8000/hmi/](http://localhost:8000/hmi/).

For agent development or advanced integration of `bluebird-dt`, we recommend the Jupyter notebooks in the `examples/` directories of `bluebird-dt` (for core Digital Twin use) or `bluebird-gymnasium` (for Agent development).   To run these, change to the `examples/` directory and run the command:
```
uv run jupyter notebook
```

## Documentation

The documentation of the latest release is available at [https://docs.projectbluebird.ai](https://docs.projectbluebird.ai).

Alternatively, to build the full web-based docs for other versions, run the following command from this directory:

```bash
./scripts/docs-serve
```

then navigate your browser to [http://localhost:8010](http://localhost:8010).

## Contributing

Please see the guidelines [here](CONTRIBUTING.md) if you would like to contribute to BluebirdATC.