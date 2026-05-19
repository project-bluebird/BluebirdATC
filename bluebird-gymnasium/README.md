# Bluebird Gymnasium

A suite of gymnasium environments for air traffic control (ATC).
The environments are based on [bluebird-dt](https://github.com/project-bluebird/BluebirdATC/tree/main/bluebird-dt) (an ATC simulator).

The environments support research in agent-based learning (e.g. reinforcement learning) for ATC.
It supports either single agent or multi-agents scenarios.

## Installation

The default installation instructions are based on the use of `uv`, a package and dependency manager. However, if you want to perform a `conda` based installation, please visit the [alternate installation instructions](./ALTERNATE_INSTALLATION.md#instructions) which contains installation and steps and how to use the package. 

### Default Installation Instructions

`bluebird-gymnasium` is available on pypi, therefore it can be installed using

```
pip install bluebird-gymnasium
```

or, if using [UV](https://docs.astral.sh/uv/), you can add it to your environment using
```
uv add bluebird-gymnasium
```

## Usage

### Basic Usage

bluebird-gymnasium currently supports the following environments/airspace:
X sector, Y sector, I sector, Xplus sector and Springfield sector.

To instantiate a X sector environment with the default config, run:

```bash
python
>>> import gymnasium as gym
>>> import bluebird_gymnasium
>>> env = gym.make("SectorXEnv-v0")
```

### Sample RL Agents

Below, an example agent that takes random actions.

```bash
import gymnasium as gym
import bluebird_gymnasium

env = gym.make("SectorXEnv-v0")
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```

