# `bluebird_gymnasium` documentation

A suite of [gymnasium](https://github.com/Farama-Foundation/Gymnasium) API compliant environments for air traffic control (ATC).
The environments are based on the digital twin simulator, [bluebird-dt](https://github.com/project-bluebird/BluebirdATC/tree/main/bluebird-dt){ target="_blank" rel="noopener" }, serving as a wrapper around it. The environments support research in learning-based agents (e.g., reinforcement learning and imitation learning) for ATC. It supports either single agent (aka centralized) or multi-agent (aka decentralized) agent set up.

In addition to the environment specification, the package also contains a set of state spaces (encodings), reward functions, and discrete action spaces definitions that are configurable, thus enabling users to set up custom problems according to their defined specifications. Furthermore, users develop new reward functions. See [Rewards](rewards.md) for more information.

## Installation

`bluebird-gymnasium` is available on pypi, therefore it can be installed using

```
pip install bluebird-gymnasium
```

or, if using [UV](https://docs.astral.sh/uv/), you can add it to your environment using
```
uv add bluebird-gymnasium
```

## Simple usage

To instantiate an environment in the suite, for example, the X sector environment, enter the commands below (after the installation and activation of the virtual environment).

```python
import gymnasium as gym
import bluebird_gymnasium
env = gym.make("SectorXEnv-v0")
```

For more details, please see the Getting started section below.

### Sample agent

Below, an example agent that takes random actions.

```python
import gymnasium as gym
import bluebird_gymnasium

env = gym.make("SectorXEnv-v0")
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```


## Additional examples

Additional examples are available as [rendered jupyter notebooks](examples/simple_demo.ipynb).
