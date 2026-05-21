# Environment Configuration

The environment configuration contains a set of parameters that define how a `bluebird_gymnasium` environment is set up when instantiated. It is defined in the class `bluebird_gymnasium.envs.EnvConfig`.

## Default Configuration
Each environment contains a default environment configuration which can be retrieved by calling the class method `get_default_env_config`.

The sample code snippet below fetches the default configuration for the `SectorXPlusEnv` environment.

```python
from bluebird_gymnasium.envs import SectorXPlusEnv
env_config = SectorXPlusEnv.get_default_env_config()
```

## Instantiate an Environment

`bluebird_gymnasium` environments are instantiated with an environment configuration. If unset at the environment instance creation, the default configuration is used.

The sample code snippet below demonstrates how to instantiate the `SectorXPlusEnv` environment.

```python
from bluebird_gymnasium.envs import SectorXPlusEnv
env_config = SectorXPlusEnv.get_default_env_config()

# method 1: direct object creation interface
env_1 = SectorXPlusEnv(config=env_config)

# method 2: gymnasium `make` function interface
import gymnasium as gym
import bluebird_gymnasium.envs
env_2 = gym.make("SectorXPlusEnv-v0", config=env_config)
```

## Save and Load Configurations from Disk
The configuration can be saved to and loaded from a disk as a JSON file.

The sample code snippet below saves a configuration to disk.
```python
import json
from dataclasses import asdict

from bluebird_gymnasium.envs import SectorXPlusEnv

env_config = SectorXPlusEnv.get_default_env_config()

with open("env_config.json", "w") as fp:
    json.dump(asdict(env_config), fp, indent=4)
```

The sample code snippet below loads a configuration from disk.
```python
import json

from bluebird_gymnasium.envs import EnvConfig

with open("env_config.json", "r") as fp:
    env_config_dict = json.load(fp)
    env_config = EnvConfig(**env_config_dict)
```

## Configuration Components

Each configuration comprises sub-configurations, primarily defined as dictionaries to configure various aspects of the environment.

### Agent Focused Sub-Configurations
- `.action_config`: defines the action space for an environment instance.
- `.reward_config`: defines a set of reward functions and their respective weights used to compute the scalarized reward per aircraft per step.
- `.state_repr_config`: defines the state encoding method to use in representing each aircraft state.
- `.view_config`: defines the parameters that set up the gymnasium environment for either a single-agent (centralized) or multi-agent (decentralized).
- `.forward_fixes_config`: defines the parameters used to configure an aircraft's route information based on the number of forward fixes. The config is exploited by the actions execution and aircraft state representation.

### Digital Twin Related Sub-Configurations

- `.airspace_config`: defines the airspace parameters used by the underlying `bluebird_dt` digital twin.
- `.radar_config`: defines the parameters used for visualizing the digital twin via a matplotlib-based or SVG plot.
- `.scenario_config`: defines the parameters used to instantiate a scenario to run in the `bluebird_dt` digital twin.
- `.simulation_log_config`: defines parameters used for specifying logging in the digital twin.

### Other Non Sub-Configuration Parameters

Other parameters in the environment configuration include:
- `.scenario_duration`: defines the total time (in seconds) to run a scenario within an episode.
- `.scenario_sec_per_step`: defines the time duration (in seconds) to elapse per simulation step. Defaults to 6. Note, the number of RL steps within an episode is computed as: `scenario_duration / scenario_sec_per_step`.
- `.diagnostic_level`:
