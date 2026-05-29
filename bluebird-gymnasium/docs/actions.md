# Actions

The action space for all environments in `bluebird_gymnasium` is defined as discrete. The actions are configurable and are specified using the available Air Traffic Control (ATC) instruction set defined in `bluebird_dt`.


In the `bluebird_dt` digital twin, the simulator has been designed to reflect the real-world ATC instructions. For further information, please refer to the [bluebird_dt action page](../atc/clearances.md). Also, please see the basic [introduction to real-world ATC instructions](../atc/clearances.md) for more information.

The ATC instructions serve as the action set exposed to an agent to perform the task of ATC.

## Instruction to Bluebird Gymnasium Discrete Action

The ATC instructions in the `bluebird_dt` simulator are mapped to discrete actions in `bluebird_gymnasium` using the `ActionParser` class that parses the actions specified in the environment configuration (see [Action Configuration](#action-configuration) below). By default, action 0 maps to a no operation (`NOOP`; also known as do nothing) action. The active actions specified in the action configuration then incrementally map to other integer values starting from 1.

The action mapping differs slightly for the single (centralised) and multi-agent (decentralised) setups. For multi-agent, aircraft are monitored and controlled by separate agents (one agent to one aircraft). Therefore, the total number of discrete actions for each agent is `1 + number of active actions` (i.e., one extra action for the `NOOP` action). For single-agent, there is a one agent to N aircraft mapping (with N specified by the `num_sampled_aircraft` in `centralized_params` in `view_config`). Therefore, the total number of discrete actions for such an agent is `1 + (N x number of active actions)`.


## Action Configuration

### Lateral Actions

- `simple_heading_left`: A relative heading action (measured in degrees) that instructs an aircraft to turn left. It can be set to a list of integer values in multiples of 5 for an environment instance or a boolean value `True` to map to a default list `[10]` or `False` to disable the action.
- `simple_heading_right`: A relative heading action (measured in degrees) that instructs an aircraft to turn right. It can be set to a list of integer values in multiples of 5 for an environment instance or a boolean value `True` to map to a default list `[10]` or `False` to disable the action.
- `simple_heading_fly_parallel`: An absolute heading action (measured in degrees) that instructs an aircraft to fly parallel to a segment of its route. It can be set as a list of integers or a boolean. As a list of integers, it starts from 1 and increments by 1, with each index denoting a segment of the route, starting with the current segment. As a boolean, it is disabled when set to `False`. When set to `True`, it instructs the aircraft to fly parallel to the current route segment (defaults to list `[1]`). Activates or deactivates the action based on a boolean value.
- `simple_heading_maintain_current`: An action that instructs an aircraft to maintain its current heading.
- `simple_route_direct`: An action that instructs an aircraft to fly on its defined route. It is useful when an aircraft has been instructed to fly on a heading that takes it off-route. It can be set as a list of integers or a boolean. As a list of integers, it starts from 1 and increments by 1, with each index denoting the forward fix to route direct to. As a boolean, it is disabled when set to `False`. When set to `True`, it instructs the aircraft to its next forward route fix and continue on-route (defaults to list `[1]`).

### Vertical Actions

- `simple_fl_descent`: A relative action that instructs an aircraft to descend below its current altitude. Set to a list of integers in multiples of 10 for an environment instance, or `True` for a default list `[10]`.
- `simple_fl_climb`: A relative action that instructs an aircraft to climb above its current altitude. Set to a list of integers in multiples of 10 for an environment instance, or `True` for a default list `[10]`.
- `simple_fl_intermediate`: An absolute action that instructs an aircraft to climb or descend to an intermediate altitude that avoids safety violations. Activates or deactivates the action based on a boolean value.
- `simple_fl_exit`: An absolute action that instructs an aircraft to climb or descend to its exit flight level. Activates or deactivates the action based on a boolean value.

### Speed Actions

- `simple_speed_decrease`: A relative speed action that instructs an aircraft to reduce its speed below its current calibrated airspeed (CAS). It can be set to a list of integer values in multiples of 10 for specific instructions or a boolean value to disable the action.
- `simple_speed_increase`: A relative speed action that instructs an aircraft to increase its speed above its current calibrated airspeed (CAS). It can be set to a list of integer values in multiples of 10 for specific instructions or a boolean value to disable the action.
- `simple_speed_maintain_current`: An absolute action that instructs an aircraft to maintain its current calibrated airspeed (CAS). It can be activated or deactivated based on a boolean value.
- `simple_speed_choose_own`: An absolute action that instructs an aircraft to choose its own current calibrated airspeed (CAS). It can be activated or deactivated based on a boolean value.

### Other Actions

- `simple_outcomm`: defines an action that instructs an aircraft to transfer an aircraft out of the sector.


## Sample Configuration

The actions specification below will define the action space for a gymnasium environment instantiated using the configuration. Note that an any action not specified in the configuration is disabled by default.

```json
{
    ...
    "action_config": {
        "simple_heading_left": [
            10,
        ],
        "simple_heading_right": [
            10,
        ],
        "simple_heading_fly_parallel": True,
        "simple_fl_descent": False,
        "simple_fl_climb": False,
        "simple_fl_intermediate": False,
        "simple_fl_exit": False,
        "simple_speed_increase": False,
        "simple_speed_decrease": False,
        "simple_route_direct": False,
        "simple_outcomm": False,
    },
    ...
}
```

The above sample configuration defines an action space that can issue instructions to:
- turn aircraft left by 10 degrees,
- turn aircraft right by 10 degrees,
- and set the heading to the angle of the current segment of the route.

In a multi-agent set up, the action space for the above configuration will be set to 4 discrete actions (`3 + 1 NOOP action`) per agent. For a single agent set up, with `num_sampled_aircraft` set to 4 (for example), the action space will be set to 13 discrete actions (i.e., `(4 aircraft x 3 actions) + 1 NOOP action`).
