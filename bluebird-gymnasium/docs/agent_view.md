# Agent View

Each environment in `bluebird_gymnasium` can support single agent (centralized) or multi-agent (decentralized) views.

In a single agent view, an environment expects a single external agent interacting with it. The `step` method expects a scalar integer for the `action` argument. The `step` and `reset` methods produce a single quantity per step for the observed state, reward, done flag, truncated flag, and a Python dictionary of auxiliary information.

In a decentralised multi-agent environment, multiple agents interact with it. The environment expects a Python dictionary of actions from the agents, where each key is an agent id and each value is an action. The `step` and `reset` methods return a dictionary of observed state, rewards, done flags, truncated flags, and auxiliary information. Each dictionary contains a key-value pair for an agent, with the key being the agent id.



## State Representation

### Single Agent View
The environment allows an external agent to monitor and control all controllable aircraft in the scenario at any time step. At each time step, the environment produces a state by concatenating the state encoding vectors of all aircraft present in the scenario (subject to the constraint below).

**Constraint:**
The single agent must monitor and control all controllable aircraft at any time step, but this is problematic due to the variable number of aircraft. Therefore, the agent designer must choose the maximum number of aircraft the agent expects to monitor and control at any given time step, which is specified in the environment configuration using the `num_sampled_aircraft` parameter. Also, a sampling strategy, specified using the `sample_strategy` parameter, determines the ordering of aircraft encodings in the state representation (see the different types of sampling strategy below).

If the number of aircraft in a scenario exceeds the specified `num_sampled_aircraft`, only a subset is exposed to the agent in the state representation and for control. The chosen subset is determined by the `sample_strategy`. Conversely, if the number of aircraft is less than `num_sampled_aircraft`, all are selected and ordered by the `sample_strategy`. Then, the global state representation is a concatenation of aircraft encodings, padded with zeros to match the fixed state representation size. Zero-padded features represent ‘dummy aircraft’. If the agent selects an action for a dummy aircraft, it’s ignored for the given time step.

**Selection Strategy:**
- "earliest_entries": aircraft are selected and ordered based on the time step they were first included in the state representation. Ordered in ascending order. It is represented by the constant `CentralizedSampler.EARLIEST`.
- "latest_entries": aircraft are selected and ordered based on the time step they were first included in the state representation. Ordered in descending order. It is represented by the constant `CentralizedSampler.LATEST`.
- "random_step": aircraft are selected and ordered based at random per time step. It is represented by the constant `CentralizedSampler.RANDOM_STEP`.
- "random_episodal": an order mask is stochastically generated at the start of the episode, kept fixed and applied at every time step during the episode to order the aircraft. The mask is only regenerated at the start of each new episode. It is represented by the constant `CentralizedSampler.RANDOM_EPISODAL`.

Note: `CentralizedSampler` is located in `bluebird_gymnasium.envs.__init__.py`

 ### Multi-Agent View
In the multi-agent (decentralized) set up, where a single aircraft is controlled by a single agent, an agent receives the encoded state for the aircraft it was assigned.


## Usage

To set up an environment instance as a single agent view, see sample configuration snippet below.

```json
{
    ...
    "view_config": {
        "type": "centralized",
        "centralized_params": {
            "num_sampled_aircraft": 5,
            "sample_strategy": latest_entries
        },
    },
    ...
}
```

To set up an environment instance as a multi-agent view, see sample configuration snippet below.

```json
{
    ...
    "view_config": {
        "type": "decentralized",
        "decentralized_params": {},
    },
    ...
}
```
