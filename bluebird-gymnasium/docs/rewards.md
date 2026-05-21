# Reward Functions

The goal of a reinforcement learning agent (single or multi-agent) or any agent for tactical air traffic control (ATC) is to monitor aircraft flying and issue instructions (when necessary) to ensure **safe, orderly and efficient** navigation of aircraft through a given sector (defined as an environment in bluebird_gymnasium). Such behaviours are induced via reward functions that the agent seeks to optimise. Each reward function returns a scalar reward signal that can be positive (to incentivize good behaviours) or negative (to penalize bad behaviours). At any given time step, a reward value is computed for each aircraft within the sector. The final reward returned by the environment for the time step is the mean of the rewards across all aircraft.

Given the nature of the problem, an agent usually optimises multiple rewards objectives that are linearly combined to achieve the ATC goal. Common behaviours are safety, reduce fuel burn per aircraft, fly an efficient track path to reduce distance travelled through a sector (shortest part), fly aircraft at their preferred flight level and so on. Each reward induces certain behaviours. However, not all rewards/objectives are equal. For example, safety is the most critical objective of ATC. Thus, the linear combination of rewards are weighted, with the most important reward (e.g., safety) being assigned the highest weight and vice versa.

To support the development of agents, a set of common ATC reward functions has been implemented within `bluebird_gymnasium`, which researchers can leverage (see [pre-defined reward functions](#predefined-reward-functions) for more information). In addition, the repository supports the implementation of user-defined (custom) reward functions (see the [custom reward functions](#custom-reward-functions) for more information).


## Predefined Reward Functions

The reward functions are located in `bluebird_gymnasium.rewards`, spanning across safety, conflict resolution, efficient navigation, instruction/action discipline, optimal flight profile (climb or descend), and route centre navigation. Within each objective, one or more reward function variants have been implemented, which can be exploited by researchers.

- Safety (`bluebird_gymnasium.rewards.safety`): Contains a collection of safety reward functions that penalise the agent for any aircraft that is in an unsafe (relative to other aircraft) state. The minimum safety requirement is defined as aircraft being laterally separated by at least 5 nautical miles distance or vertically separated by at least 1000 feet (aka 10 flight level) at any given time.
- Efficient navigation (`bluebird_gymnasium.rewards.expeditious`): Reward or penalise an agent for an aircraft navigating efficiently (reduced time or track miles within a sector) or inefficiently (increased time or track miles within a sector).
- Action discipline (`bluebird_gymnasium.rewards.action_penalty`): Penalise an agent for spamming actions/instructions or repeating unnecessary instructions to aircraft.
- Optimal flight profile (`bluebird_gymnasium.rewards.climb_descent`): Contains a collection of reward functions that reward or penalise an agent for aircraft either flying at their preferred flight level or not. When an aircraft enters a sector, it may need to exit the sector at a different flight level or maintain its current flight level at the exit. The flight level that an aircraft is required to exit the sector is referred to as its exit flight level defined in the [coordination](../atc/coordinations.md). Depending on the aircraft's [current flight level](../atc/aircraft.md#current-flight-level) and its defined exit flight level, an agent may need to climb (increase their flight level) or descend (reduce their flight level) or overfly (maintain their flight level) before the exit. If an aircraft needs to climb, then the goal is to climb as early as possible. However, if an aircraft needs to descend before exiting the sector, the aircraft should fly at a higher flight level for as long as possible, then start descending at a location that will enable the aircraft to reach the target flight level **just before exiting the sector** (this is known as [top of descent](https://pilotinstitute.com/how-to-calculate-descent/)).
- Route centre navigation (`bluebird_gymnasium.rewards.lateral_centreline_distance`): Contains a collection of reward functions that reward the agent for aircraft that fly on (or stay close to) their route centre. If an aircraft deviates significantly away from its route centre, then the agent is penalised.
- Conflict resolution (`bluebird_gymnasium.rewards.conflict_resolution`): Contains a collection of reward functions that penalise an agent for not resolving a conflict between an aircraft and another aircraft that prevents them from achieving their optimal flight profile.
- Position status (`bluebird_gymnasium.rewards.position`): Contains a collection of reward functions that penalise an agent when an aircraft incorrectly leaves the sector (i.e., not exiting the sector at the correct location).


## Usage

As earlier mentioned, an agent needs to optimise for several objectives expressed as multiple reward functions that are linearly combined in a weighted fashion. The reward function(s) to use is specified in the environment configuration (which is used to instantiate an environment). The environment configuration is specified in a JSON file which is loaded from disk. See an example of a sample configuration for the `Springfield` environment, with emphasis on the reward configuration section.

```json
{
    ...
    "reward_config": {
        "fns": [
            "position_status_const",
            "lateral_centreline_distance_linear",
            "safety_simple_avoidance_exp",
        ],
        "coeffs": [1.0, 1.0, 2.0],
    },
    ...
}
```

In the above sample configuration, the reward functions to use are defined in the `"fns"` key of the dictionary which contains a list of pre-defined reward functions. In addition, the weight of each reward function in the linear combination is specified in `"coeffs"`. Note that the number of elements in the `"fns"` list should match that of `"coeffs"` list.

An instantiated environment (using the configuration) will produce a reward per aircraft that is based on incorrect sector exit penalisation, penalisation for flying away from the defined route centre of the aircraft and penalisation for an safety violation. In essence, an agent is optimised to prevent any safety violation, ensure aircraft on or close to its defined route and ensure aircraft correctly exits the sector.


## Custom Reward Functions

To create a user-defined (custom) reward function that can be specified in an environment configuration and used, follow the steps below:

1. Create a file (python module) in the reward function package, in `bluebird_gymnasium/rewards/custom/`.

2. Implement your reward function(s) logic in the file , with the following parameters: the gym environment, callsign (the identifier of the aircraft to evaluate by reward function), action (taken by the agent at the current time step) and a python `kwargs`

3. Import your reward function in `bluebird_gymnasium/rewards/__init__.py` and add the reward function to `registry_reward_fn`.

4. Now you can add your reward function to the environment configuration like any other pre-defined reward function.
