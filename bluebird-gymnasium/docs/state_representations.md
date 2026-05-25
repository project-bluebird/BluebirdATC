# State representations

In `bluebird_gymnasium`, each aircraft within or near the boundary of a sector in an environment is encoded using a state representation function. The representation is generated as a 1D vector, using information about the aircraft from the digital twin. The complete state representation emitted by the environment to an agent differs between the single agent (centralised) and multi-agent (decentralised) set up.

In the single agent set up (where a single agent monitors and controls all aircraft), the set of 1D vectors across encoded aircraft states are concatenated into one global vector. Whereas, in the multi-agent set up, where a single aircraft is controlled by a single agent, an agent receives the encoded state for the aircraft it was assigned.


`bluebird_gymnasium` contains several predefined aircraft state representations in `bluebird_gymnasium/state_representations/`. The state representation to use in an environment instance can be specified via the environment configuration (see an [example](#usage) below). Also, the package supports the development of custom (user-defined) aircraft state representations (see the [custom state representations](#custom-state-representations) section below).


## Predefined Aircraft State Representations

The list below describes the pre-defined encodings that could be used to generate an aircraft's state representation.

- Full (`bluebird_gymnasium.state_repr.full`): uses the absolute scalar values of an aircraft such as heading, speed, flight level, entry flight level, exit flight level, sector entry position, sector exit position, and so on.
- Relative (`bluebird_gymnasium.state_repr.relative`): generates relative values from the absolute values of an aircraft's state. For example, the difference between flight level and exit flight level, distance to the exit location from the current position, and so on. The encoding enables a crude form of generalisation as it produces an environment/sector agnostic features (similar relative measures across different environments would produce similar features regardless of the properties of the environment/sector). 
- Minimal (`bluebird_gymnasium.state_repr.minimal`): defines a simple encoding that combines both relative and absolute features.
- Extra minimal (`bluebird_gymnasium.state_repr.extraminimal`): an even simpler version of `Minimal` with fewer features.

Note that each state representation comes in two variants, one with clipped range of feature values and the other with unbounded feature value range. For example, the relative representation contains both `RelativeRepresentationRaw` (the variant with unclipped features) and `RelativeRepresentation` (the variant with clipped features).


## Components

Aircraft state representations consist of at least two components. The first is a set of features that captures information about an aircraft's current state. The second component consists of features that capture information about the aircraft's relationships with other (neighbour) aircraft in the environment. For example, such features could include the distance from the aircraft to the other aircraft, the bearing from the aircraft to the other aircraft, the speed difference between both aircraft, and so on.

To keep a fixed feature size for the neighbour aircraft features, a parameter called `k_nearest_aircraft` is used to define the number of other aircraft to capture in the encoding. If there are no neighbour aircraft, then the features allocated for neighbour aircraft are filled with zeros. To disable the representation of neighbour aircraft, the `k_nearest_aircraft` parameter should be set to zero. See an example in the [usage](#usage) section below.


## Usage

To use a specific state representation (for aircraft encoding) in an environment instance, specify the state representation in the configuration used to instantiate the environment. In a sample environment, a `RelativeRepresentation` can be specified using:

```json
{
    ...
    "state_repr_config": {
        "encoder_cls": "relative",
        "k_nearest_aircraft": 0
    },
    ...
}
```


## Custom State Representations

To implement a custom aircraft state representation that can be specified in an environment configuration, follow the steps below:

1. Create a file in the `bluebird_gymnasium/state_repr/custom/`.

2. Implement your state representation class following the structure of the pre-defined state representations.

3. Import your state representation in `bluebird_gymnasium/state_repr/__init__.py and add the new state representation `registry_repr`.

4. Now you can add your aircraft state representation to the environment configuration like any other pre-defined ones.
