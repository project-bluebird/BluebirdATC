# Design of the digital twin

The digital twin code consists of:

### Core classes

Representing things like *Aircraft*, *Sector*, *Fix* etc. described above, as well as *Environment*, a container class that holds the current state of all the Aircraft and the Airspace in the simulation.

### Scenario and airspace generators

Code to define simulation scenarios, in very simple artificial sectors, or more complex/realistic airspaces.

### Simulation and event management

Keep track of and evolve the current state of the simulation, including writing to logfiles.

### Predictors

Simulate how aircraft will move from one time step to the next.

### Utility functions

Code to read and write data files, perform geometric calculations etc.
