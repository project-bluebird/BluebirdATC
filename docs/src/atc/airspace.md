# Airspace

The airspace above is made up of different concepts to provide structure to the task of air traffic control.

The Bluebird digital twin encapsulates only those concepts relevant for upper airspace, frequently above FL195, intentionally avoiding concepts only relevant to approach to airports.

## Controlled airspace

Controlled airspace is that which requires pilots to obtain ATC clearance to enter, maintain two-way communication and comply with ATC instructions.

### Uncontrolled airspace

Uncontrolled airspace is any airspace that is not controlled airspace.

## Sector

Refers to a collection of volumes, pre-defined three-dimensional polyhedron, managed by one controller.
Aircraft must stay within their sector before their exit co-ordination is reached.

The sectors in the Bluebird digital twin are all assumed to be controlled airspace, and therefore aircraft must followed all the instructions issued by air traffic control.

## Waypoint

Refers to a specific geographical location (defined using latitude and longitude) used as a "landmark" in an airspace.
An aircraft's route will be made up of these
Air traffic controllers may issue clearances for aircraft to fly to a waypoint in their route.

### Fix

A fix represents exact geographical coordinates (latitude and longitude) in space.

### VOR

A VOR (VHF Omnidirectional range) emits 360 radials radiating outward like spokes on a bicycle wheel.
By turning to a VOR, a pilot can see exactly what radial they are on and fly directly towards or away from the station.

In the context of the Bluebird digital twin, no difference is made between a VOR and a FIX.

### NDB

A NDB (Non-Directional Beacon) is a low frequency radio transmitter that sends signals in all directions such that an aircraft system's can point the aircraft directly to the beacon.

In the context of the Bluebird digital twin, no difference is made between a NDB and a FIX.
