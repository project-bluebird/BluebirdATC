# Aircraft

An aircraft is a physical aeroplane, helicopter, or other machine capable of flight.
In this documentation though, we will refer to aircraft undergoing a flight, therefore having a flight plan.

## Aircraft properties

#### Callsign
A callsign, sometimes referred to as aircraft identification, is an string identifier for an individual flight which are used by ATC for radiotelephony communication.

The field usually consists of the aircraft registration or the company designator operating the aircraft followed by a flight number.

#### Position
Refers to the current position of an aircraft defined using the Geographic Coordinate System (GCS), expressed as made of latitude and longitude coordinates.

#### Aircraft type

The aircraft type is the manufacturer's designator for the model. A list of aircraft type designators is available [in the ICAO website](https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search).

#### Current Heading
The direction in which the longitudinal axis of an aircraft is pointed, expressed in degrees from North.

#### Selected Heading
Refers to the heading that the pilot of an aircraft has instructed the aircraft's flight computer to fly. In the digital twin this will be driven by instructions from the agent. However, in the real world, due to pilot error or different flight modes, this may not always be correct or available.

#### Track 
The projection on the Earth’s surface of the path of an aircraft, the direction of which path at any point is usually expressed in degrees from North.

The heading of an aircraft may be different than its track due to the wind. This difference is called drift.

#### Current Flight Level
Refers to the [flight level](./index.md#flight-levels) of an aircraft at a given time.

#### Selected Flight Level
Refers to the [flight level](./index.md#flight-levels) that the pilot has instructed the aircraft's flight computer to fly, and therefore the level the aircraft will fly to. This should be the same as the cleared flight level. However, in the real world, a pilot may choose a different flight level.

#### Route Following
Refers to a state in which the aircraft is being controlled by its pilot to navigate based on the list of defined fixes in its route. The aircraft is expected to fly from one fix to the next until it stops. Only when a [cleared heading](#cleared-heading) instruction is issued does the aircraft stop following its route. In such a scenario, the pilot is expected to fly the aircraft at the angle that was issued in the cleared heading instruction.

#### Indicated Airspeed
Indicated Airspeed (IAS) refers to the raw speed read directly from the aircraft's cockpit instruments, measured in [knots](index.md#knots).

#### Calibrated Airspeed
Calibrated Airspeed (CAS) refers to IAS readings corrected for instrumentation and positioning errors. Hence, relative to IAS, CAS is a more accurate measure of an aircraft's speed. It is measured in [knots](index.md#knots).

#### Selected Calibrated Airspeed
Refers to the CAS that the pilot of an aircraft chooses to fly after being issued a cleared CAS instruction by an ATCO. This should be the same as the cleared CAS. However, in the real world, there may be reasons for a pilot to choose a different CAS (e.g., weather conditions, pilot error, etc.). It is measured in [knots](index.md#knots).

Note that in reality, aircraft will not report their selected calibrated airspeed, rather reporting their selected indicated airspeeds, but this is a simplification which has been made in the Digital Twin.

#### True Airspeed
True Airspeed (TAS) refers to the speed of an aircraft relative to the air mass that it is flying through measured in [knots](index.md#knots). 
Compared to calibrated air speed, TAS accounts for the effects of air compressibility.
For a constant CAS, as air density decreases with increasing altitude, the TAS increases.

#### Ground Speed
Ground speed refers to the speed of an aircraft relative to the ground. It can be viewed as the correction of TAS accounting for wind, measured in [knots](index.md#knots).

$$\overrightarrow{GS} = \overrightarrow{TAS} + \overrightarrow{wind}$$

#### Vertical Speed
Refers to the speed at which an aircraft changes its altitude when climbing or descending. It is measured in feet per minute (FPM).

#### Mach Number
Refers to the speed of an aircraft measured as the ratio of its true airspeed to the speed of sound. It serves as the primary measure of an aircraft's speed at higher altitudes (typically above 240 flight level), where the aircraft's indicated airspeed measurement becomes unreliable due to atmospheric and temperature conditions. Being a decimal ratio, the measurement is simply known as "Mach".

## Flight plan

A flight plan, as described by ICAO Annex 2, provides information relative to an intended flight or portion of a flight, to be provided to air traffic service units.

For simplicity, the digital twin does not work with full ICAO flight plans, only a simplification with the elements described below.

#### Unexpanded route

The unexpanded route is the aircraft's route description using standard ICAO abbreviations, waypoint and airways.
This is what would be filed as part of Item 15 of an ICAO flight plan.

#### Origin

The ICAO code of the departure airport the aircraft is flying from.

#### Destination

The ICAO code of the destination airport the aircraft is flying to.

#### Requested flight level

The requested cruise flight level the aircraft wants to climb to initially.
The digital twin currently only supports a single requested flight level for each flight plan, restricting flight planned requestes to change level.

#### Route
Refers to a series of fixes that an aircraft is required to fly through.

##### Filed Route
Refers to the predetermined flight path (collection of fixes) of the aircraft, based on the pilot's original request. This is usually part of the aircraft's flight plan information.

##### Current Route
Refers to the route that the aircraft is currently flying at a given time. It is usually the same as the filed route or a subset of it if the aircraft had previously gone off route (e.g., due to a cleared heading instruction) and a route direct instruction was issued to return the aircraft back to its route.

#### Filed true air speed

The requested true airspeed the aircraft wants to fly initially upon reaching their cruise altitude.
The digital twin currently only supports a single requested flight level for each flight plan, restricting flight planned requestes to change true airspeed.

## ATC information

#### Cleared Flight Level
Refers to the flight level (altitude) that an aircraft has been instructed by an ATCO to fly at.

#### Cleared Calibrated Airspeed
Refers to the CAS that an aircraft has been instructed by an ATCO to fly. It is measured in [knots](index.md#knots).

#### Cleared Mach
Refers to the Mach that an aircraft has been instructed by an ATCO to fly.

#### Cleared Heading
Refers to the heading that an aircraft has been instructed by an ATCO to fly. Whenever an aircraft is issued a heading instruction, it will continue to fly at the given angle (and thus may go off its defined route) until another instruction is issued to return the aircraft to its route. Therefore, when a heading instruction is issued, the route following status is false.

## Additional concepts

#### Bank Angle
Refers to the angle between an aircraft's wings and the horizon (horizontal plane, the imaginary line where the sky meets the ground or sea), which represents a sideways tilt during a turn. It is measured in degrees.

Heuristically, the bank angle is usually measured as 15% of [TAS](#true-airspeed) in order to approximately achieve the standard 3 degrees rate of turn. For example, an aircraft flying at a TAS of 100 knots will require a bank angle of 15 degrees. Note that slower aircraft (lower TAS) require a shallow bank angle, and vice versa.

#### Rate of Turn
Rate of turn (or turn rate) refers to the speed at which an aircraft changes its heading. The default heuristic popularly used in ATC is that aircraft turn at a constant rate of 3 degrees per second. Therefore, a full 360-degree turn would take 2 minutes.

Turns can stem from an aircraft requiring a turn when navigating to its next fix during [route following](#route-following) or when a heading instruction is issued by an ATCO where the new heading is different from the aircraft's current heading.

The digital twin assumes all turns are fly-by turns, meaning the aircraft begins turning before reaching the waypoint, crating a smoth, arced path to the next route segment.

#### Top Of Descent
The top of descent (TOD) for an aircraft descending to a flight level before exiting a sector (current flight level > exit flight level) is the location in its flight trajectory where an instruction to descend to the exit flight level needs to be issued. It can be calculated by first determining the distance from the exit location where the descent should begin, and then projecting backwards from the exit location to the TOD location based on this distance.

Why not instruct an aircraft to descend immediately upon entering the sector?

It is generally preferred to fly aircraft at higher altitudes as it is more efficient (one of ATC’s goals). Therefore, it is usually better to fly them at higher altitudes until it’s absolutely necessary to descend. However, aircraft may sometimes be required to descend early than their target descent altitude to resolve a conflict with another aircraft that could lead to a safety breach.

There are heuristics used to estimate the top of descent based on the altitude difference between the aircraft’s current altitude and its exit flight level. Without accounting for wind, a simple heuristic is:

```
((exit flight level - current flight level) / 10) * 3
```

For example, an aircraft at current flight level 320 (i.e., 32,000 feet) and an exit flight level of 280 (i.e., 28,000 feet) would have a top of descent distance from the exit location as:
```
tod = ((320 - 280) / 10) * 3 = 12 nautical miles
```

See more information about [top of descent here](https://pilotinstitute.com/how-to-calculate-descent/).

#### Trajectory Prediction (TP)
Refers to the process of predicting the future flight path of an aircraft given its current position and other information such as heading, route following status, flight level, selected flight level, and so on. The prediction is executed using a model which can either be mathematically defined or data-driven.
