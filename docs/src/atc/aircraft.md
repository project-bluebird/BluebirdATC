### Aircraft

#### Aircraft
An aeroplane, helicopter, or other machine capable of flight. In its most minimal form, an aircraft has a position, heading, speed, and a callsign.

#### Callsign
A string identifier of an aircraft.

#### Position
Refers to the current position of an aircraft defined using the Geographic Coordinate System (GCS) made of latitude and longitude.

#### Route
Refers to a collection of fixes that an aircraft is required to fly through. A straight line, also known as a segment, is used to connect each two consecutive fixes in a route. The collection of segments defines the route's centreline.

#### Segment
Refers to a straight line that connects two consecutive fixes in a route.

#### Heading
Refers to the direction the nose of an aircraft is pointing towards. It is also known as a vector in ATC, and is measured in degrees.

#### Current Heading
Refers to the heading an aircraft is currently flying.

#### Cleared Heading
Refers to the heading that an aircraft has been instructed by an ATCO to fly. Whenever an aircraft is issued a heading instruction, it will continue to fly at the given angle (and thus may go off its defined route) until another instruction is issued to return the aircraft to its route. Therefore, when a heading instruction is issued, the route following status is false.

#### Selected Heading
Refers to the heading that the pilot of an aircraft chooses to fly after been issued a cleared heading instruction from an ATCO. This should be the same as the cleared flight level. However, in the real world, there may be reasons for a pilot to choose a different flight level (e.g., weather conditions, pilot error, etc.).

#### Entry Coordination
Refers to a term used to describe a group of information that describes the entry agreement of an aircraft into a sector.

#### Exit Coordination
Refers to a term used to describe a group of information that describes the exit agreement of an aircraft out of a sector.

#### Entry Fix
Refers to the fix at which an aircraft should enter a sector. This is a predetermined agreement contained in the entry coordination of the aircraft.

#### Exit Fix
Refers to the fix at which an aircraft should enter a sector. This is a predetermined agreement contained in the exit coordination of the aircraft.

#### Entry Flight Level
Refers to the flight level (altitude) of the aircraft that needs to be at when it enters a sector in the airspace. This is a predetermined agreement contained in the entry coordination of the aircraft.

#### Exit Flight Level
Refers to the flight level (altitude) of the aircraft that needs to be at when it leaves a sector in the airspace. This is a predetermined agreement contained in the exit coordination of the aircraft.

#### Current Flight Level
Refers to the flight level (altitude) of an aircraft at a given time. It is calculated using the aircraft's current altitude in feet divided by 100 (i.e., `feet / 100`).

#### Cleared Flight Level
Refers to the flight level (altitude) that an aircraft has been instructed by an ATCO to fly at.

#### Selected Flight Level
Refers to the flight level (altitude) that the pilot of an aircraft chooses to fly after being issued a cleared flight level instruction by an ATCO. This should be the same as the cleared flight level. However, in the real world, there may be reasons for a pilot to choose a different flight level (e.g., weather conditions, pilot error, etc.).

#### Route Following
Refers to a state in which the aircraft is being controlled by its pilot to navigate based on the list of defined fixes in its route. The aircraft is expected to fly from one fix to the next until it stops. Only when a [cleared heading](#cleared-heading) instruction is issued does the aircraft stop following its route. In such a scenario, the pilot is expected to fly the aircraft at the angle that was issued in the cleared heading instruction.

#### Route Centre
Refers to the collection of straight lines connecting the consecutive fixes of an aircraft's [route](#route). For example, an aircraft that is positioned between two of its route fixes (i.e., the aircraft is at a [segment](#segment)) and is flying exactly one the line that connect the fixes, then the aircraft is positioned on the route's centre. Otherwise, the aircraft is flying off its route centreline. Note that when an aircraft is route following, then it will be fly on its route's centre.

#### Filed Route
Refers to the predetermined flight path (collection of fixes) of the aircraft, based on the pilot's original request. This is usually part of the aircraft's flight plan information.

#### Current Route
Refers to the route that the aircraft is currently flying at a given time. It is usually the same as the filed route or a subset of it if the aircraft had previously gone off route (e.g., due to a cleared heading instruction) and a route direct instruction was issued to return the aircraft back to its route.

#### Indicated Airspeed
Indicated Airspeed (IAS) refers to the raw speed read directly from the aircraft's cockpit instruments. It serves as a good measure of the aircraft's speed at lower altitudes, and for takeoff and landing situations. It is measured in [knots](#knots).

#### Calibrated Airspeed
Calibrated Airspeed (CAS) refers to IAS readings corrected for instrumentation and positioning errors. Hence, relative to IAS, CAS is a more accurate measure of an aircraft's speed. It is measured in [knots](#knots).

#### Cleared Calibrated Airspeed
Refers to the CAS that an aircraft has been instructed by an ATCO to fly. It is measured in [knots](#knots).

#### Selected Calibrated Airspeed
Refers to the CAS that the pilot of an aircraft chooses to fly after being issued a cleared CAS instruction by an ATCO. This should be the same as the cleared CAS. However, in the real world, there may be reasons for a pilot to choose a different CAS (e.g., weather conditions, pilot error, etc.). It is measured in [knots](#knots).

#### True Airspeed
True Airspeed (TAS) refers to the speed of an aircraft relative to the air mass that it is flying through. For a constant CAS, TAS increases or decreases as an aircraft climbs (increases altitude) or descend (decreases altitude). It is measured in [knots](#knots).

#### Ground Speed
Ground speed refers to the speed of an aircraft relative to the ground. It can be viewed as the correction of TAS accounting for wind (i.e., `TAS +/- wind`, -ve for headwind and +ve for tailwind). It is measured in [knots](#knots).

#### Knots
Refers to the unit of measure of indicated airspeed, calibrated airspeed, true airspeed, and ground speed of an aircraft. Knots stands for "Nautical Miles per Hour".

#### Vertical Speed
Refers to the speed at which an aircraft changes its altitude when climbing or descending. It is measured in feet per minute (FPM).

#### Mach Number
Refers to the speed of an aircraft measured as the ratio of its true airspeed to the speed of sound. It serves as the primary measure of an aircraft's speed at higher altitudes (typically above 240 flight level), where the aircraft's indicated airspeed measurement becomes unreliable due to atmospheric and temperature conditions. Being a decimal ratio, the measurement is simply known as "Mach".

#### Cleared Mach
Refers to the Mach that an aircraft has been instructed by an ATCO to fly.

#### Selected Mach
Refers to the Mach that the pilot of an aircraft chooses to fly after being issued a cleared Mach instruction by an ATCO. This should be the same as the cleared Mach. However, in the real world, there may be reasons for a pilot to choose a different Mach (e.g., weather conditions, pilot error, etc.).

#### Tailwind
Refers to wind that blows in the direction of an aircraft's travel, pushing it from behind. This push increases the ground speed but does not change how the plane flies through the air (TAS).

#### Headwind
Refers to wind that blows against the direction of an aircraft's travel, pushing it from the front (or pulling it from behind). This reduces the ground speed of the aircraft but does not change how the plan flies through the air (TAS).

#### Bank Angle
Refers to the angle between an aircraft's wings and the horizon (horizontal plane, the imaginary line where the sky meets the ground or sea), which represents a sideways tilt during a turn. It is measured in degrees.

Heuristically, the bank angle is usually measured as 15% of [TAS](#true-airspeed) in order to approximately achieve the standard 3 degrees rate of turn. For example, an aircraft flying at a TAS of 100 knots will require a bank angle of 15 degrees. Note that slower aircraft (lower TAS) require a shallow bank angle, and vice versa.

#### Rate of Turn
Rate of turn (or turn rate) refers to the speed at which an aircraft changes its heading. The default heuristic popularly used in ATC is that aircraft turn at a constant rate of 3 degrees per second. Therefore, a full 360-degree turn would take 2 minutes.

Turns can stem from an aircraft requiring a turn when navigating to its next fix during [route following](#route-following) or when a heading instruction is issued by an ATCO where the new heading is different from the aircraft's current heading.


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
