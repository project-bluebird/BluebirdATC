ATC clearances are issued by an Air Traffic Controller (ATCO) to an aircraft (i.e., to the pilot), which results in the modification of the flight trajectory of the aircraft. 
The instructions are broadly classified into lateral, vertical and speed instructions. 
Lateral instructions modify the heading or instruct the aircraft to fly on its defined route. 
Vertical instructions modify the altitude (measured in feet or flight level; `flight level = feet / 100`) of an aircraft. 
Speed instructions modify the calibrated airspeed (CAS) measured in knots, causing the aircraft to reduce or increase its speed. Instructions can be specified in absolute or relative terms. For example, if an aircraft’s current heading is 150 degrees and needs to fly 170 degrees, a relative instruction could be issued to turn the aircraft right by 20 degrees or an absolute instruction could be issued to set the aircraft’s heading to 170 degrees.

For information about the types of actions which can be issued and the values of these see [the API reference for the Action class](../bluebird-dt/source.md#bluebird_dt.core.Action.__init__)

**Value Range:** As the instructions modify the heading, altitude and speed of an aircraft, it is important to highlight the range of values common to commercial aircraft. The range of heading is between 0 and 360 degrees. The range of altitude depends on the sector’s geometry within a volume of airspace. However, for most real-world settings it falls within the range of 80 and 480 flight levels (8000 feet and 48000 feet). The range of speed is usually between 250 and 320 nautical miles per hour (knots).

**Value Constraint:** Although the heading, altitude, and speed are real/integer values, the issued instruction for each category is constrained to be in multiples of X. The value of a heading instruction should be in multiples of 5. The value of a vertical action should be in multiples of 10 flight level (1000 feet). The value of a speed action should be in multiples of 5.
