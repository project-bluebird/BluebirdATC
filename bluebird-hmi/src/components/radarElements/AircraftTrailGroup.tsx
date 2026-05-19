import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect, useMemo } from "react";
import { selectAircraft, selectTime } from "slices/dynamicDataSlice";
import { selectEvolvePeriod } from "slices/scenarioSlice";
import {
  Aircraft,
  ColourProfile,
  LineWidthProfile,
  OpacityProfile,
} from "utils/types";
import {
  getAircraftColour,
  getAircraftOpacity,
} from "utils/profiles/ColourSelectors";

interface AircraftTrailGroupProps {
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
  lineWidthProfile: LineWidthProfile;
  colourByFL: boolean;
  selectedColourScaleName: string;
  projection: GeoProjection;
  filterByFlightLevel: boolean;
}

interface TrailData {
  callsign: string;
  index: number;
  transform: string;
}

export default function AircraftTrailGroup(props: AircraftTrailGroupProps) {
  const {
    colourProfile,
    opacityProfile,
    lineWidthProfile,
    colourByFL,
    selectedColourScaleName,
    projection,
    filterByFlightLevel,
  } = props;

  const evolvePeriod = useAppSelector(selectEvolvePeriod);
  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const selectedAircraft: string = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  const time = useAppSelector(selectTime);
  const dragEndTimes = useAppSelector(
    (state) => state.dynamicData.dragEndTimes,
  );

  // Memoize color and opacity calculations
  const aircraftColours = useMemo(() => {
    return allAircraft.map((aircraft) => {
      const colour = getAircraftColour(
        aircraft,
        colourProfile,
        colourByFL,
        selectedColourScaleName,
        selectedAircraft,
      );
      return {
        callsign: aircraft.callsign,
        stroke: colour,
        fill: colour,
        opacity: getAircraftOpacity(
          allAircraft,
          aircraft,
          opacityProfile,
          selectedAircraft,
          filterByFlightLevel,
        ),
      };
    });
  }, [
    allAircraft,
    colourProfile,
    colourByFL,
    selectedColourScaleName,
    selectedAircraft,
    opacityProfile,
    filterByFlightLevel,
  ]);

  useEffect(() => {
    const aircraftTrailGroup = d3.select("#aircraftTrailGroup");

    if (allAircraft.length === 0) {
      aircraftTrailGroup.selectAll("*").remove();
      return;
    }

    const numberAircraftPositions = allAircraft[0].lats.length - 1;
    const numberTrailDots = 5; // Default number of trailing dots
    const trailLength = 30; // Length of trail [seconds]
    const dt = trailLength / numberTrailDots; // Time between trail dots [seconds]

    // Calculate the trail dot positions
    const calculateTransform = (aircraft: Aircraft, n: number): string => {
      const x = ((n + 1) * dt) / evolvePeriod;
      const m = Math.min(Math.floor(x), numberAircraftPositions - 1); // Clamp index to valid range

      const trail_lon =
        aircraft.lons[m] + (aircraft.lons[m + 1] - aircraft.lons[m]) * (x - m);
      const trail_lat =
        aircraft.lats[m] + (aircraft.lats[m + 1] - aircraft.lats[m]) * (x - m);
      const coords = projection([trail_lon, trail_lat]);
      return `translate(${coords[0]}, ${coords[1]})`;
    };

    // Prepare the trail dots data with a stable key based on callsign and dot index
    const trailData = allAircraft.flatMap((aircraft) =>
      Array.from({ length: numberTrailDots }, (_, n) => ({
        callsign: aircraft.callsign,
        index: n, // Dot index (n) is also part of the key to differentiate dots for the same aircraft
        transform: calculateTransform(aircraft, n),
      })),
    );

    // Join the data to the DOM, using callsign and dot index as the stable key
    const trailDots = aircraftTrailGroup
      .selectAll("circle")
      .data(trailData, (d: TrailData) => `${d.callsign}-${d.index}`);

    // Enter: Handle new dots
    trailDots
      .enter()
      .append("circle")
      .attr("class", (d) => `trail-dot-${d.callsign}`)
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 1)
      .attr("transform", (d) => d.transform)
      .attr("stroke", (d) => {
        const colorData = aircraftColours.find(
          (a) => a.callsign === d.callsign,
        );
        return colorData ? colorData.stroke : "none";
      })
      .attr("stroke-width", lineWidthProfile.previousPositionLineWidth)
      .attr("fill", "none")
      .attr("opacity", (d) => {
        const colorData = aircraftColours.find(
          (a) => a.callsign === d.callsign,
        );
        return colorData ? colorData.opacity : 1;
      });

    // Update: Update existing dots
    trailDots
      .attr("transform", (d) => d.transform)
      .attr("stroke", (d) => {
        const colorData = aircraftColours.find(
          (a) => a.callsign === d.callsign,
        );
        return colorData ? colorData.stroke : "none";
      })
      .attr("opacity", (d) => {
        const colorData = aircraftColours.find(
          (a) => a.callsign === d.callsign,
        );
        return colorData ? colorData.opacity : 1;
      });

    // Exit: Remove old dots
    trailDots.exit().remove();
  }, [
    allAircraft,
    evolvePeriod,
    projection,
    aircraftColours,
    lineWidthProfile.previousPositionLineWidth,
    time,
  ]);

  return <g id="aircraftTrailGroup" className="aircraft-trail-group" />;
}
