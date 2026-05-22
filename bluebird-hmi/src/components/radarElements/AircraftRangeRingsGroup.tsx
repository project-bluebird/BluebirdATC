import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useCallback, useEffect, useMemo } from "react";
import { selectAircraft } from "slices/dynamicDataSlice";
import {
  Aircraft,
  ColourProfile,
  LineWidthProfile,
  OpacityProfile,
  ToolCallsignStore,
} from "utils/types";
import {
  getRangeRingPixelRadius,
  nauticalMilesToKm,
} from "utils/DistanceUtils";
import {
  getAircraftColour,
  getAircraftOpacity,
} from "utils/profiles/ColourSelectors";

interface AircraftRangeRingsGroupProps {
  toolCallsigns: ToolCallsignStore;
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
  lineWidthProfile: LineWidthProfile;
  colourByFL: boolean;
  selectedColourScaleName: string;
  projection: GeoProjection;
  filterByFlightLevel: boolean;
}

interface AircraftWithCoords extends Aircraft {
  coords: [number, number];
}

export default function AircraftRangeRingsGroup(
  props: AircraftRangeRingsGroupProps,
) {
  const {
    toolCallsigns,
    colourProfile,
    opacityProfile,
    lineWidthProfile,
    colourByFL,
    selectedColourScaleName,
    projection,
    filterByFlightLevel,
  } = props;

  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const selectedAircraft: string = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );

  // Memoise the nm to km calc
  const rangeRingRadiusKm = useMemo(() => nauticalMilesToKm(5), []);

  // Memoise range ring pixel radius
  const memoizedGetRangeRingPixelRadius = useCallback(
    (aircraft: Aircraft) => {
      return getRangeRingPixelRadius(projection, aircraft, rangeRingRadiusKm);
    },
    [projection, rangeRingRadiusKm], // Dependencies: projection and rangeRingRadiusKm
  );

  // Memoise the projection calcs on the filtered aircraft
  const filteredAircraft = useMemo(
    () =>
      allAircraft.filter((aircraft) =>
        toolCallsigns.rangeRings.includes(aircraft.callsign),
      ),
    [allAircraft, toolCallsigns.rangeRings],
  );

  const allAircraftWithCoords: AircraftWithCoords[] = useMemo(() => {
    return filteredAircraft.map((aircraft: Aircraft) => ({
      ...aircraft,
      coords: projection([aircraft.lon, aircraft.lat]),
    }));
  }, [filteredAircraft, projection]);

  // Memoise colour and opacity calculations
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
    const aircraftRangeRingsGroup = d3.select("#aircraftRangeRingsGroup");

    aircraftRangeRingsGroup
      .selectAll("circle")
      .data(allAircraftWithCoords, (aircraft: Aircraft) => aircraft.callsign)
      .join(
        (enter) =>
          enter
            .append("circle")
            .attr("cx", 0)
            .attr("cy", 0)
            .attr("transform", (aircraft: AircraftWithCoords) => {
              const { coords } = aircraft;
              return coords ? `translate(${coords[0]}, ${coords[1]})` : null;
            })
            .attr("r", (aircraft: Aircraft) =>
              memoizedGetRangeRingPixelRadius(aircraft),
            )
            .attr("stroke", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.stroke : null;
            })
            .attr("opacity", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.opacity : null;
            })
            .attr("stroke-width", lineWidthProfile.rangeRingLineWidth)
            .style("fill", "none"),
        (update) =>
          update
            .attr("transform", (aircraft: AircraftWithCoords) => {
              const { coords } = aircraft;
              return coords ? `translate(${coords[0]}, ${coords[1]})` : null;
            })
            .attr("r", (aircraft: Aircraft) =>
              memoizedGetRangeRingPixelRadius(aircraft),
            )
            .attr("stroke", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.stroke : null;
            })
            .attr("opacity", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.opacity : null;
            }),
        (exit) => exit.remove(),
      );
  }, [
    allAircraft,
    colourProfile,
    colourByFL,
    selectedColourScaleName,
    selectedAircraft,
    opacityProfile,
    filterByFlightLevel,
    toolCallsigns.rangeRings,
    lineWidthProfile.rangeRingLineWidth,
    projection,
    rangeRingRadiusKm,
    aircraftColours,
    memoizedGetRangeRingPixelRadius,
    allAircraftWithCoords,
  ]);

  return (
    <g id="aircraftRangeRingsGroup" className="aircraft-range-rings-group" />
  );
}
