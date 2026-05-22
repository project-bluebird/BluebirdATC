import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect, useMemo } from "react";
import { selectAircraft } from "slices/dynamicDataSlice";
import {
  Aircraft,
  ColourProfile,
  LineWidthProfile,
  ToolCallsignStore,
} from "utils/types";
import { getAircraftNextPixelPosition } from "utils/DistanceUtils";
import { getAircraftColour } from "utils/profiles/ColourSelectors";

interface AircraftVectorLinesGroupProps {
  toolCallsigns: ToolCallsignStore;
  colourProfile: ColourProfile;
  lineWidthProfile: LineWidthProfile;
  colourByFL: boolean;
  selectedColourScaleName: string;
  projection: GeoProjection;
  vectorLength: number;
}

interface AircraftWithCoords extends Aircraft {
  currentCoords: [number, number];
  futureCoords: [number, number];
}

export default function AircraftVectorLinesGroup(
  props: AircraftVectorLinesGroupProps,
) {
  const {
    toolCallsigns,
    colourProfile,
    lineWidthProfile,
    colourByFL,
    selectedColourScaleName,
    projection,
    vectorLength,
  } = props;

  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);

  // Memoise projection and future position calculations
  const filteredAircraft = useMemo(() => {
    return allAircraft.filter((aircraft) =>
      toolCallsigns.vectorLines.includes(aircraft.callsign),
    );
  }, [allAircraft, toolCallsigns.vectorLines]);

  const aircraftWithProjectedCoords = useMemo(() => {
    return filteredAircraft.map((aircraft: Aircraft) => {
      const currentCoords = projection([aircraft.lon, aircraft.lat]);
      const futureDelta = getAircraftNextPixelPosition(
        projection,
        aircraft,
        vectorLength,
      );

      return {
        ...aircraft,
        currentCoords,
        futureCoords: [
          currentCoords[0] + futureDelta[0],
          currentCoords[1] + futureDelta[1],
        ],
      };
    });
  }, [filteredAircraft, projection, vectorLength]);

  // Memoise colour calculations
  const aircraftColours = useMemo(() => {
    return allAircraft.map((aircraft) => {
      const colour = getAircraftColour(
        aircraft,
        colourProfile,
        colourByFL,
        selectedColourScaleName,
      );
      return {
        callsign: aircraft.callsign,
        stroke: colour,
        fill: colour,
      };
    });
  }, [allAircraft, colourProfile, colourByFL, selectedColourScaleName]);

  useEffect(() => {
    const aircraftVectorLinesGroup = d3.select("#aircraftVectorLinesGroup");

    aircraftVectorLinesGroup
      .selectAll("line")
      .data(
        aircraftWithProjectedCoords,
        (aircraft: Aircraft) => aircraft.callsign,
      )
      .join(
        (enter) =>
          enter
            .append("line")
            .attr(
              "x1",
              (aircraft: AircraftWithCoords) => aircraft.currentCoords[0],
            )
            .attr(
              "y1",
              (aircraft: AircraftWithCoords) => aircraft.currentCoords[1],
            )
            .attr(
              "x2",
              (aircraft: AircraftWithCoords) => aircraft.futureCoords[0],
            )
            .attr(
              "y2",
              (aircraft: AircraftWithCoords) => aircraft.futureCoords[1],
            )
            .attr("stroke", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.stroke : null;
            })
            .attr("stroke-width", lineWidthProfile.vectorLineWidth),

        (update) =>
          update
            .attr(
              "x1",
              (aircraft: AircraftWithCoords) => aircraft.currentCoords[0],
            )
            .attr(
              "y1",
              (aircraft: AircraftWithCoords) => aircraft.currentCoords[1],
            )
            .attr(
              "x2",
              (aircraft: AircraftWithCoords) => aircraft.futureCoords[0],
            )
            .attr(
              "y2",
              (aircraft: AircraftWithCoords) => aircraft.futureCoords[1],
            )
            .attr("stroke", (aircraft: Aircraft) => {
              const color = aircraftColours.find(
                (a) => a.callsign === aircraft.callsign,
              );
              return color ? color.stroke : null;
            })
            .attr("stroke-width", lineWidthProfile.vectorLineWidth),

        (exit) => exit.remove(),
      );
  }, [
    aircraftColours,
    aircraftWithProjectedCoords,
    lineWidthProfile.vectorLineWidth,
  ]);

  return (
    <g id="aircraftVectorLinesGroup" className="aircraft-vector-lines-group" />
  );
}
