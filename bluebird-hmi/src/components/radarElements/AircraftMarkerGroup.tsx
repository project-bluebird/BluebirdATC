//import { useSelectAircraftMutation } from "api/api";
import { useAppSelector } from "app/hooks";
import {
  aircraftIconMarker,
  aircraftPlusMarker,
  aircraftStarMarker,
} from "assets/svgIcons/svgPaths";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useCallback, useEffect, useMemo } from "react";
import { useDispatch } from "react-redux";
import { selectAircraft, selectTime } from "slices/dynamicDataSlice";
import { setSelectedAircraft } from "slices/hmiDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
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

import { radius } from "./ActionHighlights";

interface AircraftMarkerGroupProps {
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
  lineWidthProfile: LineWidthProfile;
  colourByFL: boolean;
  selectedColourScaleName: string;
  aircraftMarkerName: string;
  projection: GeoProjection;
  filterByFlightLevel: boolean;
}

export default function AircraftMarkerGroup(props: AircraftMarkerGroupProps) {
  const {
    colourProfile,
    opacityProfile,
    colourByFL,
    selectedColourScaleName,
    aircraftMarkerName,
    projection,
    filterByFlightLevel,
  } = props;

  const dispatch = useDispatch();
  const sectorId = useAppSelector(selectIndividualSectorIds);
  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const selectedAircraft: string = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  const time = useAppSelector(selectTime);
  // const [selectAircraftMut] = useSelectAircraftMutation();

  const projectedCoordinates = useMemo(() => {
    return allAircraft.map((aircraft) => ({
      callsign: aircraft.callsign,
      coords: projection([aircraft.lon, aircraft.lat]),
      heading: aircraft.heading || 0,
    }));
  }, [allAircraft, projection]);

  // Memoize color and opacity calculations
  const aircraftColors = useMemo(() => {
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

  // Memoize event handler for left click
  const handleLeftClick = useCallback(
    (event: React.MouseEvent<SVGPathElement, MouseEvent>) => {
      if (event.button === 0) {
        const craft = d3
          .select(event.currentTarget as SVGPathElement)
          .datum() as Aircraft;
        // selectAircraftMut({
        //     callsign: craft.callsign,
        //     sectorId: sectorId.join(";"),
        // });
        dispatch(setSelectedAircraft(craft.callsign));
      }
    },
    [sectorId, dispatch],
  );

  useEffect(() => {
    const aircraftMarkerGroup = d3.select("#aircraftMarkerGroup");
    aircraftMarkerGroup.selectAll("*").remove();

    const aircraftIdent = aircraftMarkerGroup
      .selectAll<SVGPathElement, Aircraft>(".aircraft")
      .data(allAircraft, (aircraft: Aircraft) => aircraft.callsign)
      .join("g")
      .attr("transform", (aircraft: Aircraft) => {
        const projectionData = projectedCoordinates.find(
          (ac) => ac.callsign === aircraft.callsign,
        );
        return `translate(${projectionData.coords}) rotate(${projectionData.heading})`;
      })
      .attr(
        "class",
        (aircraft: Aircraft) => `aircraft marker-${aircraft.callsign}`,
      )
      .attr("opacity", (aircraft: Aircraft) => {
        const colourData = aircraftColors.find(
          (ac) => ac.callsign === aircraft.callsign,
        );
        return colourData.opacity;
      })
      .on("mousedown", handleLeftClick);

    aircraftIdent
      .filter((x) => x.squawk_identing)
      .append("circle")
      .attr("r", 50 / radius(projection))
      .attr("stroke", (aircraft) => {
        const colourData = aircraftColors.find(
          (ac) => ac.callsign === aircraft.callsign,
        );
        return colourData.stroke;
      })
      .attr("stroke-width", 3)
      .style("fill", "none")
      .append("animate")
      .attr("attributeName", "visibility")
      .attr("values", "visible;hidden")
      .attr("dur", "0.8s")
      .attr("repeatCount", "indefinite");

    const aircraftEnter = aircraftIdent.append<SVGPathElement>("path");

    // Set marker: star marker or aircraft marker
    if (aircraftMarkerName === "starMarker") {
      aircraftEnter
        .attr("d", (acft) => aircraftStarMarker.join(""))
        .attr("stroke", (aircraft: Aircraft) => {
          const colourData = aircraftColors.find(
            (ac) => ac.callsign === aircraft.callsign,
          );
          return colourData.stroke;
        });
    } else if (aircraftMarkerName === "aircraftIcon") {
      aircraftEnter
        .attr("d", aircraftIconMarker)
        .attr("stroke", (aircraft: Aircraft) => {
          const colourData = aircraftColors.find(
            (ac) => ac.callsign === aircraft.callsign,
          );
          return colourData.stroke;
        })
        .attr("fill", (aircraft: Aircraft) => {
          const colourData = aircraftColors.find(
            (ac) => ac.callsign === aircraft.callsign,
          );
          return colourData.fill;
        });
    }
  }, [
    allAircraft,
    aircraftMarkerName,
    projectedCoordinates,
    aircraftColors,
    handleLeftClick,
    projection,
  ]);
  return <g id="aircraftMarkerGroup" className="aircraft-marker-group" />;
}
