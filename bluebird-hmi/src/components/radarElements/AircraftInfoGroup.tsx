//import { useSelectAircraftMutation } from "api/api";
import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { selectAircraft } from "slices/dynamicDataSlice";
import { setSelectedAircraft } from "slices/hmiDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
import {
  Aircraft,
  ColourProfile,
  Coordinate,
  LineWidthProfile,
  OpacityProfile,
} from "utils/types";
import { InfoBlockOptions } from "utils/Constants";
import {
  getAircraftColour,
  getAircraftOpacity,
  getExitFlightLevelColour,
  getSelectedFlightLevelColour,
} from "utils/profiles/ColourSelectors";

interface AircraftInfoGroupProps {
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
  lineWidthProfile: LineWidthProfile;
  scale: number;
  colourByFL: boolean;
  selectedColourScaleName: string;
  infoBlockState: string;
  projection: GeoProjection;
  resetTrigger: number;
  setRecentlySelectedAircraft: (boolean) => void;
  filterByFlightLevel: boolean;
}

export default function AircraftInfoGroup(props: AircraftInfoGroupProps) {
  const {
    colourProfile,
    opacityProfile,
    lineWidthProfile,
    scale,
    colourByFL,
    selectedColourScaleName,
    infoBlockState,
    projection,
    resetTrigger,
    setRecentlySelectedAircraft,
    filterByFlightLevel,
  } = props;

  const dispatch = useDispatch();
  const individualSectorIds = useAppSelector(selectIndividualSectorIds);
  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const selectedAircraft: string = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  //   const [selectAircraftMut] = useSelectAircraftMutation();

  const [infoBlockOffsets, setInfoBlockOffsets] = useState<{
    [key: string]: Coordinate;
  }>({});
  const [infoBlockDragScales, setInfoBlockDragScales] = useState<{
    [key: string]: number;
  }>({});

  // Reset the positions of the radar info blocks for all aircraft.
  useEffect(() => {
    setInfoBlockOffsets({});
    setInfoBlockDragScales({});
  }, [resetTrigger]);

  useEffect(() => {
    // Size of info block depends on choice of infoBlockState.
    // Create function to get the [width,height] of block in pixels
    function getAircraftInfoBlockSize(
      aircraft: Aircraft,
      infoBlockState: string,
    ) {
      const simpleBoxWidth = 75;
      const fullBoxWidth = 105;
      const narrowBoxWidth = 98;
      const boxRowHeight = 12;
      const shortBoxHeight = 30;
      const middleBoxHeight = 42;
      const tallBoxHeight = 54;

      if (infoBlockState === InfoBlockOptions.SimplifiedBlock) {
        return [simpleBoxWidth, shortBoxHeight];
      }

      let boxHeight =
        aircraft.entry_flight_level !== null && aircraft.bay !== "INCOMM"
          ? tallBoxHeight
          : middleBoxHeight;

      boxHeight =
        infoBlockState === InfoBlockOptions.HideGroundSpeed
          ? boxHeight - boxRowHeight
          : boxHeight;

      const boxWidth =
        aircraft.selected_flight_level < 100 ? narrowBoxWidth : fullBoxWidth;

      return [boxWidth, boxHeight];
    }
    const selectionBoxPadding = [-2, 8];

    // Create group for AircraftInfo block
    const aircraftInfoParent = d3.select("#aircraftInfoParent");
    // Cleanup previous render
    aircraftInfoParent.selectAll("*").remove();

    aircraftInfoParent.style(
      "display",
      infoBlockState === InfoBlockOptions.HideBlock ? "none" : "block",
    );

    let startPosX: number | null, startPosY: number | null;
    let dragCallsign: string;

    function dragBlock(event: MouseEvent) {
      if (!startPosX || !startPosY) {
        return;
      }
      const dragDistanceX = event.clientX - startPosX;
      const dragDistanceY = event.clientY - startPosY;

      let currentPosX = 0;
      let currentPosY = 0;
      if (dragCallsign in infoBlockOffsets) {
        const offset = infoBlockOffsets[dragCallsign];
        currentPosX = offset[0];
        currentPosY = offset[1];
      }
      // Adding a single key-value pair
      setInfoBlockOffsets((prevOffsets) => {
        return {
          ...prevOffsets,
          [dragCallsign]: [
            currentPosX + dragDistanceX,
            currentPosY + dragDistanceY,
          ],
        };
      });
      // Also record the "scale" at the point that the dragging happens
      setInfoBlockDragScales((prevScale) => {
        return { ...prevScale, [dragCallsign]: scale };
      });
    }

    // after right-click-and-drag on the info block, an offset will be stored in the
    // infoBlockOffsets object, keyed by callsign. Look them up here, and use to
    // modify the current coordinates.
    function getInfoBlockCoords(coords: Coordinate | null, callsign: string) {
      if (callsign in infoBlockOffsets) {
        const offset: number[] = infoBlockOffsets[callsign];
        const origScale: number = infoBlockDragScales[callsign];
        if (coords) {
          return coords.map((e, i) => e + offset[i] * (scale / origScale));
        }
      }
      // No modification to existing coords - return whatever coordinates we were given
      return coords;
    }

    // Return [x1, y1, x2, y2] for a line from the aircraft marker to the info block.
    function getInfoBlockLine(aircraft: Aircraft): number[] {
      if (!(aircraft.callsign in infoBlockOffsets)) {
        return [0, 0, 0, 0];
      }

      // the start point of the line will be the negative of the 'offset'
      const offset: Coordinate = infoBlockOffsets[aircraft.callsign];
      const origScale: number = infoBlockDragScales[aircraft.callsign];
      // scale this according to the zoom scale
      const scaledOffset = offset.map((e: number) => e * (scale / origScale));
      // the end point of the line will depend on where the info block is
      // relative to the aircraft marker.  Note that the size of the
      // rectangle around the info block is (105x54) when 4 lines and 105x42 when 3 lines

      // Get the aircraft info block height
      const [boxWidth, boxHeight] = getAircraftInfoBlockSize(
        aircraft,
        infoBlockState,
      );

      // Aircraft location
      const px = -scaledOffset[0];
      const py = -scaledOffset[1];
      let x2: number;
      let y2: number;

      // let the line be perpendicular to the box if possible, otherwise snap to the nearest corner
      const xPadding = selectionBoxPadding[0];
      if (px >= xPadding && px <= boxWidth + xPadding) {
        x2 = px;
      } else if (px <= xPadding) {
        x2 = xPadding;
      } else {
        x2 = boxWidth + xPadding;
      }

      const yPadding = selectionBoxPadding[1];
      if (py >= yPadding && py <= boxHeight + yPadding) {
        y2 = py;
      } else if (py <= yPadding) {
        y2 = yPadding;
      } else {
        y2 = boxHeight + yPadding;
      }

      return [px, py, x2, y2];
    }

    // Bind aircraft info to svg group
    const aircraftInfoGroup = aircraftInfoParent
      .selectAll(".aircraft")
      .data(allAircraft)
      .join("g")
      .attr("transform", (aircraft: Aircraft) => {
        const coords = projection([aircraft.lon, aircraft.lat]);
        const newCoords = getInfoBlockCoords(coords, aircraft.callsign);
        const transformString = `translate(${newCoords})`;
        return transformString;
      })
      .attr("opacity", (aircraft: Aircraft) =>
        getAircraftOpacity(
          allAircraft,
          aircraft,
          opacityProfile,
          selectedAircraft,
          filterByFlightLevel,
        ),
      );

    // Append aircraft callsign
    aircraftInfoGroup
      .append("text")
      .attr("x", 0)
      .attr("y", 22)
      .attr("font-family", "sans-serif")
      .attr("font-size", "15px")
      .attr("fill", (aircraft: Aircraft) =>
        getAircraftColour(
          aircraft,
          colourProfile,
          colourByFL,
          selectedColourScaleName,
        ),
      )
      .text((aircraft: Aircraft) => aircraft.callsign);

    // Append flight level
    aircraftInfoGroup
      .append("text")
      .attr("x", 0)
      .attr("y", 34)
      .attr("font-family", "sans-serif")
      .attr("font-size", "15px")
      .attr("fill", (aircraft: Aircraft) =>
        getAircraftColour(
          aircraft,
          colourProfile,
          colourByFL,
          selectedColourScaleName,
        ),
      )
      .text((aircraft: Aircraft) => Math.round(aircraft.flight_level));

    // For full block, show ground speed.
    if (infoBlockState === InfoBlockOptions.ShowBlock) {
      aircraftInfoGroup.each(function (aircraft: Aircraft) {
        if (aircraft.true_air_speed !== null) {
          d3.select(this)
            .append("text")
            .attr("x", 0)
            .attr("y", 46)
            .attr("font-family", "sans-serif")
            .attr("font-size", "15px")
            .attr("fill", (aircraft: Aircraft) =>
              getAircraftColour(
                aircraft,
                colourProfile,
                colourByFL,
                selectedColourScaleName,
              ),
            )
            .text(
              (aircraft: Aircraft) => "G" + aircraft.true_air_speed.toFixed(0),
            ); // ground speed. Using true air speed for now.
        }
      });

      // For ShowBlock and HideGroundSpeed, show NFL
      if (
        infoBlockState === InfoBlockOptions.ShowBlock ||
        infoBlockState === InfoBlockOptions.HideGroundSpeed
      ) {
        aircraftInfoGroup.each(function (aircraft: Aircraft) {
          if (
            aircraft.entry_flight_level !== null &&
            aircraft.bay !== "INCOMM"
          ) {
            const aircraftColour = getAircraftColour(
              aircraft,
              colourProfile,
              colourByFL,
              selectedColourScaleName,
            );
            d3.select(this)
              .append("text")
              .attr("x", 0)
              .attr("y", 58)
              .attr("font-family", "sans-serif")
              .attr("font-size", "15px")
              .attr("fill", aircraftColour)
              .text(aircraft.entry_flight_level);
          }
        });

        // Append intention code, taking into account Simplified and Hide Ground Speed option
        aircraftInfoGroup
          .append("text")
          .attr("x", 46)
          .attr("y", 34)
          .attr("font-family", "sans-serif")
          .attr("font-size", "15px")
          .attr("fill", (aircraft: Aircraft) =>
            getAircraftColour(
              aircraft,
              colourProfile,
              colourByFL,
              selectedColourScaleName,
            ),
          )
          .text((aircraft: Aircraft) => aircraft.intention_code);

        // Append XFL for all infoBlockStates
        aircraftInfoGroup.each(function (aircraft: Aircraft) {
          if (aircraft.exit_flight_level !== null) {
            d3.select(this)
              .append("text")
              .attr("x", 73)
              .attr("y", 22)
              .attr("font-family", "sans-serif")
              .attr("font-size", "15px")
              .attr(
                "fill",
                getExitFlightLevelColour(
                  aircraft,
                  colourProfile,
                  colourByFL,
                  selectedColourScaleName,
                ),
              )
              .text(Math.round(0.1 * aircraft.exit_flight_level));
          }
        });
        // SFL for the ShowBlock and HideGroundSpeed options
        if (
          infoBlockState === InfoBlockOptions.ShowBlock ||
          infoBlockState === InfoBlockOptions.HideGroundSpeed
        ) {
          aircraftInfoGroup
            .append("text")
            .attr("x", 73)
            .attr("y", 34)
            .attr("font-family", "sans-serif")
            .attr("font-size", "15px")
            .attr("fill", (aircraft: Aircraft) =>
              getSelectedFlightLevelColour(
                aircraft,
                colourProfile,
                colourByFL,
                selectedColourScaleName,
              ),
            )
            .text((aircraft: Aircraft) =>
              Math.round(aircraft.selected_flight_level),
            );
        }
      }
    }
    // Selection box
    let isDragging = false;
    aircraftInfoGroup
      .append("rect")
      .attr("x", selectionBoxPadding[0])
      .attr("y", selectionBoxPadding[1])
      .attr(
        "width",
        (aircraft: Aircraft) =>
          getAircraftInfoBlockSize(aircraft, infoBlockState)[0],
      )
      // height depends on whether NFL is shown
      .attr(
        "height",
        (aircraft: Aircraft) =>
          getAircraftInfoBlockSize(aircraft, infoBlockState)[1],
      )
      .attr("fill", "transparent")
      .attr("stroke", (aircraft: Aircraft) => {
        if (aircraft.callsign === selectedAircraft && colourByFL) {
          return getAircraftColour(
            aircraft,
            colourProfile,
            colourByFL,
            selectedColourScaleName,
          );
        } else {
          return aircraft.callsign === selectedAircraft
            ? colourProfile.selectionColour
            : "none";
        }
      })
      .attr("stroke-width", lineWidthProfile.selectionBoxLineWidth)
      .on("mousedown", function (event) {
        if (event.button === 2) {
          // Right button
          const craft = d3.select(this).datum() as Aircraft;
          startPosX = event.clientX;
          startPosY = event.clientY;
          dragCallsign = craft.callsign;
          isDragging = true;
          document.addEventListener("mousemove", aircraftMouseMove);
          document.addEventListener("mouseup", aircraftMouseUp);
        } /*  else if (event.button === 0) {
                    const craft = d3.select(this).datum() as Aircraft;
                    selectAircraftMut({
                        tenantId: tenantId,
                        callsign: craft.callsign,
                        sectorId: individualSectorIds.join(";"),
                    });
                    dispatch(setSelectedAircraft(craft.callsign));
                    // Set recentlySelectedAircraft to true for 100 ms to remove any flicker of the heading line
                    setRecentlySelectedAircraft(true);
                    setTimeout(() => {
                        setRecentlySelectedAircraft(false);
                    }, 100);
                } */
      });

    function aircraftMouseMove(event: MouseEvent) {
      if (isDragging) {
        dragBlock(event);
      }
    }

    function aircraftMouseUp() {
      if (isDragging) {
        isDragging = false;
        startPosX = null;
        startPosY = null;
        document.removeEventListener("mousemove", aircraftMouseMove);
        document.removeEventListener("mouseup", aircraftMouseUp);
      }
    }

    // Append line between info box and aircraft marker
    aircraftInfoGroup
      .filter((aircraft: Aircraft) => aircraft.callsign in infoBlockOffsets)
      .append("line")
      .attr("stroke", (aircraft: Aircraft) =>
        getAircraftColour(
          aircraft,
          colourProfile,
          colourByFL,
          selectedColourScaleName,
        ),
      )
      .attr("stroke-width", lineWidthProfile.infoBoxLineWidth)
      .attr("x1", (aircraft: Aircraft) => getInfoBlockLine(aircraft)[0])
      .attr("y1", (aircraft: Aircraft) => getInfoBlockLine(aircraft)[1])
      .attr("x2", (aircraft: Aircraft) => getInfoBlockLine(aircraft)[2])
      .attr("y2", (aircraft: Aircraft) => getInfoBlockLine(aircraft)[3]);
  }, [
    individualSectorIds,
    allAircraft,
    selectedAircraft,
    infoBlockDragScales,
    infoBlockOffsets,
    colourProfile,
    opacityProfile,
    lineWidthProfile,
    scale,
    colourByFL,
    infoBlockState,
    selectedColourScaleName,
    projection,
    dispatch,
    //  selectAircraftMut,
    setRecentlySelectedAircraft,
    filterByFlightLevel,
  ]);

  return <g id="aircraftInfoParent" className="aircraft-info-group" />;
}
