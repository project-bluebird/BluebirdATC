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
} from "utils/profiles/ColourSelectors";

import { callsignOnlyInfoBlock } from "components/infoBlock/CallsignOnlyBlock";
import { defaultBlock } from "components/infoBlock/DefaultBlock";
import { hideGroundSpeedBlock } from "components/infoBlock/NoGroundSpeedBlock";
import { simplifiedInfoBlock } from "components/infoBlock/SimplifiedBlock";

const SIMPLE_BOX_WIDTH = 73;
const FULL_BOX_WIDTH = 105;
const NARROW_BOX_WIDT = 98;

const BOX_ROW_HEIGHT = 12;
const SHORT_BOX_HEIGHT = 30;
const MIDDLE_BOX_HEIGHT = 42;
const TALL_BOX_HEIGHT = 54;

///////////////////////
//// HELPER FUNCTIONS
///////////////////////
function getAircraftInfoBlockSize(aircraft: Aircraft, infoBlockState: string) {
  if (infoBlockState === InfoBlockOptions.SimplifiedBlock) {
    return [SIMPLE_BOX_WIDTH, SHORT_BOX_HEIGHT];
  }

  let boxHeight =
    aircraft.entry_flight_level !== null && aircraft.bay !== "INCOMM"
      ? TALL_BOX_HEIGHT
      : MIDDLE_BOX_HEIGHT;

  boxHeight =
    infoBlockState === InfoBlockOptions.HideGroundSpeed
      ? boxHeight - BOX_ROW_HEIGHT
      : boxHeight;

  let boxWidth =
    aircraft.selected_flight_level < 100 ? NARROW_BOX_WIDT : FULL_BOX_WIDTH;

  return [boxWidth, boxHeight];
}

export interface AircraftInfoBlockSimpleProps {
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
  lineWidthProfile: LineWidthProfile;
  colourByFL: boolean;
  scale: number;
  selectedColourScaleName: string;
  infoBlockState: string;
  projection: GeoProjection;
  resetTrigger: number;
  filterByFlightLevel: boolean;
}

export default function AircraftInfoBlockSimple(
  props: AircraftInfoBlockSimpleProps,
) {
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
    filterByFlightLevel,
  } = props;

  const dispatch = useDispatch();
  const individualSectorIds = useAppSelector(selectIndividualSectorIds);
  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const selectedAircraft: string = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  //  const [selectAircraftMut] = useSelectAircraftMutation();

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
    const aircraftInfoParent = d3.select("#aircraftInfoSimpleParent");
    aircraftInfoParent.selectAll("*").remove();
    aircraftInfoParent.style(
      "display",
      infoBlockState === InfoBlockOptions.HideBlock ? "none" : "block",
    );

    // Setup group
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
    function getInfoBlockLine(
      aircraft: Aircraft,
      boxWidth: number,
      boxHeight: number,
    ): number[] {
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

    ///////////////////////
    //// CREATE INFO BLOCKS
    ///////////////////////

    const selectionBoxPadding = [-2, 8];
    let boxWidth: number, boxHeight: number;

    aircraftInfoGroup.each(function (aircraft: Aircraft) {
      const group = d3.select(this);
      const aircraftColour = getAircraftColour(
        aircraft,
        colourProfile,
        colourByFL,
        selectedColourScaleName,
      );

      switch (infoBlockState) {
        case InfoBlockOptions.ShowBlock:
          defaultBlock(
            aircraft,
            group,
            aircraftColour,
            colourProfile,
            colourByFL,
            selectedColourScaleName,
          );
          break;

        case InfoBlockOptions.HideGroundSpeed:
          hideGroundSpeedBlock(
            aircraft,
            group,
            aircraftColour,
            colourProfile,
            colourByFL,
            selectedColourScaleName,
          );
          break;

        case InfoBlockOptions.SimplifiedBlock:
          simplifiedInfoBlock(aircraft, group, aircraftColour);
          break;
      }

      [boxWidth, boxHeight] = getAircraftInfoBlockSize(
        aircraft,
        infoBlockState,
      );

      ///////////////////////
      //// SELECTION BOX
      ///////////////////////

      // Set boxHeight and boxWidth

      // Note: always plot a rectangle to allow consistent left/right clicking
      // in gaps between text. Only display the selected aircraft box though.
      group
        .append("rect")
        .attr("x", selectionBoxPadding[0])
        .attr("y", selectionBoxPadding[1])
        .attr("width", boxWidth)
        .attr("height", boxHeight)
        .attr("fill", "transparent")
        .attr("stroke", aircraftColour)
        .attr(
          "stroke-width",
          aircraft.callsign === selectedAircraft
            ? lineWidthProfile.selectionBoxLineWidth
            : 0,
        );

      ///////////////////////
      //// EVENT HANDLERS
      ///////////////////////
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
          document.removeEventListener("pointermove", aircraftMouseMove);
          document.removeEventListener("pointerup", aircraftMouseUp);
        }
      }
      let isDragging = false;
      group.on("pointerdown", function (event, aircraft: Aircraft) {
        if (event.button === 2 || event.pointerType === "touch") {
          startPosX = event.clientX;
          startPosY = event.clientY;
          dragCallsign = aircraft.callsign;
          isDragging = true;
          document.addEventListener("pointermove", aircraftMouseMove);
          document.addEventListener("pointerup", aircraftMouseUp);
        }

        if (event.button === 0 || event.pointerType === "touch") {
          //selectAircraftMut({
          //    callsign: aircraft.callsign,
          //    sectorId: individualSectorIds.join(";"),
          //});
          dispatch(setSelectedAircraft(aircraft.callsign));
        }
      });
      ///////////////////////
      //// LINES BETWEEN OFFSET
      //// BLOCKS AND AIRCRAFT
      ///////////////////////
      if (
        aircraft.callsign in infoBlockOffsets &&
        boxHeight !== 0 &&
        boxWidth !== 0
      ) {
        const infoBlockDims = getInfoBlockLine(aircraft, boxWidth, boxHeight);
        group
          .append("line")
          .attr("stroke", aircraftColour)
          .attr("stroke-width", lineWidthProfile.infoBoxLineWidth)
          .attr("x1", infoBlockDims[0])
          .attr("y1", infoBlockDims[1])
          .attr("x2", infoBlockDims[2])
          .attr("y2", infoBlockDims[3]);
      }
    });
  }, [
    allAircraft,
    colourByFL,
    colourProfile,
    dispatch,
    filterByFlightLevel,
    infoBlockDragScales,
    infoBlockOffsets,
    infoBlockState,
    lineWidthProfile.infoBoxLineWidth,
    lineWidthProfile.selectionBoxLineWidth,
    opacityProfile,
    projection,
    scale,
    individualSectorIds,
    //  selectAircraftMut,
    selectedAircraft,
    selectedColourScaleName,
  ]);

  return (
    <g id="aircraftInfoSimpleParent" className="aircraft-info-simple-group" />
  );
}
