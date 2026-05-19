import booleanOverlap from "@turf/boolean-overlap";
import * as turf from "@turf/turf";
import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect, useRef } from "react";
import { selectActions, selectAircraft } from "slices/dynamicDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
import {
  Aircraft,
  ColourProfile,
  Coordinate,
  LineWidthProfile,
  OpacityProfile,
} from "utils/types";
import { getRangeRingPixelRadius } from "utils/DistanceUtils";
import { toRadians } from "utils/HeadingUtils";

import { isActionForCurrentSector } from "components/overlays/ActionLog";

const VIRTUAL_MARGIN_SIZE = 5;

interface AircraftActionHighlightsProps {
  colourProfile: ColourProfile;
  lineWidthProfile: LineWidthProfile;
  opacityProfile: OpacityProfile;
  projection: GeoProjection;
}

export function radius(projection: GeoProjection): number {
  return (50 * 4000) / projection.scale();
}

class ActionBox {
  aircraft: Aircraft;
  clearances: string[];

  // Represents the radius of the circle used to generate the position of the label, although not neccesarily the distance between the aircraft and the left most coordinates are it may be displaced dependent on the position
  radius: number;

  // Represents the global coordinates of the top left corner of the action highlight rectangle. Note the global coordinates have the origin at the top left corner of the screen
  coordinates: Coordinate;

  // Represents the translation from the aircrafts position to the top left most coordinate of the action highlight rectangle
  translation: Coordinate;
  colour: string;
  width = 0;
  height = 0;
  angle = 24;
  text: { content: string[]; colour: string } = {
    content: [],
    colour: "black",
  };

  constructor(
    aircraft: Aircraft,
    clearances: string[],
    projection: GeoProjection,
    color = "none",
  ) {
    this.aircraft = aircraft;
    this.text.content = clearances;
    this.radius = getRangeRingPixelRadius(
      projection,
      this.aircraft,
      3 * radius(projection),
    );
    this.updateCoordinates(projection);
    this.colour = color;
  }

  updateTranslation() {
    this.translation = [
      Math.sin(toRadians(this.angle)) * this.radius,
      Math.cos(toRadians(this.angle)) * -this.radius,
    ];

    if (this.angle > 180 && this.angle <= 360) {
      this.translation[0] -= this.width;
    }

    if (this.angle < 90 || this.angle > 270) {
      this.translation[1] -= this.height;
    }
  }
  updateCoordinates(projection: GeoProjection) {
    this.updateTranslation();
    this.coordinates = projection([this.aircraft.lon, this.aircraft.lat]).map(
      (value, index) => value + this.translation[index],
    );
  }
  updateAngle(actionBoxes: ActionBox[], projection: GeoProjection) {
    const others = actionBoxes.filter(
      (x) => x.aircraft.callsign !== this.aircraft.callsign,
    );
    // After rendering, this needs to be done at least once to calculate the translation based on the size of the box
    this.updateCoordinates(projection);

    while (
      others.some(
        (actionBox) => checkOverlap(this, actionBox) && this.angle < 360,
      )
    ) {
      this.angle += 12;
      this.updateCoordinates(projection);
    }

    if (this.angle >= 360) {
      console.log(`${this.aircraft.callsign} could not be deconflicted`);
    }
  }
}

function ActionBoxIntoPolygon(rect: ActionBox) {
  return turf.polygon([
    [
      [
        rect.coordinates[0] - VIRTUAL_MARGIN_SIZE,
        rect.coordinates[1] - VIRTUAL_MARGIN_SIZE,
      ],
      [
        rect.coordinates[0] + rect.width + VIRTUAL_MARGIN_SIZE,
        rect.coordinates[1] - VIRTUAL_MARGIN_SIZE,
      ],
      [
        rect.coordinates[0] + rect.width + VIRTUAL_MARGIN_SIZE,
        rect.coordinates[1] + rect.height + VIRTUAL_MARGIN_SIZE,
      ],
      [
        rect.coordinates[0] - VIRTUAL_MARGIN_SIZE,
        rect.coordinates[1] + rect.height + VIRTUAL_MARGIN_SIZE,
      ],
      [
        rect.coordinates[0] - VIRTUAL_MARGIN_SIZE,
        rect.coordinates[1] - VIRTUAL_MARGIN_SIZE,
      ],
    ],
  ]);
}

const checkOverlap = (rect1: ActionBox, rect2): boolean => {
  const polygon1 = ActionBoxIntoPolygon(rect1);
  const polygon2 = ActionBoxIntoPolygon(rect2);

  return booleanOverlap(polygon1, polygon2);
};

export default function ActionHighlights({
  colourProfile,
  lineWidthProfile,
  opacityProfile,
  projection,
}: AircraftActionHighlightsProps) {
  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const actions = useAppSelector(selectActions);
  const prevActionsLength = useRef(0);
  const isInitialRender = useRef(true);
  const individualSectorIds = useAppSelector(selectIndividualSectorIds);

  useEffect(() => {
    const actionBoxes: ActionBox[] = [];

    // Highlight aircraft on action
    if (isInitialRender.current) {
      // Stop highlighting on page refresh
      prevActionsLength.current = actions.length;
      isInitialRender.current = false;
    } else if (
      actions.length > 0 &&
      actions.length !== prevActionsLength.current
    ) {
      const newActions = actions[actions.length - 1].actions;

      // Create action box objects
      for (const aircraft of allAircraft) {
        const clearances = newActions
          .filter(
            (action) =>
              action.callsign === aircraft.callsign &&
              typeof action.text_representation?.clearance === "string",
          )
          .filter((action) =>
            isActionForCurrentSector(individualSectorIds, action),
          )
          .map((action) => action.text_representation.clearance);
        if (clearances.length > 0) {
          actionBoxes.push(
            new ActionBox(
              aircraft,
              clearances,
              projection,
              colourProfile.actionLogColour,
            ),
          );
        }
      }
    }

    // Create group for action highlights
    const parentGroup = d3
      .select("#aircraftActionHighlights")
      .attr("opacity", opacityProfile.actionLogOpacity)
      .style("stroke", colourProfile.actionLogBorderColour);

    // Cleanup previous render
    parentGroup.selectAll("*").remove();

    // Bind info to svg groups
    const bindInfo = (parentElement) => {
      return parentElement
        .selectAll(".aircraft")
        .data(actionBoxes)
        .join("g")
        .attr("transform", (actionBox: ActionBox) => {
          const coords = projection([
            actionBox.aircraft.lon,
            actionBox.aircraft.lat,
          ]);
          return `translate(${coords})`;
        });
    };

    // Parent for circle and line elements
    const highlightGroup = bindInfo(parentGroup);

    // Parent for info box elements
    const infoGroup = bindInfo(parentGroup);

    const highlightCircleRadius = radius(projection) * 1.2;

    const highlightDuration = 5000;

    // Append circle around aircraft
    highlightGroup
      .append("circle")
      .attr("r", (actionBox: ActionBox) =>
        getRangeRingPixelRadius(
          projection,
          actionBox.aircraft,
          highlightCircleRadius,
        ),
      )
      .attr("stroke-width", lineWidthProfile.actionIndicatorLineWidth)
      .style("fill", "none");

    // Append line from circle
    const lineElement = highlightGroup.append("line");

    // Create group for action info box
    const actionBoxGroup = infoGroup.append("g");

    // Plot action box rectangle to contain text
    const rectElement = actionBoxGroup
      .append("rect")
      .attr(
        "id",
        (actionBox: ActionBox) => `${actionBox.aircraft.callsign}-info`,
      )
      .attr("stroke-width", lineWidthProfile.actionIndicatorLineWidth)
      .attr("fill", (actionBox: ActionBox) => actionBox.colour);

    // Plot action text
    const textElement = actionBoxGroup.append("text").attr("y", 1);

    textElement
      .selectAll("tspan")
      .data((actionBox: ActionBox) => [actionBox])
      .enter()
      .append("tspan")
      .style("stroke", (actionBox: ActionBox) => actionBox.text.colour)
      .selectAll("tspan")
      .data((actionBox: ActionBox) => actionBox.text.content)
      .enter()
      .append("tspan")
      .text((d: string) => d)
      .attr("x", 5)
      .attr("dy", "1.2em");

    // Fit box to text
    rectElement.each(function (actionBox: ActionBox, i) {
      const rect = d3.select(this);
      const text = d3.select(textElement.nodes()[i]);
      const padding = text.text() ? 10 : 0;

      // Calculate the bounding box of the text element
      const bbox = text.node().getBBox();

      rect
        .attr("width", bbox.width + padding)
        .attr("height", bbox.height + padding);

      const rectBbox = document
        .getElementById(`${actionBox.aircraft.callsign}-info`)
        .getBoundingClientRect();

      actionBox.width = rectBbox.width;
      actionBox.height = rectBbox.height;

      actionBox.updateAngle(actionBoxes, projection);

      // Position info box
      actionBoxGroup.attr(
        "transform",
        (actionBox: ActionBox) => `translate(${actionBox.translation})`,
      );

      // Position line element
      lineElement
        .attr("x1", 0)
        .attr("x2", 0)
        .attr(
          "y1",
          (actionBox: ActionBox) =>
            -getRangeRingPixelRadius(
              projection,
              actionBox.aircraft,
              highlightCircleRadius,
            ),
        )
        .attr("y2", (actionBox: ActionBox) => -actionBox.radius * 1.2)
        .attr("stroke-width", lineWidthProfile.actionIndicatorLineWidth)
        .attr(
          "transform",
          (actionBox: ActionBox) => `rotate(${actionBox.angle})`,
        );
    });

    // Remove all groups after the designated duration
    highlightGroup.transition().duration(highlightDuration).remove();
    actionBoxGroup.transition().duration(highlightDuration).remove();

    if (prevActionsLength.current < actions.length) {
      // If the state needs to be flushed
      setTimeout(() => {
        prevActionsLength.current = actions.length;
      }, highlightDuration);
    }
  }, [
    actions,
    allAircraft,
    colourProfile,
    lineWidthProfile,
    opacityProfile,
    projection,
    individualSectorIds,
  ]);

  return <g id="aircraftActionHighlights" className="aircraft-action-group" />;
}
