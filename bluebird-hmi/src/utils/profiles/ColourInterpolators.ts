import * as d3 from "d3";
import {
  colourBarMaxFL,
  colourBarMinFL,
  defaultColourScaleName,
} from "utils/Constants";
import { colourScaleFunctions } from "utils/profiles/ColourScales";

interface ColourScaleOptions {
  value: number;
  colourScaleString: string;
  minValue: number;
  maxValue: number;
}

// Using d3-scale-chromatic, interpolate between a min and max
// on a provided colour scale and return the correct colour.
// First ensure value is within specified range, then return a
// colour based on the correct scale function, and the min/max values
export const getColourForValue = ({
  colourScaleString,
  value,
  minValue,
  maxValue,
}: ColourScaleOptions): string => {
  const clampedValue = Math.max(minValue, Math.min(maxValue, value));
  const scaleFunction = colourScaleFunctions[colourScaleString];
  const colourScale = d3
    .scaleSequential(scaleFunction)
    .domain([minValue, maxValue]);
  const colour = d3.color(colourScale(clampedValue)).formatHex();

  return colour;
};

// Given default min and max FL limits for the colour bar, get
// a colour for a given rounded FL (nearest 10 FL)
export function getColourForFL(
  flightLevel: number,
  colourScaleString?: string,
  minFL = colourBarMinFL,
  maxFL = colourBarMaxFL,
): string {
  const roundedFlightLevel = Math.round(flightLevel / 10) * 10;

  return getColourForValue({
    colourScaleString: colourScaleString || defaultColourScaleName,
    value: roundedFlightLevel,
    minValue: minFL,
    maxValue: maxFL,
  });
}
