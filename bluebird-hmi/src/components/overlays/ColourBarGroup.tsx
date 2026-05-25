import * as d3 from "d3";
import { useEffect } from "react";
import { ColourProfile } from "utils/types";
import { colourBarMaxFL, colourBarMinFL } from "utils/Constants";
import { colourScaleFunctions } from "utils/profiles/ColourScales";

interface ColourBarGroupProps {
  colourProfile: ColourProfile;
  selectedColourScaleName: string;
}

export default function ColourBarGroup(props: ColourBarGroupProps) {
  const { colourProfile, selectedColourScaleName } = props;

  // Define size and location of colour bar
  const colourBarXLocation = 250;
  const colourBarYLocation = 40;
  const colourBarWidth = 20;
  const colourBarHeight = 300;

  // Use default min and max values from constants file
  const minColourBarValue = colourBarMinFL;
  const maxColourBarValue = colourBarMaxFL;

  // Note that this colour scale should match that used in ColourInterpolators.ts
  const scaleFunction = colourScaleFunctions[selectedColourScaleName];
  const colourScale = d3
    .scaleSequential(scaleFunction)
    .domain([minColourBarValue, maxColourBarValue]);

  useEffect(() => {
    // Create group for ColourBar
    const colourBarGroup = d3.select("#colourBarGroup");

    // If colour-bar-group already exists, update the existing one
    let gradient = colourBarGroup.select("linearGradient");
    if (gradient.empty()) {
      // If not, create a new gradient
      gradient = colourBarGroup
        .append("defs")
        .append("linearGradient")
        .attr("id", "colourGradient")
        .attr("x1", "0%")
        .attr("y1", "100%")
        .attr("x2", "0%")
        .attr("y2", "0%");
    }

    const stops = gradient
      .selectAll("stop")
      .data(d3.ticks(minColourBarValue, maxColourBarValue, 10));

    // Update existing stops
    stops
      .attr(
        "offset",
        (d) =>
          `${((d - minColourBarValue) / (maxColourBarValue - minColourBarValue)) * 100}%`,
      )
      .attr("stop-color", (d) => colourScale(d));

    // Enter new stops if needed
    stops
      .enter()
      .append("stop")
      .attr(
        "offset",
        (d) =>
          `${((d - minColourBarValue) / (maxColourBarValue - minColourBarValue)) * 100}%`,
      )
      .attr("stop-color", (d) => colourScale(d));

    // Exit any extra stops
    stops.exit().remove();

    // Update the color bar
    let rect = colourBarGroup.select("rect");
    if (rect.empty()) {
      rect = colourBarGroup
        .append("rect")
        .attr("width", colourBarWidth)
        .attr("height", colourBarHeight);
    }
    rect.style("fill", "url(#colourGradient)");

    // Create/update axis scale and axis
    const axisScale = d3
      .scaleLinear()
      .domain([maxColourBarValue, minColourBarValue])
      .range([0, colourBarHeight]);
    const axisTicks = [
      minColourBarValue,
      ...axisScale
        .ticks(5)
        .filter(
          (value) => value !== minColourBarValue && value !== maxColourBarValue,
        ),
      maxColourBarValue,
    ];
    const axis = d3
      .axisRight(axisScale)
      .tickValues(axisTicks)
      .tickSize(5)
      .tickFormat(d3.format(".0f"));

    let axisGroup = colourBarGroup.select<SVGGElement>(".axis-group");
    if (axisGroup.empty()) {
      axisGroup = colourBarGroup
        .append("g")
        .attr("class", "axis-group")
        .attr("transform", `translate(${colourBarWidth + 10}, 0)`);
    }
    axisGroup.call(axis).style("color", colourProfile.colourBarAxisColour);

    // Update text label
    let label = colourBarGroup.select(".axis-label");
    if (label.empty()) {
      label = colourBarGroup
        .append("text")
        .attr("class", "axis-label")
        .attr("x", -4)
        .attr("y", -15)
        .attr("font-family", "sans-serif")
        .attr("font-size", "10px");
    }
    label.attr("fill", colourProfile.colourBarAxisColour).text("FLIGHT LEVEL");
  }, [
    minColourBarValue,
    maxColourBarValue,
    selectedColourScaleName,
    colourBarWidth,
    colourBarHeight,
    colourScale,
    colourProfile.colourBarAxisColour,
  ]);

  return (
    <svg
      id="colourBarGroup"
      className="colour-bar-group"
      overflow="visible"
      style={{
        position: "absolute",
        left: `${colourBarXLocation}px`,
        top: `${colourBarYLocation}px`,
      }}
    />
  );
}
