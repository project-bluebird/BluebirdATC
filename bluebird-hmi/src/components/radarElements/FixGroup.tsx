import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect, useMemo } from "react";
import { selectFixes, selectScenarioName } from "slices/staticDataSlice";
import { Fix } from "utils/types";

interface FixGroupProps {
  colourProfile: object;
  lineWidthProfile: object;
  showFixNames: boolean;
  projection: GeoProjection;
}

interface FixWithCoords extends Fix {
  coords: [number, number];
}

export default function FixGroup(props: FixGroupProps) {
  const { colourProfile, lineWidthProfile, showFixNames, projection } = props;

  const fixes = useAppSelector(selectFixes);
  const scenarioName = useAppSelector(selectScenarioName);
  const fixData: FixWithCoords[] = useMemo(() => {
    return fixes
      .filter((fix: Fix) => fix.visible)
      .map((fix: Fix) => ({
        ...fix,
        coords: projection([fix.lon, fix.lat]),
      }));
  }, [fixes, projection]);

  useEffect(() => {
    const fixGroup = d3.select("#fixGroup");
    const fixSelection = fixGroup
      .selectAll(".fix")
      .data(fixData, (d: FixWithCoords) => d.name);

    // Enter: Append either rect or circle based on the condition
    fixSelection
      .enter()
      .append((d) =>
        d.name.length === 4 && scenarioName !== "ATC Game vs Falcon"
          ? document.createElementNS(d3.namespaces.svg, "rect")
          : document.createElementNS(d3.namespaces.svg, "circle"),
      )
      .attr("class", "fix")
      .attr("stroke", colourProfile["fixColour"])
      .attr("stroke-width", lineWidthProfile["fixLineWidth"])
      .attr("fill", (d) =>
        d.name.length === 4 && scenarioName !== "ATC Game vs Falcon"
          ? "none"
          : colourProfile["fixColour"],
      )
      .attr("x", (d) => (d.name.length === 4 ? d.coords[0] - 2 : null))
      .attr("y", (d) => (d.name.length === 4 ? d.coords[1] - 2 : null))
      .attr("width", (d) => (d.name.length === 4 ? 4 : null))
      .attr("height", (d) => (d.name.length === 4 ? 4 : null))
      .attr("cx", (d) => (d.name.length !== 4 ? d.coords[0] : null))
      .attr("cy", (d) => (d.name.length !== 4 ? d.coords[1] : null))
      .attr("r", (d) => (d.name.length !== 4 ? 0.7 : null));

    // Update rects
    fixSelection
      .filter("rect")
      .attr("stroke", colourProfile["fixColour"])
      .attr("x", (d) => d.coords[0] - 2)
      .attr("y", (d) => d.coords[1] - 2)
      .attr("width", 4)
      .attr("height", 4);

    // Update circles
    fixSelection
      .filter("circle")
      .attr("stroke", colourProfile["fixColour"])
      .attr("cx", (d) => d.coords[0])
      .attr("cy", (d) => d.coords[1])
      .attr("r", 0.7);

    // Exit
    fixSelection.exit().remove();

    // Handle fix names
    const textSelection = fixGroup
      .selectAll("text.fix-name")
      .data(showFixNames ? fixData : [], (d: FixWithCoords) => d.name);

    // Enter
    textSelection
      .enter()
      .append("text")
      .attr("class", "fix-name")
      .attr("x", (d) => d.coords[0] - 10)
      .attr("y", (d) => d.coords[1] + 11)
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", colourProfile["fixColour"])
      .text((d) => d.name);

    // Update
    textSelection
      .attr("fill", colourProfile["fixColour"])
      .attr("x", (d) => d.coords[0] - 10)
      .attr("y", (d) => d.coords[1] + 11)
      .text((d) => d.name);

    // Exit
    textSelection.exit().remove();
  }, [
    colourProfile,
    lineWidthProfile,
    showFixNames,
    fixes,
    projection,
    scenarioName,
    fixData,
  ]);

  return <g id="fixGroup" className="fix-group" />;
}
