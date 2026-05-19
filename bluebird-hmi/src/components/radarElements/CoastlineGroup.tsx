import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import coastlineJson from "data/MERGED_COASTLINE_dm84c.json";
import { useEffect, useMemo } from "react";
import { ColourProfile, Coordinate, LineWidthProfile } from "utils/types";
import {
  getFeatureCoordinates,
  offsetCoordinates,
} from "utils/GeometryParsing";
import { createCoordinatePaths } from "utils/PathCreation";

interface CoastlineGroupProps {
  colourProfile: ColourProfile;
  lineWidthProfile: LineWidthProfile;
  projection: GeoProjection;
  lonLatOffset?: Coordinate;
  altSvgId?: string;
}

export default function CoastlineGroup(props: CoastlineGroupProps) {
  const {
    colourProfile,
    lineWidthProfile,
    projection,
    lonLatOffset,
    altSvgId,
  } = props;

  // Memoise coastline coordinates and paths
  const coastlineCoordinates = useMemo(() => {
    const rawCoastlineCoordinates: Coordinate[][] = coastlineJson.features.map(
      getFeatureCoordinates,
    ) as Coordinate[][];
    return rawCoastlineCoordinates.map((coordinateArray: Coordinate[]) =>
      offsetCoordinates(coordinateArray, lonLatOffset),
    );
  }, [lonLatOffset]);

  const coastlinePaths = useMemo(() => {
    return createCoordinatePaths(coastlineCoordinates, projection);
  }, [coastlineCoordinates, projection]);

  const svgId = !altSvgId ? "coastlineGroup" : altSvgId;

  useEffect(() => {
    const coastlineGroup = d3.select("#" + svgId);

    coastlineGroup
      .selectAll("path")
      .data(coastlinePaths)
      .join(
        // Enter phase: Add new paths
        (enter) =>
          enter
            .append("path")
            .attr("d", (d) => d)
            .attr("fill", colourProfile.landColour)
            .attr("stroke", colourProfile.coastlineColour)
            .attr("stroke-width", lineWidthProfile.coastlineLineWidth),

        // Update phase: Update existing paths
        (update) =>
          update
            .attr("d", (d) => d)
            .attr("fill", colourProfile.landColour)
            .attr("stroke", colourProfile.coastlineColour)
            .attr("stroke-width", lineWidthProfile.coastlineLineWidth),

        // Exit phase: Remove old paths
        (exit) => exit.remove(),
      );
  }, [colourProfile, lineWidthProfile, coastlinePaths, svgId]);

  return <g id={svgId} className="coastline-group" />;
}
