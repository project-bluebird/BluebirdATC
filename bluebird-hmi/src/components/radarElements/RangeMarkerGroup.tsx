import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import rangeMarkerJson from "data/custom_range_markers.json";
import { useEffect, useMemo } from "react";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
import {
  ColourProfile,
  Coordinate,
  GeojsonFeature,
  LineWidthProfile,
} from "utils/types";
import {
  getFeatureCoordinates,
  getRangeMarkerTickCoordinates,
} from "utils/GeometryParsing";
import { createCoordinatePaths } from "utils/PathCreation";

interface RangeMarkerGroupProps {
  colourProfile: ColourProfile;
  lineWidthProfile: LineWidthProfile;
  projection: GeoProjection;
}

export default function RangeMarkerGroup(props: RangeMarkerGroupProps) {
  const { colourProfile, lineWidthProfile, projection } = props;

  const individualSectorIds = useAppSelector(selectIndividualSectorIds);

  // Memoise range marker features/coordinates
  const sectorRangeMarkerFeatures: GeojsonFeature[] = useMemo(() => {
    return rangeMarkerJson.features.filter(
      (feature) =>
        feature.properties.sectorId === individualSectorIds.join(";"),
    );
  }, [individualSectorIds]);

  const rangeMarkerCoordinates: Coordinate[][] = useMemo(() => {
    return sectorRangeMarkerFeatures.map(
      getFeatureCoordinates,
    ) as Coordinate[][];
  }, [sectorRangeMarkerFeatures]);

  const tickCoordinates: Coordinate[][] = useMemo(() => {
    return getRangeMarkerTickCoordinates(rangeMarkerCoordinates);
  }, [rangeMarkerCoordinates]);

  // Memoise range marker paths
  const rangeMarkerPaths = useMemo(() => {
    return createCoordinatePaths(rangeMarkerCoordinates, projection);
  }, [rangeMarkerCoordinates, projection]);

  const svgId = "rangeMarkerGroup";

  useEffect(() => {
    const rangeMarkerGroup = d3.select("#" + svgId);

    // Append paths
    rangeMarkerGroup
      .selectAll("path")
      .data(rangeMarkerPaths)
      .join(
        (enter) =>
          enter
            .append("path")
            .attr("d", (d) => d)
            .attr("fill", "none")
            .attr("stroke", colourProfile.sectorBoundaryColour)
            .attr("stroke-width", lineWidthProfile.rangeMarkerLineWidth),

        (update) =>
          update
            .attr("d", (d) => d)
            .attr("fill", "none")
            .attr("stroke", colourProfile.sectorBoundaryColour)
            .attr("stroke-width", lineWidthProfile.rangeMarkerLineWidth),

        (exit) => exit.remove(),
      );

    // Labels and anchors for range markers
    const tickLabels: string[] = ["3", "5", "10"];
    const tickLabelAlignment: string[] = ["middle", "middle", "end"];

    // Flatten tickCoordinates to match the tickLabels and alignments
    const flatTickCoordinates = tickCoordinates.flat();
    const tickData = flatTickCoordinates.map((coord, index) => ({
      coord,
      label: tickLabels[index % tickLabels.length],
      alignment: tickLabelAlignment[index % tickLabelAlignment.length],
    }));

    // Append ticks/labels
    const tickSelection = rangeMarkerGroup.selectAll("text").data(tickData);

    tickSelection.join(
      (enter) =>
        enter
          .append("text")
          .attr("font-family", "sans-serif")
          .attr("font-size", "10px")
          .attr("fill", colourProfile.sectorBoundaryColour)
          .attr("transform", (d) => {
            const coords = projection([d.coord[0], d.coord[1]]);
            return `translate(${coords})`;
          })
          .text((d) => d.label)
          .style("text-anchor", (d) => d.alignment),

      (update) =>
        update
          .attr("transform", (d) => {
            const coords = projection([d.coord[0], d.coord[1]]);
            return `translate(${coords})`;
          })
          .text((d) => d.label)
          .style("text-anchor", (d) => d.alignment),

      (exit) => exit.remove(),
    );
  }, [
    colourProfile,
    lineWidthProfile,
    rangeMarkerPaths,
    tickCoordinates,
    projection,
    svgId,
  ]);

  return <g id={svgId} className="range-marker-group" />;
}
