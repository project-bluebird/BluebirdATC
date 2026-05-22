import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useCallback, useEffect, useMemo } from "react";
import { selectSectors } from "slices/staticDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
import { ColourProfile, Coordinate, LineWidthProfile } from "utils/types";

interface SectorBoundaryGroupProps {
  colourProfile: ColourProfile;
  lineWidthProfile: LineWidthProfile;
  projection: GeoProjection;
}

export default function SectorBoundaryGroup(props: SectorBoundaryGroupProps) {
  const { colourProfile, lineWidthProfile, projection } = props;

  const sectors = useAppSelector(selectSectors);
  const individualSectors = useAppSelector(selectIndividualSectorIds);
  const selectedSectorData = useMemo(
    () =>
      individualSectors
        ? individualSectors
            .map((individual_id) => sectors[individual_id])
            .flat()
        : [],
    [sectors, individualSectors],
  );

  const getProjectedPoints = useCallback(
    (sectorPoints: Coordinate[] | undefined) => {
      if (sectorPoints) {
        return sectorPoints
          .map((sectorPoint: Coordinate) => {
            const coords = projection([sectorPoint[1], sectorPoint[0]])?.join(
              ",",
            );
            return coords;
          })
          .join(" ");
      }
      console.info(
        "Got an undefined instead of a list of coordinates to render as sector boundaries. This could be because no sector is selected.",
      );
      return [];
    },
    [projection],
  );

  useEffect(() => {
    const sectorGroup = d3.select("#sectorGroup");

    const drawPolygons = (data, className, strokeColor, strokeWidth) => {
      sectorGroup
        .selectAll(`.${className}`)
        .data(data)
        .join(
          (enter) =>
            enter
              .append("polygon")
              .attr("class", className)
              .attr("points", getProjectedPoints)
              .attr("fill", "none")
              .attr("stroke", strokeColor)
              .attr("stroke-width", strokeWidth),
          (update) =>
            update
              .attr("points", getProjectedPoints)
              .attr("stroke", strokeColor)
              .attr("stroke-width", strokeWidth),
          (exit) => exit.remove(),
        );
    };

    // Clear all polygons to ensure fresh redraw
    sectorGroup.selectAll("polygon").remove();

    // Draw non-selected sectors first
    Object.keys(sectors).forEach((sectorKey) => {
      if (sectorKey && !individualSectors.includes(sectorKey)) {
        drawPolygons(
          sectors[sectorKey],
          `polygon-${sectorKey}`,
          colourProfile.sectorBoundaryColour,
          lineWidthProfile.sectorBoundaryLineWidth,
        );
      }
    });

    // Draw selected sector on top
    if (Array.isArray(selectedSectorData)) {
      drawPolygons(
        selectedSectorData,
        "polygon-selected",
        colourProfile.sectorHighlightBoundaryColour,
        lineWidthProfile.sectorHighlightBoundaryLineWidth,
      );
    }
  }, [
    colourProfile,
    lineWidthProfile,
    projection,
    sectors,
    getProjectedPoints,
    selectedSectorData,
    individualSectors,
  ]);

  return <g id="sectorGroup" className="sector-group" />;
}
