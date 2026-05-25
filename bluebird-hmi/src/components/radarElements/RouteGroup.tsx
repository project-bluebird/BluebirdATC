import { useAppSelector } from "app/hooks";
import * as d3 from "d3";
import { GeoProjection } from "d3-geo";
import { useEffect } from "react";
import { selectAircraft } from "slices/dynamicDataSlice";
import { selectFixes } from "slices/staticDataSlice";
import {
  Aircraft,
  ColourProfile,
  Fix,
  LineWidthProfile,
  ToolCallsignStore,
} from "utils/types";
import { createRoutePathWithNames } from "utils/PathCreation";
import { getAircraftColour } from "utils/profiles/ColourSelectors";

interface RouteGroupProps {
  toolCallsigns: ToolCallsignStore;
  colourProfile: ColourProfile;
  colourByFL: boolean;
  lineWidthProfile: LineWidthProfile;
  selectedColourScaleName: string;
  profileName: string;
  projection: GeoProjection;
}

export default function RouteGroup(props: RouteGroupProps) {
  const {
    toolCallsigns,
    colourProfile,
    colourByFL,
    lineWidthProfile,
    selectedColourScaleName,
    profileName,
    projection,
  } = props;

  const allAircraft: Aircraft[] = useAppSelector(selectAircraft);
  const fixes = useAppSelector(selectFixes);

  useEffect(() => {
    // Create routePaths for each aircraft in toolCallsigns.routes
    const routePaths: string[][] = [];
    const routeColours: string[] = [];
    let fixNamesOnRoutes: string[] = [];

    // Use a Set to store unique routes (based on their coordinate paths)
    const uniqueRoutes = new Set<string>();

    toolCallsigns.routes.forEach((callsign: string) => {
      const aircraft = allAircraft.find(
        (a: Aircraft) => a.callsign === callsign,
      );
      if (aircraft !== undefined) {
        const routeColour =
          profileName === "presentation"
            ? getAircraftColour(
                aircraft,
                colourProfile,
                colourByFL,
                selectedColourScaleName,
              )
            : colourProfile.routeColour;

        // Get route paths and updated fix names on the routes
        const [coordPath, updatedFixNamesOnRoutes] = createRoutePathWithNames(
          aircraft.route,
          fixes,
          fixNamesOnRoutes,
          projection,
        );

        // Check if the route is unique by using a stringified version of the path
        const routeString = JSON.stringify(coordPath);
        if (!uniqueRoutes.has(routeString)) {
          uniqueRoutes.add(routeString);

          routePaths.push(coordPath);
          routeColours.push(routeColour);
          fixNamesOnRoutes = updatedFixNamesOnRoutes;
        }
      }
    });

    // Append routes
    const routeGroup = d3.select("#routeGroup");

    const routeSelection = routeGroup.selectAll(".route-path").data(routePaths);

    routeSelection
      .enter()
      .append("path")
      .attr("class", "route-path")
      .attr("d", (d: string[]) => d)
      .attr("fill", "none")
      .attr("stroke", (d, i) => routeColours[i])
      .attr("stroke-width", lineWidthProfile.routeLineWidth);

    routeSelection
      .attr("d", (d: string[]) => d)
      .attr("stroke", (d, i) => routeColours[i])
      .attr("stroke-width", lineWidthProfile.routeLineWidth);

    routeSelection.exit().remove();

    // Filter fixes to only those in the routes
    const filteredFixes = fixes.filter((fix: Fix) =>
      fixNamesOnRoutes.includes(fix.name),
    );

    // Append fixes
    const fixSelection = routeGroup.selectAll(".fix").data(filteredFixes);

    fixSelection
      .enter()
      .append("text")
      .attr("class", "fix")
      .attr("x", (fix: Fix) => {
        const coords = projection([fix.lon, fix.lat]);
        return coords[0];
      })
      .attr("y", (fix: Fix) => {
        const coords = projection([fix.lon, fix.lat]);
        return coords[1];
      })
      .attr("dx", -10)
      .attr("dy", 11)
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", colourProfile["fixColour"])
      .text((fix: Fix) => fix.name);

    fixSelection
      .attr("x", (fix: Fix) => {
        const coords = projection([fix.lon, fix.lat]);
        return coords[0];
      })
      .attr("y", (fix: Fix) => {
        const coords = projection([fix.lon, fix.lat]);
        return coords[1];
      })
      .text((fix: Fix) => fix.name);

    fixSelection.exit().remove();
  }, [
    toolCallsigns,
    allAircraft,
    fixes,
    colourProfile,
    lineWidthProfile,
    colourByFL,
    selectedColourScaleName,
    profileName,
    projection,
  ]);

  return <g id="routeGroup" className="route-group" />;
}
