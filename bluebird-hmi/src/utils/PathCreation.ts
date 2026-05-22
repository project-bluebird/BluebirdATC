import { Coordinate, Fix } from "utils/types";

// Create d3 formatted path string from coordinates. This greatly aids plotting.
// Coordinates[] is a continuous path, Coordinates[][] is n discrete paths
export function createCoordinatePaths(
  coordinates: Coordinate[][],
  projection: d3.GeoProjection,
): string[] {
  const paths: string[] = [];
  for (const featureCoords of coordinates) {
    const pixelCoords = featureCoords.map(([x, y]: Coordinate) => {
      return projection([x, y]).join(",");
    });

    const path = "M" + pixelCoords.join("L");
    paths.push(path);
  }

  return paths;
}

// Use createCoordinatePaths function to get a d3 path for the route, i.e.
// the path from the coordinates of a comma-separated array of fixes. The
// fix names of these fixes are also returned, and added to the array of
// fix names on route.
export function createRoutePathWithNames(
  route: string[],
  fixes: Fix[],
  fixNamesOnRoutes: string[],
  projection: d3.GeoProjection,
): [string[], string[]] {
  const fixCoords: Coordinate[] = [];
  route.forEach(function (fixName) {
    // Get the fix
    const routeFix: Fix | undefined = fixes.find((obj) => obj.name === fixName);

    // If the fix in the route exists in all fixes, store the fix coordinates
    // Also add the fix name to the current list of fixes on route (i.e. to display)
    if (routeFix) {
      fixCoords.push([routeFix.lon, routeFix.lat]);
      fixNamesOnRoutes = [...fixNamesOnRoutes, fixName];
    }
  });

  // Create d3 coordinate path for easy plotting
  const coordPath: string[] = createCoordinatePaths([fixCoords], projection);

  return [coordPath, fixNamesOnRoutes];
}
