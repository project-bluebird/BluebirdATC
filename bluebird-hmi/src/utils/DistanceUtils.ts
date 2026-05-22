import { Aircraft, Coordinate } from "utils/types";
import { earthRadiusKm } from "utils/Constants";
import { toDegrees, toRadians } from "utils/HeadingUtils";

// Using equi-rectangular approximation as in www.movable-type.co.uk/scripts/latlong.html
export function pairedHaversine(
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number,
): number {
  lon1 = toRadians(lon1);
  lat1 = toRadians(lat1);
  lon2 = toRadians(lon2);
  lat2 = toRadians(lat2);

  const x = (lon2 - lon1) * Math.cos(0.5 * (lat2 + lat1));
  const y = lat2 - lat1;
  const distanceKm = earthRadiusKm * Math.sqrt(x * x + y * y);

  return distanceKm;
}

// Convert nautical miles to kilometres
export function nauticalMilesToKm(distanceNm: number): number {
  return distanceNm * 1.852;
}

// Convert kilometres to nautical miles
export function kmToNauticalMiles(distanceKm: number): number {
  return distanceKm / 1.852;
}

// Move an aircraft from one position to another
export function moveLocation(
  lon: number,
  lat: number,
  headingDeg: number,
  distanceKm: number,
  earthRadiusKm: number,
): Coordinate {
  // Convert to radians
  const latRad = toRadians(lat);
  const lonRad = toRadians(lon);
  const headingRad = toRadians(headingDeg);

  // Compute distances and headings
  const cosLatRad = Math.cos(latRad);
  const sinLatRad = Math.sin(latRad);

  const normDist = distanceKm / earthRadiusKm;
  const cosNormDist = Math.cos(normDist);
  const sinNormDist = Math.sin(normDist);

  const cosHeadingRad = Math.cos(headingRad);
  const sinHeadingRad = Math.sin(headingRad);

  // New latitude and longitude
  let latNew = Math.asin(
    sinLatRad * cosNormDist + cosLatRad * sinNormDist * cosHeadingRad,
  );
  let lonNew =
    lonRad +
    Math.atan2(
      sinHeadingRad * sinNormDist * cosLatRad,
      cosNormDist - sinLatRad * Math.sin(latNew),
    );

  // Convert to degrees
  latNew = toDegrees(latNew);
  lonNew = toDegrees(lonNew);

  return [lonNew, latNew];
}

// Get projected distances between two points in pixels, given a d3 GeoProjection
export function getProjectedDistances(
  projection: d3.GeoProjection,
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number,
): Coordinate {
  const projection1 = projection([lon1, lat1]);
  const projection2 = projection([lon2, lat2]);

  const deltaLonPixels = projection2[0] - projection1[0];
  const deltaLatPixels = projection2[1] - projection1[1];

  return [deltaLonPixels, deltaLatPixels];
}

// Get projected distance a vector length in mins away from an aircraft
export function getAircraftNextPixelPosition(
  projection: d3.GeoProjection,
  aircraft: Aircraft,
  vectorLengthMins: number,
): Coordinate {
  const vectorLengthHours = vectorLengthMins / 60;
  const distanceKm = nauticalMilesToKm(
    aircraft.true_air_speed * vectorLengthHours,
  );
  const futurePosition = moveLocation(
    aircraft.lon,
    aircraft.lat,
    aircraft.ground_track,
    distanceKm,
    earthRadiusKm,
  );

  return getProjectedDistances(
    projection,
    aircraft.lon,
    aircraft.lat,
    futurePosition[0],
    futurePosition[1],
  );
}

// Get the range ring radius in pixels around an aircraft given its size in kilometers
export function getRangeRingPixelRadius(
  projection: d3.GeoProjection,
  aircraft: Aircraft,
  ringRadiusKm: number,
): number {
  const rangeRingPosition = moveLocation(
    aircraft.lon,
    aircraft.lat,
    aircraft.heading,
    ringRadiusKm,
    earthRadiusKm,
  );
  const projectedDistances = getProjectedDistances(
    projection,
    aircraft.lon,
    aircraft.lat,
    rangeRingPosition[0],
    rangeRingPosition[1],
  );

  return Math.hypot(projectedDistances[0], projectedDistances[1]);
}

// Function to check whether aircraft is close to selected aircraft in terms of FL
export function isVerticallyClose(
  aircraft: Aircraft,
  selectedAircraftObject: Aircraft,
  verticalThreshold: number,
): boolean {
  return (
    Math.abs(aircraft.flight_level - selectedAircraftObject.flight_level) <
    verticalThreshold
  );
}

export function calculateSeparation(aircraftPair: Aircraft[]): {
  distanceLateralNm: number;
  distanceFL: number;
} {
  const craft1 = aircraftPair[0];
  const craft2 = aircraftPair[1];

  // Calculate lateral separation between two aircraft in Nm
  const distanceLateralKm: number = pairedHaversine(
    craft1.lon,
    craft1.lat,
    craft2.lon,
    craft2.lat,
  );
  const distanceLateralNm: number = kmToNauticalMiles(distanceLateralKm);

  // Calculate the flight level difference
  const distanceFL: number = Math.abs(
    craft1.flight_level - craft2.flight_level,
  );

  return { distanceLateralNm, distanceFL };
}
