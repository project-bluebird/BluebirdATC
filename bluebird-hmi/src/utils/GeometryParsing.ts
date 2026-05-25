import { Coordinate, GeojsonFeature } from "utils/types";

// Extract coordinates from standard geojson data structure
export function getFeatureCoordinates(feature: GeojsonFeature) {
  return feature.geometry.coordinates;
}

// Offset coordinate list by a fixed amount
export function offsetCoordinates(
  coordinateArray: Coordinate[],
  lonLatOffset: Coordinate = [0, 0],
) {
  return coordinateArray.map((coordinate: Coordinate) => [
    coordinate[0] + lonLatOffset[0],
    coordinate[1] + lonLatOffset[1],
  ]);
}

// Extract & adjust coordinates of range marker label locations from range marker line paths
export function getRangeMarkerTickCoordinates(
  rangeMarkerCoordinates: Coordinate[][],
) {
  const allTickCoordinates: Coordinate[][] = [];
  rangeMarkerCoordinates.forEach((pathCoordinates: Coordinate[]) => {
    const tickCoordinates: Coordinate[] = [];

    const firstPair = pathCoordinates[3];
    const secondPair = pathCoordinates[6];
    const lastPair = pathCoordinates[9];

    const tickBottom = pathCoordinates[1][1];
    const heightAdjustment = 0.25 * (lastPair[1] - tickBottom);
    const horizontalAdjustment = 0.04 * (lastPair[0] - secondPair[0]);

    const adjustedFirstPair: Coordinate = [
      firstPair[0],
      lastPair[1] - heightAdjustment,
    ];
    const adjustedSecondPair: Coordinate = [
      secondPair[0],
      lastPair[1] - heightAdjustment,
    ];
    const adjustedLastPair: Coordinate = [
      lastPair[0] - horizontalAdjustment,
      lastPair[1] - heightAdjustment,
    ];

    tickCoordinates.push(adjustedFirstPair);
    tickCoordinates.push(adjustedSecondPair);
    tickCoordinates.push(adjustedLastPair);

    allTickCoordinates.push(tickCoordinates);
  });

  return allTickCoordinates;
}
