export function toRadians(degrees: number): number {
  return degrees * (Math.PI / 180);
}

export function toDegrees(radians: number): number {
  return radians * (180 / Math.PI);
}

export function wrappedLongitudeDelta(
  lon1Deg: number,
  lon2Deg: number,
): number {
  let deltaLon = lon1Deg - lon2Deg;
  if (deltaLon > 180) {
    deltaLon -= 360;
  } else if (deltaLon < -180) {
    deltaLon += 360;
  }

  return deltaLon;
}

export function formatHeading(heading: number | string): string {
  return typeof heading === "number" ? Math.round(heading).toString() : heading;
}
