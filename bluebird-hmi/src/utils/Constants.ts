// Related to the geographic projection (WGS-84 elipsoid)
export const earthRadiusKm = 6378.137;
export const earthFlattening = 1 / 298.257223563;
// Longitude and Latitude of reference point for projection
export const theta0 = -(27.0 / 60.0 + 41.0 / 3600.0);
export const phi0 = 55.0;
// Scale factor for svg attributes such as text and line widths
export const projectionScaleFactor = 800;
// conversion factor here to mitigate "loss of precision" errors.
export const radiansPerDegree = Math.PI / 180;
// game settings
export const minLateralSeparationNm = 5.0;
export const minFlightLevelSeparation = 10;
export const warningLateralSeparationNm = 10.0;
export const warningFlightLevelSeparation = 15;
// colour bar when colouring by Flight Level
export const colourBarMinFL = 210;
export const colourBarMaxFL = 450;
export const defaultColourScaleName = "d3.interpolateWarm";
export const defaultLonLatOffset = [-40, 0];
// Text to speech play back rate multiplier
export const TTSPlaybackRate = 1.5;

export enum InfoBlockOptions {
  ShowBlock = "Show block",
  HideBlock = "Hide block",
  HideGroundSpeed = "Hide ground speed",
  SimplifiedBlock = "Simplified block",
}

export const defaultInfoBlockState = InfoBlockOptions.ShowBlock;
