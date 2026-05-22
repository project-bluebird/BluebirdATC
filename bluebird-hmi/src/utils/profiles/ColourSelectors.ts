import { Aircraft, ColourProfile, OpacityProfile } from "utils/types";
import { isVerticallyClose } from "utils/DistanceUtils";
import { getColourForFL } from "utils/profiles/ColourInterpolators";
import { gameColourProfile } from "utils/profiles/ColourProfiles";

export function getAircraftColour(
  aircraft: Aircraft,
  colourProfile: ColourProfile,
  isColourByFL = false,
  colourScaleName?: string,
  selectedAircraft: string = null,
): string {
  if (isColourByFL) {
    return getColourForFL(aircraft.flight_level, colourScaleName);
  }

  switch (aircraft.bay) {
    case "PENDING":
      return colourProfile.aircraftPendingColour;
    case "INCOMM":
      return colourProfile.aircraftIncommColour;
    case "OUTCOMM":
      return colourProfile.aircraftOutcommColour;
    // This is traffic belonging to other sectors in the scenario or purely background traffic
    default:
      return colourProfile.aircraftBackgroundColour;
  }
}

export function getSelectedFlightLevelColour(
  aircraft: Aircraft,
  colourProfile: ColourProfile,
  isColourByFL = false,
  colourScaleName?: string,
): string {
  if (isColourByFL) {
    return getColourForFL(aircraft.flight_level, colourScaleName);
  }

  switch (aircraft.bay) {
    case "PENDING":
      return colourProfile.sflPendingColour;
    case "INCOMM":
      return colourProfile.sflIncommColour;
    case "OUTCOMM":
      return colourProfile.sflOutcommColour;
    default:
      return colourProfile.aircraftBackgroundColour;
  }
}

export function getExitFlightLevelColour(
  aircraft: Aircraft,
  colourProfile: ColourProfile,
  isColourByFL = false,
  colourScaleName?: string,
): string {
  if (isColourByFL) {
    return getColourForFL(aircraft.flight_level, colourScaleName);
  }

  switch (aircraft.bay) {
    case "PENDING":
      return colourProfile.xflPendingColour;
    case "INCOMM":
      return colourProfile.xflIncommColour;
    case "OUTCOMM":
      return colourProfile.xflOutcommColour;
    default:
      return colourProfile.xflIncommColour;
  }
}

// Function to get opacity level of aircraft based on the flight level filter and the selected aircraft
export function getAircraftOpacity(
  allAircraft: Aircraft[],
  aircraft: Aircraft,
  opacityProfile: OpacityProfile,
  selectedAircraft: string,
  filterByFlightLevel: boolean,
) {
  const selectedAircraftObject = allAircraft.find(
    (a: Aircraft) => a.callsign === selectedAircraft,
  );
  // If filtering button is not depressed and aircraft object is undefined return full opacity
  if (!filterByFlightLevel || selectedAircraftObject === undefined) {
    return opacityProfile.fullOpacity;
  }

  // If aircraft is selected aircraft, return full opacity
  if (aircraft.callsign === selectedAircraftObject.callsign) {
    return opacityProfile.fullOpacity;
  }

  // If button is depressed and aircraft is vertically close to the selected aircraft, return full opacity
  const verticalThreshold = 10;
  if (
    filterByFlightLevel &&
    isVerticallyClose(aircraft, selectedAircraftObject, verticalThreshold)
  ) {
    return opacityProfile.fullOpacity;
  }

  // Otherwise return opacity of 0 (do not display)
  return opacityProfile.noOpacity;
}

// Function to get the label opacity level for flight level filter
export function getLabelOpacity(
  label: number,
  filterByFlightLevel: boolean,
  aircraft: Aircraft,
  flightState: string,
  opacityProfile: OpacityProfile,
) {
  if (!filterByFlightLevel) {
    return opacityProfile.fullOpacity;
  } else if (flightState !== "level") {
    return opacityProfile.fullOpacity;
  } else if (
    filterByFlightLevel &&
    Math.abs(aircraft.flight_level - label) >= 10
  ) {
    return opacityProfile.noOpacity;
  }
}

// Change alpha channel of a hex colour
export function changeAlpha(hexColor: string, alpha: number): string {
  const sanitizedHex = hexColor.replace(/^#/, "");
  const rgbaColor = `rgba(${parseInt(sanitizedHex.slice(0, 2), 16)}, ${parseInt(
    sanitizedHex.slice(2, 4),
    16,
  )}, ${parseInt(sanitizedHex.slice(4, 6), 16)}, ${alpha})`;

  return rgbaColor;
}
