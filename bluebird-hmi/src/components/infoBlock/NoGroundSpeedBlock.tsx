import { Aircraft, ColourProfile } from "utils/types";
import {
  getExitFlightLevelColour,
  getSelectedFlightLevelColour,
} from "utils/profiles/ColourSelectors";

import { appendText } from "./builders";

export function hideGroundSpeedBlock(
  aircraft: Aircraft,
  group: d3.Selection<d3.BaseType | SVGElement, unknown, null, undefined>,
  aircraftColour: string,
  colourProfile: ColourProfile,
  colourByFL: boolean,
  selectedColourScaleName: string,
) {
  // Callsign
  appendText(group, 0, 22, aircraft.callsign, aircraftColour);

  // Flight level
  appendText(group, 0, 34, Math.round(aircraft.flight_level), aircraftColour);

  // Entry flight level
  if (aircraft.entry_flight_level !== null && aircraft.bay === "PENDING") {
    appendText(group, 0, 46, aircraft.entry_flight_level, aircraftColour);
  }

  // Intention code
  appendText(group, 46, 34, aircraft.intention_code, aircraftColour);

  // Exit flight level
  if (aircraft.exit_flight_level !== null) {
    const exitFlightLevelColour = getExitFlightLevelColour(
      aircraft,
      colourProfile,
      colourByFL,
      selectedColourScaleName,
    );
    appendText(
      group,
      73,
      22,
      Math.round(0.1 * aircraft.exit_flight_level),
      exitFlightLevelColour,
    );
  }

  // Selected flight level
  appendText(
    group,
    73,
    34,
    Math.round(aircraft.selected_flight_level),
    getSelectedFlightLevelColour(
      aircraft,
      colourProfile,
      colourByFL,
      selectedColourScaleName,
    ),
  );
}
