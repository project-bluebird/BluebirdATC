import { Aircraft } from "utils/types";

import { appendText } from "components/infoBlock/builders";

export function simplifiedInfoBlock(
  aircraft: Aircraft,
  group: d3.Selection<d3.BaseType | SVGElement, unknown, null, undefined>,
  aircraftColour: string,
) {
  // Callsign
  appendText(group, 0, 22, aircraft.callsign, aircraftColour);

  // Flight level
  appendText(group, 0, 34, Math.round(aircraft.flight_level), aircraftColour);
}
