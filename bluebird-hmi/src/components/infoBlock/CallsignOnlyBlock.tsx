import { Aircraft } from "utils/types";

import { appendText } from "./builders";

export function callsignOnlyInfoBlock(
  aircraft: Aircraft,
  group: d3.Selection<d3.BaseType | SVGElement, unknown, null, undefined>,
  aircraftColour: string,
) {
  // Callsign
  appendText(group, 0, 22, aircraft.callsign, aircraftColour);
}
