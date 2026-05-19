import * as d3 from "d3";

// Dictionary mapping colour scale strings to d3 functions
export const colourScaleFunctions: Record<string, (t: number) => string> = {
  "d3.interpolateWarm": d3.interpolateWarm,
  "d3.interpolateCool": d3.interpolateCool,
  "d3.interpolateYlGnBu": d3.interpolateYlGnBu,
  "d3.interpolateYlOrRd": d3.interpolateYlOrRd,
  "d3.interpolateTurbo": d3.interpolateTurbo,
  "d3.interpolateViridis": d3.interpolateViridis,
  "d3.interpolateInferno": d3.interpolateInferno,
  "d3.interpolateMagma": d3.interpolateMagma,
  "d3.interpolatePlasma": d3.interpolatePlasma,
  "d3.interpolateCividis": d3.interpolateCividis,
  "d3.interpolateCubehelixDefault": d3.interpolateCubehelixDefault,
  "d3.interpolateGnBu": d3.interpolateGnBu,
  "d3.interpolateOrRd": d3.interpolateOrRd,
};
