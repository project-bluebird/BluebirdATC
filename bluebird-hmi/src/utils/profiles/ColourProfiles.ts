import { ColourProfile, OpacityProfile } from "utils/types";
import { defaultColourScaleName } from "utils/Constants";

export const radarColourProfile: ColourProfile = {
  backgroundColour: "#000000",
  sflPendingColour: "#eda539ad",
  sflIncommColour: "#eda539ff",
  sflOutcommColour: "#eda53952",
  xflPendingColour: "#5598f8cc",
  xflIncommColour: "#5598f8ff",
  xflOutcommColour: "#5598f859",
  aircraftPendingColour: "#009bb0",
  aircraftIncommColour: "#00b028",
  aircraftOutcommColour: "#1434A4",
  aircraftBackgroundColour: "#81a65d59",
  coastlineColour: "#136155",
  landColour: "none",
  colourBarAxisColour: "#ffffff",
  sectorBoundaryColour: "#136155",
  sectorHighlightBoundaryColour: "#0d8835",
  routeColour: "#ffffff",
  selectionColour: "#6bb87cff",
  fixColour: "#55D7D7",
  planColour: "#FF00FF",
  windowColour: "#FFFF00",
  actionIndicatorColour: "#eda53977",
  actionLogColour: "#ffffff",
  actionLogBorderColour: "#ffffff",
  actionLogIconColour: "#828282",
  actionLogIconHoverColour: "#000000",
  defaultColourScaleName: defaultColourScaleName,
};

export const presentationColourProfile: ColourProfile = {
  backgroundColour: "#D3E7F6",
  sflPendingColour: "#eda539cc",
  sflIncommColour: "#eda539ff",
  sflOutcommColour: "#eda53977",
  xflPendingColour: "#5598f8cc",
  xflIncommColour: "#5598f8ff",
  xflOutcommColour: "#5598f877",
  aircraftPendingColour: "#009bb0",
  aircraftIncommColour: "#00b028",
  aircraftOutcommColour: "#1434A4",
  aircraftBackgroundColour: "#989898",
  coastlineColour: "#2C5680",
  landColour: "#FFFFFF",
  colourBarAxisColour: "#000000",
  sectorBoundaryColour: "#134532",
  sectorHighlightBoundaryColour: "#0c2e21",
  routeColour: "#FB9902",
  selectionColour: "#6bb87cff",
  fixColour: "#136155",
  planColour: "#B90194",
  windowColour: "#1102F7",
  actionIndicatorColour: "#eda53977",
  actionLogColour: "#ffffff",
  actionLogBorderColour: "#134532",
  actionLogIconColour: "#828282",
  actionLogIconHoverColour: "#000000",
  defaultColourScaleName: defaultColourScaleName,
};

export const gameColourProfile: ColourProfile = {
  ...radarColourProfile,
  aircraftSelectedColour: "#8fffadff",
  aircraftProblemColour: "#ff0000ff",
  aircraftWarningColour: "#ff9500ff",
  exitLineColour: "#ff9500ff",
};

export const radarOpacityProfile: OpacityProfile = {
  actionLogOpacity: 0.7,
  noOpacity: 0,
  fullOpacity: 1.0,
  overlayOpacity: 0.6,
  gameDrawerOpacity: 0.45,
};

export const presentationOpacityProfile: OpacityProfile = {
  ...radarOpacityProfile,
  actionLogOpacity: 1.0,
};

export const gameOpacityProfile: OpacityProfile = {
  ...radarOpacityProfile,
};

export const stripBorderColourProfile: ColourProfile = {
  east: "orange",
  west: "blue",
};
