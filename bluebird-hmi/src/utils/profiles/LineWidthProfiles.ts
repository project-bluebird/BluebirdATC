import { LineWidthProfile } from "utils/types";

export const radarLineWidthProfile: LineWidthProfile = {
  actionIndicatorLineWidth: 3,
  coastlineLineWidth: 1,
  fixLineWidth: 1,
  infoBoxLineWidth: 1,
  planLineWidth: 1,
  previousPositionLineWidth: 2,
  rangeMarkerLineWidth: 1.5,
  rangeRingLineWidth: 1.5,
  routeLineWidth: 1,
  sectorBoundaryLineWidth: 2,
  sectorHighlightBoundaryLineWidth: 3,
  selectionBoxLineWidth: 2,
  vectorLineWidth: 1.5,
};

export const presentationLineWidthProfile: LineWidthProfile = {
  actionIndicatorLineWidth: 3,
  coastlineLineWidth: 2,
  fixLineWidth: 2,
  infoBoxLineWidth: 2,
  planLineWidth: 2,
  previousPositionLineWidth: 2,
  rangeMarkerLineWidth: 2,
  rangeRingLineWidth: 2,
  routeLineWidth: 2,
  sectorBoundaryLineWidth: 2,
  sectorHighlightBoundaryLineWidth: 4,
  selectionBoxLineWidth: 2,
  vectorLineWidth: 2,
};

export const gameLineWidthProfile: LineWidthProfile = {
  ...radarLineWidthProfile,
  headingLineWidth: 2,
  overlayTriangleLineWidth: 2,
};
