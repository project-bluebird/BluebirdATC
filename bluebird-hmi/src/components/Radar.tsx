import { useAppSelector } from "app/hooks";
import bblogo from "assets/bluebird_logo.png";
import ActionHighlights from "components/radarElements/ActionHighlights";
import AircraftMarkerGroup from "components/radarElements/AircraftMarkerGroup";
import CoastlineGroup from "components/radarElements/CoastlineGroup";
import FixGroup from "components/radarElements/FixGroup";
import ActionLog from "components/overlays/ActionLog";
import ColourBarGroup from "components/overlays/ColourBarGroup";
import classes from "components/Radar.module.css";
import RangeMarkerGroup from "components/radarElements/RangeMarkerGroup";
import RouteGroup from "components/radarElements/RouteGroup";
import SectorBoundaryGroup from "components/radarElements/SectorBoundaryGroup";
import * as d3 from "d3";
import { Box, Typography } from "@mui/material";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { selectActions } from "slices/dynamicDataSlice";
import {
  selectExists,
  selectFixes,
  selectProjectionCentre,
  selectScenarioName,
  selectSectors,
} from "slices/staticDataSlice";
import { selectBandboxSectorId } from "slices/scenarioSlice";
import { TimedActions, ToolCallsignStore } from "utils/types";
import {
  selectColourProfile,
  selectLineWidthProfile,
  selectOpacityProfile,
} from "utils/profiles/ProfileSelectors";

import AircraftRangeRingsGroup from "components/radarElements/AircraftRangeRingsGroup";
import AircraftTrailGroup from "components/radarElements/AircraftTrailGroup";
import AircraftVectorLinesGroup from "components/radarElements/AircraftVectorLinesGroup";
import AircraftInfoBlockSimple from "components/infoBlock/AircraftInfoBlockSimple";

interface RadarProps {
  toolCallsigns: ToolCallsignStore;
  showFixNames: boolean;
  showSectorBoundary: boolean;
  showCoastline: boolean;
  showRangeMarkers: boolean;
  showActionLog: boolean;
  showActionHighlight: boolean;
  colourByFL: boolean;
  resetTrigger: number;
  vectorLength: number;
  profileName: string;
  aircraftMarkerName: string;
  selectedColourScaleName: string;
  infoBlockState: string;
  showFuelBurn?: boolean;
  showTechnicalSafety?: boolean;
  showNumClearances?: boolean;
  showComplexityGraph?: boolean;
  headingKeyCount?: number;
  filterByFlightLevel?: boolean;
}

export default function Radar(props: RadarProps) {
  const {
    toolCallsigns,
    showFixNames,
    showSectorBoundary,
    showCoastline,
    showRangeMarkers,
    showActionLog,
    showActionHighlight,
    colourByFL,
    resetTrigger,
    vectorLength,
    profileName,
    aircraftMarkerName,
    selectedColourScaleName,
    infoBlockState,
    filterByFlightLevel,
  } = props;

  // Flag for first rendering, to avoid double-refresh on Firefox.
  const isInitialRender = useRef(true);
  const actions: TimedActions[] = useAppSelector(selectActions);
  const [width] = useState(window.innerWidth);
  const [height] = useState(window.innerHeight);
  const scaleFactor = 20000;
  const [scale, setScale] = useState(scaleFactor);
  const [translateX, setTranslateX] = useState(width / 2);
  const [translateY, setTranslateY] = useState(height / 2);
  const [initialTranslateX, setInitialTranslateX] = useState(0);
  const [initialTranslateY, setInitialTranslateY] = useState(0);

  // Get the scenarioName
  const scenarioName = useAppSelector(selectScenarioName);

  // Get the projection centre, if there is one
  const projectionCentre = useAppSelector(selectProjectionCentre);

  // Find display centre: either centre of fixes, or centre of sector
  const staticDataExists = useAppSelector(selectExists);
  const fixes = useAppSelector(selectFixes);
  const sectors = useAppSelector(selectSectors);
  const bandboxSectorId = useAppSelector(selectBandboxSectorId);

  // Take centre point as centre of fixes
  let lats: number[] = fixes.map(({ lat }) => lat);
  let lons: number[] = fixes.map(({ lon }) => lon);

  // Find the centre of the lats and lons
  const maxLat: number = Math.max(...lats);
  const minLat: number = Math.min(...lats);
  const latCentre: number = (maxLat + minLat) / 2;
  const maxLon: number = Math.max(...lons);
  const minLon: number = Math.min(...lons);
  const lonCentre: number = (maxLon + minLon) / 2;

  // Reference point for stereographic projection - if not set
  // via static data, take the centre of the fixes as fallback.
  let [theta0, phi0] = [0, 0];
  if (projectionCentre) {
    [theta0, phi0] = projectionCentre;
  } else if (!isNaN(lonCentre) && !isNaN(latCentre)) {
    [theta0, phi0] = [lonCentre, latCentre];
  }
  // Rotations for projection - needed for in order
  // to have lines of lon/lat roughly vertical/horizontal
  const [lambda, phi, gamma] = [-1 * theta0, phi0, 0];

  // D3 GeoProjection, to convert lon/lat to pixel positions

  let projection = d3
    .geoStereographic()
    .scale(scale)
    .center([theta0, phi0])
    .rotate([lambda, phi, gamma])
    .translate([translateX, translateY]);

  // Create d3 zoom function
  const scaleZoom = useCallback(
    (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
      if (event.sourceEvent === null) return;
      setScale(event.transform.k * scaleFactor);
      setTranslateX(event.transform.k * (width * 0.5) + event.transform.x);
      setTranslateY(event.transform.k * (height * 0.5) + event.transform.y);
    },
    [setScale, setTranslateX, setTranslateY, scaleFactor, width, height],
  );
  const zoom = useMemo(() => d3.zoom().on("zoom", scaleZoom), [scaleZoom]);

  // For stereographic projection, ensure that we are centred on the average lon,lat
  // of the fixes, as opposed to the centre of the projection (defined by theta0, phi0).
  useEffect(() => {
    if (!isNaN(lonCentre) && !isNaN(latCentre)) {
      console.log("calculating screen position ", lonCentre, latCentre);
      // Calculate the screen coordinates of the initial position
      const initialScreenCoords = projection([lonCentre, latCentre]);
      // Set these values so that first pan/zoom is centred on the correct location
      setInitialTranslateX(width / 2 - initialScreenCoords[0]);
      setInitialTranslateY(height / 2 - initialScreenCoords[1]);
      // We also have to set the following so the view on first render is centred correctly.
      setTranslateX(width - initialScreenCoords[0]);
      setTranslateY(height - initialScreenCoords[1]);
    }
  }, [lonCentre, latCentre]); // eslint-disable-line react-hooks/exhaustive-deps

  // Select colourProfile, lineWidthProfile and opacityProfile
  const colourProfile = selectColourProfile(profileName);
  const lineWidthProfile = selectLineWidthProfile(profileName);
  const opacityProfile = selectOpacityProfile(profileName);

  // Background colour - default is black
  const [backgroundColour, setBackgroundColour] = useState(
    colourProfile.backgroundColour,
  );

  // reload the page when we change sector - seems to be the most reliable
  // way of ensuring we are centred in the right place, with scroll/zoom enabled.
  useEffect(() => {
    if (isInitialRender.current) {
      isInitialRender.current = false;
    }
  }, [bandboxSectorId, staticDataExists]);

  useEffect(() => {
    // Set background colour for chosen profile
    setBackgroundColour(
      profileName === "presentation" && !showCoastline
        ? "white"
        : colourProfile.backgroundColour,
    );
  }, [profileName, showCoastline, colourProfile]);

  useEffect(() => {
    // Create initial group for canvas and make it zoom-able.
    const svgElement = d3.select("#radar");

    svgElement.call(zoom);
    // For stereographic projection, ensure that zooming centers on the mouse cursor
    // rather than the centre of the projection.

    const initialTransform = d3.zoomIdentity.translate(
      initialTranslateX,
      initialTranslateY,
    );
    svgElement.call(
      zoom.transform as unknown as (
        selection: d3.Selection<SVGSVGElement, unknown, null, undefined>,
        transform,
      ) => void,
      initialTransform,
    );
  }, [scaleZoom, zoom, initialTranslateX, initialTranslateY]);

  // Only render radar when projection has been initialised correctly. This prevents console errors.
  if (
    projection !== undefined &&
    !isNaN(projection.center()[0]) &&
    !isNaN(projection.center()[1])
  ) {
    return (
      <>
        <svg id="radar" className={classes.radar}>
          <rect width="100%" height="100%" fill={backgroundColour} />
          {showCoastline && (
            <>
              <CoastlineGroup
                colourProfile={colourProfile}
                lineWidthProfile={lineWidthProfile}
                projection={projection}
              />
            </>
          )}
          <FixGroup
            colourProfile={colourProfile}
            lineWidthProfile={lineWidthProfile}
            showFixNames={showFixNames}
            projection={projection}
          />
          {showSectorBoundary && (
            <SectorBoundaryGroup
              colourProfile={colourProfile}
              lineWidthProfile={lineWidthProfile}
              projection={projection}
            />
          )}
          {showRangeMarkers && (
            <RangeMarkerGroup
              colourProfile={colourProfile}
              lineWidthProfile={lineWidthProfile}
              projection={projection}
            />
          )}
          {toolCallsigns.routes.length > 0 && (
            <RouteGroup
              toolCallsigns={toolCallsigns}
              colourProfile={colourProfile}
              colourByFL={colourByFL}
              lineWidthProfile={lineWidthProfile}
              selectedColourScaleName={selectedColourScaleName}
              profileName={profileName}
              projection={projection}
            />
          )}
          <AircraftInfoBlockSimple
            colourProfile={colourProfile}
            opacityProfile={opacityProfile}
            lineWidthProfile={lineWidthProfile}
            colourByFL={colourByFL}
            scale={scale}
            selectedColourScaleName={selectedColourScaleName}
            infoBlockState={infoBlockState}
            projection={projection}
            resetTrigger={resetTrigger}
            filterByFlightLevel={filterByFlightLevel}
          />
          <AircraftMarkerGroup
            colourProfile={colourProfile}
            opacityProfile={opacityProfile}
            lineWidthProfile={lineWidthProfile}
            colourByFL={colourByFL}
            selectedColourScaleName={selectedColourScaleName}
            aircraftMarkerName={aircraftMarkerName}
            projection={projection}
            filterByFlightLevel={filterByFlightLevel}
          />

          <AircraftTrailGroup
            colourProfile={colourProfile}
            opacityProfile={opacityProfile}
            lineWidthProfile={lineWidthProfile}
            colourByFL={colourByFL}
            selectedColourScaleName={selectedColourScaleName}
            projection={projection}
            filterByFlightLevel={filterByFlightLevel}
          />

          <AircraftRangeRingsGroup
            toolCallsigns={toolCallsigns}
            colourProfile={colourProfile}
            opacityProfile={opacityProfile}
            lineWidthProfile={lineWidthProfile}
            colourByFL={colourByFL}
            selectedColourScaleName={selectedColourScaleName}
            projection={projection}
            filterByFlightLevel={filterByFlightLevel}
          />

          <AircraftVectorLinesGroup
            toolCallsigns={toolCallsigns}
            colourProfile={colourProfile}
            lineWidthProfile={lineWidthProfile}
            colourByFL={colourByFL}
            selectedColourScaleName={selectedColourScaleName}
            vectorLength={vectorLength}
            projection={projection}
          />

          {showActionHighlight && (
            <ActionHighlights
              colourProfile={colourProfile}
              lineWidthProfile={lineWidthProfile}
              projection={projection}
              opacityProfile={opacityProfile}
            />
          )}
        </svg>
        {colourByFL && (
          <ColourBarGroup
            colourProfile={colourProfile}
            selectedColourScaleName={selectedColourScaleName}
          />
        )}
        {showActionLog && actions.length > 0 && (
          <ActionLog
            colourProfile={colourProfile}
            opacityProfile={opacityProfile}
          />
        )}
      </>
    );
  } else {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          minHeight: "100vh",
          backgroundColor: "black",
        }}
      >
        <img src={bblogo} alt="App logo" width="80%" />
        <Typography variant="h5" sx={{ mt: 2, color: "white" }}>
          Welcome to BluebirdATC
        </Typography>
        <Typography variant="body1" color="white">
          Use the left sidebar to load a scenario.
        </Typography>
      </Box>
    );
  }
}
