import Box from "@mui/material/Box";
import { useAppSelector } from "app/hooks";
import RadarDrawer from "components/radarDrawer/RadarDrawer";
import Radar from "components/Radar";
import { ScenarioSelectModal } from "components/scenarioSelection/ScenarioSelectModal";
import {
  addCallsigns,
  removeAllCallsigns,
  toggleCallsign,
} from "components/ToolStoreHandlers";
import { initialToolCallsignStore } from "utils/initialState";
import { useCallback, useEffect, useState } from "react";
import { selectAircraft } from "slices/dynamicDataSlice";
import { selectScenarioName } from "slices/staticDataSlice";
import {
  selectCategory,
  selectIndividualSectorIds,
  selectRunning,
} from "slices/scenarioSlice";
import { ToolCallsignStore } from "utils/types";
import { compareArraysAsSets } from "utils/CompareArrays";
import { defaultColourScaleName, defaultInfoBlockState } from "utils/Constants";

export default function RadarPage() {
  document.title = "Radar";
  const drawerWidth = 250;

  // Get data from redux
  const individualSectorId = useAppSelector(selectIndividualSectorIds);
  const isRunning = useAppSelector(selectRunning);
  const scenarioCategory = useAppSelector(selectCategory);
  const scenarioName = useAppSelector(selectScenarioName);
  const selectedAircraft = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  const allAircraft = useAppSelector(selectAircraft);
  const allCallsigns: string[] = allAircraft.map(
    (aircraft) => aircraft.callsign,
  );
  // Set up variables to control tools that do not function by callsign
  const [showActionLog, setActionLog] = useState(false);
  const [showActionHighlight, setActionHighlight] = useState(false);
  const [showFixNames, setShowFixNames] = useState(false);
  const [showSectorBoundary, setShowSectorBoundary] = useState(true);
  const [showCoastline, setShowCoastline] = useState(false);
  const [showRangeMarkers, setShowRangeMarkers] = useState(false);
  const initialVectorLength = 5;
  const [vectorLength, setVectorLength] = useState(initialVectorLength);
  const [profileName, setProfileName] = useState("default");
  const [aircraftMarkerName, setAircraftMarkerName] = useState("starMarker");
  const [colourByFL, setColourByFL] = useState(false);
  const [selectedColourScaleName, setSelectedColourScaleName] =
    useState<string>(defaultColourScaleName);
  const [infoBlockState, setInfoBlockState] = useState<string>(
    defaultInfoBlockState,
  );

  // modal for selecting scenarios
  const [isScenarioModalOpen, setScenarioModalOpen] = useState(false);

  // Set up store to keep track of which tools are active (by callsign)
  const [toolCallsigns, setToolCallsigns] = useState<ToolCallsignStore>(
    initialToolCallsignStore,
  );

  // We need to keep track of whether the global tools are on or off
  const [showGlobalVectors, setShowGlobalVectors] = useState(false);
  const [showGlobalRings, setShowGlobalRings] = useState(false);

  // Set up functions to set state
  const toggleActionLog = () => setActionLog((prevState) => !prevState);

  const toggleFixNames = () => {
    setShowFixNames((prevState) => !prevState);
  };
  const toggleSectorBoundary = () => {
    setShowSectorBoundary((prevState) => !prevState);
  };
  const toggleCoastline = () => {
    setShowCoastline((prevState) => !prevState);
  };
  const toggleGlobalVectors = () => {
    setShowGlobalVectors((prevState) => !prevState);
  };
  const toggleGlobalRings = () => {
    setShowGlobalRings((prevState) => !prevState);
  };
  const toggleRangeMarkers = () => {
    setShowRangeMarkers((prevState) => !prevState);
  };
  const toggleColourByFL = () => {
    setColourByFL((prevState) => !prevState);
  };

  const toggleShowActionHighlight = () => {
    setActionHighlight((n) => !n);
  };

  // Set up function to toggle between default and presentation profiles
  const toggleProfile = () => {
    const newProfileName =
      profileName === "default" ? "presentation" : "default";
    setProfileName(newProfileName);
  };

  // Set up function to toggle between a star marker and an aircraft icon marker
  const toggleMarker = () => {
    const newAircraftMarkerName =
      aircraftMarkerName === "starMarker" ? "aircraftIcon" : "starMarker";
    setAircraftMarkerName(newAircraftMarkerName);
  };

  // Reset mapping tools to initial state
  const resetMappingTools = useCallback(() => {
    setShowSectorBoundary(true);
    setShowFixNames(false);
    setShowCoastline(false);
    setShowRangeMarkers(false);
  }, [scenarioCategory]);

  // Reset theme tools to initial state
  const resetThemeTools = () => {
    setColourByFL(false);
    setProfileName(() => "default");
    setAircraftMarkerName(() => "starMarker");
    setSelectedColourScaleName(defaultColourScaleName);
    setInfoBlockState(defaultInfoBlockState);
  };

  // Set up a reset trigger to pass to Radar.tsx
  const [resetTrigger, setResetTrigger] = useState(0);

  // Reset ATC tools
  const resetATCTools = useCallback(() => {
    // update the resetTrigger that we pass to Radar.tsx
    setResetTrigger((trigger) => trigger + 1);
    // reset tools to initial state
    setActionLog(false);
    setActionHighlight(false);
    setShowGlobalVectors(false);
    setShowGlobalRings(false);
    setVectorLength(() => initialVectorLength);
    // reset tools that are applied by callsign
    removeAllCallsigns(setToolCallsigns, "vectorLines");
    removeAllCallsigns(setToolCallsigns, "rangeRings");
    removeAllCallsigns(setToolCallsigns, "routes");
  }, [initialVectorLength, setToolCallsigns]);

  // Reset tools when sectorId changes
  useEffect(() => {
    resetATCTools();
    resetMappingTools();
  }, [individualSectorId, resetATCTools, resetMappingTools]);

  // Create handlers for handling the global vector and range ring buttons
  const handleGlobalVectors = () => {
    toggleGlobalVectors();
    if (!showGlobalVectors) {
      addCallsigns(setToolCallsigns, "vectorLines", allCallsigns);
    } else {
      removeAllCallsigns(setToolCallsigns, "vectorLines");
    }
  };

  const handleGlobalRings = () => {
    toggleGlobalRings();
    if (!showGlobalRings) {
      addCallsigns(setToolCallsigns, "rangeRings", allCallsigns);
    } else {
      removeAllCallsigns(setToolCallsigns, "rangeRings");
    }
  };

  // Create vectors useEffect to update global callsign stores if allCallsigns updates
  useEffect(() => {
    if (showGlobalVectors) {
      const areSetsEqual = compareArraysAsSets(
        toolCallsigns.vectorLines,
        allCallsigns,
      );

      if (!areSetsEqual) {
        setToolCallsigns((prevState) => {
          return {
            ...prevState,
            vectorLines: allCallsigns,
          };
        });
      }
    }
  }, [
    allCallsigns,
    individualSectorId,
    showGlobalVectors,
    toolCallsigns.vectorLines,
  ]);

  // Create rangeRings useEffect to update global callsign stores if allCallsigns updates
  useEffect(() => {
    if (showGlobalRings) {
      const areSetsEqual = compareArraysAsSets(
        toolCallsigns.rangeRings,
        allCallsigns,
      );

      if (!areSetsEqual) {
        setToolCallsigns((prevState) => {
          return {
            ...prevState,
            rangeRings: allCallsigns,
          };
        });
      }
    }
  }, [
    allCallsigns,
    individualSectorId,
    showGlobalRings,
    toolCallsigns.rangeRings,
  ]);

  // prevent right click bringing up context menu at any point
  document.addEventListener("contextmenu", preventContextMenu);
  function preventContextMenu(event: MouseEvent) {
    event.preventDefault();
    event.stopPropagation();
    return false;
  }

  return (
    <>
      <Box sx={{ display: "flex" }} style={{ background: "#000000" }}>
        <Box sx={{ width: drawerWidth, flexShrink: 0 }}>
          <RadarDrawer
            drawerWidth={drawerWidth}
            scenarioCategory={scenarioCategory}
            vectorLength={vectorLength}
            showActionLog={showActionLog}
            showGlobalVectors={showGlobalVectors}
            showGlobalRings={showGlobalRings}
            showFixNames={showFixNames}
            showSectorBoundary={showSectorBoundary}
            showCoastline={showCoastline}
            showRangeMarkers={showRangeMarkers}
            colourByFL={colourByFL}
            showActionHighlight={showActionHighlight}
            toggleShowActionHighlight={toggleShowActionHighlight}
            profileName={profileName}
            aircraftMarkerName={aircraftMarkerName}
            toggleGlobalVectors={() => {
              handleGlobalVectors();
            }}
            toggleGlobalRangeRings={() => {
              handleGlobalRings();
            }}
            toggleIndividualVectors={() => {
              toggleCallsign(setToolCallsigns, "vectorLines", selectedAircraft);
            }}
            toggleIndividualRangeRings={() => {
              toggleCallsign(setToolCallsigns, "rangeRings", selectedAircraft);
            }}
            toggleIndividualRoutes={() => {
              toggleCallsign(setToolCallsigns, "routes", selectedAircraft);
            }}
            toggleActionLog={toggleActionLog}
            toggleFixNames={toggleFixNames}
            toggleSectorBoundary={toggleSectorBoundary}
            toggleCoastline={toggleCoastline}
            toggleRangeMarkers={toggleRangeMarkers}
            toggleColourByFL={toggleColourByFL}
            toggleProfile={toggleProfile}
            toggleMarker={toggleMarker}
            resetThemeTools={resetThemeTools}
            resetMappingTools={resetMappingTools}
            resetATCTools={resetATCTools}
            setVectorLength={setVectorLength}
            selectedColourScaleName={selectedColourScaleName}
            setSelectedColourScaleName={setSelectedColourScaleName}
            infoBlockState={infoBlockState}
            setInfoBlockState={setInfoBlockState}
            openScenarioSelect={setScenarioModalOpen}
            isScenarioModalOpen={isScenarioModalOpen}
          />
        </Box>
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            width: `calc(100% - ${drawerWidth}px)`,
            minHeight: "100vh",
          }}
        >
          <Radar
            key={scenarioName}
            toolCallsigns={toolCallsigns}
            showFixNames={showFixNames}
            showSectorBoundary={showSectorBoundary}
            showCoastline={showCoastline}
            showRangeMarkers={showRangeMarkers}
            showActionLog={showActionLog}
            showActionHighlight={showActionHighlight}
            resetTrigger={resetTrigger}
            vectorLength={vectorLength}
            profileName={profileName}
            aircraftMarkerName={aircraftMarkerName}
            colourByFL={colourByFL}
            selectedColourScaleName={selectedColourScaleName}
            infoBlockState={infoBlockState}
          />
        </Box>
      </Box>
      <ScenarioSelectModal
        isOpen={isScenarioModalOpen}
        onClose={() => setScenarioModalOpen(false)}
      />
    </>
  );
}
