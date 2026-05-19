import Divider from "@mui/material/Divider";
import Drawer from "@mui/material/Drawer";
import { useAppSelector } from "app/hooks";
import { ScenarioControlPanel } from "components/panel/ScenarioControlPanel";
import ATCToolControls from "components/radarDrawer/ATCToolControls";
import DrawerTime from "components/radarDrawer/DrawerTime";
import MapControls from "components/radarDrawer/MapControls";
import ThemeControls from "components/radarDrawer/ThemeControls";
import { MiniScenarioPanel } from "components/radarDrawer/ScenarioManager";
import { selectTime } from "slices/dynamicDataSlice";

interface RadarDrawerProps {
    drawerWidth: number;
    scenarioCategory: string;
    vectorLength: number;
    showActionLog: boolean;
    showGlobalVectors: boolean;
    showActionHighlight: boolean;
    showGlobalRings: boolean;
    showFixNames: boolean;
    showSectorBoundary: boolean;
    showCoastline: boolean;
    showRangeMarkers: boolean;
    colourByFL: boolean;
    profileName: string;
    aircraftMarkerName: string;
    toggleGlobalVectors: () => void;
    toggleIndividualVectors: () => void;
    toggleGlobalRangeRings: () => void;
    toggleIndividualRangeRings: () => void;
    toggleIndividualRoutes: () => void;
    toggleActionLog: () => void;
    toggleShowActionHighlight: () => void;
    toggleFixNames: () => void;
    toggleSectorBoundary: () => void;
    toggleCoastline: () => void;
    toggleRangeMarkers: () => void;
    toggleColourByFL: () => void;
    toggleProfile: () => void;
    toggleMarker: () => void;
    resetThemeTools: () => void;
    resetMappingTools: () => void;
    resetATCTools: () => void;
    setVectorLength: (vectorLength: number) => void;
    selectedColourScaleName: string;
    setSelectedColourScaleName: (colourScaleName: string) => void;
    infoBlockState: string;
    setInfoBlockState: (blockState: string) => void;
    openScenarioSelect: (boolean) => void;
    isScenarioModalOpen: boolean;

}

export default function RadarDrawer(props: RadarDrawerProps) {
    const {
        drawerWidth,
        scenarioCategory,
        vectorLength,
        showActionLog,
        showGlobalVectors,
        showGlobalRings,
        showActionHighlight,
        showFixNames,
        showSectorBoundary,
        showCoastline,
        showRangeMarkers,
        colourByFL,
        profileName,
        aircraftMarkerName,
        toggleActionLog,
        toggleGlobalVectors,
        toggleIndividualVectors,
        toggleGlobalRangeRings,
        toggleIndividualRangeRings,
        toggleShowActionHighlight,
        toggleIndividualRoutes,
        toggleFixNames,
        toggleSectorBoundary,
        toggleCoastline,
        toggleRangeMarkers,
        toggleColourByFL,
        toggleProfile,
        toggleMarker,
        resetThemeTools,
        resetMappingTools,
        resetATCTools,
        setVectorLength,
        selectedColourScaleName,
        setSelectedColourScaleName,
        infoBlockState,
        setInfoBlockState,
        openScenarioSelect,
        isScenarioModalOpen,
    } = props;

    const time = useAppSelector(selectTime);
    
    return (
        <Drawer
            slotProps={{
                paper: { sx: { width: drawerWidth - 3 } },
            }}
            // Apply styles to remove scrollbars
            sx={{
                "& .MuiDrawer-paper::-webkit-scrollbar": {
                    display: "none",
                },
                "& .MuiDrawer-paper": {
                    msOverflowStyle: "none", // IE and Edge
                    scrollbarWidth: "none", // Firefox
                },
            }}
            variant="permanent"
            anchor="left"
        >
            <DrawerTime text={time} />
            <MiniScenarioPanel 
                isScenarioModalOpen={isScenarioModalOpen}
            />
            <ScenarioControlPanel
                openScenarioSelect={openScenarioSelect}
            />
            <Divider />
            <ATCToolControls
                vectorLength={vectorLength}
                showActionLog={showActionLog}
                showActionHighlight={showActionHighlight}
                toggleShowActionHighlight={toggleShowActionHighlight}
                showGlobalVectors={showGlobalVectors}
                showGlobalRings={showGlobalRings}
                toggleActionLog={toggleActionLog}
                toggleGlobalVectors={toggleGlobalVectors}
                toggleIndividualVectors={toggleIndividualVectors}
                toggleGlobalRangeRings={toggleGlobalRangeRings}
                toggleIndividualRangeRings={toggleIndividualRangeRings}
                toggleIndividualRoutes={toggleIndividualRoutes}
                resetATCTools={resetATCTools}
                setVectorLength={setVectorLength}
            />
            <Divider />
            <MapControls
                scenarioCategory={scenarioCategory}
                showFixNames={showFixNames}
                showSectorBoundary={showSectorBoundary}
                showCoastline={showCoastline}
                showRangeMarkers={showRangeMarkers}
                toggleFixNames={toggleFixNames}
                toggleSectorBoundary={toggleSectorBoundary}
                toggleCoastline={toggleCoastline}
                toggleRangeMarkers={toggleRangeMarkers}
                resetMappingTools={resetMappingTools}
            />
            <Divider />
            <ThemeControls
                colourByFL={colourByFL}
                profileName={profileName}
                aircraftMarkerName={aircraftMarkerName}
                toggleColourByFL={toggleColourByFL}
                toggleProfile={toggleProfile}
                toggleMarker={toggleMarker}
                resetThemeTools={resetThemeTools}
                selectedColourScaleName={selectedColourScaleName}
                setSelectedColourScaleName={setSelectedColourScaleName}
                infoBlockState={infoBlockState}
                setInfoBlockState={setInfoBlockState}
            />
            <Divider />
        </Drawer>
    );
}
