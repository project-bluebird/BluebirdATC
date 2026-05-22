import { ExpandLess, ExpandMore } from "@mui/icons-material";
import {
  Box,
  Collapse,
  List,
  ListItem,
  ListItemText,
  Typography,
} from "@mui/material";
import { useAppSelector } from "app/hooks";
import SectionDivider from "components/panel/items/SectionDivider";
import DrawerButton from "components/radarDrawer/DrawerButton";
import DrawerSwitch from "components/radarDrawer/DrawerSwitch";
import VectorLengthButtons from "components/radarDrawer/VectorLengthButtons";
import { useState } from "react";
//import { selectSelectedAircraft } from "slices/dynamicDataSlice";

interface ATCToolControlsProps {
  vectorLength: number;
  showActionLog: boolean;
  showActionHighlight: boolean;
  showGlobalVectors: boolean;
  showGlobalRings: boolean;
  toggleActionLog: () => void;
  toggleShowActionHighlight: () => void;
  toggleGlobalVectors: () => void;
  toggleIndividualVectors: () => void;
  toggleGlobalRangeRings: () => void;
  toggleIndividualRangeRings: () => void;
  toggleIndividualRoutes: () => void;
  resetATCTools: () => void;
  setVectorLength: (vectorLength: number) => void;
}

export default function ATCToolControls(props: ATCToolControlsProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const selectedAircraft = useAppSelector(
    (state) => state.hmiData.selectedAircraft,
  );
  const isAircraftSelected = selectedAircraft !== "";

  const {
    vectorLength,
    showActionLog,
    showActionHighlight,
    showGlobalVectors,
    showGlobalRings,
    toggleActionLog,
    toggleShowActionHighlight,
    toggleGlobalVectors,
    toggleIndividualVectors,
    toggleGlobalRangeRings,
    toggleIndividualRangeRings,
    toggleIndividualRoutes,
    resetATCTools,
    setVectorLength,
  } = props;

  return (
    <List disablePadding>
      <ListItem
        onClick={() => {
          setIsExpanded(!isExpanded);
        }}
      >
        <ListItemText primary="ATC Tools" />
        {isExpanded ? <ExpandLess /> : <ExpandMore />}
      </ListItem>

      <Collapse in={isExpanded} timeout="auto" unmountOnExit>
        <DrawerSwitch
          isChecked={showActionLog}
          onChange={() => toggleActionLog()}
          label="Action Log"
          disabled={false}
        />
        <DrawerSwitch
          isChecked={showActionHighlight}
          onChange={() => toggleShowActionHighlight()}
          label="Action Highlights"
          disabled={false}
        />
        <SectionDivider />
        <Typography sx={{ mt: 0.75, textAlign: "center" }}>
          Vector Length (mins)
        </Typography>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 1,
            px: 2,
          }}
        >
          <VectorLengthButtons
            vectorLength={vectorLength}
            setVectorLength={setVectorLength}
          />
        </Box>
        <DrawerSwitch
          isChecked={showGlobalVectors ? true : false}
          onChange={() => {
            toggleGlobalVectors();
          }}
          label="Global Vectors"
          disabled={false}
        />
        <DrawerSwitch
          isChecked={showGlobalRings ? true : false}
          onChange={() => {
            toggleGlobalRangeRings();
          }}
          label="Global Range Rings"
          disabled={false}
        />
        <SectionDivider />
        <DrawerButton
          text={"Toggle Aircraft Vector"}
          onClick={() => {
            toggleIndividualVectors();
          }}
          disabled={showGlobalVectors || !isAircraftSelected}
        />
        <DrawerButton
          text={"Toggle Aircraft Range Ring"}
          onClick={() => {
            toggleIndividualRangeRings();
          }}
          disabled={showGlobalRings || !isAircraftSelected}
        />
        <DrawerButton
          text={"Toggle Aircraft Route"}
          onClick={() => {
            toggleIndividualRoutes();
          }}
          disabled={!isAircraftSelected}
        />
        <DrawerButton
          text={"Reset ATC Tools"}
          onClick={() => {
            resetATCTools();
          }}
        />
      </Collapse>
    </List>
  );
}
