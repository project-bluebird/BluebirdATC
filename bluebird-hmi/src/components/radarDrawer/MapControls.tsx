import { ExpandLess, ExpandMore } from "@mui/icons-material";
import {
  Collapse,
  List,
  ListItem,
  ListItemText,
  Typography,
} from "@mui/material";
import { useAppSelector } from "app/hooks";
import DrawerButton from "components/radarDrawer/DrawerButton";
import DrawerSwitch from "components/radarDrawer/DrawerSwitch";
import { useState } from "react";
import { selectCategory } from "slices/scenarioSlice";

interface MapControlsProps {
  scenarioCategory: string;
  showFixNames: boolean;
  showSectorBoundary: boolean;
  showCoastline: boolean;
  showRangeMarkers: boolean;
  toggleFixNames: () => void;
  toggleSectorBoundary: () => void;
  toggleCoastline: () => void;
  toggleRangeMarkers: () => void;
  resetMappingTools: () => void;
}

export default function MapControls(props: MapControlsProps) {
  const {
    scenarioCategory,
    showFixNames,
    showSectorBoundary,
    showCoastline,
    showRangeMarkers,
    toggleFixNames,
    toggleSectorBoundary,
    toggleCoastline,
    toggleRangeMarkers,
    resetMappingTools,
  } = props;

  const [areMappingControlsOff, setMappingControls] = useState(false);

  return (
    <List disablePadding>
      <ListItem
        onClick={() => {
          setMappingControls(!areMappingControlsOff);
        }}
      >
        <ListItemText primary="Mapping Options" />
        {areMappingControlsOff ? <ExpandLess /> : <ExpandMore />}
      </ListItem>
      <Collapse in={areMappingControlsOff} timeout="auto" unmountOnExit>
        <DrawerSwitch
          isChecked={showCoastline}
          onChange={() => {
            toggleCoastline();
          }}
          label="Coastline"
          disabled={false}
        />

        <DrawerSwitch
          isChecked={showFixNames}
          onChange={() => {
            toggleFixNames();
          }}
          label={"Fix Names"}
          disabled={false}
        />
        <DrawerSwitch
          isChecked={showSectorBoundary}
          onChange={() => {
            toggleSectorBoundary();
          }}
          label="Sector boundaries"
          disabled={false}
        />
        <DrawerSwitch
          isChecked={showRangeMarkers}
          onChange={() => {
            toggleRangeMarkers();
          }}
          label={"Range Markers"}
          disabled={false}
        />
        <DrawerButton
          text={"Reset Mapping Options"}
          onClick={() => {
            resetMappingTools();
          }}
        />
      </Collapse>
    </List>
  );
}
