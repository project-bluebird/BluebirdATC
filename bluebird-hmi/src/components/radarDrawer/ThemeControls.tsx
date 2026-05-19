import { ExpandLess, ExpandMore } from "@mui/icons-material";
import { Collapse, List, ListItem, ListItemText } from "@mui/material";
import ColourScaleDropdown from "components/radarDrawer/ColourScaleDropdown";
import DrawerBlockDropdown from "components/radarDrawer/DrawerBlockDropdown";
import DrawerButton from "components/radarDrawer/DrawerButton";
import DrawerSwitch from "components/radarDrawer/DrawerSwitch";
import { useState } from "react";
import { InfoBlockOptions } from "utils/Constants";

interface ThemeControlProps {
  colourByFL: boolean;
  profileName: string;
  aircraftMarkerName: string;
  toggleColourByFL: () => void;
  toggleProfile: () => void;
  toggleMarker: () => void;
  resetThemeTools: () => void;
  selectedColourScaleName: string;
  setSelectedColourScaleName: (colourScaleName: string) => void;
  infoBlockState: string;
  setInfoBlockState: (blockState: string) => void;
}

export default function ThemeControls(props: ThemeControlProps) {
  const {
    colourByFL,
    profileName,
    aircraftMarkerName,
    toggleColourByFL,
    toggleProfile,
    toggleMarker,
    resetThemeTools,
    selectedColourScaleName,
    setSelectedColourScaleName,
    infoBlockState,
    setInfoBlockState,
  } = props;

  const [areThemeControlsOff, setThemeControls] = useState(false);

  return (
    <List disablePadding>
      <ListItem
        onClick={() => {
          setThemeControls(!areThemeControlsOff);
        }}
      >
        <ListItemText primary="Theme Options" />
        {areThemeControlsOff ? <ExpandLess /> : <ExpandMore />}
      </ListItem>
      <Collapse in={areThemeControlsOff} timeout="auto" unmountOnExit>
        <DrawerSwitch
          isChecked={profileName === "default" ? false : true}
          onChange={() => {
            toggleProfile();
          }}
          label="Presentation Mode"
          disabled={false}
        />
        <DrawerSwitch
          isChecked={aircraftMarkerName === "starMarker" ? false : true}
          onChange={() => {
            toggleMarker();
          }}
          label="Aircraft Icons"
          disabled={false}
        />
        <DrawerSwitch
          isChecked={colourByFL}
          onChange={() => {
            toggleColourByFL();
          }}
          label={"Colour by FL"}
          disabled={false}
        />
        <ColourScaleDropdown
          selectedColourScaleName={selectedColourScaleName}
          setSelectedColourScaleName={setSelectedColourScaleName}
        />
        <DrawerBlockDropdown
          infoBlockState={infoBlockState}
          setInfoBlockState={setInfoBlockState}
          blockOptions={InfoBlockOptions}
          label="Info Block Options"
        />
        <DrawerButton
          text={"Reset Theme Options"}
          onClick={() => {
            resetThemeTools();
          }}
        />
      </Collapse>
    </List>
  );
}
