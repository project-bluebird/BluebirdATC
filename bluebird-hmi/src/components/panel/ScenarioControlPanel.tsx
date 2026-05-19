import Box from "@mui/material/Box";
import { ExpandLess, ExpandMore, Pause, PlayArrow } from "@mui/icons-material";
import DangerousIcon from "@mui/icons-material/Dangerous";
import SaveIcon from "@mui/icons-material/Save";
import {
  Collapse,
  Divider,
  List,
  ListItem,
  ListItemText,
  Typography,
} from "@mui/material";
import {
  useCloseMutation,
  usePauseMutation,
  useSaveMutation,
  useSetEvolvePeriodMutation,
  useSetTickFrequencyMutation,
  useStartMutation,
} from "api/api";
import { useAppDispatch, useAppSelector } from "app/hooks";
import { PanelButtonNoText } from "components/panel/items/PanelButton";
import PanelInput from "components/panel/items/PanelInput";
import DrawerButton from "components/radarDrawer/DrawerButton";
import TimeSlider from "components/radarDrawer/TimeSlider";
import { useState } from "react";
import { clearLocalPosition } from "slices/dynamicDataSlice";
import {
  selectEvolvePeriod,
  selectRunning,
  selectExists,
  selectTickFrequencyPeriod,
} from "slices/scenarioSlice";

import SectionDivider from "./items/SectionDivider";

interface ScenarioControlPanelProps {
  openScenarioSelect: (boolean) => void;
}

export function ScenarioControlPanel(props: ScenarioControlPanelProps) {
  const { openScenarioSelect } = props;

  const [isExpanded, setIsExpanded] = useState(false);
  const [start] = useStartMutation();
  const [pause] = usePauseMutation();
  const [close, { isLoading, isSuccess, isError }] = useCloseMutation();
  const [save] = useSaveMutation();
  const [setEvolvePeriod] = useSetEvolvePeriodMutation();
  const [setTickFrequencyPeriod] = useSetTickFrequencyMutation();

  const dispatch = useAppDispatch();

  const exists = useAppSelector(selectExists);
  const isRunning = useAppSelector(selectRunning);
  const evolvePeriod = useAppSelector(selectEvolvePeriod);
  const tickFrequency = useAppSelector(selectTickFrequencyPeriod);

  const setRadarPeriod = () => {
    const evolve_period = (
      document.getElementById("evolvePeriodInput") as HTMLInputElement | null
    ).value;
    setEvolvePeriod({
      evolvePeriod: Number(evolve_period),
    });
  };

  const setTickFrequency = () => {
    const tick_frequency_period = (
      document.getElementById(
        "tickFrequencyPeriodInput",
      ) as HTMLInputElement | null
    ).value;
    setTickFrequencyPeriod({
      tickFrequency: Number(tick_frequency_period),
    });
  };

  const exitScenario = async () => {
    try {
      const result = await close().unwrap();
      console.log("close succeeded", result);
      dispatch({ type: "resetStore/resetStore" });
    } catch (err) {
      console.error("close failed", err);
    }
  };

  const pauseStartScenario = () => {
    if (isRunning) {
      pause();
    } else {
      dispatch(clearLocalPosition());
      start({ tickFrequencyPeriod: tickFrequency });
    }
  };
  return (
    <List disablePadding>
      {exists ? (
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-evenly",
            alignItems: "center",
            p: 1,
          }}
        >
          <PanelButtonNoText
            icon={isRunning ? <Pause /> : <PlayArrow />}
            onClick={pauseStartScenario}
          />
          <PanelButtonNoText icon={<SaveIcon />} onClick={() => save()} />
          <PanelButtonNoText icon={<DangerousIcon />} onClick={exitScenario} />
        </Box>
      ) : (
        <>
          <DrawerButton
            text={"Load New Scenario"}
            onClick={() => openScenarioSelect(true)}
          />
        </>
      )}
      <Divider />
      <List disablePadding>
        <ListItem
          onClick={() => {
            setIsExpanded(!isExpanded);
          }}
        >
          <ListItemText primary="Scenario Options" />
          {isExpanded ? <ExpandLess /> : <ExpandMore />}
        </ListItem>

        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
          {exists ? (
            <>
              <SectionDivider />
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 1,
                  px: 2,
                }}
              >
                <Typography variant="body2">
                  {" "}
                  {`Simulation Speed: x ${evolvePeriod / tickFrequency}`}{" "}
                </Typography>

                <PanelInput
                  fieldId={"evolvePeriodInput"}
                  fieldLabel={"Radar Period"}
                  disabled={isRunning}
                  onClick={setRadarPeriod}
                  default={evolvePeriod ? evolvePeriod.toString() : ""}
                />
                <PanelInput
                  fieldId={"tickFrequencyPeriodInput"}
                  fieldLabel={"Tick Period"}
                  disabled={isRunning}
                  onClick={setTickFrequency}
                  default={tickFrequency ? tickFrequency.toString() : ""}
                />
              </Box>
              <SectionDivider />
              <TimeSlider />
            </>
          ) : (
            <Box sx={{ alignItems: "center", gap: 1, px: 2 }}>
              <ListItemText primary="No scenario running" />
            </Box>
          )}
        </Collapse>
      </List>
    </List>
  );
}
