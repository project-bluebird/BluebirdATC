import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import PersonIcon from "@mui/icons-material/Person";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import {
  Button,
  Divider,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListSubheader,
  Typography,
} from "@mui/material";
import List from "@mui/material/List";
import { useAppSelector } from "app/hooks";
import React, { JSX, useRef, useState } from "react";
import { selectActions } from "slices/dynamicDataSlice";
import { selectIndividualSectorIds } from "slices/scenarioSlice";
import {
  Action,
  ColourProfile,
  OpacityProfile,
  TimedActions,
} from "utils/types";

interface ActionLogProps {
  colourProfile: ColourProfile;
  opacityProfile: OpacityProfile;
}

export function isActionForCurrentSector(
  current_sectors: string[],
  action: Action,
): boolean {
  return (
    !action.sector ||
    action.sector.find((individual_sector) =>
      current_sectors.includes(individual_sector),
    ) !== undefined
  );
}

function filterTimedActions(
  actions: TimedActions[],
  current_sector: string[],
): TimedActions[] {
  return actions
    .map((x) => {
      return {
        time: x.time,
        actions: x.actions.filter((x) =>
          isActionForCurrentSector(current_sector, x),
        ),
      } as TimedActions;
    })
    .filter((x) => x.actions.length !== 0);
}

function groupActionsByAgent(actions: Action[]): Action[][] {
  const groupedObjects = {};

  actions.forEach((action: Action) => {
    const agentType = action["agent"];

    if (typeof action.text_representation?.clearance === "string") {
      if (agentType in groupedObjects) {
        groupedObjects[agentType].push(action);
      } else {
        groupedObjects[agentType] = [action];
      }
    }
  });
  return Object.values(groupedObjects);
}

function setIcon(actions: Action[]): JSX.Element | null {
  const iconRef = {
    human: <PersonIcon />,
    falcon: <SmartToyIcon />,
  };
  let icon = null;

  Object.keys(iconRef).forEach((agent) => {
    if (actions.every((action) => action.agent.includes(agent))) {
      icon = iconRef[agent];
    }
  });
  return icon;
}

export default function ActionLog(props: ActionLogProps) {
  const { colourProfile, opacityProfile } = props;

  const rawActions: TimedActions[] = useAppSelector(selectActions);
  const individualSectorIds = useAppSelector(selectIndividualSectorIds);

  const actions = rawActions.map((item) => ({
    ...item,
    actions: item.actions.map((action) => ({
      ...action,
      agent: action.agent || "No agent",
    })),
  }));

  const [isOpen, setIsOpen] = useState(true);
  const listContainerRef = useRef(null);
  const margin = 15;

  const toggleList = () => {
    setIsOpen(!isOpen);
  };

  const getListWidth = () => {
    if (listContainerRef.current) {
      return listContainerRef.current.offsetWidth + margin;
    }
    return 300;
  };

  const borderStyle = {
    border: "solid",
    borderColor: colourProfile.actionLogBorderColour,
    borderWidth: 2,
  };

  return (
    <div
      id="actionLog"
      style={{
        display: "flex",
        alignItems: "flex-start",
        flexDirection: "column",
        position: "absolute",
        top: 0,
        right: isOpen ? 0 : `-${getListWidth()}px`,
        maxHeight: "40%",
        margin: `${margin}px`,
        transition: "right 0.5s ease-in-out",
        opacity: opacityProfile.actionLogOpacity,
      }}
    >
      <Button
        onClick={toggleList}
        sx={{
          ...borderStyle,
          borderRight: "none",
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
          height: "52px",
          position: "absolute",
          top: 0,
          left: "-62px",
          zIndex: 2,
          color: colourProfile.actionLogIconColour,
          bgcolor: colourProfile.actionLogColour,
          "&:hover": {
            color: colourProfile.actionLogIconHoverColour,
            bgcolor: colourProfile.actionLogColour,
          },
        }}
      >
        {isOpen ? <ArrowForwardIosIcon /> : <ArrowBackIosNewIcon />}
      </Button>
      <ListSubheader
        style={{
          ...borderStyle,
          width: "100%",
          minWidth: "300px",
          borderLeft: "none",
          borderBottom: "none",
        }}
      >
        Issued Actions
      </ListSubheader>
      <List
        dense={true}
        sx={{
          ...borderStyle,
          width: "max-content",
          minWidth: "300px",
          zIndex: 1,
          display: "flex",
          flexDirection: "column-reverse",
          bgcolor: colourProfile.actionLogColour,
          padding: 0,
          overflowY: "auto",
          scrollbarGutter: "stable",
          borderTop: "none",
        }}
        ref={listContainerRef}
      >
        {[...filterTimedActions(actions, individualSectorIds)]
          .reverse()
          .map((item) =>
            groupActionsByAgent(item.actions).map((groupedActions, index) => (
              <React.Fragment key={index}>
                <ListItem
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                  }}
                >
                  <ListItemText
                    sx={{ marginBottom: 0 }}
                    primary={
                      <React.Fragment>
                        {groupedActions.map((action, entry) => (
                          <Typography key={entry}>
                            {action.text_representation.clearance}
                          </Typography>
                        ))}
                      </React.Fragment>
                    }
                  />
                  <div
                    style={{
                      width: "100%",
                      display: "flex",
                      justifyContent: "space-between",
                    }}
                  >
                    <ListItemText secondary={item.time.replace(/\.\d+/, "")} />
                    <ListItemIcon sx={{ minWidth: 0 }}>
                      {setIcon(groupedActions)}
                    </ListItemIcon>
                  </div>
                </ListItem>
                {index !== actions.length - 1 && (
                  <Divider key={`divider-${index}`} />
                )}
              </React.Fragment>
            )),
          )}
        <Divider></Divider>
      </List>
    </div>
  );
}
