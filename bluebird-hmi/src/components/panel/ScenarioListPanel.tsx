import AirplaneTicketIcon from "@mui/icons-material/AirplaneTicket";
import FolderIcon from "@mui/icons-material/Folder";
import { MenuList } from "@mui/material";
import { useState } from "react";

import { useListScenariosQuery } from "../../api/api";
import PanelButton from "../../components/panel/items/PanelButton";
import PanelLink from "./items/PanelLink";
import PanelProgress from "./items/PanelProgress";

interface ScenarioListPanelProps {
  category: string;
  handleSelectScenario: (scenario: string) => void;
}

export default function ScenarioListPanel(props: ScenarioListPanelProps) {
  const { category, handleSelectScenario } = props;

  const [currentPath, setCurrentPath] = useState("");

  const { data, isError, isLoading } = useListScenariosQuery(
    { category: props.category },
    { skip: false, refetchOnMountOrArgChange: true },
  );

  const handleNestedScenarioList = (scenarioList: [string]) => {
    // If we have a list of scenario paths, we want to return list of "directories"
    const nextSteps = new Map();
    scenarioList.forEach((path) => {
      if (path.startsWith(currentPath)) {
        const pathPrefixLength =
          currentPath.length > 0 ? currentPath.length + 1 : 0;
        const remainingPath = path.substring(pathPrefixLength);
        const nextPath = remainingPath.split("/")[0];
        if (nextPath) {
          const isDirectory = remainingPath !== nextPath;
          nextSteps.set(nextPath, isDirectory);
        }
      }
    });
    const nextStepsList = Array.from(nextSteps)
      .map(([pathElement, isDirectory]) => ({ pathElement, isDirectory }))
      .sort((a, b) => a.pathElement.localeCompare(b.pathElement));
    return nextStepsList;
  };

  const handleDirectoryClick = (pathElement: string) => {
    const pathSep = currentPath.length > 0 ? "/" : "";
    setCurrentPath(currentPath + pathSep + pathElement);
  };

  if (isError) {
    return (
      <MenuList>
        <>An error occurred.</>
      </MenuList>
    );
  }
  if (isLoading) {
    return <PanelProgress text={"Loading list of scenarios"} />;
  }
  if (data) {
    const scenarioPaths = handleNestedScenarioList(data);
    const pathsToDisplay = scenarioPaths.map(({ pathElement, isDirectory }) =>
      isDirectory ? (
        <PanelButton
          key={`${props.category}_folder_${pathElement}`}
          text={pathElement}
          icon={<FolderIcon />}
          onClick={() => {
            handleDirectoryClick(pathElement);
          }}
        />
      ) : (
        <PanelButton
          key={`${props.category}_file_${pathElement}`}
          text={pathElement}
          onClick={() => handleSelectScenario(pathElement)}
          icon={<AirplaneTicketIcon />}
        />
      ),
    );
    return (
      <MenuList sx={{ maxHeight: 400, overflowY: "auto" }}>
        {pathsToDisplay}
      </MenuList>
    );
  }
  return <>No data.</>;
}
