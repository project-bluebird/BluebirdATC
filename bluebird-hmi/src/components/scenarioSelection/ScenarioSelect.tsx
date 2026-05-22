import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import CancelIcon from "@mui/icons-material/Cancel";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Paper from "@mui/material/Paper";
import ScenarioListPanel from "components/panel/ScenarioListPanel";
import PanelHeading from "components/panel/items/PanelHeading";
import PanelButton from "components/panel/items/PanelButton";

interface ScenarioSelectProps {
  category: string;
  handleSelectScenario: (string) => void;
  handleBack: () => void;
  handleCancel: () => void;
}

export function ScenarioSelect(props: ScenarioSelectProps) {
  const { category, handleSelectScenario, handleBack, handleCancel } = props;

  return (
    <Box className={"page"}>
      <Paper className={"column"}>
        <PanelHeading text="Scenario selection" />
        <Divider />
        <ScenarioListPanel
          category={category}
          handleSelectScenario={handleSelectScenario}
        />
        <Divider />
        <PanelButton
          onClick={handleBack}
          text={"Back"}
          icon={<ArrowBackIcon />}
        />
        <Divider />
        <PanelButton
          onClick={handleCancel}
          text={"Cancel"}
          icon={<CancelIcon />}
        />
      </Paper>
    </Box>
  );
}
