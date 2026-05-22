import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import CancelIcon from "@mui/icons-material/Cancel";
import TrackChangesIcon from "@mui/icons-material/TrackChanges";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Paper from "@mui/material/Paper";
import ScenarioInfoPanel from "components/panel/ScenarioInfoPanel";
import PanelHeading from "components/panel/items/PanelHeading";
import PanelButton from "components/panel/items/PanelButton";

interface ScenarioLoadProps {
  category: string;
  scenarioName: string;
  handleBack: () => void;
  handleCancel: () => void;
  handleLoad: () => void;
}

export function ScenarioLoad(props: ScenarioLoadProps) {
  const { category, scenarioName, handleBack, handleCancel, handleLoad } =
    props;

  return (
    <Box className={"page"}>
      <Paper className={"column"}>
        <PanelHeading text="Scenario info" />
        <Divider />
        <ScenarioInfoPanel category={category} scenarioName={scenarioName} />
        <Divider />
        <PanelButton
          text={"Load"}
          icon={<TrackChangesIcon />}
          onClick={handleLoad}
        />
        <Divider />
        <PanelButton
          text={"Back"}
          icon={<ArrowBackIcon />}
          onClick={handleBack}
        />
        <Divider />
        <PanelButton
          text={"Cancel"}
          icon={<CancelIcon />}
          onClick={handleCancel}
        />
      </Paper>
    </Box>
  );
}
