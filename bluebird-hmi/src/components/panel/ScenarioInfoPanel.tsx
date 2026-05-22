import { MenuList } from "@mui/material";
import PanelData from "components/panel/items/PanelData";

export default function ScenarioInfoPanel(props: {
  category: string;
  scenarioName: string;
}) {
  return (
    <MenuList>
      <PanelData icon={undefined} text={`Category: ${props.category}`} />
      <PanelData icon={undefined} text={`Scenario: ${props.scenarioName}`} />
    </MenuList>
  );
}
