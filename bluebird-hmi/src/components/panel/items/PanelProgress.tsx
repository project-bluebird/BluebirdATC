import { CircularProgress } from "@mui/material";
import Grid from "@mui/material/Grid";
import MenuList from "@mui/material/MenuList";
import PanelData from "components/panel/items/PanelData";

interface PanelProgressProps {
  text: string;
}

export default function PanelProgress(props: PanelProgressProps) {
  return (
    <Grid
      container
      spacing={0}
      sx={{ justifyContent: "center", alignItems: "center" }}
    >
      <Grid size={1}>
        <MenuList>
          <CircularProgress />
        </MenuList>
      </Grid>
      <Grid size={6}>
        <MenuList>
          <PanelData icon={undefined} text={props.text} />
        </MenuList>
      </Grid>
      <Grid size={1}></Grid>
    </Grid>
  );
}
