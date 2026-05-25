import { Typography } from "@mui/material";
import Grid from "@mui/material/Grid";
import Switch from "@mui/material/Switch";
import { StyledMenuIcon } from "components/panel/items/StyledMenuIcon";
import { ReactElement } from "react";

interface PanelSwitchProps {
  isChecked: boolean;
  onChange: () => void;
  label: string;
  icon?: ReactElement;
}

export default function PanelSwitch(props: PanelSwitchProps) {
  const { isChecked, onChange, label, icon } = props;
  const iconGridWidth = 1.5;

  return (
    <Grid container sx={{ alignItems: "center" }}>
      <Grid size={iconGridWidth}>
        <StyledMenuIcon icon={icon} />
      </Grid>
      <Grid size={12 - 2 * iconGridWidth}>
        <Typography>{label}</Typography>
      </Grid>
      <Grid size={iconGridWidth - 0.5}>
        <Switch checked={isChecked} onChange={onChange} />
      </Grid>
    </Grid>
  );
}
