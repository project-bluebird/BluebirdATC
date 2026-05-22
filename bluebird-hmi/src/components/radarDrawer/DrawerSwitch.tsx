import { Typography } from "@mui/material";
import Grid from "@mui/material/Grid";
import Switch from "@mui/material/Switch";

interface DrawerSwitchProps {
  isChecked: boolean;
  onChange: () => void;
  label: string;
  disabled: boolean;
}

export default function DrawerSwitch(props: DrawerSwitchProps) {
  return (
    <Grid container sx={{ alignItems: "center" }}>
      <Grid size={1} />
      <Grid size={8} sx={{ textAlign: "left" }}>
        <Typography
          sx={{
            color: props.disabled
              ? (theme) => theme.palette.text.disabled
              : (theme) => theme.palette.text.primary,
          }}
        >
          {props.label}
        </Typography>
      </Grid>
      <Grid size={1}>
        <Switch
          checked={props.isChecked}
          onChange={props.onChange}
          disabled={props.disabled}
        />
      </Grid>
    </Grid>
  );
}
