import { MenuItem } from "@mui/material";
import Box from "@mui/material/Box";

interface DrawerButtonProps {
  text: string;
  onClick?: () => void;
  onMouseDown?: () => void;
  onMouseUp?: () => void;
  onTouchStart?: () => void;
  onTouchEnd?: () => void;
  disabled?: boolean;
}

export default function DrawerButton(props: DrawerButtonProps) {
  return (
    <Box display="flex" justifyContent="center" alignItems="center">
      <MenuItem
        className={"DrawerButton"}
        onClick={props.onClick}
        onMouseDown={props.onMouseDown}
        onMouseUp={props.onMouseUp}
        onTouchStart={props.onTouchStart}
        onTouchEnd={props.onTouchEnd}
        style={{
          whiteSpace: "normal",
          width: "100%",
          padding: "8px",
          justifyContent: "center",
        }}
        disabled={props.disabled}
      >
        {props.text}
      </MenuItem>
    </Box>
  );
}
