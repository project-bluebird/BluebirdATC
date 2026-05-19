import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import { styled } from "@mui/material/styles";

const StyledMenuItem = styled(MenuItem)(() => ({
  opacity: 1.0,
  pointerEvents: "none",
  textAlign: "left",
}));

interface PanelHeadingProps {
  text: string;
}

// Panel headings are menu items, left aligned, without pointer events
export default function PanelHeading({ text }: PanelHeadingProps) {
  return (
    <StyledMenuItem>
      <ListItemText>{text}</ListItemText>
    </StyledMenuItem>
  );
}
