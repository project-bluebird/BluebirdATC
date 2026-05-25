import AccessTimeIcon from "@mui/icons-material/AccessTime";
import IconButton from "@mui/material/IconButton";
import { styled } from "@mui/material/styles";

const StyledIconButton = styled(IconButton)(() => ({
  borderRadius: 0,
}));

interface PanelClockButtonProps {
  onClick: (time: string) => void;
}

export default function PanelClockButton({ onClick }: PanelClockButtonProps) {
  const handleClick = () => {
    const now = new Date();
    const timeString = now.toLocaleTimeString("en-US", { hour12: false });

    // When click, update state with current time string
    onClick(timeString);
  };

  return (
    <StyledIconButton size="small" onClick={handleClick}>
      <AccessTimeIcon fontSize="small" />
    </StyledIconButton>
  );
}
