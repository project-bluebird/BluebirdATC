import Box from "@mui/material/Box";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import { styled } from "@mui/system";

interface VectorButtonProps {
  vectorLength: number;
  setVectorLength: (vectorLength: number) => void;
}

export default function VectorLengthButtons(props: VectorButtonProps) {
  const { vectorLength, setVectorLength } = props;

  const SmallToggleButton = styled(ToggleButton)(({ theme }) => ({
    padding: theme.spacing(2),
    minWidth: 0,
    maxWidth: 45,
    maxHeight: 30,
  }));

  return (
    <Box sx={{ m: 1.25, mx: "auto" }}>
      <ToggleButtonGroup
        fullWidth={false}
        value={vectorLength}
        exclusive
        onChange={(_, newVectorLength) => {
          if (newVectorLength !== null) {
            setVectorLength(newVectorLength);
          }
        }}
      >
        <SmallToggleButton value={1}>1</SmallToggleButton>
        <SmallToggleButton value={2}>2</SmallToggleButton>
        <SmallToggleButton value={3}>3</SmallToggleButton>
        <SmallToggleButton value={5}>5</SmallToggleButton>
        <SmallToggleButton value={10}>10</SmallToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
}
