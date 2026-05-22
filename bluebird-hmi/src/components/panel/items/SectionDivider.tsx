import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";

export default function SectionDivider() {
  return (
    <Box sx={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <Divider sx={{ width: "50%" }} />
    </Box>
  );
}
