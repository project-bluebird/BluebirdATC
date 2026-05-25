import CancelIcon from "@mui/icons-material/Cancel";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Paper from "@mui/material/Paper";
import CategoryListPanel from "components/panel/CategoryListPanel";
import PanelHeading from "components/panel/items/PanelHeading";
import PanelButton from "components/panel/items/PanelButton";

interface CategorySelectProps {
  handleSelectCategory: (string) => void;
  handleCancel: () => void;
}

export function CategorySelect(props: CategorySelectProps) {
  const { handleSelectCategory, handleCancel } = props;

  return (
    <Box className={"page"}>
      <Paper className={"column"}>
        <PanelHeading text="Category selection" />
        <Divider />
        <CategoryListPanel handleSelectCategory={handleSelectCategory} />
        <Divider />
        <PanelButton
          onClick={handleCancel}
          text={"Cancel"}
          icon={<CancelIcon />}
        />
      </Paper>
    </Box>
  );
}
