import SourceIcon from "@mui/icons-material/Source";
import { MenuList } from "@mui/material";
import { useListScenarioCategoriesQuery } from "api/api";
import PanelButton from "components/panel/items/PanelButton";
import PanelProgress from "components/panel/items/PanelProgress";

interface CategoryListPanelProps {
  handleSelectCategory: (string) => void;
}

export default function CategoryListPanel(props: CategoryListPanelProps) {
  const { handleSelectCategory } = props;
  const { data, isError, isLoading } = useListScenarioCategoriesQuery();

  if (isError) {
    return (
      <MenuList>An error occurred fetching scenario category data.</MenuList>
    );
  }
  if (isLoading) {
    return <PanelProgress text={"Loading category data"} />;
  }
  if (data) {
    return (
      <MenuList>
        {data.map((category: string) => (
          <PanelButton
            key={category}
            text={category}
            icon={<SourceIcon />}
            onClick={() => handleSelectCategory(category)}
          />
        ))}
      </MenuList>
    );
  }
  return <>No data.</>;
}
