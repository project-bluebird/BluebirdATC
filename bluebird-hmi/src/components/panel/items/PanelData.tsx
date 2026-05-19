import { MenuList } from "@mui/material";
import Grid from "@mui/material/Grid";
import ListItemText from "@mui/material/ListItemText";
import { StyledMenuIcon } from "components/panel/items/StyledMenuIcon";
import { ReactElement } from "react";
import truncateString from "utils/TruncateText";

interface PanelDataProps {
    text: string;
    icon: ReactElement;
}

// Panel data are menu items, center aligned, without pointer events, with icons
export default function PanelData({ text, icon }: PanelDataProps) {
    const iconGridWidth = 1.5;
    return (
        <Grid container spacing={0} sx={{justifyContent: "center", alignItems: "center"}}>
            <Grid size={iconGridWidth}>
                <MenuList>
                    <StyledMenuIcon icon={icon} />
                </MenuList>
            </Grid>
            <Grid size={12 - 2 * iconGridWidth}>
                <ListItemText>{truncateString(text, 100)}</ListItemText>
            </Grid>
        </Grid>
    );
}
