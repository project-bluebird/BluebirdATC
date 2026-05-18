import { ButtonBase } from "@mui/material";
import Grid from "@mui/material/Grid";
import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import MenuList from "@mui/material/MenuList";
import { styled } from "@mui/material/styles";
import { StyledMenuIcon } from "components/panel/items/StyledMenuIcon";
import { ReactElement } from "react";
import truncateString from "utils/TruncateText";

const StyledMenuItem = styled(MenuItem)(() => ({
    opacity: 1.0,
    textAlign: "center",
}));

interface PanelButtonProps {
    text: string;
    icon: ReactElement;
    disabled?: boolean;
    onClick?: () => void;
}

// Panel buttons are menu items that can be disabled, center aligned, with pointer events, with icons
export default function PanelButton({ text, icon, disabled = false, onClick }: PanelButtonProps) {
    const iconGridWidth = 1.5;
    return (
        <Grid container spacing={10} sx={{ justifyContent: "center", alignItems: "center" }}>
            <Grid size = {{ xs: iconGridWidth}} >
                <MenuList>
                    <StyledMenuIcon icon={icon} />
                </MenuList>
            </Grid>
            <Grid size = {{ xs: 12 - 2 * iconGridWidth}}>
                <MenuList>
                <StyledMenuItem disabled={disabled} onClick={onClick && onClick}>
                    <ListItemText>{truncateString(text, 100)}</ListItemText>
                </StyledMenuItem>{" "}
                </MenuList>
            </Grid>
        </Grid>
    );
}

interface PanelButtonNoTextProps {
    icon: ReactElement;
    disabled?: boolean;
    onClick?: () => void;
}

export function PanelButtonNoText({ icon, disabled = false, onClick }: PanelButtonNoTextProps) {
    return (
        <ButtonBase onClick={onClick && onClick} disabled={disabled}>
            <StyledMenuIcon icon={icon} />
        </ButtonBase>
    );
}
