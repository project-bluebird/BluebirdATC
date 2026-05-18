import ListItemText from "@mui/material/ListItemText";
import {MenuItem, MenuList} from "@mui/material";
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
        <MenuList>
        <StyledMenuItem>
            <ListItemText>{text}</ListItemText>
        </StyledMenuItem>
        </MenuList>
    );
}
