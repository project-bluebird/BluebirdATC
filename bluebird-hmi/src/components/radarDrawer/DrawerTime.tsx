import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";

interface DrawerTimeProps {
    text: string;
    colour?: string;
}

export default function DrawerTime(props: DrawerTimeProps) {
    const { text, colour = "black" } = props;

    return (
        <Toolbar variant="dense">
            <Typography
                className={"DrawerTime"}
                variant="h6"
                noWrap
                component="div"
                color={colour}
                sx={{
                    textAlign:"center",
                    flexGrow: 1
                }}
            >
                {text ? text : "No scenario running"}
            </Typography>
        </Toolbar>
    );
}
