import { Box, Typography } from "@mui/material";
import FormControl from "@mui/material/FormControl";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";

interface DrawerBlockDropdownProps<E> {
    infoBlockState: string;
    setInfoBlockState: (blockState: string) => void;
    blockOptions: E;
    label: string;
}

export default function DrawerBlockDropdown<E>(props: DrawerBlockDropdownProps<E>) {
    const handleChange = (event: SelectChangeEvent) => {
        props.setInfoBlockState(event.target.value);
    };

    return (
        <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
            <FormControl
                sx={{ m: 1, minWidth: 140, maxWidth: 160, display: "flex", flexDirection: "column" }}
                size="small"
            >
                <Typography sx={{ m: 0.5 }}>{props.label}</Typography>
                <Select
                    labelId="info-block-dropdown"
                    id="info-block-dropdown"
                    value={props.infoBlockState}
                    onChange={handleChange}
                >
                    {Object.values(props.blockOptions).map((stateOption) => (
                        <MenuItem key={stateOption} value={stateOption}>
                            {stateOption}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
        </Box>
    );
}
