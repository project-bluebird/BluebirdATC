import { Typography } from "@mui/material";
import FormControl from "@mui/material/FormControl";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import { colourScaleFunctions } from "utils/profiles/ColourScales";

interface ColourScaleDropdownProps {
    selectedColourScaleName: string;
    setSelectedColourScaleName: (colourScaleName: string) => void;
}

export default function ColourScaleDropdown(props: ColourScaleDropdownProps) {
    const handleChange = (event: SelectChangeEvent) => {
        props.setSelectedColourScaleName(event.target.value);
    };

    return (
        <FormControl sx={{ m: 1, minWidth: 140, maxWidth: 200 }} size="small">
            <Typography sx={{ m: 0.5 }}>Colour Scale Options</Typography>
            <Select
                labelId="colour-scale-dropdown"
                id="colour-scale-dropdown"
                value={props.selectedColourScaleName}
                onChange={handleChange}
            >
                {Object.keys(colourScaleFunctions).map((key) => (
                    <MenuItem key={key} value={key}>
                        {key.slice(14)}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}
