import Grid from "@mui/material/Grid";
import {
    Checkbox,
    FormControl,
    InputLabel,
    ListItemText,
    MenuItem,
    MenuList,
    Select,
    SelectChangeEvent,
} from "@mui/material";
import { StyledMenuIcon } from "components/panel/items/StyledMenuIcon";
import { ReactElement } from "react";

interface PanelSelectProps {
    id: string;
    icon: ReactElement;
    label: string;
    labelId: string;
    value: string | string[];
    onChange: (event: SelectChangeEvent<string | string[]>) => void;
    choices: string[];
    isRunning: boolean;
}

export default function PanelSelect(props: PanelSelectProps) {
    const is_multiselect: boolean = Array.isArray(props.value);

    const iconGridWidth = 1.5;

    return (
        <Grid container spacing={0} sx={{justifyContent:"flex-start", alignItems: "center"}}>
            <Grid size={iconGridWidth}>
                <MenuList>
                    <StyledMenuIcon icon={props.icon} />
                </MenuList>
            </Grid>
            <Grid size={12 - 2 * iconGridWidth}>
                <MenuList>
                    <FormControl disabled={props.isRunning} size="small" style={{ width: "60%" }}>
                        <InputLabel id={props.id}>{props.label}</InputLabel>
                        <Select
                            labelId={props.labelId}
                            id={props.id}
                            value={props.value}
                            label={props.label}
                            onChange={props.onChange}
                            multiple={is_multiselect}
                            renderValue={(selected) => (Array.isArray(selected) ? selected.join(", ") : selected)}
                        >
                            {props.choices.map((choice, index) => (
                                <MenuItem key={index} value={choice}>
                                    {is_multiselect ? <Checkbox checked={props.value.includes(choice)} /> : <></>}
                                    <ListItemText primary={choice} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </MenuList>
            </Grid>
        </Grid>
    );
}
