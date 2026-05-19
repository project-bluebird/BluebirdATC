import SyncIcon from "@mui/icons-material/Sync";
import Grid from "@mui/material/Grid";
import MenuList from "@mui/material/MenuList";
import TextField from "@mui/material/TextField";
import { IconButton, Button } from "@mui/material";
import PanelButton from "components/panel/items/PanelButton";
import { ChangeEvent, useState } from "react";

interface PanelInputProps {
    fieldId: string;
    fieldLabel: string;
    disabled?: boolean;
    onClick?: () => void;
    /** The default value to fill on initialisation of the component. DON'T USE TO WRITE TO THE API */
    default: string;
}

export default function PanelInput(props: PanelInputProps) {
    const [inputValue, setInputValue] = useState<string>(props.default);
    const [error, setError] = useState<string>("");

    const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const inputValue = event.target.value;

        // Only allow alphanumeric input, not starting with 0
        if (/^(?:[a-zA-Z1-9_][a-zA-Z0-9_]*)?$/.test(inputValue)) {
            setInputValue(inputValue);
            setError("");
        } else {
            setError("a-z, A-Z, 0-9, _ only, min 1 s");
        }
    };

  return (
    <Grid
      container
      spacing={3}
      sx = {{ mb: 1.5, alignItems: "center", justifyContent: "flex-start"}}
    >
      {/* text input */}
      <Grid size={{ xs: 6 }}>
        <TextField
          error={!!error}
          id={props.fieldId}
          label={props.fieldLabel}
          variant="outlined"
          size="small"
          disabled={props.disabled}
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={(event: React.KeyboardEvent<HTMLDivElement>) => {
            event.stopPropagation();
          }}
          helperText={error || ""}
          fullWidth
        />
      </Grid>

      {/* refresh icon */}
      <Grid size="auto">
        <IconButton
          size="small"
          disabled={props.disabled || !!error || !inputValue}
          onClick={props.onClick}
        >
          <SyncIcon fontSize="small" />
        </IconButton>
      </Grid>
    </Grid>
  );
/*
    return (
        <Grid container spacing={1} columns={20} alignItems="center" justifyContent="flex-start">
            <Grid size={{xs: 6}} >
                
                    <TextField
                        error={error !== "" ? true : false}
                        id={props.fieldId}
                        label={props.fieldLabel}
                        variant="outlined"
                        size="small"
                        disabled={props.disabled}
                        value={inputValue}
                        onChange={handleInputChange}
                        onKeyDown={(event: React.KeyboardEvent<HTMLDivElement>) => {
                            event.stopPropagation();
                        }}
                        helperText={error !== "" ? error : ""}
                    />
                
            </Grid>
            <Grid size={{xs: 4}}>
                
                    <PanelButton
                        text="Update"
                        icon={<SyncIcon />}
                        disabled={props.disabled || !!error || !inputValue}
                        onClick={props.onClick}
                        
                    />
                
            </Grid>
        </Grid>
    );

    */
}
