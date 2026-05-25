import { initialToolCallsignStore } from "utils/initialState";
import { ToolCallsignStore } from "utils/types";

// Toggle single callsign in or out of ToolCallsignStore for a given tool
export const toggleCallsign = (
  setToolCallsigns: React.Dispatch<React.SetStateAction<ToolCallsignStore>>,
  toolName: keyof ToolCallsignStore,
  selectedAircraft: string,
) => {
  if (selectedAircraft === "") {
    return;
  }

  setToolCallsigns((prevState) => ({
    ...prevState,
    [toolName]: prevState[toolName].includes(selectedAircraft)
      ? prevState[toolName].filter((callsign) => callsign !== selectedAircraft)
      : [...prevState[toolName], selectedAircraft],
  }));
};

// Add array of new callsigns to ToolCallsignStore for a given tool
export const addCallsigns = (
  setToolCallsigns: React.Dispatch<React.SetStateAction<ToolCallsignStore>>,
  toolName: keyof ToolCallsignStore,
  newCallsigns: string[],
) => {
  setToolCallsigns((prevState) => {
    const uniqueCallsigns = Array.from(
      new Set([...prevState[toolName], ...newCallsigns]),
    );

    return {
      ...prevState,
      [toolName]: uniqueCallsigns,
    };
  });
};

// Remove all callsigns for a given tool in ToolCallsignStore
export const removeAllCallsigns = (
  setToolCallsigns: React.Dispatch<React.SetStateAction<ToolCallsignStore>>,
  toolName: keyof ToolCallsignStore,
) => {
  setToolCallsigns((prevState) => {
    return {
      ...prevState,
      [toolName]: [],
    };
  });
};

// Reset/remove all callsigns in ToolCallsignStore for all tools
export const resetToolCallsigns = (
  setToolCallsigns: React.Dispatch<React.SetStateAction<ToolCallsignStore>>,
) => {
  setToolCallsigns(initialToolCallsignStore);
};
