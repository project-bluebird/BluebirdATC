import { createSlice } from "@reduxjs/toolkit";
import { RootState } from "app/store";
import { Aircraft, TimedActions } from "utils/types";

export interface DynamicDataState {
  status: "idle" | "loading" | "failed";
  time: string;
  actions: TimedActions[];
  aircraft: Aircraft[];
  localPositions: Record<string, { lat: number; lon: number }>;
}

const initialState: DynamicDataState = {
  status: "idle",
  time: "",
  actions: [],
  aircraft: [],
  localPositions: {},
};

export const dynamicDataSlice = createSlice({
  name: "dynamicData",
  initialState,
  reducers: {
    update: (state, action) => {
      state.status = "idle";
      state.time = action.payload.time.replace(/\.\d+/, ""); // remove everything after the decimal point
      state.actions = action.payload.actions;
      // Update aircraft array from Starling, except for local positions
      state.aircraft = action.payload.aircraft.map((aircraft) => {
        const override = state.localPositions[aircraft.callsign];
        return override
          ? { ...aircraft, lat: override.lat, lon: override.lon }
          : aircraft;
      });

      // temp mapping
      const callsignToAircraft = Object.fromEntries(
        state.aircraft.map((craft) => [craft.callsign, craft]),
      );
    },

    // setSelectedAircraft: (state, action) => {
    //     state.selectedAircraft = action.payload;
    // },

    updateAircraftLocalPosition: (state, action) => {
      const { callsign, lat, lon } = action.payload;
      // Update aircraft array
      const aircraft = state.aircraft.find((a) => a.callsign === callsign);
      if (aircraft) {
        aircraft.lat = lat;
        aircraft.lon = lon;
      }
      // Record the local position
      state.localPositions[callsign] = { lat, lon };
    },

    clearLocalPosition: (state) => {
      state.localPositions = {};
    },
  },
});

export const {
  update,
  //   setSelectedAircraft,
  updateAircraftLocalPosition,
  clearLocalPosition,
} = dynamicDataSlice.actions;

export const selectTime = (state: RootState): string => state.dynamicData.time;
export const selectActions = (state: RootState): TimedActions[] =>
  state.dynamicData.actions;
export const selectAircraft = (state: RootState): Aircraft[] =>
  state.dynamicData.aircraft;
//export const selectSelectedAircraft = (state: RootState): string => state.dynamicData.selectedAircraft;

export default dynamicDataSlice.reducer;
