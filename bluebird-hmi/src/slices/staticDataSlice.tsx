import { createSlice } from "@reduxjs/toolkit";
import { RootState } from "app/store";
import { Coordinate, Fix } from "utils/types";

export interface StaticDataState {
  status: "idle" | "loading" | "failed";
  exists: boolean;
  scenarioName: string;
  fixes: Fix[];
  sectors: Record<string, Coordinate[][]>;
  projection_centre: number[] | null;
}

const initialState: StaticDataState = {
  status: "idle",
  exists: false,
  scenarioName: "",
  fixes: [],
  sectors: {},
  projection_centre: [],
};

export const staticDataSlice = createSlice({
  name: "staticData",
  initialState,
  reducers: {
    update: (state, action) => {
      state.status = "idle";
      state.exists = action.payload.exists;
      state.scenarioName = action.payload.scenarioName;
      state.fixes = action.payload.fixes;
      state.sectors = action.payload.sectors;
      state.projection_centre = action.payload.projection_centre;
    },
  },
});

export const { update } = staticDataSlice.actions;

export const selectExists = (state: RootState): boolean =>
  state.staticData.exists;
export const selectScenarioName = (state: RootState): string =>
  state.staticData.scenarioName;
export const selectFixes = (state: RootState): Fix[] => state.staticData.fixes;
export const selectSectors = (
  state: RootState,
): Record<string, Coordinate[][]> => state.staticData.sectors;
export const selectProjectionCentre = (
  state: RootState,
): [number, number] | null => state.staticData.projection_centre;

export default staticDataSlice.reducer;
