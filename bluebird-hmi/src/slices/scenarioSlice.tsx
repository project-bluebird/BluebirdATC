import { createSlice } from "@reduxjs/toolkit";
import { RootState } from "app/store";

export interface ScenarioState {
  status: "idle" | "loading" | "failed";
  category: string;
  scenario: string;
  individualSectorIds: string[] | null;
  bandboxSectorId: string | null;
  exists: boolean;
  iterations: number;
  running: boolean;
  evolve_period: number;
  tick_frequency_period: number;
  kill: boolean;
  reload: boolean;
}

const initialState: ScenarioState = {
  status: "idle",
  category: null,
  scenario: null,
  individualSectorIds: [],
  bandboxSectorId: null,
  exists: false,
  iterations: 0,
  running: false,
  evolve_period: 6.0, // period evolved with each evolve call
  tick_frequency_period: 6.0, // frequency of evolve calls as a period in seconds
  kill: false,
  reload: false,
};

export const scenarioSlice = createSlice({
  name: "scenario",
  initialState,
  reducers: {
    changeSectorId: (state, action) => {
      state.individualSectorIds = action.payload.individualSectorIds;
      state.bandboxSectorId = action.payload.bandboxSectorId;
    },
    changeEvolvePeriod: (state, action) => {
      state.evolve_period = action.payload.evolvePeriod;
    },
    changeTickFrequencyPeriod: (state, action) => {
      state.tick_frequency_period = action.payload.tick_frequency_period;
    },
    update: (state, action) => {
      state.status = "idle";
      state.exists = action.payload.exists;
      state.category = action.payload.category;
      state.scenario = action.payload.scenario;
      state.iterations = action.payload.iterations;
      state.running = action.payload.running;
      state.evolve_period = action.payload.evolve_period;
      state.tick_frequency_period = action.payload.tick_frequency_period;
      state.kill = action.payload.kill;
      state.reload = action.payload.reload;
    },
  },
});

export const {
  changeSectorId,
  update,
  changeEvolvePeriod,
  changeTickFrequencyPeriod,
} = scenarioSlice.actions;

export const selectIndividualSectorIds = (state: RootState): string[] | null =>
  state.scenario.individualSectorIds;
export const selectBandboxSectorId = (state: RootState): string | null =>
  state.scenario.bandboxSectorId;
export const selectExists = (state: RootState): boolean =>
  state.scenario.exists;
export const selectCategory = (state: RootState): string =>
  state.scenario.category;
export const selectScenario = (state: RootState): string =>
  state.scenario.scenario;
export const selectIterations = (state: RootState): number =>
  state.scenario.iterations;
export const selectRunning = (state: RootState): boolean =>
  state.scenario.running;
export const selectEvolvePeriod = (state: RootState): number =>
  state.scenario.evolve_period;
export const selectTickFrequencyPeriod = (state: RootState): number =>
  state.scenario.tick_frequency_period;
export const selectKill = (state: RootState): boolean => state.scenario.kill;
export const selectStatus = (state: RootState): string => state.scenario.status;
export const selectReload = (state: RootState): boolean =>
  state.scenario.reload;

export default scenarioSlice.reducer;
