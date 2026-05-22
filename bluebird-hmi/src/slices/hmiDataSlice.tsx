import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface HMIDataState {
  selectedAircraft: string | null;
}

const initialState: HMIDataState = {
  selectedAircraft: null,
};

export const hmiDataSlice = createSlice({
  name: "hmiData",
  initialState,
  reducers: {
    setSelectedAircraft: (state, action: PayloadAction<string | null>) => {
      state.selectedAircraft = action.payload;
    },
  },
});

export const { setSelectedAircraft } = hmiDataSlice.actions;
export default hmiDataSlice.reducer;
