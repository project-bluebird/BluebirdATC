import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { REHYDRATE } from "redux-persist";

const API_URL = import.meta.env.VITE_BACKEND_URL as string;

export const emptySplitApi = createApi({
  reducerPath: "api",
  // use dynamic base query for webapp deployment
  baseQuery: fetchBaseQuery({ baseUrl: API_URL || "http://localhost:8000" }),
  keepUnusedDataFor: 0.5,
  // Rehydration (handle window reload)
  extractRehydrationInfo(action, { reducerPath }) {
    if (action.type === REHYDRATE) {
      if (action.payload) {
        return action.payload[reducerPath];
      }
    }
  },
  endpoints: () => ({}),
});
