import {
  Action,
  AnyAction,
  combineReducers,
  configureStore,
  Reducer,
  ThunkAction,
} from "@reduxjs/toolkit";
import { setupListeners } from "@reduxjs/toolkit/query";
import { api } from "api/api";
import {
  FLUSH,
  PAUSE,
  PERSIST,
  persistReducer,
  PURGE,
  REGISTER,
  REHYDRATE,
} from "redux-persist";
import localForage from "localforage";
import {
  createStateSyncMiddleware,
  initMessageListener,
} from "redux-state-sync";
//import { thunk } from "redux-thunk";
import dynamicDataReducer from "slices/dynamicDataSlice";
import staticDataReducer from "slices/staticDataSlice";
import scenarioDataReducer from "slices/scenarioSlice";
import hmiDataReducer from "slices/hmiDataSlice";

const syncConfig = {
  blacklist: [
    "api/executeQuery/fulfilled",
    "api/executeMutation/fulfilled",
    "api/executeQuery/rejected",
    "persist/PERSIST",
    "persist/REHYDRATE",
  ],
};
const middlewares = [createStateSyncMiddleware(syncConfig)];

const appReducer = combineReducers({
  dynamicData: dynamicDataReducer,
  staticData: staticDataReducer,
  scenario: scenarioDataReducer,
  hmiData: hmiDataReducer,
  [api.reducerPath]: api.reducer,
});

function storage(dbName) {
  const db = localForage.createInstance({
    name: dbName,
  });
  return {
    db,
    getItem: db.getItem,
    setItem: db.setItem,
    removeItem: db.removeItem,
  };
}

const rootReducer: Reducer = (state: RootState, action: AnyAction) => {
  if (action.type === "resetStore/resetStore") {
    console.log("Resetting store");
    const api_state = state[api.reducerPath];
    state = {
      [api.reducerPath]: api_state,
    } as RootState;
  }
  return appReducer(state, action);
};

const persistConfig = {
  key: "root",
  storage: storage("persistantstorage"),
  serialize: false,
  deserialize: false,
};

const persistantReducer = persistReducer(persistConfig, rootReducer);
export const store = configureStore({
  reducer: persistantReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
    })
      .concat(middlewares as any) //eslint-disable-line
      .concat(api.middleware),
});
initMessageListener(store);
setupListeners(store.dispatch);

export type AppDispatch = typeof store.dispatch;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;
