import { emptySplitApi as api } from "./emptyApi";
const injectedRtkApi = api.injectEndpoints({
  endpoints: (build) => ({
    index: build.query<IndexApiResponse, IndexApiArg>({
      query: () => ({ url: `/` }),
    }),
    listScenarioCategories: build.query<
      ListScenarioCategoriesApiResponse,
      ListScenarioCategoriesApiArg
    >({
      query: () => ({ url: `/list_scenario_categories` }),
    }),
    listScenarios: build.query<ListScenariosApiResponse, ListScenariosApiArg>({
      query: (queryArg) => ({ url: `/list_scenarios/${queryArg.category}` }),
    }),
    scenarioInfo: build.query<ScenarioInfoApiResponse, ScenarioInfoApiArg>({
      query: (queryArg) => ({
        url: `/info/${queryArg.category}/${queryArg.scenarioName}`,
      }),
    }),
    close: build.mutation<CloseApiResponse, CloseApiArg>({
      query: () => ({ url: `/close`, method: "POST" }),
    }),
    evolve: build.mutation<EvolveApiResponse, EvolveApiArg>({
      query: (queryArg) => ({
        url: `/evolve/${queryArg.timeDelta}`,
        method: "POST",
      }),
    }),
    start: build.mutation<StartApiResponse, StartApiArg>({
      query: (queryArg) => ({
        url: `/start/${queryArg.tickFrequencyPeriod}`,
        method: "POST",
      }),
    }),
    environment: build.query<EnvironmentApiResponse, EnvironmentApiArg>({
      query: (queryArg) => ({
        url: `/environment`,
        params: {
          no_airspace: queryArg.noAirspace,
          last_n_observations: queryArg.lastNObservations,
        },
      }),
    }),
    windField: build.query<WindFieldApiResponse, WindFieldApiArg>({
      query: () => ({ url: `/wind_field` }),
    }),
    forecastWindField: build.query<
      ForecastWindFieldApiResponse,
      ForecastWindFieldApiArg
    >({
      query: () => ({ url: `/forecast_wind_field` }),
    }),
    staticData: build.query<StaticDataApiResponse, StaticDataApiArg>({
      query: () => ({ url: `/static_data` }),
    }),
    dynamicData: build.query<DynamicDataApiResponse, DynamicDataApiArg>({
      query: (queryArg) => ({ url: `/dynamic_data/${queryArg.sectorId}` }),
    }),
    actions: build.mutation<ActionsApiResponse, ActionsApiArg>({
      query: (queryArg) => ({
        url: `/actions`,
        method: "POST",
        body: queryArg.actionInput,
      }),
    }),
    runnerStatus: build.query<RunnerStatusApiResponse, RunnerStatusApiArg>({
      query: () => ({ url: `/status` }),
    }),
    save: build.mutation<SaveApiResponse, SaveApiArg>({
      query: () => ({ url: `/save`, method: "POST" }),
    }),
    setEvolvePeriod: build.mutation<
      SetEvolvePeriodApiResponse,
      SetEvolvePeriodApiArg
    >({
      query: (queryArg) => ({
        url: `/evolve_period/${queryArg.evolvePeriod}`,
        method: "POST",
      }),
    }),
    setTickFrequency: build.mutation<
      SetTickFrequencyApiResponse,
      SetTickFrequencyApiArg
    >({
      query: (queryArg) => ({
        url: `/tick_frequency/${queryArg.tickFrequency}`,
        method: "POST",
      }),
    }),
    pause: build.mutation<PauseApiResponse, PauseApiArg>({
      query: () => ({ url: `/pause`, method: "POST" }),
    }),
    rewind: build.mutation<RewindApiResponse, RewindApiArg>({
      query: (queryArg) => ({
        url: `/rewind/${queryArg.newTime}`,
        method: "POST",
      }),
    }),
    getSelectedAircraft: build.query<
      GetSelectedAircraftApiResponse,
      GetSelectedAircraftApiArg
    >({
      query: (queryArg) => ({ url: `/selected_aircraft/${queryArg.sectorId}` }),
    }),
    load: build.mutation<LoadApiResponse, LoadApiArg>({
      query: (queryArg) => ({
        url: `/load/${queryArg.category}/${queryArg.scenarioName}`,
        method: "POST",
      }),
    }),
  }),
  overrideExisting: false,
});
export { injectedRtkApi as api };
export type IndexApiResponse = /** status 200 Successful Response */ string;
export type IndexApiArg = void;
export type ListScenarioCategoriesApiResponse =
  /** status 200 Successful Response */ any;
export type ListScenarioCategoriesApiArg = void;
export type ListScenariosApiResponse =
  /** status 200 Successful Response */ any;
export type ListScenariosApiArg = {
  category: string;
};
export type ScenarioInfoApiResponse = /** status 200 Successful Response */ any;
export type ScenarioInfoApiArg = {
  category: string;
  scenarioName: string;
};
export type CloseApiResponse = /** status 200 Successful Response */ boolean;
export type CloseApiArg = void;
export type EvolveApiResponse = /** status 200 Successful Response */ boolean;
export type EvolveApiArg = {
  timeDelta: number;
};
export type StartApiResponse = /** status 200 Successful Response */ boolean;
export type StartApiArg = {
  tickFrequencyPeriod: number;
};
export type EnvironmentApiResponse = /** status 200 Successful Response */ any;
export type EnvironmentApiArg = {
  noAirspace?: boolean;
  lastNObservations?: number;
};
export type WindFieldApiResponse =
  /** status 200 Successful Response */ WindField | null;
export type WindFieldApiArg = void;
export type ForecastWindFieldApiResponse =
  /** status 200 Successful Response */ WindField | null;
export type ForecastWindFieldApiArg = void;
export type StaticDataApiResponse = /** status 200 Successful Response */ any;
export type StaticDataApiArg = void;
export type DynamicDataApiResponse = /** status 200 Successful Response */ any;
export type DynamicDataApiArg = {
  sectorId: string | null;
};
export type ActionsApiResponse = /** status 200 Successful Response */ boolean;
export type ActionsApiArg = {
  actionInput: Action[];
};
export type RunnerStatusApiResponse = /** status 200 Successful Response */ any;
export type RunnerStatusApiArg = void;
export type SaveApiResponse = /** status 200 Successful Response */ boolean;
export type SaveApiArg = void;
export type SetEvolvePeriodApiResponse =
  /** status 200 Successful Response */ boolean;
export type SetEvolvePeriodApiArg = {
  evolvePeriod: number;
};
export type SetTickFrequencyApiResponse =
  /** status 200 Successful Response */ boolean;
export type SetTickFrequencyApiArg = {
  tickFrequency: number;
};
export type PauseApiResponse = /** status 200 Successful Response */ boolean;
export type PauseApiArg = void;
export type RewindApiResponse = /** status 200 Successful Response */ boolean;
export type RewindApiArg = {
  newTime: string;
};
export type GetSelectedAircraftApiResponse =
  /** status 200 Successful Response */ {
    [key: string]: any;
  };
export type GetSelectedAircraftApiArg = {
  sectorId: string;
};
export type LoadApiResponse = /** status 200 Successful Response */ boolean;
export type LoadApiArg = {
  category: string;
  scenarioName: string;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: any;
  ctx?: object;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type Pos2D = {
  lat: number;
  lon: number;
};
export type Area = {
  boundary: Pos2D[];
  boundary_vertices?: Pos2D[] | null;
};
export type Volume = {
  area: Area;
  min_fl: number;
  max_fl: number;
  sector_name?: string | null;
  description?: string | null;
  airspace_id?: string | null;
};
export type Sector = {
  volumes: Volume[];
  area_of_responsibility?: Volume[] | null;
  conditional_volume_dict?: {
    [key: string]: Volume;
  } | null;
};
export type Fixes = {
  places: {
    [key: string]: Pos2D;
  };
};
export type AirwayLeg = {
  upper_limit_fl: number;
  lower_limit_fl: number;
  p0: Pos2D;
  p0_identifier: string;
  p1: Pos2D;
  p1_identifier: string;
};
export type Airway = {
  identifier: string;
  legs?: AirwayLeg[] | null;
};
export type Airspace = {
  sectors: {
    [key: string]: Sector;
  };
  fixes: Fixes;
  airways?: {
    [key: string]: Airway;
  } | null;
};
export type AirspaceRead = {
  sectors: {
    [key: string]: Sector;
  };
  fixes: Fixes;
  airways?: {
    [key: string]: Airway;
  } | null;
  /** The configuration of the Airspace.
    A key:value pair. The key is the bandboxed sector name and the value is a
    list of the individual sector names which together form the bandboxed sector. */
  airspace_configuration: {
    [key: string]: string[];
  };
};
export type Route = {
  filed: string[];
  current?: string[] | null;
};
export type FlightPlan = {
  route: Route;
  unexpanded_route?: string;
  origin?: string | null;
  dest?: string | null;
  milcivil?: ("M" | "C") | null;
  sector_crossing_seq?: string | null;
  requested_flight_level?: number | null;
  filed_true_airspeed?: number | null;
  intention_code?: string | null;
  assigned_squawk?: string | null;
  start_datetime?: string | null;
  end_datetime?: string | null;
};
export type Pilot = {
  callsign: string;
};
export type ClearanceAndResponse = {
  clearance: string | null;
  pilot_response: string | null;
};
export type Action = {
  callsign: string;
  kind: string;
  value?: number | number | string | string[] | [number, string] | null;
  agent?: string | null;
  voice_representation?: ClearanceAndResponse | null;
  text_representation?: ClearanceAndResponse | null;
  sector?: string[] | null;
};
export type Instructions = {
  fl?: number | null;
  mach?: number | null;
  cas?: number | null;
  vertical_speed?: number | null;
  heading?: number | null;
  on_route?: boolean;
  speed_action?: Action | null;
  vertical_speed_action?: Action | null;
  vertical_action?: Action | null;
  lateral_action?: Action | null;
};
export type Aircraft = {
  lat: number;
  lon: number;
  fl: number;
  heading: number;
  flight_plan: FlightPlan | null;
  callsign: string;
  ufid?: string | null;
  rate_of_turn?: number | null;
  aircraft_type?: string | null;
  operation_params?: {
    [key: string]: any;
  } | null;
  controllable?: boolean;
  simulated?: boolean;
  current_sector?: string | null;
  random_seed?: number | null;
  pilot?: Pilot | null;
  squawk?: string | null;
  wake_vortex?: string | null;
  last_passed_filed_idx?: number | null;
  last_passed_current_idx?: number | null;
  squawk_ident_until?: number | null;
  cleared_instructions?: Instructions | null;
  selected_instructions?: Instructions | null;
  percentile_rank_dict?: {
    [key: string]: [number, null];
  } | null;
  speed_tas?: number | null;
  vertical_speed?: number | null;
  heading_changing_to?: number | null;
  next_fix_index?: number | null;
  ground_speed?: number | null;
  ground_track_angle?: number | null;
  predictor_params?: {
    [key: string]: any;
  } | null;
};
export type WindField = {
  u_comp: any[];
  v_comp: any[];
  pressure_array: any[];
  lat_array: any[];
  lon_array: any[];
  interpolation_method?: "trilinear" | "nearest" | "fl_interpolation";
  wind_speed?: any[] | number | null;
  wind_direction?: any[] | number | null;
};
export type Environment = {
  time: number;
  airspace: Airspace;
  aircraft: {
    [key: string]: Aircraft;
  };
  wind_field?: WindField | null;
  forecast_wind_field?: WindField | null;
  airspace_bandboxing_dict?: {
    [key: string]: string[];
  } | null;
  start_time?: number | null;
};
export type Coordination = {
  callsign: string;
  from_sector: string | null;
  to_sector: string | null;
  fl: number;
  fix: string;
  direction: "Horizontal" | "Down" | "Up";
  level_by?: boolean;
  level_by_details?: {
    [key: string]: number;
  } | null;
  secondary_coord_conditions?: string | null;
  datetime?: string | null;
};
export type EnvironmentRead = {
  time: number;
  airspace: AirspaceRead;
  aircraft: {
    [key: string]: Aircraft;
  };
  wind_field?: WindField | null;
  forecast_wind_field?: WindField | null;
  airspace_bandboxing_dict?: {
    [key: string]: string[];
  } | null;
  start_time?: number | null;
  /** Expose coordinations as list for serialization. */
  coordinations: Coordination[];
};
export const {
  useIndexQuery,
  useListScenarioCategoriesQuery,
  useListScenariosQuery,
  useScenarioInfoQuery,
  useCloseMutation,
  useEvolveMutation,
  useStartMutation,
  useEnvironmentQuery,
  useWindFieldQuery,
  useForecastWindFieldQuery,
  useStaticDataQuery,
  useDynamicDataQuery,
  useActionsMutation,
  useRunnerStatusQuery,
  useSaveMutation,
  useSetEvolvePeriodMutation,
  useSetTickFrequencyMutation,
  usePauseMutation,
  useRewindMutation,
  useGetSelectedAircraftQuery,
  useLoadMutation,
} = injectedRtkApi;
