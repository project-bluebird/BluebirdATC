export interface Aircraft {
  callsign: string;
  type: string;
  controlling_sector: string;
  previous_sector: string;
  heading: number;
  ground_track: number;
  cleared_heading: number;
  cleared_cas: number;
  cleared_mach: number;
  true_air_speed: number | null;
  ground_speed: number;
  filed_true_airspeed: number;
  cleared_flight_level: number;
  selected_flight_level: number;
  requested_flight_level: number;
  route: string[];
  flight_level: number;
  lats: number[];
  lons: number[];
  entry_time: string;
  squawk: string;
  assigned_squawk: string;
  squawk_identing: boolean;
  wake_vortex: string;
  rate_climb_descent: number;
  max_rate_climb_descent: number;
  unexpanded_route: string;
  lat: number;
  lon: number;
  entry_flight_level: number | null;
  exit_flight_level: number | null;
  bay: string | null;
  intention_code: string;
  route_status?: object | null;
  coordinations?: string[] | null;
  controllable?: boolean;
}

export interface Fix {
  lat: number;
  lon: number;
  name: string;
  visible: boolean;
}

export interface ClearanceAndResponse {
  clearance: string;
  pilot_response: string;
}

export interface Action {
  agent: string;
  callsign: string;
  kind: string;
  value: number | string | [number, string] | null;
  text_representation?: ClearanceAndResponse | null;
  voice_representation?: ClearanceAndResponse | null;
  sector?: string[];
}

export interface TimedActions {
  time: string;
  actions: Action[];
}

export type Coordinate = Array<number>;

export interface ColourProfile {
  [key: string]: string;
}

export interface OpacityProfile {
  [key: string]: number;
}

export interface LineWidthProfile {
  [key: string]: number;
}

export interface ToolCallsignStore {
  vectorLines: string[];
  rangeRings: string[];
  routes: string[];
  plans: string[];
}

export interface GeojsonPointGeometry {
  type: string;
  coordinates: Coordinate; // Longitude and latitude for a point
}

export interface GeojsonLineStringGeometry {
  type: string;
  coordinates: Coordinate[]; // Array of points
}

export interface GeojsonPolygonGeometry {
  type: string;
  coordinates: Coordinate[][]; // Array of LineString (which is an array of points)
}

// Union type for the geometry
type GeojsonGeometry =
  | GeojsonPointGeometry
  | GeojsonLineStringGeometry
  | GeojsonPolygonGeometry;

export interface GeojsonProperties {
  name?: string;
  sectorId?: string;
  line_type?: string;
  text?: null;
  marker?: null;
  fill_colour?: string;
  fill_alpha?: number;
  line_colour?: string;
  line_width?: number;
  marker_size?: number;
  marker_type?: string;
  dashed?: string;
}

export interface GeojsonFeature {
  type: string;
  properties: GeojsonProperties;
  geometry: GeojsonGeometry;
}

export interface GeojsonFeatureCollection {
  type: string;
  features: GeojsonFeature[];
}
