from __future__ import annotations

import functools
import logging
import os
import typing
from collections import defaultdict
from datetime import datetime, timedelta

from typing_extensions import Self

from bluebird_dt.core import Action, Aircraft, Coordination, WindField
from bluebird_dt.logger import ContextFilter, CustomFormatter, logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import Predictor
from bluebird_dt.simulator.simconfig import SaveConfig, SimulatorConfig
from bluebird_dt.utility.convert import timestamp_to_string
from bluebird_dt.utility.paths import LOG_DIR

if typing.TYPE_CHECKING:
    from bluebird_dt.scenario_manager.infinite import Infinite
    from bluebird_dt.scenario_manager.regular import Regular, RegularScenarioManagerConfig
    from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager, SpringfieldScenarioManagerConfig
    from bluebird_dt.scenario_manager.tactical import Tactical, TacticalScenarioManagerConfig
    from bluebird_dt.scenario_manager.two_aircraft import TwoAircraft, TwoAircraftScenarioManagerConfig


class Simulator:
    """
    Implementation of the `Simulator` interface.
    """

    logging_context: ContextFilter | None = None
    logging_file_handler: logging.FileHandler | None = None
    category: str | None
    scenario_name: str | None
    save_log_to_file: bool
    initialization_datetime: datetime
    manager: EnvironmentManager[Aircraft, WindField, WindField]
    projection_centre: tuple[float, float] | None
    use_wind: bool
    use_forecast: bool
    autosave: bool
    log_filename: str | None = None
    predictor: Predictor | None = None
    simulated_sectors: list[str] | typing.Literal["ALL"]
    save_interval: timedelta
    last_save_time: datetime

    def __init__(
        self,
        scenario_manager: SpringfieldScenarioManager | Infinite | TwoAircraft | Tactical | Regular,
        env_manager: EnvironmentManager,
        projection_centre: tuple[float, float] | None = None,
        category: str | None = None,
        scenario_name: str | None = None,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
    ):
        """
        Initialise a simulation instance from a given scenario. It is not recommended to use this directly, instead use
        the Simulator.from_category() or ScenarioManager.to_simulator() functions instead.

        Parameters
        ----------
        scenario_manager : SpringfieldScenarioManager | Infinite | TwoAircraft | Tactical | Regular
            The scenario manager instance to use for this simulator.
        env_manager : EnvironmentManager
            The environment manager instance to use for this simulator.
        projection_centre : tuple[float, float] | None
            The centre point for coordinate projection. Default is None.
        category: str, optional
            Category of the simulation. If None, no scenario manager or environment manager will be setup.
            Defaults to None.
        scenario_name: str, optional
            Scenario name. Defaults to None.
        use_wind: bool
            Whether the wind, if available, is present in the scenario. Defaults to True.
        use_forecast: bool
            Whether the forecasted wind, if available, is present in the scenario. Defaults to True.
        autosave: bool
            The scenario will autosave every 5 minutes if True. Defaults to True.
        attach_context_to_logger: bool
            Adds the scenario name and scenario category as context to the active logger. This should be set to False if
            you are initialising multiple simulator classes in the same logger as then the context will be meaningless.
            Defaults to True.
        save_log_to_file: bool
            The log will be saved to file on exit if True. Defaults to True.
        log_filename: str, optional
            The name of the log directory. If None, then {category}_{scenario_name}_{the_datetime} is used.
        predictor: Predictor, optional
            The Predictor to use for the simulation. If None the default predictor for the
            scenario type will be used.
        simulated_sectors: list[str] | typing.Literal["ALL"], default="ALL"
            The sectors to be simulated. If "ALL", all sectors will be simulated. If a list, only the sectors names in
            the list will be simulated. Currently only applicable for real world scenarios.

        Returns
        -------
        Simulator
        """

        from bluebird_dt.scenario_manager import ScenarioManager

        if not isinstance(scenario_manager, ScenarioManager):
            raise TypeError(
                f"Expected a type which implements the ScenarioManager class. Got {type(scenario_manager).__name__}"
            )

        if not isinstance(env_manager, EnvironmentManager):
            raise TypeError(f"env_manager must be of type EnvironmentManager, got {type(env_manager).__name__}")

        self.category = category
        self.scenario_name = scenario_name
        self.save_log_to_file = save_log_to_file
        self.use_wind = use_wind
        self.use_forecast = use_forecast
        self.simulated_sectors = simulated_sectors
        self.initialization_datetime = datetime.now()
        self.scenario_manager = scenario_manager
        self.manager = env_manager
        self.projection_centre = projection_centre

        if attach_context_to_logger:
            self.logging_context = ContextFilter(
                {
                    "scenario_name": self.scenario_name,
                    "scenario_category": self.category,
                }
            )
            logger.addFilter(self.logging_context)

        if log_filename is None:
            the_datetime_formatted = self.initialization_datetime.strftime("%Y_%m_%d__%H_%M_%S")
            # Windows can't handle ':' character in file-paths
            sanitised_name = scenario_name.replace(":", "_") if scenario_name is not None else None
            self.log_filename = f"{category}_{sanitised_name}_{the_datetime_formatted}"
        else:
            self.log_filename = log_filename

        if save_log_to_file:
            self.setup_logging()

        self.predictor = predictor
        self.autosave = autosave
        self.save_interval = timedelta(minutes=5.0)
        self.last_save_time = datetime.now()

    def setup_logging(self):
        """
        The location of the base logfile directory can be overridden by derived classes.
        """
        os.makedirs(os.path.join(LOG_DIR, "runtime_logs"), exist_ok=True)
        self.logging_file_handler = logging.FileHandler(
            os.path.join(LOG_DIR, "runtime_logs", self.log_filename + ".log")
        )
        self.logging_file_handler.setFormatter(CustomFormatter())
        logger.addHandler(self.logging_file_handler)

    @classmethod
    def from_category(
        cls,
        category: str,
        scenario_name: str,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
    ) -> Self:
        """
        Initialize a simulator from a given scenario category and name.

        Parameters
        ----------
        scenario_name: str
            Scenario name.
        category: str
            Category of the simulation.
        use_wind: bool
            Whether the wind, if available, is present in the scenario. Defaults to True.
        use_forecast: bool
            Whether the forecasted wind, if available, is present in the scenario. Defaults to True.
        autosave: bool
            The scenario will autosave every 5 minutes if True. Defaults to True.
        attach_context_to_logger: bool
            Adds the scenario name and scenario category as context to the active logger. This should be set to False if
            you are initialising multiple simulator classes in the same logger as then the context will be meaningless.
            Defaults to True.
        save_log_to_file: bool
            The log will be saved to file on exit if True. Defaults to True.
        log_filename: str, optional
            The name of the log directory. If None, then {category}_{scenario_name}_{the_datetime} is used.
        predictor: Predictor, optional
            The Predictor to use for the simulation. If None the default predictor for the
            scenario type will be used.
        simulated_sectors: list[str] | typing.Literal["ALL"], default="ALL"
            The sectors to be simulated. If "ALL", all sectors will be simulated. If a list, only the sectors names in
            the list will be simulated. Currently only applicable for real world scenarios.

        Returns
        -------
        Simulator
        """
        from bluebird_dt.scenario_manager.infinite import Infinite
        from bluebird_dt.scenario_manager.springfield import (
            SpringfieldScenarioManager,
        )
        from bluebird_dt.scenario_manager.two_aircraft import TwoAircraft

        match category:
            # lots of additional steps for the artificial airspace
            case "Artificial":
                return TwoAircraft.setup(
                    scenario_name=scenario_name,
                    log_filename=log_filename,
                    predictor=predictor,
                    use_wind=use_wind,
                    use_forecast=use_forecast,
                    autosave=autosave,
                    attach_context_to_logger=attach_context_to_logger,
                    save_log_to_file=save_log_to_file,
                    simulated_sectors=simulated_sectors,
                )
            case "Infinite":
                return Infinite.setup(
                    scenario_name=scenario_name,
                    log_filename=log_filename,
                    predictor=predictor,
                    use_wind=use_wind,
                    use_forecast=use_forecast,
                    autosave=autosave,
                    attach_context_to_logger=attach_context_to_logger,
                    save_log_to_file=save_log_to_file,
                    simulated_sectors=simulated_sectors,
                )
            case "Springfield":
                return SpringfieldScenarioManager.setup(
                    scenario_name=scenario_name,
                    log_filename=log_filename,
                    predictor=predictor,
                    use_wind=use_wind,
                    use_forecast=use_forecast,
                    autosave=autosave,
                    attach_context_to_logger=attach_context_to_logger,
                    save_log_to_file=save_log_to_file,
                    simulated_sectors=simulated_sectors,
                )
            case "Flight School":
                # Specify the parameters for the "Flight School" competition here
                return Infinite.setup(
                    scenario_name="Xplus-Sector",
                    log_filename=log_filename,
                    predictor=predictor,
                    use_wind=use_wind,
                    use_forecast=use_forecast,
                    autosave=autosave,
                    attach_context_to_logger=attach_context_to_logger,
                    save_log_to_file=save_log_to_file,
                    simulated_sectors=simulated_sectors,
                    random_seed=None,
                    num_starter_aircraft=2,
                    initial_spawn_rate=0.005,
                    spawn_rate_increment=0.005,
                    spawn_rate_increase_interval=30,
                    max_spawn_rate=0.1,
                    total_time_seconds=3600.0,
                )
            case _:
                raise ValueError(f"Unknown scenario category: {category}")

    def evolve(self, delta: float) -> bool:
        """
        Increment the simulation by a given time delta (seconds).

        Parameters
        ----------
        delta: float
            Time period to increment environment by

        Returns
        -------
        bool
        """
        if delta <= 0:
            raise ValueError("Time delta must be positive. Received: {}.", delta)

        if self.logging_context is not None:
            self.logging_context.set("timestamp", timestamp_to_string(self.manager.environment.time))

        # allow scenario manager to update events if needed
        if self.scenario_manager is not None:
            self.manager = self.scenario_manager.update(self.manager)

        self.manager.evolve(delta)

        if self.autosave:
            self.save(autosave=True)

        return True

    def request_coordination(self, status: str, coordination: Coordination) -> bool:
        """
        Add single CoordinationRequest to EnvironmentManager

        Parameters
        ----------
        status: str
            'Accept', 'Reject', 'Offer', 'CounterOffer' or 'RemoveRequest'
        coord: Coordination
            The requested coordination

        Returns
        -------
        bool
        """
        try:
            self.manager.request_coord(status, coordination)
            return True

        except Exception as err:
            logger.warning(f"WARNING! An error occurred requesting a coordination: {err}")
            return False

    def get_coord_requests(
        self,
        callsign: str | None = None,
        to_sector: str | None = None,
        from_sector: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Get coordination requests matching any combination of callsign to_sector and from_sector.

        Return as list of dictionaries.

        Parameters
        ----------
        callsign: str, optional
            Coordination requests will be filtered to this callsign, if not None
        to_sector: str, optional
            Coordination requests will be filtered to this to_sector, if not None
        from_sector: str, optional
            Coordination requests will be filtered to this from_sector, if not None

        Returns
        -------
        list of dictionary
            Each dictionary is a coordination request of the form
            { (callsign, from_sector, to_sector) : {status: <'Accepted' or 'Declined' or 'Request' or 'Delete'>,
                                                    coord: <a coordination> }}
        """
        coord_requests = self.manager.get_coord_requests(
            callsign=callsign, from_sector=from_sector, to_sector=to_sector
        )

        return [{"status": cr.status, "coord": cr.coord.to_json()} for cr in coord_requests]

    @functools.lru_cache(maxsize=1)
    def environment(
        self,
        sim_time: float,  # noqa: ARG002 -- time only used to enable time-based caching
        no_airspace: bool = False,
        last_n_observations: int = 0,
    ) -> dict[str, typing.Any]:
        """
        Get the current environment state. Excludes wind field data to reduce the size of data returned.

        Parameters
        ----------
        sim_time: float
            Environment time in unix time (seconds).
            Only used to enable time-based caching
        no_airspace: bool, default is False
            If true, airspace will not be included in return data
        last_n_observations: int, default is 0
            Number of last n observations to include in returned 'observations'

        Returns
        -------
        dict
            Dictionary describing the current state of the environment
        """
        env = self.manager.environment

        # only pass coordinations for aircraft in the airspace (there may be
        # coordinations for aircraft which haven't arrived yet
        coordinations = [
            coord.data() for coord in env.coordinations.values() if coord.callsign in set(env.aircraft.keys())
        ]

        # gather the basic env data, remove wind fields data to reduce the size of the data
        env_data = {
            "time": env.time,
            "start_time": env.start_time,
            "aircraft": {callsign: aircraft.data() for (callsign, aircraft) in env.aircraft.items()},
            "coordinations": coordinations,
        }

        # include the airspace if requested
        if not no_airspace:
            env_data["airspace"] = env.airspace.data()

        # include the last n observations
        if last_n_observations > 0:
            env_data["observations"] = {}

            for callsign, trajectory in self.manager.all_trajectories.items():
                observations = trajectory[-last_n_observations:]

                if len(observations) > 0:
                    env_data["observations"][callsign] = observations

        return env_data

    @functools.lru_cache(maxsize=1)
    def static_data(
        self,
        sim_time: float,  # noqa: ARG002 -- time only used to enable time-based caching
    ) -> dict[str, typing.Any]:
        """
        Get the static scenario data for a given sector.

        Parameters
        ----------
        sim_time: float
            Environment time in unix time (seconds).
            Only used to enable time-based caching

        Returns
        -------
        dict
            `Static` data for the scenario
        """
        func_start_time = datetime.now()
        airspace = self.manager.get_sector_airspace(
            None, local_fixes=self.category not in ["Basic Training", "Artificial", "Replay"]
        )

        # extract the fixes from all aircraft that are in the scenario
        all_aircraft_fixes = {
            fix for route_filed in self.manager.event_handler.flight_df.route_filed for fix in route_filed
        }

        # If we replaced environment via the API, we might not have Events, but
        # still have aircraft - add fixes from their routes as well.
        flight_plans = [
            ac.flight_plan for ac in self.manager.environment.aircraft.values() if ac.flight_plan is not None
        ]
        fixes_from_routes = {fix for fp in flight_plans for fix in fp.route.filed}

        # add and remove duplicates
        all_aircraft_fixes = all_aircraft_fixes.union(fixes_from_routes)

        # filter the airspace fixes to only those that appear in the aircraft fixes
        # and airport fixes (4 letters)
        fixes: list[dict[str, typing.Any]] = []

        for i, (name, pos) in enumerate(airspace.fixes.places.items()):
            # include all fixes in replay
            if self.category in ["Basic Training", "Replay", "Springfield"] or (
                name.isalpha()
                and (
                    name in all_aircraft_fixes
                    or (
                        # airports start with E and have 4 letters (until we have proper data filtering)
                        len(name) == 4 and name.startswith("E")
                    )
                )
            ):
                fixes.append(
                    {"id": i, "name": name, "lat": pos.lat, "lon": pos.lon, "visible": airspace.fixes.visibility[name]}
                )

        sectors = {}
        for name, sector in airspace._individual_sectors.items():
            sectors[name] = [
                [[p[1], p[0]] for p in volume.area.boundary.exterior.coords][:-1] for volume in sector.volumes
            ]

        bays = self.manager.get_bays_names()

        func_end_time = datetime.now()
        logger.debug(f"Static_data_took {func_end_time - func_start_time}")

        return {
            "scenario_name": self.scenario_name,
            "bay_names": bays,
            "sectors": sectors,
            "fixes": fixes,
            "projection_centre": self.projection_centre,
        }

    @functools.lru_cache(maxsize=1)
    def dynamic_data(
        self,
        sim_time: float,  # noqa: ARG002 -- time only used to enable time-based caching
        sector_id: str | None = None,
    ) -> dict[str, typing.Any]:
        """
        Get the volatile scenario data for a given sector.

        Parameters
        ----------
        sim_time: float
             Environment time in unix time (seconds).
            Only used to enable time-based caching
        sector_id: str, optional
             Semicolon separated list of individual sectors for which dynamic data is to be returned.

        Returns
        -------
        dict
            `Dynamic` data for the scenario. Data that updates regularly as a scenario evolves.
        """
        func_start_time = datetime.now()
        # radar displays all aircraft
        environment = self.manager.environment

        if sector_id is not None:
            sector_id = self.manager.environment.airspace.sector_name_from_list_of_individual_sectors(
                sector_id.split(";")
            )

        n_trail_dots = 6

        aircraft_data = []

        all_trajectories = self.manager.all_trajectories

        # only return aircraft data if sector_id in airspace
        for n, (callsign, aircraft) in enumerate(environment.aircraft.items()):
            trail_lats = []
            trail_lons = []

            if all_trajectories[callsign] is not None:
                trail = all_trajectories[callsign][-n_trail_dots:]
                n_replicates = n_trail_dots - len(trail)

                # Ensure there are always `n_train_dots` points in the trails
                if len(trail) > 0:
                    trail += [trail[-1]] * n_replicates
                    trail.reverse()
                    trail_lats = [p[0] for p in trail]
                    trail_lons = [p[1] for p in trail]
                else:
                    # just use current lat and long of aircraft as no trail yet
                    trail_lats = [self.manager.environment.aircraft[callsign].lat] * n_trail_dots
                    trail_lons = [self.manager.environment.aircraft[callsign].lon] * n_trail_dots

            nfl = None
            xfl = None

            if sector_id is None or sector_id == "ALL":
                bay = None
            else:
                bay = self.manager.get_assigned_bay(sector_id, callsign)

                # get next sector considering bandboxing
                next_sector = self.manager.environment.next_sector_of_aircraft(callsign)

                # include the entry and exit fls only if we will next be or are currently controlling the aircraft
                if sector_id in [aircraft.current_sector, next_sector]:
                    # entry coordination is 'to' this sector
                    entry_coord = self.manager.environment.entry_coordination(sector_id, callsign)

                    # exit coord is 'from' this sector
                    exit_coord = self.manager.environment.exit_coordination(sector_id, callsign)

                    if entry_coord is not None:
                        nfl = entry_coord.fl

                    if exit_coord is not None:
                        xfl = exit_coord.fl

            entry_time = "--"  # TODO: fill this, if possible, with predicted entry time to sector

            # get values from flight plan if flight plan is not None
            if aircraft.flight_plan is None:
                filed_route = []
                unexpanded_route = []
                requested_flight_level = None
                filed_true_airspeed = None
                intention_code = None
                assigned_squawk = None
            else:
                filed_route = aircraft.flight_plan.route.filed
                unexpanded_route = aircraft.flight_plan.unexpanded_route
                requested_flight_level = aircraft.flight_plan.requested_flight_level
                filed_true_airspeed = aircraft.flight_plan.filed_true_airspeed
                intention_code = aircraft.flight_plan.intention_code
                assigned_squawk = str(aircraft.flight_plan.assigned_squawk)

            assigned_squawk = None if assigned_squawk is None else assigned_squawk.zfill(4)
            squawk_identing = (
                False if aircraft.squawk_ident_until is None else aircraft.squawk_ident_until > environment.time
            )

            aircraft_data.append(
                {
                    "id": n,
                    "callsign": callsign,
                    "entry_time": entry_time,
                    "squawk": "--" if aircraft.squawk is None else aircraft.squawk,
                    "assigned_squawk": assigned_squawk,
                    "squawk_identing": squawk_identing,
                    "wake_vortex": aircraft.wake_vortex,
                    "type": aircraft.aircraft_type,
                    "controlling_sector": aircraft.current_sector,
                    "previous_sector": aircraft.previous_sector,
                    "bay": bay,
                    "route_direct": aircraft.on_route,
                    "flight_level": aircraft.fl,
                    "cleared_flight_level": aircraft.cleared_fl,
                    "selected_flight_level": aircraft.selected_fl,
                    "requested_flight_level": requested_flight_level,
                    "heading": aircraft.heading,
                    "ground_track": aircraft.ground_track_angle,
                    "cleared_heading": aircraft.cleared_instructions.heading,
                    "true_air_speed": aircraft.speed_tas,
                    "ground_speed": aircraft.ground_speed,
                    "filed_true_airspeed": filed_true_airspeed,
                    "cleared_cas": aircraft.cleared_instructions.cas,
                    "cleared_mach": aircraft.cleared_instructions.mach,
                    "rate_climb_descent": aircraft.vertical_speed,
                    "max_rate_climb_descent": "--",
                    "route": filed_route,
                    "unexpanded_route": unexpanded_route,
                    "lat": aircraft.lat,
                    "lon": aircraft.lon,
                    "lats": trail_lats,
                    "lons": trail_lons,
                    "entry_flight_level": nfl,
                    "exit_flight_level": xfl,
                    "intention_code": intention_code,
                    "coordinations": [
                        coord.to_json() for coord in self.manager.environment.coordinations.get(callsign)
                    ],
                    # "coordinations": [],  # dynamic data can't always cope with size of coordinations
                    "controllable": aircraft.controllable,
                }
            )
        # transform clearance log into format expected by HMI
        action_log = defaultdict(list)
        for log in self.manager.event_logger.clearances_log:
            action_log[log["datetime"]].append(
                {
                    "callsign": log["callsign"],
                    "kind": log["kind"],
                    "value": log["value"],
                    "agent": log["agent"],
                    "text_representation": {
                        "clearance": log["text_clearance"],
                        "pilot_response": log["text_pilot_response"],
                    },
                    "voice_representation": {
                        "clearance": log["voice_clearance"],
                        "pilot_response": log["voice_pilot_response"],
                    },
                    "sector": log["sector"],
                }
            )

        timed_action_log = []
        for time, action_list in sorted(action_log.items()):
            timed_action_log.append(
                {
                    "time": time.isoformat(timespec="microseconds")[:-6].replace("T", " "),
                    "actions": action_list,
                }
            )

        func_end_time = datetime.now()
        logger.debug(f"Dynamic_data_took {func_end_time - func_start_time}")
        return {
            "time": timestamp_to_string(environment.time).replace("T", " "),
            "actions": timed_action_log,
            "aircraft": aircraft_data,
        }

    def action(self, actions: list[dict[str, typing.Any]]) -> bool:
        """
        Add actions to the queue.

        Parameters
        ----------
        actions: list[dict[str, Any]]
            List of dictionaries. Each dictionary represents a single Action

        Returns
        -------
        bool
        """
        try:
            actions_parsed: list[Action] = []

            for act in actions:
                # The sector may be passed as the bandbox configuration, which for consistency, parsed on
                # the edge to be a list of individual sectors, e.g. ["25", "26"] representing the sector frequencies.

                sector_unparsed: list[str] | str | None = act.get("sector", None)
                sector: list[str] | None = (
                    self.manager.environment.airspace.expand_bandbox_sector(sector_unparsed)
                    if isinstance(sector_unparsed, str)
                    else sector_unparsed
                )

                actions_parsed.append(
                    Action(
                        act["callsign"],
                        act["kind"],
                        act["value"],
                        act["agent"],
                        sector=sector,
                    )
                )

            self.manager.receive_actions(actions_parsed)

        except Exception as err:
            logger.warning(
                f"WARNING! An error occurred sending actions: {err}",
                stacklevel=2,
            )
            return False

        return True

    def config(
        self,
    ) -> SaveConfig[
        RegularScenarioManagerConfig
        | TacticalScenarioManagerConfig
        | SpringfieldScenarioManagerConfig
        | TwoAircraftScenarioManagerConfig
        | InfiniteScenarioManagerConfig
    ]:
        """
        Obtain the configuration this instance of a simulator is running with.

        Returns
        -------
        SaveConfig[
            RegularScenarioManagerConfig
            | TacticalScenarioManagerConfig
            | SpringfieldScenarioManagerConfig
            | TwoAircraftScenarioManagerConfig
            | InfiniteScenarioManagerConfig
        ]
        """
        return SaveConfig(
            scenario_name=self.scenario_name,
            scenario_category=self.category,
            save_real_datetime=self.last_save_time,
            load_real_datetime=self.initialization_datetime,
            simulator=SimulatorConfig(
                projection_centre=self.projection_centre,
            ),
            save_simulator_datetime=self.manager.environment.datetime,
            scenario=self.scenario_manager.config() if self.scenario_manager is not None else None,
            environment_manager=self.manager.config(),
        )

    def close(self):
        """
        Clean up after the simulator, otherwise it doesn't get garbage collected.
        """
        self.scenario_manager.close()

        if self.logging_context:
            logger.removeFilter(self.logging_context)

        if self.logging_file_handler:
            logger.removeHandler(self.logging_file_handler)

        self.environment.cache_clear()
        self.dynamic_data.cache_clear()
        self.static_data.cache_clear()

    def save(self, autosave: bool = False) -> bool:
        """
        Save simulator state to JSON.

        Parameters
        ----------
        autosave: bool
            Whether autosave mode is being used

        Returns
        -------
        bool
        """
        # if we're in autosave mode and enough time hasn't passed, don't save
        if autosave and (datetime.now() - self.last_save_time < self.save_interval):
            return False

        # update the last save time
        self.last_save_time = datetime.now()

        sim_config = self.config()

        # Guard against the unlikely event that autosave true but save_logs_to_file is false
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, self.manager.event_logger.log_name + ".tar.gz")
        with open(log_path, "wb") as tar:
            tar.write(self.manager.write_logs_to_buffer(sim_config).getvalue())
            logger.info(f"Log saved to {log_path}")

        return True
