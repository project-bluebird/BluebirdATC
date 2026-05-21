import os
import typing
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

import pandas as pd
from pydantic import BaseModel
from typing_extensions import override

from bluebird_dt.airspace_generator import SpringfieldAirspace
from bluebird_dt.core import Aircraft, Environment, FlightPlan, Pos2D, Route, WindField
from bluebird_dt.events import EventHandler, EventLogger
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import LinearPredictor, Predictor
from bluebird_dt.scenario_manager.outcomm_handler import OutcommHandler
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator.simulator import Simulator
from bluebird_dt.utility.paths import SPRINGFIELD_DIR
from bluebird_dt.utility.scenario_utils import convert_string_to_lists

TAircraft = TypeVar("TAircraft", bound=Aircraft)
TWindField = TypeVar("TWindField", bound=WindField)
TForecastWindField = TypeVar("TForecastWindField", bound=WindField)
TEventLogger = TypeVar("TEventLogger", bound=EventLogger)
TEventHandler = TypeVar("TEventHandler", bound=EventHandler[Aircraft])
TSimulator = TypeVar("TSimulator", bound=Simulator)


class SpringfieldScenarioManagerConfig(BaseModel):
    """
    Configuration of the springfield scenario manager
    """

    scenario_manager: typing.Literal["springfield"] = "springfield"


class SpringfieldScenarioManager(
    ScenarioManager[SpringfieldScenarioManagerConfig],
    Generic[TAircraft, TWindField, TForecastWindField, TEventLogger, TEventHandler],
):
    """
    Constructs a ScenarioManager from a Springfield scenario.
    The scenario will be encoded in a JSON file which we parse here to
    create aircraft events.
    """

    scenario_name: str
    vertical_buffer_distance: float | int
    lateral_buffer_distance: float | int
    airport_fix_distance_threshold: float
    initialise_with_event_handler: bool
    outcomm_handler: OutcommHandler | None
    typeof_environment_manager: type[EnvironmentManager[TAircraft, TWindField, TForecastWindField]]
    typeof_aircraft: type[TAircraft]
    typeof_event_logger: type[TEventLogger]
    typeof_event_handler: type[TEventHandler]
    scenario_directory: str = os.path.join(SPRINGFIELD_DIR, "scenarios")
    projection_centre: tuple[float, float] = (0.186029, 51.888057)
    event_handler_ignore_flags: EventHandler.IgnoreFlags

    def __init__(
        self,
        scenario_name: str,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
        airport_fix_distance_threshold: float = 5.0,
        initialise_with_event_handler: bool = True,
        typeof_environment_manager: type[
            EnvironmentManager[TAircraft, TWindField, TForecastWindField]
        ] = EnvironmentManager,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        typeof_event_handler: type[TEventHandler] = EventHandler[Aircraft],
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        scenario_name: str
            Name of the scenario.
        typeof_environment_manager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        airport_fix_distance_threshold: float
            Distance from airport (4-letter) fix at which we can say aircraft has landed.
        initialise_with_event_handler: bool, default is True
            Initialise the environment with the EventHandler
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the event logger, specify here.
        typeof_event_handler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        """
        self.scenario_name = scenario_name
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.airport_fix_distance_threshold = airport_fix_distance_threshold
        self.initialise_with_event_handler = initialise_with_event_handler
        self.typeof_environment_manager = typeof_environment_manager
        self.typeof_aircraft = typeof_aircraft
        self.typeof_event_logger = typeof_event_logger
        self.typeof_event_handler = typeof_event_handler
        self.event_handler_ignore_flags = typeof_event_handler.IgnoreFlags(
            radar_if_simmed=True,
            flight_if_simmed=False,
            clearance_if_simmed=False,
            coordination_if_simmed=False,
            sectors_if_simmed=False,
            incomm_if_simmed=False,
            aircraft_internals_if_simmed=True,
            ac_attribute_if_simmed=False,
            airspace_config_updates=True,
        )
        self.outcomm_handler = OutcommHandler(require_exit_fl=False)

    @classmethod
    def setup(
        cls,
        scenario_name: str,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
        typeof_environment_manager: type[
            EnvironmentManager[TAircraft, TWindField, TForecastWindField]
        ] = EnvironmentManager,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        typeof_event_handler: type[TEventHandler] = EventHandler[Aircraft],
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """Setup Springfield scenarios based on scenario name.

        Parameters
        ----------
        scenario_name: str
            The scenario name
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
        env_manager_class: type, optional
            if specified, use this class (maybe a subclass of BluebirdATC EventManager).

        Returns
        -------
        Simulator
            A fully configured simulator instance
        """
        return cls(
            scenario_name,
            lateral_buffer_distance=500,
            vertical_buffer_distance=500,
            initialise_with_event_handler=True,
            typeof_aircraft=typeof_aircraft,
            typeof_event_logger=typeof_event_logger,
            typeof_event_handler=typeof_event_handler,
            typeof_environment_manager=typeof_environment_manager,
        ).to_simulator(
            predictor=predictor,
            log_filename=log_filename,
            category="Springfield",
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            simulated_sectors=simulated_sectors,
            typeof_simulator=typeof_simulator,
        )

    @override
    def create_event_handler(self) -> TEventHandler:
        """
        Generate event_handler for the scenario

        Parameters
        ----------

        Returns
        -------
        EventHandler
        """

        event_handler = self.typeof_event_handler(
            ignore=self.event_handler_ignore_flags,
            typeof_aircraft=self.typeof_aircraft,
        )

        self.__event_handler_flights_init(event_handler)
        self.__event_handler_events_init(event_handler)

        return event_handler

    def create_env_manager(
        self,
        predictor: Predictor | None = None,
        log_filename: str | None = None,
    ) -> EnvironmentManager[TAircraft, TWindField, TForecastWindField]:
        """
        Create an EnvironmentManager for springfield.

        Parameters
        ----------
        predictor: Predictor, optional
            Predictor to simulate aircraft trajectories
            If None, LinearPredictor is used in simple performance mode
        log_filename: str or None
            Name of file logs will be saved to. If None, defaults to datetime logger created.

        Returns
        ----------
        EnvironmentManager
            EnvironmentManager for springfield scenarios
        """

        logger.info(
            """
        ===================================================================
        Creating Springfield Scenario.
        """
        )

        # create instance of base Predictor class if no predictor passed
        if predictor is None:
            predictor = LinearPredictor(dt=1, fix_proximity_threshold=0.5, fixes=SpringfieldAirspace._fixes_init)

        # create event handler from the events list
        event_handler = self.create_event_handler()

        em = self.typeof_environment_manager(
            airspace=SpringfieldAirspace._airspace_init(),
            event_handler=event_handler,
            predictor=predictor,
            time=0,
            penumbra_fl=int(self.vertical_buffer_distance),
            penumbra_lat=self.lateral_buffer_distance,
            log_filename=log_filename,
        )

        if self.initialise_with_event_handler:
            em.initialise_env_with_event_handler()

        return em

    @override
    def update(
        self, env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField]
    ) -> EnvironmentManager[TAircraft, TWindField, TForecastWindField]:
        """
        Update the environment and/or coordinations.

        Currently, this method does the following, on every time step:
        * Outcomms any aircraft that have left the sector via the correct edge of the boundary
         - optionally also require that they meet the exit FL
        * Checks for all aircraft whether they have been incommed, and if so, set on_route to False
        * Calls a method that checks for aircraft arriving at an airport (4-letter fix), in order
        to remove them from the environment.


        Parameters
        ----------
        env_manager: EnvironmentManager
            An environment manager containing the environment and coordinations

        Returns
        -------
        EnvironmentManager
        """
        environment = env_manager.environment

        # outcomm any aircraft that are no longer in sector
        if self.outcomm_handler is not None:
            self.outcomm_handler.update(env_manager)

        # any aircraft that have been incommed (current_sector is not "background")
        # should have on_route flag set to False.
        # (Note that we want this to remain False even after they outcomm.)
        for a in environment.aircraft.values():
            if a.current_sector != "background":
                a.on_route = False

        # remove aircraft from the scenario if they "land",
        # i.e. the last fix on their route is an airport, and they get close to that fix.
        aircraft_to_remove = self._check_aircraft_landing(environment, check_exit_fl=False)

        for callsign in aircraft_to_remove:
            del environment.aircraft[callsign]

        return env_manager

    def _check_aircraft_landing(self, environment: Environment, check_exit_fl: bool = False) -> list[str]:
        """
        Check whether aircraft with a four-letter fix at the end of their route have reached some
        threshold (e.g. 5NM) from that final fix, and if so, remove them (as they have landed).

        Parameters
        ==========
        environment: Environment
            the current simulation environment
        check_exit_fl: bool
            if True, only add the aircraft to the "to-be-removed" list if it meets exit FL.

        Returns
        =======
        aircraft_to_remove: list[str]
            callsigns of aircraft to remove from scenario
        """
        aircraft_to_remove = []
        for callsign, a in environment.aircraft.items():
            last_fix = a.flight_plan.route.filed[-1]
            if len(last_fix) == 4:
                exit_coordination = environment.exit_coordination(a.current_sector, a.callsign)
                # check whether aircraft is within distance threshold of fix
                fix_location = environment.airspace.fixes.places[last_fix]
                aircraft_location = Pos2D(lat=a.lat, lon=a.lon)
                if aircraft_location.distance(fix_location) < self.airport_fix_distance_threshold:
                    # if we are not checking fl, or if we don't have an exit coordination,
                    # append to list.   Otherwise, require FLs to match in order to append.
                    logger.debug(f"{callsign} is within lateral distance threshold to destination airport.")
                    if (not check_exit_fl) or (exit_coordination is None) or (a.fl == exit_coordination.fl):
                        logger.debug(f"Removing aircraft {callsign} from environment.")
                        aircraft_to_remove.append(callsign)
        return aircraft_to_remove

    def __events_csv_path(self) -> str:
        return os.path.join(SPRINGFIELD_DIR, "scenarios", self.scenario_name, "events.csv")

    def __event_handler_events_init(self, event_handler: EventHandler[TAircraft]):
        events_csv_path = self.__events_csv_path()
        df_events = pd.read_csv(events_csv_path)

        for row in df_events.itertuples():
            assert isinstance(row.callsign, str)
            event_datetime = datetime.fromisoformat(row.datetime)

            # process foreground events (incomm)
            if row.event_type == "incomm":
                callsign = row.callsign

                # there are a few cases where the aircraft is already in the sector at the start of the scenario even
                # though there is an incomm event. in this case, just continue on and warn
                sector_name = row.sector
                event_handler.add_incomm_event(event_datetime, callsign, sector_name=sector_name)

            # flag other event types
            else:
                logger.warning(
                    f"WARNING: Skipping unhandled event type {row.event_type}",
                    f"for {self.scenario_name} - {row.callsign}",
                )

        logger.debug(event_handler.incomm_df)

    def __coordinations_csv_path(self) -> str:
        return os.path.join(SPRINGFIELD_DIR, "scenarios", self.scenario_name, "coordination.csv")

    def __event_handler_coordination_init(
        self, event_handler: TEventHandler, aircraft_spawn_times: dict[str, datetime]
    ):
        coordination_path = self.__coordinations_csv_path()

        df_coordination = pd.read_csv(coordination_path)
        for row in df_coordination.itertuples():
            if (aircraft_spawn_time := aircraft_spawn_times.get(row.callsign)) is not None:
                event_handler.add_coordination_event(
                    aircraft_spawn_time - timedelta(seconds=1),
                    row.callsign,
                    row.from_sector,
                    row.to_sector,
                    row.fl,
                    row.fix,
                    row.direction,
                )

    def __flight_csv_path(self) -> str:
        return os.path.join(SPRINGFIELD_DIR, "scenarios", self.scenario_name, "flight.csv")

    def __event_handler_flights_init(self, event_handler: TEventHandler):
        """
        Loads the flights

        This function calls on :func:`~springield.SpringfieldScenarioManager.__event_handler_coordination_init` to load
        the coordinations of the loaded aircraft.

        Parameters
        ----------
        event_handler: EventHandler
            The event handler to load flights, and their respective coordinations, to.

        """
        scenario_csv_path = self.__flight_csv_path()
        df_scenarios = pd.read_csv(scenario_csv_path)
        df_scenarios = convert_string_to_lists(df_scenarios, ["expanded_route"])

        flight_spawn_times: dict[str, datetime] = {}

        # first, create all the add aircraft events for foreground,and, potentially, background aircraft
        for row in df_scenarios.itertuples():
            aircraft_spawn_time, aircraft = self._generate_aircraft_and_coordination_to_add(row)
            aircraft_spawn_time = datetime.fromtimestamp(aircraft_spawn_time, timezone.utc).replace(tzinfo=None)

            event_handler.add_aircraft(aircraft_spawn_time, aircraft)

            if (callsign := aircraft.callsign) is not None:
                flight_spawn_times[callsign] = aircraft_spawn_time

        self.__event_handler_coordination_init(event_handler, flight_spawn_times)

    def _generate_aircraft_and_coordination_to_add(self, row: pd.Series) -> tuple[float, Aircraft]:  # type: ignore
        """
        Generate an aircraft, its associated entry and exit coordinations and the time the aircraft is to
        be added to the environment, from a row of the scenarios dataframe.

        Parameters
        ----------
        row: pd.Series
            A row of the scenarios dataframe

        Returns
        -------
        tuple(float, Aircraft, tuple(Coordination, Coordination))
            3 element tuple. First element is the time that aircraft should be added to the scenario, second
            element is the aircraft to be added, the element is a tuple containing the entry and exit
            coordination.
        """
        # extract the route and apply any SIDs and STARs to it
        route_fixes = (
            row.expanded_route
        )  # , route_fix_types = apply_sid_or_star(row.route, row.route_type, self.df_sids, SID_TYPE)

        # TODO: apply stars to route (there are none in the data)
        # route_fixes, route_fix_types = apply_sid_or_star(route_fixes, route_fix_types, self.df_sids, STAR_TYPE)

        # initial check: the route is allowed to have one fix iif it does not spawn at that fix.
        if len(route_fixes) == 1 and row.offset_range == 0:
            msg = f"WARNING: Skipping aircraft event for {self.scenario_name} - {row.callsign} because it has only one"
            msg += f" fix in the route {route_fixes} and it spawns at that fix."
            logger.warning(msg)
            raise UserWarning(msg)

        # current: at this level, cleared: changing to this level, requested: want to go to this level by the xfl
        cleared_fl = row.cleared_fl
        current_fl = row.start_fl

        sector_name = "background" if row.sector_name == "BKGND" else row.sector_name

        # add in the departure and destination airport, if already not in the route
        # TODO: currently disabled. need to find out if this is needed
        if False:
            if route_fixes[0] != row.departure_airport:
                route_fixes.insert(0, row.departure_airport)
            if route_fixes[-1] != row.destination_airport:
                route_fixes.append(row.destination_airport)

        route = Route(route_fixes)
        flight_plan = FlightPlan(
            route,
            unexpanded_route=row.route,
            origin=row.departure_airport,
            dest=row.destination_airport,
            milcivil="C",
            requested_flight_level=row.requested_fl,
            intention_code=None,
            assigned_squawk=str(row.ssr_assigned),
        )

        initial_heading = row.initial_heading
        target_heading = row.target_heading
        on_route = row.on_route

        squawk = None if row.ssr_set == 0 else str(row.ssr_set)  # expected as a string

        # build the aircraft
        seed = getattr(row, "random_seed", None)
        aircraft = Aircraft(
            row.start_lat,
            row.start_lon,
            current_fl,
            initial_heading,
            flight_plan,
            row.callsign,
            current_sector=sector_name,
            aircraft_type=row.aircraft_type,
            ufid=row.callsign,
            squawk=squawk,
            wake_vortex=row.wake_vortex_category,
            selected_fl=row.cleared_fl,
            random_seed=int(seed) if seed is not None and not pd.isna(seed) else None,
        )

        # if the aircraft is on route, then set it as such and calculate and set the next fix index
        if on_route:
            # otherwise, the next fix is the offset fix
            next_fix_idx = route_fixes.index(row.offset_fix) if row.offset_fix in route_fixes else 0

            aircraft.on_route = True
            aircraft.next_fix_index = next_fix_idx

        else:
            aircraft.on_route = False  # this is mapped to aircraft.selected_instructions.on_route
            # by default, the cleared instruction has on route set to true, so we need to turn this off as well.
            aircraft.cleared_instructions.on_route = False

        if initial_heading != target_heading:
            aircraft.heading_changing_to = target_heading
            aircraft.selected_instructions.heading = target_heading
            aircraft.cleared_instructions.heading = target_heading

        # if the aircraft isn't at its cleared flight level, then update it
        if cleared_fl != current_fl:
            aircraft.cleared_fl = cleared_fl
            aircraft.selected_fl = cleared_fl

        # explicitly set the rate of turn to 3 degrees/sec
        aircraft.rate_of_turn = 3.0

        # lastly, parse the start time and create the event
        aircraft_start_time = datetime.fromisoformat(row.start_datetime).timestamp()

        return aircraft_start_time, aircraft

    @staticmethod
    def list_scenarios() -> list[str]:
        """
        Gives a list of all the scenarios available to load from the directory
        :attr:`~springfield.SpringfieldScenarioManager.scenario_directory`.

        Returns
        -------
        list[str]
            A list of scenario names, the selected of which is in turn passed to
            :func:`~springield.SpringfieldScenarioManager.__init__`
        """
        scenario_directory: str = SpringfieldScenarioManager.scenario_directory

        return sorted(
            [
                folder
                for folder in os.listdir(scenario_directory)
                if os.path.isdir(os.path.join(scenario_directory, folder))
            ]
        )

    @override
    def config(self) -> SpringfieldScenarioManagerConfig:
        return SpringfieldScenarioManagerConfig()

    def to_simulator(
        self,
        category: str | None = None,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """
        Create a Simulator instance for Springfield scenarios.

        Parameters
        ----------
        category : str | None, optional
            Category of the simulation. Default is None.
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
        simulated_sectors: list[str] | typing.Literal["ALL"], optional
            The sectors to be simulated. If "ALL", all sectors will be simulated. If a list, only the sectors names in
            the list will be simulated. Currently only applicable for real world scenarios. Defaults to "ALL".

        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        if log_filename is None:
            the_datetime_formatted = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            # Windows can't handle ':' character in file-paths
            sanitised_name = self.scenario_name.replace(":", "_")
            log_filename = f"{category}_{sanitised_name}_{the_datetime_formatted}"

        env_manager = self.create_env_manager(log_filename=log_filename, predictor=predictor)

        return typeof_simulator(
            scenario_manager=self,
            env_manager=env_manager,
            projection_centre=self.projection_centre,
            scenario_name=self.scenario_name,
            category=category,
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            log_filename=log_filename,
            predictor=predictor,
            simulated_sectors=simulated_sectors,
        )
