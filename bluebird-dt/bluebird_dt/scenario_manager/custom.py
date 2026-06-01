import typing
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_dt.airspace_generator.airspace_loader import AirspaceLoader
from bluebird_dt.core import Aircraft, Airspace, Pos3D, Route, WindField
from bluebird_dt.events.event_handler import EventHandler
from bluebird_dt.events.event_logger import EventLogger
from bluebird_dt.logger import logger
from bluebird_dt.manager.environment_manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility.scenario_manager_utils import create_aircraft_with_coordinations, laterally_offset_start_point


class CustomScenarioManagerConfig(BaseModel):
    """
    Configuration of a custom scenario manager
    """

    scenario_manager: typing.Literal["custom"] = Field(default="custom")


TAircraft = typing.TypeVar("TAircraft", bound=Aircraft)
TWindField = typing.TypeVar("TWindField", bound=WindField)
TForecastWindField = typing.TypeVar("TForecastWindField", bound=WindField)
TEnvironmentManager = typing.TypeVar("TEnvironmentManager", bound=EnvironmentManager[Aircraft, WindField, WindField])
TEventHandler = typing.TypeVar("TEventHandler", bound=EventHandler[Aircraft])
TEventLogger = typing.TypeVar("TEventLogger", bound=EventLogger)
TSimulator = typing.TypeVar("TSimulator", bound=Simulator)
TAirspaceLoader = typing.TypeVar("TAirspaceLoader", bound=AirspaceLoader)


class Custom(
    ScenarioManager[CustomScenarioManagerConfig],
    typing.Generic[TAircraft, TWindField, TForecastWindField, TEnvironmentManager, TEventLogger, TEventHandler],
):
    """
    Aircraft generator for simple custom scenarios:
    - configurable number of Aircraft and balance of
      climbers/descenders/overfliers
    - randomly selected entry and exit Coordinations
    - randomly generated speeds
    - ensures that no two Aircraft have the same entry coordination
      (same Fix, Flight Level and time)
    - ability to randomize the start position of aircraft within an entry fix
      through a stochastic sample of lateral distance from the entry fix.
    """

    projection_centre: tuple[float, float] | None = None
    num_aircraft: int
    airspace: Airspace
    routes: list[Route]
    sector_name: str | None
    balance: list[float] | None
    speed_range: list[float] | None
    time_entry_gap: float
    random_seed: int | None
    aircraft_on_route: bool
    lateral_offset: tuple[int, int] | None
    start_time: int
    vertical_buffer_distance: float | int
    lateral_buffer_distance: float | int
    typeof_environment_manager: type[TEnvironmentManager]
    typeof_event_handler: type[TEventHandler]
    typeof_aircraft: type[TAircraft]
    typeof_event_logger: type[TEventLogger]

    def __init__(
        self,
        num_aircraft: int,
        airspace: Airspace,
        routes: list[Route],
        sector_name: str | None = None,
        balance: tuple[float, float, float] | None = None,
        speed_range: tuple[float, float] | None = None,
        time_entry_gap: float = 5,
        random_seed: int | None = None,
        aircraft_on_route: bool = False,
        lateral_offset: tuple[float, float] | None = None,
        start_time: int = 0,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
        typeof_environment_manager: type[TEnvironmentManager] = EnvironmentManager,
        typeof_event_handler: type[TEventHandler] = EventHandler,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        typeof_aircraft: type[TAircraft] = Aircraft,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        num_aircraft: float
            Number of Aircraft to generate.
        airspace: Airspace
            The Airspace the Aircraft are flying through. The generator expects it to have a single Sector
            composed of a single Volume. This is true for the I,X,Y Airspaces.
        routes: list[Route]
            The available Routes in the Airspace (choose one at random for each Aircraft Route).
        sector_name: str | None
            The name of the sector being simulated.  If not specified, use first sector in the airspace.
        balance: list[float, float, float]
            Probabilities of any given Aircraft being one of climber/descender/overflier.
            The probabilities have to sum to 1 (Multinomial distribution parameter).
        speed_range: list[float, float]
            Optional range of [min,max] speeds from which to randomly generate Aircraft speed.
            If not provided, speed of all Aircraft is set to 400.
        aircraft_on_route: bool
            If True, aircraft will automatically follow their route.  If False, they will travel on
            constant heading unless they receive other instructions.  Default is False.
        time_entry_gap: float
            Optional amount of time in seconds that must be maintained between two
            Aircraft entry Coordinations if they are at the same Fix and FL.
        random_seed: int, optional
            Optionally set the seed for the random number generator, for reproducibility.
        lateral_offset: tuple[float, float] | None
            the range (low, high) from which to sample (uniform distribution)
            a lateral offset that is applied to randomize an aircraft start
            position (spawn point). Example sensible values are (0, 10)
        env_manager_class: type, optional
            if specified, use this class (maybe a subclass of BluebirdATC EventManager).
        start_time: int
            Start time in unix time (seconds)
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        typeof_environment_manager: type[TEnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[TAircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[TEventLogger], optional
            If we want to use a derived class for the Event Logger, specify here.
        typeof_event_handler: type[TEventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        """
        if balance is None:
            balance = (1 / 3, 1 / 3, 1 / 3)
        elif len(balance) != 3:
            raise ValueError("balance must be None or a tuple of 3 values")
        elif sum(balance) != 1.0:
            # scale probabilities so they sum to 1
            balance = tuple([b / sum(balance) for b in balance])
        self.balance = balance
        if speed_range is None:
            speed_range = [400.0, 400.0]
        elif len(speed_range) != 2:
            raise ValueError("speed_range must be None or a tuple of 2 values")
        self.speed_range = speed_range
        if num_aircraft < 0:
            raise ValueError("Number of Aircraft must not be negative.")
        self.num_aircraft = num_aircraft
        self.airspace = airspace
        self.routes = routes
        self.sector_name = sector_name if sector_name else next(iter(airspace.sectors.keys()))
        if time_entry_gap < 0:
            raise ValueError("Time entry gap cannot be negative.")
        self.time_entry_gap = time_entry_gap
        self.aircraft_on_route = aircraft_on_route
        self.lateral_offset = lateral_offset
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.rng = np.random.default_rng(random_seed)
        self.typeof_environment_manager = typeof_environment_manager
        self.typeof_event_handler = typeof_event_handler
        self.typeof_aircraft = typeof_aircraft
        self.typeof_event_logger = typeof_event_logger
        self.event_handler_ignore_flags = typeof_event_handler.IgnoreFlags()
        # If users want to fully customize the scenario, specifying every aircraft,
        # we keep a list of custom aircraft, with their coordinations, and their start times.
        # i.e. [(start_time, (Aircraft, Coordination, Coordination)), ...]
        self.user_added_aircraft: list[tuple[float, tuple[Aircraft, Coordination, Coordination]]] = []

    @override
    def create_event_handler(self) -> TEventHandler:
        """
        Generate event_handler for the given Airspace.

        Returns
        ----------
        TEventHandler
            The EventHandler specifying Aircraft with unique string identifiers (callsigns) to fly
            through the Airspace and when to add them to the Environment.
        """
        # split aircraft into climbers, descenders and overfliers
        climbers, descenders, overfliers = self.rng.multinomial(self.num_aircraft, self.balance)
        journey_type = ["climb"] * climbers + ["descend"] * descenders + ["overfly"] * overfliers
        self.rng.shuffle(journey_type)
        # sector.get_bounds returns two arrays of [lat, lon, fl] - we want the last element of each.
        min_fl, max_fl = [b[2] for b in self.airspace.sectors[self.sector_name].get_bounds()]

        allowed_FLs = np.arange(min_fl, max_fl + 10, 10, dtype="float")

        # keep track of start fixes and entry coordinations to avoid clashes
        entries = defaultdict(lambda: defaultdict(list))

        # create empty event handler
        event_handler = self.typeof_event_handler(ignore=self.event_handler_ignore_flags)

        for i in range(self.num_aircraft):
            route = self.rng.choice(self.routes)

            journey = journey_type[i]
            if journey == "overfly":
                entry_fl = exit_fl = self.rng.choice(allowed_FLs)

            elif journey == "climb":
                # entry_fl < exit_fl
                # choose start FL so that Aircraft has somewhere to climb to
                entry_fl = self.rng.choice(np.arange(allowed_FLs[0], allowed_FLs[-1], 10))
                # choose exit to be higher than entry_fl AND within exit FL limits
                exit_fl = self.rng.choice(np.arange(entry_fl + 10, allowed_FLs[-1] + 10, 10))

            elif journey == "descend":
                # entry_fl > exit_fl
                # choose start FL so that Aircraft has somewhere to descend to
                entry_fl = self.rng.choice(np.arange(allowed_FLs[0] + 10, allowed_FLs[-1] + 10, 10))
                # choose exit to be lower than entry_fl AND within exit FL limits
                exit_fl = self.rng.choice(np.arange(allowed_FLs[0], entry_fl, 10))

            else:
                raise ValueError(f"Invalid journey type: {journey}")

            callsign = f"AIR{i}"
            speed = self.rng.uniform(self.speed_range[0], self.speed_range[1])
            start_fix_name = route.filed[0]
            start_fix = self.airspace.fixes.places[start_fix_name]

            if self.lateral_offset:
                # compute the start position of the aircraft
                # offset by a stochastic lateral distance.
                updated_pos = laterally_offset_start_point(
                    self.airspace,
                    route,
                    self.lateral_offset,
                    self.rng,
                )
                pos = updated_pos.pos3d(entry_fl)
            else:
                # no lateral offset
                pos = start_fix.pos3d(entry_fl)
            # Even if we are laterally offset, heading should be parallel to the route,
            # so take heading from start_fix to next_fix rather than aircraft pos to next_fix.
            heading = start_fix.bearing_to(self.airspace.fixes.places[route.filed[1]])

            aircraft, coordination_entry, coordination_exit = create_aircraft_with_coordinations(
                callsign=callsign,
                pos=pos,
                heading=heading,
                speed=speed,
                route=route,
                sector_name=self.sector_name,
                entry_fl=entry_fl,
                exit_fl=exit_fl,
                on_route=self.aircraft_on_route,
                airspace=self.airspace,
                typeof_aircraft=self.typeof_aircraft,
            )

            # check entry coordination and make sure it doesn't clash with another Aircraft entry coordination
            start_t = (
                self.start_time
            )  # start time is set by default to 0.0 seconds, which corresponds to 1970-01-01T00:00:00 UTC
            if start_t in entries[start_fix_name][entry_fl]:
                start_t = max(entries[start_fix_name][entry_fl]) + self.time_entry_gap
            entries[start_fix_name][entry_fl].append(start_t)

            # Add aircaft to event handler
            event_start_time = pd.to_datetime(start_t, unit="s")
            event_handler.add_aircraft(event_start_time, aircraft)

            # ensure coordinations are in the environment before the aircraft
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coordination_exit)
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coordination_entry)

        return event_handler

    def add_aircraft_with_coordinations(
        self,
        aircraft_start_time: float,
        callsign: str,
        pos: Pos3D,
        heading: float,
        speed: float,
        route: Route,
        entry_fl: float,
        exit_fl: float,
        on_route: bool | None = None,
        sector_name: str | None = None,
    ):
        """
        Allows the user to fully customise the scenario by adding fully specified aircraft to the scenario.
        Aircraft specified here will be appended to a list, which is then used to populate the EventHandler in
        the create_env_manager method.

        Parameters
        ----------
        aircraft_start_time: float
            time in seconds (from UTC timestamp 0) for aircraft to be added to scenario.
        callsign: str
            callsign of the new aircraft.
        pos: Pos3D
            spawn position of the new aircraft.
        heading: float
            initial heading of the new aircraft.
        speed: float
            initial speed of the new aircraft.
        route: Route
            filed route to be followed by the aircraft.
        entry_fl: float
            flight level for the entry coordination
        exit_fl: float
            flight level for the exit coordination
        on_route: bool | None
            if True, aircraft will automatically follow the route.
            if False, aircraft will continue on current heading until instructed otherwise.
            if not specified, use the instance's `self.aircraft_on_route` value.
        sector_name: str | None
            name of the sector for the coordinations.  If not specified, use the instance's
            `self.sector_name` value.
        """
        if on_route is None:
            on_route = self.aircraft_on_route

        if not sector_name:
            sector_name = self.sector_name

        self.user_added_aircraft.append(
            (
                aircraft_start_time,
                create_aircraft_with_coordinations(
                    callsign=callsign,
                    pos=pos,
                    heading=heading,
                    speed=speed,
                    route=route,
                    sector_name=sector_name,
                    entry_fl=entry_fl,
                    exit_fl=exit_fl,
                    on_route=on_route,
                    airspace=self.airspace,
                ),
            )
        )

    def create_env_manager(
        self,
        predictor: Predictor | None = None,
        log_filename: str | None = None,
        event_handler: TEventHandler | None = None,
    ) -> TEnvironmentManager:
        """
        Create event_manager for the given Airspace.
        Will also add any custom user-specified aircraft to the EventHandler.

        Parameters
        ----------
        event_handler: EventHandler, optional
            If provided, use this to initialize the EnvironmentManager.
            Default is not to provide this, so new EventHandler will be created via self.create_event_handler().
        predictor: Predictor, optional
            Aircraft Trajectory prediction used to evolve Aircraft. If None, then SimplePredictor will be created.
        log_filename: str, optional
            Name of file logs will be saved to. If None, defaults to datetime logger created.

        Returns
        ----------
        TEnvironmentManager
            Environment Manager for Tactical scenario
        """

        logger.info(
            f"""

        ===================================================================
        Creating Custom Scenario with {self.num_aircraft} aircraft.
        """
        )

        # create SimplePredictor if no Predictor passed
        if predictor is None:
            predictor = SimplePredictor(1.0, 2.0)

        if not event_handler:
            # create event handler from the events list
            event_handler = self.create_event_handler()

        # add any user-specified aircraft
        for aircraft_start_time, aircraft_with_coordinations in self.user_added_aircraft:
            aircraft, coord_entry, coord_exit = aircraft_with_coordinations
            # Add aircaft to event handler
            event_start_time = pd.to_datetime(aircraft_start_time, unit="s")
            event_handler.add_aircraft(event_start_time, aircraft)

            # ensure coordinations are in the environment before the aircraft
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coord_exit)
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coord_entry)

        em = self.typeof_environment_manager(
            airspace=self.airspace,
            event_handler=event_handler,
            predictor=predictor,
            time=self.start_time,
            penumbra_fl=int(self.vertical_buffer_distance),
            penumbra_lat=self.lateral_buffer_distance,
            log_filename=log_filename,
        )

        # set the visibility flag of fixes to True only if they are in the penumbra
        em.set_local_fixes_visibility()

        em.initialise_env_with_event_handler()

        return em

    @override
    def config(self) -> CustomScenarioManagerConfig:
        return CustomScenarioManagerConfig()

    def to_simulator(
        self,
        scenario_name: str | None = None,
        category: str | None = None,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
        env_manager: TEnvironmentManager | None = None,
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> Simulator:
        """
        Create a Simulator instance for Tactical scenarios.

        Parameters
        ----------
        scenario_name : str | None, optional
            Name of the scenario. Default is None.
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
        env_manager: TEnvironmentManager | None
            If given, use this EnvironmentManager as the Simulator's `manager`.
            Default is None, in which case a new EnvironmentManager will be created.
        typeof_simulator: type[TSimulator]
            In case we are returning a derived class from BluebirdATC's Simulator
        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        if log_filename is None:
            the_datetime_formatted = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            # Windows can't handle ':' character in file-paths
            sanitised_name = scenario_name.replace(":", "_") if scenario_name is not None else None
            log_filename = f"{category}_{sanitised_name}_{the_datetime_formatted}"

        if not env_manager:
            env_manager = self.create_env_manager(log_filename=log_filename, predictor=predictor)

        return typeof_simulator(
            scenario_manager=self,
            env_manager=env_manager,
            projection_centre=self.projection_centre,
            scenario_name=scenario_name,
            category="Custom",
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            log_filename=log_filename,
            predictor=predictor,
            simulated_sectors=simulated_sectors,
        )

    @classmethod
    def setup(
        cls,
        scenario_name: str,
        num_aircraft: int = 2,
        balance: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        speed_range: tuple[float, float] | None = None,
        aircraft_on_route: bool = False,
        lateral_offset: tuple[float, float] | None = None,
        time_entry_gap: float = 5.0,
        start_time: float = 0.0,
        random_seed: int | None = None,
        use_wind: bool = True,
        use_forecast: bool = True,
        autosave: bool = True,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        predictor: Predictor | None = None,
        simulated_sectors: list[str] | typing.Literal["ALL"] = "ALL",
        typeof_environment_manager: type[TEnvironmentManager] = EnvironmentManager,
        typeof_event_handler: type[TEventHandler] = EventHandler,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """Setup artificial scenarios based on scenario name.

        Parameters
        ----------
        scenario_name: str
            The scenario name
        random_seed: int
            If specified, set the random seed for the generator
        num_aircraft: int
            Number of aircraft to spawn
        balance: tuple[float, float, float]
            Fraction of overfliers, climbers, descenders.
        speed_range: tuple[float, float]
            Optional, if not set, aircraft speeds are set between 350 and 450 knots.
        aircraft_on_route: bool
            If True, aircraft will follow route by default, in False, they will travel
            at constant heading until instructed otherwise.  Default is False.
        lateral_offset: tuple[float, float]
            min, max values for laterally offsetting start position from route centreline.
        time_entry_gap: float
            Time between spawning aircraft
        start_time: float
            UTC time to start simulation.
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
        typeof_environment_manager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the event logger, specify here.
        typeof_event_handler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        typeof_simulator: type[Simulator], optional
            If we want to use a derived class for the Simulator, specify here.
        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        airspace, routes, sector_name = AirspaceLoader.load(scenario_name)
        return cls(
            airspace=airspace,
            routes=routes,
            sector_name=sector_name,
            num_aircraft=num_aircraft,
            balance=balance,
            speed_range=speed_range,
            aircraft_on_route=aircraft_on_route,
            lateral_offset=lateral_offset,
            time_entry_gap=time_entry_gap,
            start_time=start_time,
            random_seed=random_seed,
            typeof_aircraft=typeof_aircraft,
            typeof_event_logger=typeof_event_logger,
            typeof_event_handler=typeof_event_handler,
            typeof_environment_manager=typeof_environment_manager,
        ).to_simulator(
            log_filename=log_filename,
            predictor=predictor,
            category="Custom",
            scenario_name=scenario_name,
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            simulated_sectors=simulated_sectors,
            typeof_simulator=typeof_simulator,
        )
