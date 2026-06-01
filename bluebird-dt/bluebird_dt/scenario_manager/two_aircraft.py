import typing
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_dt.airspace_generator.airspace_loader import AirspaceLoader
from bluebird_dt.core import Aircraft, Airspace, Route, WindField
from bluebird_dt.events import EventHandler, EventLogger
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility.scenario_manager_utils import create_aircraft_with_coordinations


class TwoAircraftScenarioManagerConfig(BaseModel):
    """
    Configuration of the two aircraft scenario manager
    """

    scenario_manager: typing.Literal["two_aircraft"] = Field(default="two_aircraft")


TAircraft = typing.TypeVar("TAircraft", bound=Aircraft)
TWindField = typing.TypeVar("TWindField", bound=WindField)
TForecastWindField = typing.TypeVar("TForecastWindField", bound=WindField)
TEnvironmentManager = typing.TypeVar("TEnvironmentManager", bound=EnvironmentManager[Aircraft, WindField, WindField])
TEventHandler = typing.TypeVar("TEventHandler", bound=EventHandler[Aircraft])
TEventLogger = typing.TypeVar("TEventLogger", bound=EventLogger)
TSimulator = typing.TypeVar("TSimulator", bound=Simulator)
TAirspaceLoader = typing.TypeVar("TAirspaceLoader", bound=AirspaceLoader)


class TwoAircraft(
    ScenarioManager[TwoAircraftScenarioManagerConfig],
    typing.Generic[TAircraft, TWindField, TForecastWindField, TEnvironmentManager, TEventLogger, TEventHandler],
):
    """
    Two Aircraft travelling head on:
        - start at opposite ends of the same route
        - first Aircraft is an overflier
        - second Aircraft is an overflier, climber or descender
        - the Aircraft either travel on the same FL or have crossing entry/exit FL coordinations
    """

    projection_centre: tuple[float, float] | None = None
    event_handler_ignore_flags: EventHandler.IgnoreFlags
    airspace: Airspace
    routes: list[Route]
    sector_name: str
    total_time: float
    speed_range: tuple[float, float] | None
    start_time: int
    vertical_buffer_distance: float | int
    lateral_buffer_distance: float | int
    scenario_type: typing.Literal["random", "overflier", "climber", "descender"]
    random_seed: int | None
    typeof_environment_manager: type[TEnvironmentManager]
    typeof_event_handler: type[TEventHandler]
    typeof_aircraft: type[TAircraft]
    typeof_event_logger: type[TEventLogger]

    def __init__(
        self,
        airspace: Airspace,
        routes: list[Route],
        sector_name: str | None = None,
        total_time: float = 100.0,
        speed_range: tuple[float, float] | None = None,
        scenario_type: typing.Literal["random", "overflier", "climber", "descender"] = "random",
        random_seed: int | None = None,
        typeof_environment_manager: type[TEnvironmentManager] = EnvironmentManager,
        typeof_event_handler: type[TEventHandler] = EventHandler,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        start_time: int = 0,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        airspace: Airspace
            The airspace to be used in the environment
        routes: list[Route]
            The available Routes in the Airspace
        sector_name: str | None
            Name of Sector within the Airspace to be used for Coordinations.  If not specified,
            use the first sector in the airspace.
        total_time: float
            The total time to travel the Route for both Aircraft (in seconds). If speed_range is not provided, speed of
            both Aircraft is chosen such that the FlightPlan coordinations can be satisfied.
        speed_range: tuple[float, float] | None
            Optional range of [min,max] speeds from which to randomly choose Aircraft speed. If not provided,
            speed of both Aircraft is set based on the total_time parameter. If provided, total_time is not used.
        scenario_type: typing.Literal
            One of "random", "overflier", "climber" or "descender". If "random" - choose one of
            ["overflier", "climber", "descender"].
            Describes the behaviour of the second Aircraft.
        random_seed: int | None
            If specified, set the seed for the random number generator for reproducibility.
        env_manager_class: type, optional
            if specified, use this class (maybe a subclass of BluebirdATC EventManager).
        start_time: int
            Start time of scenario, in unix time (seconds)
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        typeof_environment_manager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the event logger, specify here.
        typeof_event_handler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        """

        if (total_time is not None) and total_time <= 0.0:
            raise ValueError("Total time must be positive.")

        self.total_time = total_time
        self.airspace = airspace
        self.sector_name = sector_name if sector_name else next(iter(airspace.sectors.keys()))
        self.routes = routes
        if speed_range is not None and len(speed_range) != 2:
            raise ValueError("speed_range must be None or a tuple of 2 values")
        self.speed_range = speed_range
        if scenario_type not in ["random", "overflier", "climber", "descender"]:
            raise ValueError(f"scenario_type must be one of {['random', 'overflier', 'climber', 'descender']}")
        self.scenario_type = scenario_type
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.typeof_environment_manager = typeof_environment_manager
        self.typeof_event_handler = typeof_event_handler
        self.typeof_aircraft = typeof_aircraft
        self.typeof_event_logger = typeof_event_logger
        self.event_handler_ignore_flags = typeof_event_handler.IgnoreFlags()
        self.rng = np.random.default_rng(random_seed)

    def get_overflier_coordination_FLs(self, allowed_FLs: list[float], aircraft_scenario: str) -> tuple[float, float]:
        """
        Generate FL for the entry and exit Coordination of overflier Aircraft.

        Parameters
        ----------
        allowed_FLs: list[float]
            List of FLs one can fly through in this Airspace.
        aircraft_scenario: str
            One of "overflier", "climber" or "descender". Describes the behaviour of the second Aircraft.
            Note that the first Aircraft is always an "overflier".

        Returns
        ----------
        list[float]
            FL of the [entry, exit] Coordination of overflier Aircraft.
        """

        coordination_FL = self.rng.choice(allowed_FLs)

        # make sure climber/descender can start below/above overflier
        # - leave lowest/highest FL band clear
        if aircraft_scenario == "descender":
            coordination_FL = min(coordination_FL, allowed_FLs[-2])
        elif aircraft_scenario == "climber":
            coordination_FL = max(coordination_FL, allowed_FLs[1])

        return coordination_FL, coordination_FL

    def get_descender_coordination_FLs(self, overflier_fl: float, allowed_FLs: list[float]) -> tuple[float, float]:
        """
        Generate FL for the entry and exit Coordination of descender Aircraft.

        Parameters
        ----------
        overflier_fl: float
            The FL the overflier is flying at
        allowed_FLs: list[float]
            List of FLs one can fly through in this Airspace.

        Returns
        ----------
        list[float]
            FL of the [entry, exit] Coordination for descender Aircraft.
        """

        # start above the overflier (up to the highest allowed entry FL limit)
        start_fl = self.rng.choice(np.arange(overflier_fl + 10, allowed_FLs[-1] + 10, 10, dtype="float"))

        # exit at or below the overflier FL to ensure paths cross
        exit_fl = self.rng.choice(np.arange(allowed_FLs[0], overflier_fl + 10, 10, dtype="float"))

        return start_fl, exit_fl

    def get_climber_coordination_FLs(self, overflier_fl: float, allowed_FLs: list[float]) -> tuple[float, float]:
        """
        Generate FL of the entry and exit Coordination of climber Aircraft.

        Parameters
        ----------
        overflier_fl: float
            The FL the overflier is flying at
        allowed_FLs: list[float]
            List of FLs one can fly through in this Airspace.

        Returns
        ----------
        list[float]
            FL of the [entry, exit] Coordination of climber Aircraft.
        """

        # start below the overflier (down to the lowest allowed entry FL)
        start_fl = self.rng.choice(np.arange(allowed_FLs[0], overflier_fl, 10, dtype="float"))

        # climb to FL at or above overflier FL to ensure paths cross
        exit_fl = self.rng.choice(np.arange(overflier_fl, allowed_FLs[-1] + 10, 10, dtype="float"))

        return start_fl, exit_fl

    @override
    def create_event_handler(self) -> TEventHandler:
        """
        Generate two Aircraft for the given Airspace.

        Returns
        ----------
        List[AddAircraftEvent]
            A list of AddAircraftEvents specifying Aircraft with unique string identifiers (callsigns) to fly
            through the Airspace and when to add them to the Environment.
        """

        # set or randomly select scenario type
        if self.scenario_type == "random":
            aircraft_scenario = self.rng.choice(["overflier", "climber", "descender"])
        else:
            aircraft_scenario = self.scenario_type

        route_fwd = self.rng.choice(self.routes)
        route_rev = Route(route_fwd.filed[::-1])
        routes = [route_fwd, route_rev]
        volume = self.airspace.sectors[self.sector_name].volumes[0]
        allowed_FLs = [float(x) for x in np.arange(volume.min_fl, volume.max_fl + 10, 10)]

        # randomly generate entry/exit coordinations for first aircraft
        coordinations_fwd = self.get_overflier_coordination_FLs(allowed_FLs, aircraft_scenario)
        overflier_fl = coordinations_fwd[0]

        # generate entry/exit coordinations for second aircraft
        if aircraft_scenario == "overflier":
            coordinations_rev = coordinations_fwd
        elif aircraft_scenario == "climber":
            coordinations_rev = self.get_climber_coordination_FLs(overflier_fl, allowed_FLs)
        elif aircraft_scenario == "descender":
            coordinations_rev = self.get_descender_coordination_FLs(overflier_fl, allowed_FLs)
        coordinations = [coordinations_fwd, coordinations_rev]

        # create empty event handler
        event_handler = self.typeof_event_handler(ignore=self.event_handler_ignore_flags)

        for i in range(2):
            callsign = f"AIR{i}"

            # speed is in knots i.e., nmi per hour
            if self.speed_range is None:
                speed = route_fwd.length(self.airspace.fixes) / (self.total_time / 3600.0)
            else:
                speed = self.rng.uniform(self.speed_range[0], self.speed_range[1])

            route = routes[i]
            entry_fl, exit_fl = coordinations[i]

            fix1 = self.airspace.fixes.places[route.filed[0]]
            heading = fix1.bearing_to(self.airspace.fixes.places[route.filed[1]])

            the_datetime = datetime(1970, 1, 1)

            pos = fix1.pos3d(entry_fl)

            aircraft, coordination_entry, coordination_exit = create_aircraft_with_coordinations(
                callsign=callsign,
                pos=pos,
                heading=heading,
                speed=speed,
                route=route,
                sector_name=self.sector_name,
                entry_fl=entry_fl,
                exit_fl=exit_fl,
                typeof_aircraft=self.typeof_aircraft,
            )

            event_handler.add_aircraft(the_datetime, aircraft)
            event_handler.add_coordination(the_datetime, coordination_exit)
            event_handler.add_coordination(the_datetime, coordination_entry)

        return event_handler

    def create_env_manager(
        self, log_filename: str | None = None, predictor: Predictor | None = None
    ) -> TEnvironmentManager:
        """
        Create event_manager for the given Airspace.

        Parameters
        ----------
        predictor: Predictor, optional
            Aircraft Trajectory prediction used to evolve Aircraft. If None, then SimplePredictor will be created.
            Distance to expand airspace lateral boundary by - UoM: NMI
        log_filename: str or None
            Name of file logs will be saved to. If None, defaults to datetime logger created.

        Returns
        ----------
        EnvironmentManager
            EnvironmentManager for Two Aircraft scenario
        """
        logger.info(
            """
===================================================================
Creating TwoAircraft Scenario
===================================================================
        """
        )

        # create SimplePredictor if no Predictor passed
        if predictor is None:
            predictor = SimplePredictor(1.0, 2.0)

        # create event handler from the events list
        event_handler = self.create_event_handler()

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
    def config(self) -> TwoAircraftScenarioManagerConfig:
        return TwoAircraftScenarioManagerConfig()

    @classmethod
    def setup(
        cls,
        scenario_name: str,
        start_time: int = 0,
        total_time: float = 100.0,
        speed_range: tuple[float, float] | None = None,
        scenario_type: typing.Literal["random", "overflier", "climber", "descender"] = "random",
        random_seed: int | None = None,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
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
        total_time: float
            Number of seconds that the scenario should run for
        start_time: int
            Start time of scenario, in unix time (seconds)
        speed_range: tuple[float, float] | None
            If specified, randomly choose speeds from within the range for the aircraft.
        scenario_type: typing.Literal["random","overflier", "climber", "descender"]
            Describes the behaviour of the second aircraft in the scenario. Default is "random".
        random_seed: int | None
            If specified, set the seed for the random generator, for reproducibility.
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
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
        typeof_environmentmanager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the event logger, specify here.
        typeof_event_handler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        typeof_simulator: type[Simulator], optional
            If we want to create a derived Simulator class, specify here.
        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        airspace, routes, sector_name = AirspaceLoader.load(scenario_name)

        # set up the simulator for "climber" scenario using TwoAircraft scenario manager
        sim = cls(
            total_time=total_time,
            start_time=start_time,
            speed_range=speed_range,
            scenario_type=scenario_type,
            random_seed=random_seed,
            airspace=airspace,
            routes=routes,
            sector_name=sector_name,
            vertical_buffer_distance=vertical_buffer_distance,
            lateral_buffer_distance=lateral_buffer_distance,
            typeof_aircraft=typeof_aircraft,
            typeof_event_logger=typeof_event_logger,
            typeof_event_handler=typeof_event_handler,
            typeof_environment_manager=typeof_environment_manager,
        ).to_simulator(
            log_filename=log_filename,
            predictor=predictor,
            category="Two Aircraft",
            scenario_name=scenario_name,
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            simulated_sectors=simulated_sectors,
            typeof_simulator=typeof_simulator,
        )

        # if needed, fast-forward to the first aircraft entry time, ensuring that it is
        # a multiple of the evolve time-step
        first_entry_time = sim.manager.event_handler.radar_df.index.min().replace(tzinfo=timezone.utc).timestamp()

        time_step = 6.0

        # if the first time is a multiple of the time step, evolve one extra step.
        # note that 0 % anything == 0 (except 0!), so this accounts for the case where the first entry time is 0
        if first_entry_time % time_step == 0:
            evolve_time = first_entry_time + time_step

        # otherwise, evolve to the smallest multiple of time_step that is higher than the entry time
        else:
            evolve_time = ((first_entry_time // time_step) + 1) * time_step

        sim.manager.evolve(evolve_time)

        return sim

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
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """
        Create a Simulator instance for Two Aircraft scenarios.

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

        env_manager = self.create_env_manager(log_filename=log_filename, predictor=predictor)

        return typeof_simulator(
            scenario_manager=self,
            env_manager=env_manager,
            projection_centre=self.projection_centre,
            scenario_name=scenario_name,
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
