import typing
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
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


class RegularScenarioManagerConfig(BaseModel):
    """
    Configuration of a regular scenario manager.
    """

    total_time: float
    number_of_aircraft: int
    scenario_manager: typing.Literal["regular"] = Field(default="regular")


TAircraft = typing.TypeVar("TAircraft", bound=Aircraft)
TWindField = typing.TypeVar("TWindField", bound=WindField)
TForecastWindField = typing.TypeVar("TForecastWindField", bound=WindField)
TEnvironmentManager = typing.TypeVar("TEnvironmentManager", bound=EnvironmentManager[Aircraft, WindField, WindField])
TEventHandler = typing.TypeVar("TEventHandler", bound=EventHandler[Aircraft])
TEventLogger = typing.TypeVar("TEventLogger", bound=EventLogger)
TSimulator = typing.TypeVar("TSimulator", bound=Simulator)
TAirspaceLoader = typing.TypeVar("TAirspaceLoader", bound=AirspaceLoader)


class Regular(ScenarioManager[RegularScenarioManagerConfig]):
    """
    Quasi-regularly spaced Aircraft emitted from Route starts.
    """

    projection_centre: tuple[float, float] | None = None
    event_handler_ignore_flags: typing.ClassVar[EventHandler.IgnoreFlags]
    total_time: float
    num_aircraft: int
    airspace: Airspace
    routes: list[Route]
    sector_name: str | None
    start_time: float
    random_seed: int | None
    vertical_buffer_distance: int | float
    lateral_buffer_distance: int | float
    typeof_environment_manager: type[TEnvironmentManager]
    typeof_event_handler: type[TEventHandler]
    typeof_aircraft: type[TAircraft]
    typeof_eventlogger: type[TEventLogger]

    def __init__(
        self,
        total_time: float,
        num_aircraft: int,
        airspace: Airspace,
        routes: list[Route],
        sector_name: str | None = None,
        start_time: float = 0,
        random_seed: int | None = None,
        vertical_buffer_distance: int | float = 500,
        lateral_buffer_distance: int | float = 20,
        typeof_environment_manager: type[TEnvironmentManager] = EnvironmentManager,
        typeof_event_handler: type[TEventHandler] = EventHandler,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        total_time: float
            Time period [sec] within which all Aircraft enter the Airspace.
            The Aircraft entry times are quasi-regularly spaced out within this interval.
        num_aircraft: float
            Number of Aircraft to generate.
        airspace: Airspace
            The Airspace the Aircraft are flying through. The generator expects it to have a single Sector
            composed of a single Volume. This is true for the I,X,Y Airspaces.
        routes: list[Route]
            The available Routes in the Airspace (choose one at random for each Aircraft Route).
        sector_name: str | None,
            The sector name to be used in Coordinations.  If not specified, use the first sector in airspace.
        start_time: int
            Start time of scenario, in unix time (seconds)
        random_seed: int | None
            If given, set the seed for the random number generator, for reproducibility.
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        """

        if total_time <= 0.0:
            raise ValueError("Total time must be positive.")

        if num_aircraft <= 0:
            raise ValueError("Number of Aircraft must be positive.")

        self.total_time = total_time
        self.num_aircraft = num_aircraft
        self.airspace = airspace
        self.routes = routes
        self.sector_name = sector_name if sector_name else next(iter(airspace.sectors.keys()))
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.typeof_environment_manager = typeof_environment_manager
        self.typeof_event_handler = typeof_event_handler
        self.typeof_event_logger = typeof_event_logger
        self.typeof_aircraft = typeof_aircraft
        self.event_handler_ignore_flags = typeof_event_handler.IgnoreFlags()
        self.rng = np.random.default_rng(random_seed)

    @override
    def create_event_handler(self) -> EventHandler:
        """
        Generate EventHandler for the given Airspace.

        Returns
        ----------
        EventHandler
        """
        # create empty event handler
        event_handler = self.typeof_event_handler(ignore=self.event_handler_ignore_flags)

        volume = self.airspace.sectors[self.sector_name].volumes[0]
        allowed_FLs = np.arange(volume.min_fl, volume.max_fl + 10, 10, dtype="float")

        # Create start times for all Aircraft ensuring that the Aircraft starts
        # are quasi-regularly spaced between start of scenario and self.total_time.
        start_times = self.rng.uniform(low=0, high=1, size=self.num_aircraft)
        total = sum(start_times)
        t = -start_times[0] * 0.5
        for i in range(len(start_times)):
            t += start_times[i]
            start_times[i] = (t / total) * self.total_time

        for i, start_t in enumerate(start_times):
            flight_time = 1800.0  # in seconds
            route = self.rng.choice(self.routes)
            speed = route.length(self.airspace.fixes) / (flight_time / 3600.0)

            # entry/exit flight level should be within the Airspace limits
            entry_fl = self.rng.choice(allowed_FLs)
            exit_fl = self.rng.choice(allowed_FLs)

            callsign = f"AIR{i}"
            pos = self.airspace.fixes.places[route.filed[0]].pos3d(entry_fl)
            heading = self.airspace.fixes.places[route.filed[0]].bearing_to(self.airspace.fixes.places[route.filed[1]])

            aircraft, coordination_entry, coordination_exit = create_aircraft_with_coordinations(
                callsign=callsign,
                pos=pos,
                heading=heading,
                speed=speed,
                route=route,
                sector_name=self.sector_name,
                entry_fl=entry_fl,
                exit_fl=exit_fl,
                on_route=False,
                typeof_aircraft=self.typeof_aircraft,
            )

            start_time = pd.to_datetime(start_t, unit="s")

            event_handler.add_aircraft(start_time, aircraft)

            # ensure coordinations are in the environment before the aircraft
            event_handler.add_coordination(start_time - timedelta(seconds=1), coordination_exit)
            event_handler.add_coordination(start_time - timedelta(seconds=1), coordination_entry)

        return event_handler

    def create_env_manager(
        self,
        predictor: Predictor | None = None,
        log_filename: str | None = None,
    ) -> TEnvironmentManager:
        """
        Create event_manager for the given Airspace.

        Parameters
        ----------
        predictor: Predictor, optional
            Aircraft Trajectory prediction used to evolve Aircraft. If None, then SimplePredictor will be created.
        log_filename: str or None
            Name of file logs will be saved to. If None, defaults to datetime logger created.

        Returns
        ----------
        EnvironmentManager
            TEnvironmentManager for Regular scenario
        """

        logger.info(
            f"""
===================================================================
Creating Regular Scenario with {self.num_aircraft} aircraft.
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
    def config(self) -> RegularScenarioManagerConfig:
        return RegularScenarioManagerConfig(total_time=self.total_time, number_of_aircraft=self.num_aircraft)

    @classmethod
    def setup(
        cls,
        scenario_name: str,
        total_time: float,
        num_aircraft: int,
        random_seed: int | None = None,
        vertical_buffer_distance: int | float = 500,
        lateral_buffer_distance: int | float = 20,
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
        typeof_event_logger: type[TEventLogger] = EventLogger,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """Setup artificial scenarios based on scenario name.

        Parameters
        ----------
        scenario_name: str
            The scenario name
        scenario_type: typing.Literal["random","overflier", "climber", "descender"]
            Describes the behaviour of the second aircraft in the scenario.
        total_time: float
            The total time in seconds for the scenario to run
        num_aircraft: int
            The total number of aircraft that will be generated, evenly spaced throughout total_time.
        random_seed: int | None
            Optionally set the seed for the random number generator.
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
        typeof_environment_manager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_handler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the Event Logger, specify here.
        typeof_simulator: type[Simulator], optional
            If we want to create a derived Simulator class, specify here.
        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        airspace, routes, sector_name = AirspaceLoader.load(scenario_name)

        sim = cls(
            airspace=airspace,
            routes=routes,
            sector_name=sector_name,
            total_time=total_time,
            num_aircraft=num_aircraft,
            random_seed=random_seed,
            vertical_buffer_distance=vertical_buffer_distance,
            lateral_buffer_distance=lateral_buffer_distance,
            typeof_aircraft=typeof_aircraft,
            typeof_event_handler=typeof_event_handler,
            typeof_event_logger=typeof_event_logger,
            typeof_environment_manager=typeof_environment_manager,
        ).to_simulator(
            log_filename=log_filename,
            predictor=predictor,
            category="Regular",
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
    ) -> Simulator:
        """
        Create a Simulator instance for Regular scenarios.

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
        typeof_simulator: type[TSimulator]
            If we want to create a derived class of Simulator, specify here

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
