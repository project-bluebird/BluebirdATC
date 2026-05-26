import random
import typing
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_dt.core import Aircraft, Airspace, Coordination, FlightPlan, Route
from bluebird_dt.events import EventHandler
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator import Simulator


class RegularScenarioManagerConfig(BaseModel):
    """
    Configuration of a regular scenario manager.
    """

    total_time: float
    number_of_aircraft: int
    scenario_manager: typing.Literal["regular"] = Field(default="regular")


class Regular(ScenarioManager[RegularScenarioManagerConfig]):
    """
    Quasi-regularly spaced Aircraft emitted from Route starts.
    """

    projection_centre: tuple[float, float] | None = None
    event_handler_ignore_flags: typing.ClassVar[EventHandler.IgnoreFlags] = EventHandler.IgnoreFlags()
    total_time: float
    num_aircraft: int
    airspace: Airspace
    routes: list[Route]
    start_time: float
    vertical_buffer_distance: int | float
    lateral_buffer_distance: int | float
    initialise_with_event_handler: bool

    def __init__(
        self,
        total_time: float,
        num_aircraft: int,
        airspace: Airspace,
        routes: list[Route],
        start_time: float = 0,
        vertical_buffer_distance: int | float = 500,
        lateral_buffer_distance: int | float = 20,
        initialise_with_event_handler: bool = True,
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
        start_time: int
            Start time of scenario, in unix time (seconds)
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        initialise_with_event_handler: bool, default is True
            Initialise the environment with the EventHandler
        """

        if total_time <= 0.0:
            raise ValueError("Total time must be positive.")

        if num_aircraft <= 0:
            raise ValueError("Number of Aircraft must be positive.")

        self.total_time = total_time
        self.num_aircraft = num_aircraft
        self.airspace = airspace
        self.routes = routes
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.initialise_with_event_handler = initialise_with_event_handler

    @override
    def create_event_handler(self) -> EventHandler:
        """
        Generate EventHandler for the given Airspace.

        Returns
        ----------
        EventHandler
        """
        # create empty event handler
        event_handler = EventHandler(ignore=self.event_handler_ignore_flags)

        sector_name = next(iter(self.airspace.sectors.keys()))
        volume = self.airspace.sectors[sector_name].volumes[0]
        allowed_FLs = np.arange(volume.min_fl, volume.max_fl + 10, 10, dtype="float")

        # Create start times for all Aircraft ensuring that the Aircraft starts
        # are quasi-regularly spaced between start of scenario and self.total_time.
        start_times = np.random.uniform(low=0, high=1, size=self.num_aircraft)
        total = sum(start_times)
        t = -start_times[0] * 0.5
        for i in range(len(start_times)):
            t += start_times[i]
            start_times[i] = (t / total) * self.total_time

        for i, start_t in enumerate(start_times):
            flight_time = 1800.0  # in seconds
            route = random.choice(self.routes)
            speed = route.length(self.airspace.fixes) / (flight_time / 3600.0)

            # entry/exit flight level should be within the Airspace limits
            entry_fl = random.choice(allowed_FLs)
            exit_fl = random.choice(allowed_FLs)

            # don't need to specify coordination times
            aircraft_entry_fix = route.filed[0]
            aircraft_exit_fix = route.filed[-1]

            callsign = f"AIR{i}"

            coordination_entry = Coordination(
                callsign=callsign,
                from_sector="background",
                to_sector=sector_name,
                fl=entry_fl,
                fix=aircraft_entry_fix,
                direction="Horizontal",
            )

            coordination_exit = Coordination(
                callsign=callsign,
                from_sector=sector_name,
                to_sector="background",
                fl=exit_fl,
                fix=aircraft_exit_fix,
                direction="Horizontal",
            )

            flight_plan = FlightPlan(route)

            pos = self.airspace.fixes.places[route.filed[0]].pos3d(entry_fl)
            heading = self.airspace.fixes.places[route.filed[0]].bearing_to(self.airspace.fixes.places[route.filed[1]])

            aircraft = Aircraft(
                pos.lat,
                pos.lon,
                pos.fl,
                heading,
                flight_plan,
                callsign,
                selected_fl=pos.fl,
                current_sector=None,
            )
            aircraft.speed_tas = speed
            aircraft.simulated = True

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
    ) -> EnvironmentManager:
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
            EnvironmentManager for Regular scenario
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

        em = EnvironmentManager(
            airspace=self.airspace,
            event_handler=event_handler,
            predictor=predictor,
            time=self.start_time,
            penumbra_fl=int(self.vertical_buffer_distance),
            penumbra_lat=self.lateral_buffer_distance,
            log_filename=log_filename,
        )

        if self.initialise_with_event_handler:
            em.initialise_env_with_event_handler()

        return em

    @override
    def config(self) -> RegularScenarioManagerConfig:
        return RegularScenarioManagerConfig(total_time=self.total_time, number_of_aircraft=self.num_aircraft)

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

        return Simulator(
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
        )
