import math
import typing
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import typing_extensions
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.core import Aircraft, Airspace, Coordination, Environment, FlightPlan, Pos2D, Pos3D, Route, WindField
from bluebird_dt.events import EventHandler, EventLogger
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.outcomm_handler import OutcommHandler
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility import geometry
from bluebird_dt.utility.artificial_airspace_defaults import AIRSPACE_SETTINGS


class InfiniteScenarioManagerConfig(BaseModel):
    """
    Configuration of the infinite scenario manager
    """

    scenario_manager: typing.Literal["infinite"] = Field(default="infinite")


TAircraft = typing_extensions.TypeVar("TAircraft", bound=Aircraft, default=Aircraft)
TWindField = typing_extensions.TypeVar("TWindField", bound=WindField, default=WindField)
TForecastWindField = typing_extensions.TypeVar("TForecastWindField", bound=WindField, default=WindField)
TEnvironmentManager = typing_extensions.TypeVar(
    "TEnvironmentManager",
    bound=EnvironmentManager[Aircraft, WindField, WindField],
    default=EnvironmentManager[Aircraft, WindField, WindField],
)
TEventHandler = typing_extensions.TypeVar("TEventHandler", bound=EventHandler[Aircraft], default=EventHandler[Aircraft])
TEventLogger = typing_extensions.TypeVar("TEventLogger", bound=EventLogger, default=EventLogger)
TSimulator = typing_extensions.TypeVar("TSimulator", bound=Simulator, default=Simulator)


class Infinite(
    ScenarioManager[InfiniteScenarioManagerConfig],
    typing.Generic[TAircraft, TWindField, TForecastWindField, TEnvironmentManager, TEventLogger, TEventHandler],
):
    """
    Generate aircraft at random points at set intervals, over an indefinite period of time.
    Optionally the rate at which aircraft spawn can increase over time.
        - start from randomly chosen seed point
        - choose random route to take it to one of the other exit points.
        - entry flight level chosen from restricted range.
        - current level matches entry level
        - exit level chosen from restricted range - limited by airspace.

    """

    initial_spawn_rate: float
    max_spawn_rate: float
    spawn_rate_increment: float
    spawn_rate_increase_interval: float
    total_time_seconds: float
    random_seed: int
    projection_centre: tuple[float, float] | None = None
    event_handler_ignore_flags: EventHandler.IgnoreFlags
    airspace: Airspace
    routes: list[Route]
    speed_range: tuple[float, float] | None
    aircraft_on_route: bool
    start_time: int
    vertical_buffer_distance: float | int
    lateral_buffer_distance: float | int
    current_spawn_rate: float
    fix_spawn_target_interval: int
    spawn_distance_threshold: float
    spawn_distance_behind_fix: float
    num_starter_aircraft: int
    max_spawn_attempts: int
    outcomm_handler: OutcommHandler | None
    last_spawn_time: float
    lats_spawn_rate_increase: float
    min_spawn_delta: float
    next_callsign_number: int
    typeof_environment_manager: type[TEnvironmentManager]
    typeof_event_handler: type[TEventHandler]
    typeof_aircraft: type[TAircraft]
    typeof_event_logger: type[TEventLogger]

    def __init__(
        self,
        airspace: Airspace,
        routes: list[Route],
        initial_spawn_rate: float = 0.01,
        max_spawn_rate: float = 0.1,
        spawn_rate_increment: float = 0.0,
        spawn_rate_increase_interval: float = 0.0,
        spawn_distance_threshold: float = 10,
        spawn_distance_behind_fix: float = 10.0,
        total_time_seconds: float | None = None,
        random_seed: int | None = None,
        automatic_outcomm: bool = True,
        num_starter_aircraft: int = 2,
        speed_range: tuple[float, float] | None = None,
        aircraft_on_route: bool = False,
        typeof_environment_manager: type[TEnvironmentManager] = EnvironmentManager,
        typeof_event_handler: type[TEventHandler] = EventHandler,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_event_logger: type[TEventLogger] = EventLogger,
        start_time: int = 0,
        max_spawn_attempts: int = 10,
        min_spawn_delta: float = 6.0,
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
        initial_spawn_rate: float
            frequency (in 1/s) at which aircraft spawn at the start of the scenario
        max_spawn_rate: float
            maximum allowed frequency at which aircraft can spawn
        spawn_rate_increment: float
            if we choose to "ramp up" the spawn rate, this determines the step size
        spawn_rate_increase_interval: float
            if we "ramp up", this determines time in seconds between increasing the spawn rate by spawn_rate_increment.
        spawn_distance_threshold: float
            Minimum distance in nautical miles for spawning an aircraft behind an existing one
        spawn_distance_behind_fix: float
            Distance behind starting fix that an aircraft will spawn
        total_time_seconds: float | None
            Optionally set the maximum time (in seconds) for the scenario, after which no aircraft will spawn.
        random_seed: int | None
            Optionally specify the seed for the random number generator
        automatic_outcomm: bool
            Specify if the scenario manager should automatically outcomm aircraft which leave the sector meeting exit
            coordination. Defaults to True
        num_starter_aircraft: int
            How many aircraft to spawn (at different fixes) at very start of the scenario.
        speed_range: tuple[float, float] | None
            Optional range of [min,max] speeds from which to randomly choose Aircraft speed. If not provided,
            speeds of all Aircraft are set to a range 350-450 knots
        aircraft_on_route: bool
            If True, spawned aircraft will not be laterally displaced from their starting fix,
            and will have "on_route" set to True.  Default is False.
        start_time: int
            Start time of scenario, in unix time (seconds)
        max_spawn_attempts: int
            How many times to try and randomly spawn an aircraft that doesn't clash with existing aircraft
        min_spawn_delta: float
            Minimum allowed time (in seconds) between spawns.  Default is 6.0.
        vertical_buffer_distance: int or float, default is 500
            Distance to expand airspace vertical boundary by - UoM: FL
        lateral_buffer_distance: int or float, default is 20
            Distance to expand airspace lateral boundary by - UoM: NMI
        typeof_environmentmanager: type[EnvironmentManager], optional
            If we want to use a derived class of env manager, specify here.
        typeof_aircraft: type[Aircraft], optional
            If we want to use a derived class for the aircraft class, specify here.
        typeof_event_logger: type[EventLogger], optional
            If we want to use a derived class for the event logger, specify here.
        typeof_eventhandler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        """

        self.airspace = airspace
        self.routes = routes
        if initial_spawn_rate < 0:
            raise ValueError("spawn rate must be greater than zero")
        self.initial_spawn_rate = initial_spawn_rate
        self.current_spawn_rate = initial_spawn_rate
        if max_spawn_rate < initial_spawn_rate:
            raise ValueError("max spawn rate must be > initial spawn rate")
        self.max_spawn_rate = max_spawn_rate
        self.spawn_rate_increment = spawn_rate_increment
        self.spawn_rate_increase_interval = spawn_rate_increase_interval
        if spawn_distance_threshold < 0 or spawn_distance_behind_fix < 0:
            raise ValueError("spawn_distance_threshold, spawn_distance_behind fix must all be >=0")
        self.spawn_distance_threshold = spawn_distance_threshold
        self.spawn_distance_behind_fix = spawn_distance_behind_fix
        self.total_time_seconds = total_time_seconds
        self.num_starter_aircraft = num_starter_aircraft
        self.speed_range = speed_range if speed_range else (350, 450)
        self.aircraft_on_route = aircraft_on_route
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.max_spawn_attempts = max_spawn_attempts
        self.typeof_environment_manager = typeof_environment_manager
        self.typeof_event_handler = typeof_event_handler
        self.typeof_aircraft = typeof_aircraft
        self.typeof_event_logger = typeof_event_logger
        self.event_handler_ignore_flags = typeof_event_handler.IgnoreFlags()
        self.rng = np.random.default_rng(random_seed)
        self.outcomm_handler = OutcommHandler() if automatic_outcomm else None

        self.last_spawn_time = 0.0
        self.last_spawn_rate_increase = 0.0
        self.min_spawn_delta = min_spawn_delta
        self.next_callsign_number = 0
        # keep a buffer of the aircraft to add, may add to env_manager in
        # the next tick after initial creation.
        self.aircraft_to_add: list[TAircraft] = []
        # dict of headings to use when offsetting spawn positions laterally
        # from each possible start fix
        self.lateral_offset_headings: dict[str, tuple[float, float]] = self.setup_lateral_offset_headings()
        # list of possible fixes to spawn aircraft at
        # (first fixes in all allowed Routes)
        self.start_fixes: list[str] = sorted({route.filed[0] for route in self.routes})
        # shuffle the starting fixes.
        self.rng.shuffle(self.start_fixes)

    def create_aircraft_with_coordinations(
        self,
        possible_routes: list[Route],
        callsign: str,
        spawn_distance_behind_fix: float,
    ) -> tuple[TAircraft, Coordination, Coordination]:
        """
        Generate an Aircraft instance with parameters in the allowed range

        Parameters
            airspace: Airspace
            possible_routes: list[Route]
                choice of routes for this aircraft, with given starting fix
            callsign: str
                the callsign of the created aircraft
            spawn_distance_behind_fix: float
                how far behind the first fix to spawn
        """
        sector_name = sorted(self.airspace.sectors.keys())[0]
        if len(self.airspace.sectors[sector_name].volumes) == 0:
            raise ValueError("Selected airspace has no Volumes.  Please choose another airspace.")
        volume = self.airspace.sectors[sector_name].volumes[0]
        possible_flight_levels = np.arange(volume.min_fl, volume.max_fl + 10, 10, dtype="float")  # ensure floats

        route = self.rng.choice(np.asarray(possible_routes))
        first_fix = self.airspace.fixes.places[route.filed[0]]

        entry_flight_level = self.rng.choice(possible_flight_levels)
        exit_flight_level = self.rng.choice(possible_flight_levels)
        speed = self.rng.uniform(self.speed_range[0], self.speed_range[1])

        # Laterally offset the aircraft from their spawning point to minimise clashes
        # Note the distance is chosen based on the length of the segment of the sector boundary
        # closest to the starting fix.
        segment = volume.area.nearest_segment_to_point(first_fix)
        segment_start = Pos2D(lat=segment[0][1], lon=segment[0][0])
        segment_end = Pos2D(lat=segment[1][1], lon=segment[1][0])
        segment_length = segment_start.distance(segment_end)
        # Use GeoHelper tool to get offset spawn positions
        gh = self.airspace.geo_helper
        # If aircraft is on route, no lateral offset from route centre line.
        if self.aircraft_on_route:
            updated_start_pos = (first_fix.lon, first_fix.lat)
            on_route = True
        else:
            # Offset by a random amount, up to half the length of the nearest sector boundary line.
            # since GeoHelper.forward takes (x,y), give it longitude then latitude
            updated_start_pos = gh.forward(
                first_fix.lon,
                first_fix.lat,
                heading=self.rng.choice(self.lateral_offset_headings[route.filed[0]]),
                distance=self.rng.uniform(0, segment_length / 2),
            )
            on_route = False

        heading = first_fix.bearing_to(self.airspace.fixes.places[route.filed[1]])
        # Now offset the spawning point "backwards" from the starting fix
        # (where "backwards" is defined as 180 degrees from aircraft heading)
        updated_start_pos = gh.forward(
            updated_start_pos[0],
            updated_start_pos[1],
            heading=(heading + 180.0) % 360,
            distance=spawn_distance_behind_fix,
        )
        # pos3d will have latitude then longitude.
        pos = Pos3D(updated_start_pos[1], updated_start_pos[0], entry_flight_level)
        flight_plan = FlightPlan(route)
        coordination_entry = Coordination(
            callsign=callsign,
            from_sector="background",
            to_sector=sector_name,
            fl=entry_flight_level,
            fix=route.filed[0],
            direction="Horizontal",
        )

        coordination_exit = Coordination(
            callsign=callsign,
            from_sector=sector_name,
            to_sector="background",
            fl=exit_flight_level,
            fix=route.filed[-1],
            direction="Horizontal",
        )

        # generate the Aircraft instance
        aircraft = self.typeof_aircraft(
            pos.lat,
            pos.lon,
            pos.fl,
            heading,
            flight_plan,
            callsign,
            selected_fl=int(pos.fl),
            current_sector=sector_name,
        )
        aircraft.speed_tas = speed
        aircraft.selected_instructions.cas = speed
        aircraft.on_route = on_route

        return aircraft, coordination_entry, coordination_exit

    def setup_lateral_offset_headings(self) -> dict[str, tuple[float, float]]:
        """
        Create a dictionary of headings - two for each spawn point which can be used to offset the aircraft position
        on spawning.

        Returns
        -------
        lateral_offset_headings: {str:[float, float]} fix name: list of two headings
            Key is starter fix, value is the two perpendicular headings to the route direction.
        """
        gh = self.airspace.geo_helper
        lateral_offset_headings: dict[str, tuple[float, float]] = {}

        for route in self.routes:
            outer, inner = route.filed[:2]
            inner_fix = self.airspace.fixes.places[inner]
            outer_fix = self.airspace.fixes.places[outer]

            # Get a perpendicular line between the inner (or boundary) fix and the outer (or spawning) fix
            perp, _, _ = geometry.get_perpendicular_line(inner_fix, outer_fix)

            # Calculate the heading from the outer (spawning) fix along this perpendicular line in one direction
            heading_1 = gh.bearing_to(lat=perp[0], lon=perp[1], lat_origin=outer_fix.lat, lon_origin=outer_fix.lon)
            # and the opposite direction
            heading_2 = (heading_1 + 180.0) % 360
            lateral_offset_headings[outer] = (heading_1, heading_2)
        return lateral_offset_headings

    def add_starting_aircraft(self, event_handler: TEventHandler) -> TEventHandler:
        """
        Generate the aircraft that will be there at the start of the scenario and add them to Eventhandler

        Parameters
        ----------
        event_handler: TEventHandler
            newly created EventHandler instance

        Returns
        -------
        TEventHandler
            containing num_starter_aircraft aircraft and coordinations
        """

        # start off with a couple of aircraft from different fixes,
        # spawning in the first couple of seconds.
        current_aircraft = []

        for i in range(self.num_starter_aircraft):
            # only allow one starter aircraft per start_fix
            if i >= len(self.start_fixes):
                logger.info(f"Will only generate {len(self.start_fixes)} starter aircraft.")
                break
            start_point = self.start_fixes[i]
            possible_routes = [route for route in self.routes if route.filed[0] == start_point]
            callsign = f"AIR-0{self.next_callsign_number}"

            # spawn these aircraft on their starting fixes, rather than behind
            aircraft, entry_coord, exit_coord = self.create_aircraft_with_coordinations(
                possible_routes, callsign=callsign, spawn_distance_behind_fix=0
            )
            start_time = pd.to_datetime(float(i), unit="s")
            aircraft.simulated = True
            # ensure coordinations are in the environment before the aircraft
            event_handler.add_coordination(start_time - timedelta(seconds=1), entry_coord)
            event_handler.add_coordination(start_time - timedelta(seconds=1), exit_coord)
            # add this aircraft to the EventHandler
            event_handler.add_aircraft(start_time, aircraft)
            # and to the list of current aircraft, to check for FL clashes
            current_aircraft.append((start_time, aircraft))
            self.next_callsign_number += 1
        return event_handler

    @override
    def create_event_handler(self) -> TEventHandler:
        """
        Generate Aircraft at random locations at regular intervals for the given Airspace.

        Returns
        ----------
        TEventHandler
            EventHandler instance for the chosen scenario
        """

        # create empty event handler
        event_handler = self.typeof_event_handler(ignore=self.event_handler_ignore_flags)

        # add starter aircraft
        event_handler = self.add_starting_aircraft(event_handler)
        logger.debug(f"Added {len(event_handler.coordination_df)} aircraft and coordinations.")
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
Creating Infinite Scenario
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

        em.initialise_env_with_event_handler()

        return em

    @override
    def config(self) -> InfiniteScenarioManagerConfig:
        return InfiniteScenarioManagerConfig()

    @override
    def update(self, env_manager: TEnvironmentManager) -> TEnvironmentManager:
        """
        If enough time has passed, spawn another aircraft

        Parameters
        ----------
        env_manager: TEnvironmentManager
            environment manager instance to update
        """

        if self.outcomm_handler is not None:
            self.outcomm_handler.update(env_manager)

        # If total_time_seconds was specified, and we have gone beyond that,
        # return without spawning any aircraft.
        if self.total_time_seconds and env_manager.environment.time - self.start_time > self.total_time_seconds:
            return env_manager

        # Has enough time passed to increment the spawn rate?
        if self.spawn_rate_increment > 0 and self.current_spawn_rate < self.max_spawn_rate:
            time_since_last_rate_increase = env_manager.environment.time - self.last_spawn_rate_increase
            if time_since_last_rate_increase >= self.spawn_rate_increase_interval:
                self.current_spawn_rate = min(self.current_spawn_rate + self.spawn_rate_increment, self.max_spawn_rate)
                self.last_spawn_rate_increase = env_manager.environment.time

        # Has enough time passed to spawn a new aircraft?
        time_since_last_spawn = env_manager.environment.time - self.last_spawn_time
        if time_since_last_spawn < self.min_spawn_delta:
            return env_manager
        # Loop through each second since last spawn, do Poisson trial on each to see if we
        # will spawn
        num_to_spawn = 0
        for _ in range(int(time_since_last_spawn)):
            num_to_spawn += self.rng.poisson(self.current_spawn_rate)

        for _ in range(num_to_spawn):
            spawn_time = env_manager.environment.time + 6
            next_callsign = f"AIR-{self.next_callsign_number:02d}"

            # spawn these aircraft behind their starting fixes
            safe_to_spawn = False
            attempt_count = 0
            while True:
                aircraft, entry_coord, exit_coord = self.create_aircraft_with_coordinations(
                    self.routes, callsign=next_callsign, spawn_distance_behind_fix=self.spawn_distance_behind_fix
                )
                safe_to_spawn = check_safe_to_spawn(aircraft, env_manager.environment, self.spawn_distance_threshold)
                attempt_count += 1
                if safe_to_spawn or attempt_count > self.max_spawn_attempts:
                    break
            # if we couldn't find a safe way in max_spawn_attempts tries, just continue.
            if not safe_to_spawn:
                continue
            self.aircraft_to_add.append((spawn_time, aircraft))
            self.next_callsign_number += 1
            self.last_spawn_time = spawn_time

            aircraft.simulated = True

            spawn_time = pd.to_datetime(float(spawn_time), unit="s")

            # ensure coordinations are in the environment before the aircraft
            env_manager.event_handler.add_coordination(spawn_time - timedelta(seconds=1), entry_coord)
            env_manager.event_handler.add_coordination(spawn_time - timedelta(seconds=1), exit_coord)
            env_manager.event_handler.add_aircraft(spawn_time, aircraft)

        # If the time has come to add aircraft to env manager, do it now.
        for ac in self.aircraft_to_add:
            if env_manager.environment.time >= ac[0]:
                env_manager.environment.aircraft[ac[1].callsign] = ac[1]
                self.aircraft_to_add.remove(ac)

        return env_manager

    @staticmethod
    def create_airspace(scenario_name: str) -> tuple[Airspace, list[Route]]:
        """
        Create specified airspace.

        Parameters
        ----------
        scenario_name: str
            This is used to identify the sector/airspace.

        Returns
        --------
        tuple[Airspace, list[Route]]
            tuple of the Airspace object and a list of allowed Routes
        """
        match scenario_name:
            case "I-Sector":
                airspace, routes = ArtificialAirspace("i").generate_airspace()
            case "X-Sector":
                airspace, routes = ArtificialAirspace("x").generate_airspace()
            case "Xplus-Sector":
                airspace, routes = ArtificialAirspace("xplus").generate_airspace()
            case "Y-Sector":
                airspace, routes = ArtificialAirspace("y").generate_airspace()
            case "Two Sector":
                airspace, routes = ArtificialAirspace("two").generate_airspace()
            case _:
                raise ValueError(f"Scenario name {scenario_name} unknown")
        return airspace, routes

    @classmethod
    def setup(
        cls,
        scenario_name: str,
        random_seed: int | None = None,
        num_starter_aircraft: int = 2,
        initial_spawn_rate: float = 0.01,
        spawn_rate_increment: float = 0,
        spawn_rate_increase_interval: float | None = None,
        max_spawn_rate: float = 0.1,
        total_time_seconds: float | None = None,
        speed_range: tuple[float, float] | None = None,
        spawn_distance_threshold: float = 10.0,
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
        num_starter_aircraft: int
            Number of aircraft to spawn at the start of the scenario
        initial_spawn_rate: float
            Average frequency (in Hz) of spawning new aircraft
        spawn_rate_increment: float
            Spawn rate will change by this number at set intervals
        spawn_rate_increase_interval: int
            If specified, time in seconds between spawn rate increases
        max_spawn_rate: float
            Spawn rate cannot exceed this value
        total_time_seconds: float | None
            Optionally specify the total time in seconds, after which no new aircraft.
        speed_range: list[float]
            Optional, if not set, aircraft speeds are set between 350 and 450 knots.
        spawn_distance_threshold: float
            Minimum spawn distance from nearest aircraft with overlapping fl range.
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
        typeof_eventhandler: type[EventHandler], optional
            If we want to use a derived class for the Event Handler, specify here.
        Returns
        -------
        Simulator
            A fully configured simulator instance
        """

        airspace, routes = cls.create_airspace(scenario_name)

        sim = cls(
            airspace=airspace,
            routes=routes,
            random_seed=random_seed,
            num_starter_aircraft=num_starter_aircraft,
            initial_spawn_rate=initial_spawn_rate,
            spawn_rate_increment=spawn_rate_increment,
            spawn_rate_increase_interval=spawn_rate_increase_interval,
            max_spawn_rate=max_spawn_rate,
            total_time_seconds=total_time_seconds,
            speed_range=speed_range,
            spawn_distance_threshold=spawn_distance_threshold,
            vertical_buffer_distance=AIRSPACE_SETTINGS["penumbra_fl"],
            lateral_buffer_distance=AIRSPACE_SETTINGS["penumbra_lat"],
            typeof_aircraft=typeof_aircraft,
            typeof_event_logger=typeof_event_logger,
            typeof_event_handler=typeof_event_handler,
            typeof_environment_manager=typeof_environment_manager,
        ).to_simulator(
            log_filename=log_filename,
            predictor=predictor,
            category="Infinite",
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
        typeof_simulator: type[TSimulator] = Simulator,
    ) -> TSimulator:
        """
        Create a Simulator instance for scenarios with infinitely spawning aircraft.

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


def check_safe_to_spawn(new_ac: TAircraft, env: Environment, spawn_distance_threshold: float = 10.0) -> bool:
    """
    Check if it is safe to spawn a new aircraft given existing aircraft in the environment.

    Safe = no existing aircraft are within `spawn_distance_threshold` laterally with level range overlap

    Parameters
    ----------
    new_ac: Aircraft
        The new aircraft to be spawned.
    env: Environment
        The existing environment.

    Returns
    -------
    bool
        True if it is safe to spawn the new aircraft, False otherwise.
    """
    new_ac_afl = new_ac.fl
    new_ac_sfl = new_ac.selected_fl

    new_ac_fl_range = __get_level_range_rounded_to_nearest_10(new_ac_afl, new_ac_sfl)
    for existing_callsign in env.aircraft:
        existing_ac = env.aircraft[existing_callsign]
        if existing_ac.current_sector == "background":
            continue
        existing_ac_afl = existing_ac.fl
        existing_ac_sfl = existing_ac.selected_fl

        existing_ac_fl_range = __get_level_range_rounded_to_nearest_10(existing_ac_afl, existing_ac_sfl)

        lateral_distance = new_ac.pos2d().distance(existing_ac.pos2d())

        level_overlap = min(new_ac_fl_range) <= max(existing_ac_fl_range) and min(existing_ac_fl_range) <= max(
            new_ac_fl_range
        )

        # Basic check: do FL ranges overlap within lateral threshold distance?
        if lateral_distance < spawn_distance_threshold and level_overlap:
            return False

        # Additional check: want to avoid AC spawning exactly at same level as existing AC
        # (for this the lateral threshold should be larger)
        if new_ac_afl == existing_ac_afl and lateral_distance < 60:
            return False

    return True


def __get_level_range_rounded_to_nearest_10(afl: float, sfl: float, xfl: float | None = None) -> tuple[int, int]:
    """
    Get the flight level range encompassing actual, selected and exit FLs for an aircraft.

    Parameters
    ----------
    afl: float
        The actual flight level.
    sfl: float
        The selected flight level.
    xfl: float | None
        Optional, exit flight level, defaults to None.

    Returns
    -------
    tuple[int, int]
        The flight level range rounded to the nearest 10.
    """
    if xfl:
        fl_min = min(afl, sfl, xfl)
        fl_max = max(afl, sfl, xfl)
    else:
        fl_min = min(afl, sfl)
        fl_max = max(afl, sfl)
    return math.floor(fl_min / 10) * 10, math.ceil(fl_max / 10) * 10
