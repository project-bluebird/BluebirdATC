import random
import typing
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_dt.core import Aircraft, Airspace, Coordination, FlightPlan, Pos2D, Route
from bluebird_dt.events.event_handler import EventHandler
from bluebird_dt.logger import logger
from bluebird_dt.manager.environment_manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility import geometry


class TacticalScenarioManagerConfig(BaseModel):
    """
    Configuration of the tactical scenario manager
    """

    scenario_manager: typing.Literal["tactical"] = Field(default="tactical")


class Tactical(ScenarioManager[TacticalScenarioManagerConfig]):
    """
    Aircraft generator for simple tactical scenarios:
    - configurable number of Aircraft and balance of
      climbers/descenders/overfliers
    - randomly selected entry and exit Coordinations
    - randomly generated speeds
    - ensures that no two Aircraft have the same entry coordination
      (same Fix, Flight Level and time)
    - ability to randomize the start position of aircraft within an entry fix
      through a stochastic sample of lateral distance from the entry fix.
    """

    # pairs of boundary then outer fixes for the x-sector, y-sector, and i-sector
    # required for lateral offset calculations
    x_sector_pairs = (
        ("GATES", "SIN"),
        ("DEMON", "SANTA"),
        ("WITCH", "SIREN"),
        ("HAUNT", "LIMBO"),
    )
    y_sector_pairs = (("CANON", "GOD"), ("BISHP", "GHOST"), ("SON", "DECAN"))
    i_sector_pairs = (("EARTH", "FIRE"), ("AIR", "SPIRIT"))

    projection_centre: tuple[float, float] | None = None
    event_handler_ignore_flags = EventHandler.IgnoreFlags()
    num_aircraft: int
    airspace: Airspace
    routes: list[Route]
    balance: list[float] | None
    speed_range: list[float] | None
    time_entry_gap: float
    lateral_offset: tuple[int, int] | None
    env_manager_class: type[EnvironmentManager] | None
    start_time: int
    vertical_buffer_distance: float | int
    lateral_buffer_distance: float | int
    initialise_with_event_handler: bool

    def __init__(
        self,
        num_aircraft: int,
        airspace: Airspace,
        routes: list[Route],
        balance: list[float] | None = None,
        speed_range: list[float] | None = None,
        time_entry_gap: float = 5,
        lateral_offset: tuple[int, int] | None = None,
        env_manager_class: type[EnvironmentManager] | None = None,
        start_time: int = 0,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
        initialise_with_event_handler: bool = True,
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
        balance: list[float, float, float]
            Probabilities of any given Aircraft being one of climber/descender/overflier.
            The probabilities have to sum to 1 (Multinomial distribution parameter).
        speed_range: list[float, float]
            Optional range of [min,max] speeds from which to randomly generate Aircraft speed.
            If not provided, speed of all Aircraft is set to 400.
        time_entry_gap: float
            Optional amount of time in seconds that must be maintained between two
            Aircraft entry Coordinations if they are at the same Fix and FL.
        lateral_offset: tuple, optional
            if present, the range (low, high) from which to sample (uniform distribution)
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
        initialise_with_event_handler: bool, default is True
            Initialise the environment with the EventHandler
        """
        if balance is None:
            balance = [1 / 3, 1 / 3, 1 / 3]

        if speed_range is None:
            speed_range = [400.0, 400.0]

        self.check_inputs_valid(num_aircraft, balance, speed_range, time_entry_gap)

        self.num_aircraft = num_aircraft
        self.airspace = airspace
        self.routes = routes
        self.balance = balance
        self.speed_range = speed_range
        self.time_entry_gap = time_entry_gap
        self.lateral_offset = lateral_offset
        if env_manager_class:
            self.env_manager_class = env_manager_class
        else:
            self.env_manager_class = EnvironmentManager
        self.start_time = start_time
        self.vertical_buffer_distance = vertical_buffer_distance
        self.lateral_buffer_distance = lateral_buffer_distance
        self.initialise_with_event_handler = initialise_with_event_handler

    @staticmethod
    def check_inputs_valid(
        num_aircraft: int,
        balance: list[float],
        speed_range: list[float],
        time_entry_gap: float,
    ) -> None:
        """
        Check the inputs have valid types.

        Parameters
        ----------
        num_aircraft: float
            Number of Aircraft to generate.
        balance: list[float, float, float]
            Probabilities of any given Aircraft being one of climber/descender/overflier.
            The probabilities have to sum to 1 (Multinomial distribution parameter).
        speed_range: list[float, float]
            Optional range of [min,max] speeds from which to randomly generate Aircraft speed.
            If not provided, speed of all Aircraft is set to 400.
        time_entry_gap: float
            Optional amount of time in seconds that must be maintained between two
            Aircraft entry Coordinations if they are at the same Fix and FL.
        """

        if num_aircraft <= 0:
            raise ValueError("Number of Aircraft must be positive.")

        if len(balance) != 3:
            raise ValueError("The balance parameter must be a list of length 3.")

        if sum(balance) != 1:
            raise ValueError("Balance probabilities must sum to 1.")

        if len(speed_range) != 2:
            raise ValueError("Speed range must be a list of length 2.")

        if time_entry_gap < 0:
            raise ValueError("Time entry gap cannot be negative.")

    def stochastic_start_pos(self, airspace: Airspace, route: Route) -> Pos2D:
        """Laterally offset the aircraft from their spawning point (start fix)

        Laterally offset the aircraft from their spawning point to minimise
        clashes and introduce stochasticity to aircraft start point in within
        a start fix.

        Parameters
        ----------
            airspace: Airspace
                The airspace to generate aircraft within offset start position.
            route: Route
                The route of the aircraft.

        Returns
        ----------
        Pos2D
            the sampled start position.
        """

        offset_low, offset_high = self.lateral_offset if self.lateral_offset is not None else (0, 0)

        lateral_offset_headings = self.set_up_lateral_start_points(airspace)

        first_fix_name = route.filed[0]
        first_fix = airspace.fixes.places[first_fix_name]
        # second_fix = airspace.fixes.places[route.filed[1]]

        gh = airspace.geo_helper
        # since GeoHelper.forward takes (x,y), give it longitude then latitude
        updated_start_pos = gh.forward(
            first_fix.lon,
            first_fix.lat,
            heading=np.random.choice(lateral_offset_headings[first_fix_name]),
            distance=np.random.uniform(offset_low, offset_high),
        )

        return Pos2D(updated_start_pos[1], updated_start_pos[0])

    def set_up_lateral_start_points(self, airspace: Airspace) -> dict[str, tuple[float, float]]:
        """
        Create a dictionary of headings - two for each spawn point which can
        be used to offset the aircraft position on spawning.

        Parameters
        ----------
        airspace: Airspace
            The current airspace

        Returns
        ----------
        dictionary
            lateral_offset_headings: {str:[float, float]} fix name: list of
            two headings which would run parallel to the sector boundary.
        """

        gh = airspace.geo_helper
        lateral_offset_headings: dict[str, tuple[float, float]] = {}

        sectors_ = list(airspace.sectors.keys())
        if "sector_x" in sectors_:
            pairs = Tactical.x_sector_pairs
        elif "sector_y" in sectors_:
            pairs = Tactical.y_sector_pairs
        elif "sector_i" in sectors_:
            pairs = Tactical.i_sector_pairs
        else:
            raise ValueError("Invalid airspace")

        for inner, outer in pairs:
            inner_fix = airspace.fixes.places[inner]
            outer_fix = airspace.fixes.places[outer]

            # Get a perpendicular line between the inner (or boundary) fix and the outer (or spawning) fix
            start, _end, _ = geometry.get_perpendicular_line(inner_fix, outer_fix)

            # Calculate the heading from the outer (spawning) fix along this perpendicular line in one direction
            heading_1 = gh.bearing_to(
                lat=start[0],
                lon=start[1],
                lat_origin=outer_fix.lat,
                lon_origin=outer_fix.lon,
            )

            # Calculate the heading from the outer (spawning) fix along this perpendicular line in the other direction
            # trick: cheaper computation below instead of computing the heading based on bearing to
            # `end` from `outer_fix`
            heading_2 = heading_1 + 180.0
            heading_2 = heading_2 if heading_2 < 360.0 else heading_2 - 360.0

            lateral_offset_headings[outer] = (heading_1, heading_2)

        return lateral_offset_headings

    @override
    def create_event_handler(self) -> EventHandler:
        """
        Generate event_handler for the given Airspace.

        Returns
        ----------
        EventHandler
            The EventHandler specifying Aircraft with unique string identifiers (callsigns) to fly
            through the Airspace and when to add them to the Environment.
        """
        # split aircraft into climbers, descenders and overfliers
        climbers, descenders, overfliers = np.random.multinomial(self.num_aircraft, self.balance)
        journey_type = ["climb"] * climbers + ["descend"] * descenders + ["overfly"] * overfliers
        random.shuffle(journey_type)

        sector_name = next(iter(self.airspace.sectors.keys()))
        volume = self.airspace.sectors[sector_name].volumes[0]
        allowed_FLs = np.arange(volume.min_fl, volume.max_fl + 10, 10, dtype="float")

        # keep track of start fixes and entry coordinations to avoid clashes
        entries = defaultdict(lambda: defaultdict(list))

        # create empty event handler
        event_handler = EventHandler(ignore=self.event_handler_ignore_flags)

        for i in range(self.num_aircraft):
            route = random.choice(self.routes)

            journey = journey_type[i]
            if journey == "overfly":
                entry_fl = exit_fl = random.choice(allowed_FLs)

            elif journey == "climb":
                # entry_fl < exit_fl
                # choose start FL so that Aircraft has somewhere to climb to
                entry_fl = random.choice(np.arange(allowed_FLs[0], allowed_FLs[-1], 10))
                # choose exit to be higher than entry_fl AND within exit FL limits
                exit_fl = random.choice(np.arange(entry_fl + 10, allowed_FLs[-1] + 10, 10))

            elif journey == "descend":
                # entry_fl > exit_fl
                # choose start FL so that Aircraft has somewhere to descend to
                entry_fl = random.choice(np.arange(allowed_FLs[0] + 10, allowed_FLs[-1] + 10, 10))
                # choose exit to be lower than entry_fl AND within exit FL limits
                exit_fl = random.choice(np.arange(allowed_FLs[0], entry_fl, 10))

            else:
                raise ValueError(f"Invalid journey type: {journey}")

            callsign = f"AIR{i}"
            speed_diff = self.speed_range[1] - self.speed_range[0]
            speed = self.speed_range[0] + speed_diff * np.random.uniform()
            start_fix_name = route.filed[0]
            start_fix = self.airspace.fixes.places[start_fix_name]

            if self.lateral_offset:
                # compute the start position of the aircraft
                # offset by a stochastic lateral distance.
                pos = self.stochastic_start_pos(
                    self.airspace,
                    route,
                )
                pos = pos.pos3d(entry_fl)
            else:
                # no lateral offset
                pos = start_fix.pos3d(entry_fl)

            heading = pos.bearing_to(self.airspace.fixes.places[route.filed[1]])

            aircraft_entry_fix = route.filed[0]
            aircraft_exit_fix = route.filed[-1]

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

            # check entry coordination and make sure it doesn't clash with another Aircraft entry coordination
            start_t = self.start_time  # start time is set to 0.0 seconds, which corresponds to 1970-01-01T00:00:00 UTC
            if start_t in entries[start_fix_name][entry_fl]:
                start_t = max(entries[start_fix_name][entry_fl]) + self.time_entry_gap
            entries[start_fix_name][entry_fl].append(start_t)

            flight_plan = FlightPlan(route)

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
            aircraft.selected_instructions.cas = speed

            # TODO: understand why this is set for diverse tactical, but not tactical?
            if self.lateral_offset:
                aircraft.on_route = False

            event_start_time = pd.to_datetime(start_t, unit="s")

            event_handler.add_aircraft(event_start_time, aircraft)

            # ensure coordinations are in the environment before the aircraft
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coordination_exit)
            event_handler.add_coordination(event_start_time - timedelta(seconds=1), coordination_entry)

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
        log_filename: str, optional
            Name of file logs will be saved to. If None, defaults to datetime logger created.

        Returns
        ----------
        EnvironmentManager
            Environment Manager for Tactical scenario
        """

        logger.info(
            f"""

        ===================================================================
        Creating Tactical Scenario with {self.num_aircraft} aircraft.
        """
        )

        # create SimplePredictor if no Predictor passed
        if predictor is None:
            predictor = SimplePredictor(1.0, 2.0)

        # create event handler from the events list
        event_handler = self.create_event_handler()

        em = self.env_manager_class(
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
    def config(self) -> TacticalScenarioManagerConfig:
        return TacticalScenarioManagerConfig()

    def to_simulator(
        self,
        scenario_name: str | None = None,
        category: str | None = None,
        use_wind: bool = True,
        use_forecast: bool = True,
        predictor: Predictor | None = None,
        attach_context_to_logger: bool = True,
        save_log_to_file: bool = True,
        log_filename: str | None = None,
        save_csv: bool = True,
        autosave_interval: timedelta | None = timedelta(minutes=1),
        save_chunk_interval: timedelta | None = None,
    ) -> Simulator:
        """
        Create a Simulator instance for Tactical scenarios.

        Parameters
        ----------
        category : str | None, optional
            Category of the simulation. Default is None.
        scenario_name : str | None, optional
            Name of the scenario. Default is None.
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
            category=category,
            scenario_name=scenario_name,
            use_wind=use_wind,
            use_forecast=use_forecast,
            predictor=predictor,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            log_filename=log_filename,
            save_csv=save_csv,
            autosave_interval=autosave_interval,
            save_chunk_interval=save_chunk_interval,
        )
