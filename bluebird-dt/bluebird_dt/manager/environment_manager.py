from __future__ import annotations

import io
import typing
from datetime import timezone
from typing import Generic, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

import bluebird_dt.predictor
from bluebird_dt.core import (
    Action,
    Aircraft,
    Airspace,
    Coordination,
    Environment,
    Fixes,
)
from bluebird_dt.core.wind import WindField
from bluebird_dt.events import EventHandler, EventLogger
from bluebird_dt.logger import logger
from bluebird_dt.utility.artificial_airspace_defaults import (
    AIRSPACE_SETTINGS as default_airspace_settings,
)
from bluebird_dt.utility.artificial_airspace_defaults import (
    AirspaceSettingsType,
)
from bluebird_dt.utility.clearance import add_phraseology

if typing.TYPE_CHECKING:
    from bluebird_dt.predictor import Predictor

from bluebird_dt.utility import constants


class PredictorConfig(BaseModel):
    """
    Predictor configuration stored with logs.
    """

    predictor_type: str
    fix_proximity: float


class EnvironmentConfig(BaseModel):
    """
    Environment manager configuration stored with logs.
    """

    penumbra_latitude: float
    penumbra_flight_level: float
    predictor: PredictorConfig


class CoordRequest(NamedTuple):
    status: str
    coord: Coordination


TAircraft = typing.TypeVar("TAircraft", bound=Aircraft)
TWindField = typing.TypeVar("TWindField", bound=WindField)
TForecastWindField = typing.TypeVar("TForecastWindField", bound=WindField)


class EnvironmentManager(Generic[TAircraft, TWindField, TForecastWindField]):
    """
    Manage and evolve the Environment.
    """

    environment: Environment[TAircraft, TWindField, TForecastWindField]
    event_handler: EventHandler[TAircraft]
    predictor: Predictor
    min_step_time: float
    penumbra_lat: float
    penumbra_fl: int
    reload_environment: bool
    event_logger: EventLogger
    _actions_to_issue: list[Action]
    contains_disjoint_polygons: bool

    def __init__(
        self,
        airspace: Airspace,
        event_handler: EventHandler[TAircraft],
        predictor: Predictor,
        time: float = 0.0,
        penumbra_lat: float = 5.0,
        penumbra_fl: int = 10,
        wind_field: TWindField | None = None,
        forecast_wind_field: TForecastWindField | None = None,
        log_filename: str | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        airspace: Airspace
            The Airspace to be controlled.
        event_handler: EventHandler
            The EventHandler applies events to the internal Environment of the EM.
        predictor: Predictor
            Aircraft Trajectory prediction used to evolve Aircraft.
        time: float, default = 0.0
            Environment start time in seconds (Posix/UNIX).
        penumbra_lat: float, optional, default = 5 NMI
            Maximum lateral distance at which an uncontrolled aircraft may leave the sector boundary before being
            removed from being observable by an agent.
        penumbra_fl: int, optional, default = 10 flight levels
            Maximum number of flight levels that an uncontrolled aircraft may leave the sector by before being removed
            from being observable by an agent.
        wind_field: WindField, optional
            A wind vector field.
        forecast_wind_field: WindField, optional
            A forecasted wind vector field.
        log_filename: str or None, default = None
            Name of file logs will be saved to. If None, defaults to
            time logger created.


        Type Variables
        --------------
        Dependent packages can implement expanded representations of the following classes.

        TAircraft: bluebird_dt.core.Aircraft
        TWindField: bluebird_dt.core.WindField
        TForecastWindField: bluebird_dt.core.WindField
        """

        self.environment = self.typeof_environment()(
            time=time,
            airspace=airspace,
            aircraft={},
            wind_field=wind_field,
            forecast_wind_field=forecast_wind_field,
        )
        self.event_handler = event_handler
        self.predictor = predictor
        # ensure the fixes in the predictor are the ones in the environment
        self.predictor.fixes = airspace.fixes

        # minimum dt [sec] needed when self.evolve() is called to ensure Predictor returns a Trajectory
        # Trajectory can only be formed if there are at least 2 control points --> we need at least 2 Predictor dt steps
        self.min_step_time = self.predictor.dt * 2

        # perform error checking to ensure that the given penumbra distances
        # can be evaluated by the cache -- i.e. we do not want a scenario in
        # which the cache can, for example, only give distances upto 4NMI away,
        # and yet the penumbra is 5NMI.
        if (penumbra_lat < 0) or (penumbra_fl < 0):
            raise ValueError("Both `penumbra_lat` and `penumbra_fl` must be non-negative")

        # store the border checking parameters
        self.penumbra_lat = penumbra_lat
        self.penumbra_fl = penumbra_fl

        # does the HMI need to reload the environment?
        self.reload_environment = False
        self.event_logger = EventLogger(log_name=log_filename)

        # actions that have been received and will be issued at the next 'evolve'
        self._actions_to_issue = []

        # coordination requests are added via the `request_coord` function
        self._coord_requests: dict[tuple[str, str | None, str | None], CoordRequest] = {}

        # some scenarios contain highly disjoint sector polygons, requiring different handling when removing aircraft
        # outside penumbra (sector-wise rather than considering whole airspace)
        self.contains_disjoint_polygons = False

    @staticmethod
    def typeof_environment() -> type[Environment[TAircraft, TWindField, TForecastWindField]]:
        return Environment

    @property
    def all_trajectories(self) -> dict[str, list[list[float]]]:
        """
        Return Trajectories flown for every aircraft

        Returns
        -------
        dict :
            key-value pair representing the callsign and trajectory of each aircraft
            {callsign: 2D numpy array of [lat, lon, fl, datetime]}
        """
        if not self.event_logger.radar_log:
            # if no radar log, return empty dict
            # transform log data into dataframe
            all_trajectories: dict[str, list[list[float]]] = {}
        else:
            # transform radar_log to {callsign: trajectory} dictionary
            radar_df = self.event_logger.radar_log_as_df()

            # change datetime to utc timestamp
            radar_df["datetime"] = radar_df["datetime"].dt.tz_localize("UTC").apply(lambda v: v.timestamp())

            # drop any duplicates which can occur with real world data
            radar_df = radar_df.drop_duplicates(subset=["datetime", "callsign"], keep="last")

            # turn into dict with callsign as key and (lat, lon, fl, env_time) as values
            all_trajectories = {
                str(key): group[["lat", "lon", "fl", "datetime"]].to_numpy().tolist()
                for key, group in radar_df.groupby("callsign")
            }

        return all_trajectories

    def request_coord(self, status: str, coord: Coordination):
        """
        Add a single coordination request.

        coordination_requests is a dict of suggested coordinations of the form
        { (callsign, from_sector, to_sector) :
                      {status: <'Accepted' or 'Declined' or 'Request' or 'Delete'>,
                       coord: <a coordination> }}
        'Accept' coord requests will be passed to the coordinations immediately
        'Reject' coord requests will remain in the dict until overwritten by another
                   coord request for the same <to_sector>
        'Offer' coord requests are a requested coordination from the 'from_sector',
                they will remain in the dict until overwritten by another coord
                request for the same <to_sector>
        'CounterOffer' coord requests are a requested coordination from the 'to_sector',
                       usually as a counter-offer to an original "Offer" request.
                       They will remain in the dict until overwritten by another coord
                       request for the same <to_sector>
        'RemoveRequest' coord requests with matching (callsign, from_sector, to_sector)
                        will be deleted from coord_requests
        """
        if status not in ["Accept", "Reject", "Offer", "CounterOffer", "RemoveRequest"]:
            logger.warning(
                f"""'status: {status}' is invalid and request for
                    callsign: {coord.callsign}, from_sector: {coord.from_sector}, to_sector: {coord.to_sector}
                    is being ignored.
                    Valid 'status' are any of ["Accepted", "Declined", "Request", "Deleted"]
                    """,
                stacklevel=2,
            )

        key = (coord.callsign, coord.from_sector, coord.to_sector)
        if status in ["Reject", "Offer", "CounterOffer"]:
            # add them to coord_requests so they can be viewed by others
            self._coord_requests[key] = CoordRequest(status, coord)
        elif status == "Accept":
            self.environment.coordinations.add(coord)
            # remove any requests with the same key if they exist
            # (usually the same coord_request with status: 'Request')
            self._coord_requests.pop(key, None)
        elif status == "RemoveRequest":
            self._coord_requests.pop(key, None)
        else:
            raise Exception(f"Unknown coord_request status {status}")

    @property
    def coord_requests(self) -> list[CoordRequest]:
        """
        Return the coordination requests.

        Returns
        ----------
        dict
            list of coordination requests
        """
        return list(self._coord_requests.values())

    def get_coord_requests(
        self,
        callsign: str | None = None,
        to_sector: str | None = None,
        from_sector: str | None = None,
    ) -> list[CoordRequest]:
        """
        Get coordination requests matching any combination of callsign to_sector and from_sector.

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
        coord_requests = self.coord_requests

        # filter to parameters if they are set
        if callsign is not None:
            # filter to specific to_sector
            coord_requests = [coord_req for coord_req in coord_requests if coord_req.coord.callsign == callsign]

        if to_sector is not None:
            # filter to specific to_sector
            coord_requests = [coord_req for coord_req in coord_requests if coord_req.coord.to_sector == to_sector]

        if from_sector is not None:
            # filter to specific from_sector
            coord_requests = [coord_req for coord_req in coord_requests if coord_req.coord.from_sector == from_sector]

        return coord_requests

    def square_penumbra_limits(self, sector_names: list[str] | None = None) -> tuple[float, float, float, float]:
        """
        The maximum and minimum of the latitude and longitude using a square
        penumbra around the sector.
        Max is calculated as maximum sector boundary point plus the penumbra.
        Min is calculated as minimum sector boundary point minus the penumbra.

        Returns
        -------
        (float, float, float, float)
            min_latitude, max_latitude, min_longitude, max_longitude
        """
        if sector_names is None:
            # use whole airspace
            boundary_points = self.environment.airspace.boundary().boundary_vertices
        else:
            boundary_points = [
                p for name in sector_names for p in self.environment.airspace.sectors[name].boundary().boundary_vertices
            ]

        max_lat = max(p.lat for p in boundary_points) + self.penumbra_lat / constants.NM_PER_DEGREE
        min_lat = min(p.lat for p in boundary_points) - self.penumbra_lat / constants.NM_PER_DEGREE
        max_lon = max(p.lon for p in boundary_points) + self.penumbra_lat / constants.NM_PER_DEGREE
        min_lon = min(p.lon for p in boundary_points) - self.penumbra_lat / constants.NM_PER_DEGREE

        return min_lat, max_lat, min_lon, max_lon

    def get_sector_airspace(self, sector_names: str | list[str] | None, local_fixes: bool = False) -> Airspace:
        """
        Return given Sector(s) as an Airspace object (with associated Fixes).

        Parameters
        ----------
        sector_names: union[str, list[str]]
            The Sector name(s). None will return all individual sectors.
        local_fixes: bool, default False
            Limit the returned airspace to only include fixes that are within its penumbra.

        Returns
        ----------
        Airspace
        """

        if isinstance(sector_names, str):
            sector_names = [sector_names]

        # dictionary of the relevant sector objects
        if sector_names is None:
            sectors = self.environment.airspace._individual_sectors
        else:
            sectors = {name: self.environment.airspace.get_sector(name) for name in sector_names}

        if local_fixes:
            min_lat, max_lat, min_lon, max_lon = self.square_penumbra_limits(sector_names)

            all_fixes = {
                fix: pos
                for fix, pos in self.environment.airspace.fixes.places.items()
                if (min_lat < pos.lat <= max_lat) and (min_lon < pos.lon <= max_lon)
            }
            visibility = {fix: self.environment.airspace.fixes.visibility[fix] for fix in all_fixes}
            fixes = Fixes(all_fixes, visibility)
        else:
            fixes = self.environment.airspace.fixes

        return Airspace(sectors, fixes, airways=self.environment.airspace.airways)

    def set_local_fixes_visibility(self, sector_names: str | list[str] | None = None) -> None:
        """
        Set the visibility of fixes to only include fixes that are within the
        sector penumbra and those in aircraft and airport fixes.

        Parameters
        ----------
        sector_names: union[str, list[str]]
            The Sector name(s) to filter by. None will use all individual sectors.
        """

        if isinstance(sector_names, str):
            sector_names = [sector_names]

        min_lat, max_lat, min_lon, max_lon = self.square_penumbra_limits(sector_names)

        # extract the fixes from all aircraft that are in the scenario
        all_aircraft_fixes = {fix for route_filed in self.event_handler.flight_df.route_filed for fix in route_filed}

        # If we replaced environment via the API, we might not have Events, but
        # still have aircraft - add fixes from their routes as well.
        flight_plans = [ac.flight_plan for ac in self.environment.aircraft.values() if ac.flight_plan is not None]
        fixes_from_routes = {fix for fp in flight_plans for fix in fp.route.filed}

        # add and remove duplicates
        all_aircraft_fixes = all_aircraft_fixes.union(fixes_from_routes)

        # Update the fixes visibility to only include fixes within penumbra
        for fix, pos in self.environment.airspace.fixes.places.items():
            within_penumbra = (min_lat < pos.lat <= max_lat) and (min_lon < pos.lon <= max_lon)
            should_be_visible = fix.isalpha() and (
                fix in all_aircraft_fixes
                or (
                    # airports start with E and have 4 letters (until we have proper data filtering)
                    len(fix) == 4 and fix.startswith("E")
                )
            )

            self.environment.airspace.fixes.visibility[fix] = within_penumbra and should_be_visible

    def observe(
        self,
        sector_name: str | None = None,  # noqa: ARG002
        local_fixes: bool = False,
    ) -> Environment[TAircraft, TWindField, TForecastWindField]:
        """
        Get the current state of the observable Environment (Airspace or a Sector).
        Note that there might be Aircraft in the Environment that are not observable.
        To access full environment with everything, use `self.environment`.

        Parameters
        ----------
        sector_name: str, optional
            Name of Sector to return current state of. If not provided, returns
            state of the entire Airspace.
        local_fixes: bool, default False
            Limit the returned environment's airspace to only include fixes that are within its penumbra.

        Returns
        ----------
        Environment
            The simulation environment
        """

        start_time = self.environment.start_time

        if local_fixes:
            sector_names = list(self.environment.airspace.sectors)
            airspace = self.get_sector_airspace(sector_names, local_fixes)
        else:
            airspace = self.environment.airspace

        # make sure we keep the original start_time
        env = self.typeof_environment()(
            time=self.environment.time,
            airspace=airspace,
            aircraft=self.environment.aircraft,
            coordinations=self.environment.coordinations.values(),
        )
        env.start_time = start_time

        return env

    def _evolve_simulated_aircraft(self, step_time: float):
        """
        Evolve the simulated aircraft by a specified amount of time.

        Parameters
        ----------
        step_time: float
            Number of seconds to evolve the simulated aircraft.
        """
        # move aircraft that are simulated
        for callsign in self.environment.simulated_aircraft():
            # calculate future trajectory
            # then get aircraft in new state from predictor

            selected_aircraft = self.environment.aircraft[callsign]

            # allow a "frozen" flag to 'freeze' simulated aircraft. Used by side-by-side game
            if not selected_aircraft.predictor_params.get("frozen", False):
                # Move the aircraft
                self.predictor.predict_aircraft(
                    selected_aircraft,
                    step_time,
                    environment_time=self.environment.time,
                    wind_field=self.environment.wind_field,
                    deepcopy_aircraft=False,
                )

    def process_actions(self) -> None:
        """
        Process actions in actions_to_issue by passing to pilot and then processing.
        """
        # issue received actions to pilots
        if len(self._actions_to_issue) > 0:
            # send to actions to correct pilots
            aircraft_with_actions = {action.callsign for action in self._actions_to_issue}

            # only issue actions to aircraft in environment
            aircraft_with_actions = aircraft_with_actions.intersection(self.environment.aircraft)

            for callsign in aircraft_with_actions:
                aircraft = self.environment.aircraft[callsign]
                aircraft_actions = [action for action in self._actions_to_issue if action.callsign == callsign]

                ignored_actions = aircraft.pilot.receive_actions(aircraft_actions, self.environment)
                if len(ignored_actions) > 0:
                    logger.warning(
                        f"Actions ignored by pilot: {[str(a) for a in ignored_actions]}",
                        stacklevel=2,
                    )

                # allow pilots to act on clearances
                aircraft.pilot.process_actions(self.environment)

    def evolve(self, step_time: float) -> Environment[TAircraft, TWindField, TForecastWindField]:
        """
        Evolve the Environment: apply Events and update Aircraft Trajectories.

        Parameters
        ----------
        step_time: float
            Amount of time [sec] to evolve the Environment for. This is equivalent
            to radar frequency (i.e., the resolution at which Aircraft locations are
            observed).
            WARNING:
            - If step_time is < 2 * self.predictor.dt the Predictor does not return a Trajectory.
            - If step_time % self.predictor.dt != 0 then the returned Trajectories won't have equidistant time.

        Returns
        ----------
        Environment
            The updated simulation environment.
        """

        # Raise error if `step_time` is not compatible with `self.predictor.dt`
        # TODO: Fix this for small predictor.dt values.
        # E.g. 6.0 % 0.1 should be zero but isn't due to floating point rounding, so throws an error here.
        if step_time < self.min_step_time or step_time % self.predictor.dt != 0:
            raise ValueError(
                f"The step_time value {step_time} is not compatible with the Predictor.dt."
                f" It must be >= {self.min_step_time} and a multiple of the Predictor.dt {self.predictor.dt}..."
            )

        # the environment time before and after being evolved (Posix/UNIX seconds)
        step_end_time = self.environment.time + step_time

        self._evolve_simulated_aircraft(step_time)

        # enact the events
        # event handler ensures actions_to_issue are actioned
        self.environment = self.event_handler.forward(self, step_time)

        self.remove_aircraft_outside_airspace_penumbra()

        # finally move the environment time to the end time
        self.environment.time = step_end_time

        self.event_logger.log_environment(self.environment)
        self.event_logger.log_clearances(self.environment.time, self._actions_to_issue)

        # resent list of actions waiting to be enacted
        self._actions_to_issue = []

        removed_aircraft = self.environment.remove_obsolete_aircraft()

        if len(removed_aircraft) > 0:
            debugging_list = [f"{aircraft.callsign} at FL{aircraft.fl}" for aircraft in removed_aircraft.values()]
            logger.debug(f"Removed aircraft {debugging_list} from environment because their track data is too old.")

        return self.environment

    def remove_aircraft_outside_airspace_penumbra(self) -> None:
        """
        Remove from the environment any aircraft outside the airspace penumbra. If self.contains_disjoint_polygons is
        True, we perform calculations using a square around each sector, rather than a square around the Airspace.

        In the Sector or Airspace cases the Penumbra used is a square around the maximum and minimum points of the
        airspace latitude and longitude, plus the penumbra_lat instance attribute
        """

        callsigns_to_remove = set()

        # If Airspace has disjoint polygons, remove aircraft when they are outside penumbra limits of previous sector
        if self.contains_disjoint_polygons:
            # Store penumbra limits for each sector
            sector_penumbra_limits = {
                sector_name: self.square_penumbra_limits([sector_name])
                for sector_name in self.environment.airspace.sectors
            }

            # for each aircraft, check penumbra limits for their previous sector
            # if there are no penumbra limits for the sector, ignore the aircraft
            for callsign, aircraft in self.environment.aircraft.items():
                if aircraft.previous_sector in sector_penumbra_limits:
                    min_lat, max_lat, min_lon, max_lon = sector_penumbra_limits[aircraft.previous_sector]

                    # Remove aircraft outside the limits
                    in_airspace_penumbra = (min_lat < aircraft.lat <= max_lat) and (min_lon < aircraft.lon <= max_lon)
                    if not in_airspace_penumbra:
                        callsigns_to_remove.add(callsign)

        # If no disjoint polygons, remove aircraft when they are outside penumbra limits of Airspace (the standard case)
        else:
            min_lat, max_lat, min_lon, max_lon = self.square_penumbra_limits()

            for callsign, aircraft in self.environment.aircraft.items():
                in_airspace_penumbra = (min_lat - 1 < aircraft.lat <= max_lat + 1) and (
                    min_lon - 1 < aircraft.lon <= max_lon + 1
                )
                if not in_airspace_penumbra:
                    callsigns_to_remove.add(callsign)

        # remove aircraft from environment
        for callsign in callsigns_to_remove:
            del self.environment.aircraft[callsign]

        # also remove any actions due to be issued to these aircraft
        self._actions_to_issue = [
            action for action in self._actions_to_issue if action.callsign not in callsigns_to_remove
        ]

    def finished(self) -> bool:
        """
        Determine whether the scenario is finished.

        Finished implies that the following criteria have all been met:
        - there are no upcoming events (i.e., no new Aircraft)
        - there aren't any Aircraft left in the Environment

        Returns
        ----------
        bool
        """

        # if there are any remaining events or any aircraft left in the airspace,
        # then we aren't finished.
        no_events_left = (
            self.environment.time > self.event_handler.radar_df.index.max().replace(tzinfo=timezone.utc).timestamp()
        )
        no_aircraft_left = len(self.environment.aircraft) == 0

        return no_events_left and no_aircraft_left

    def receive_actions(self, actions: list[Action]):
        """
        Store Actions to apply when self.evolve() is called

        Actions issued to Aircraft that are not in the environment are filtered before sending to the Pilots.
        An illegal callsign is an aircraft that is not in the Environment (it does not exist or has not appeared yet).
        Legal actions are immediately filtered to aircraft pilots (using its callsign), a clearance is added, then the
        action is issued to the aircraft.

        Parameters
        ----------
        actions: List[Action]
            List of Actions to issue to Aircraft.
        """

        if isinstance(self.predictor, bluebird_dt.predictor.RouteFollowPredictor):
            logger.warning(
                "The predictor RouteFollowPredictor does not respond to Actions...",
                stacklevel=2,
            )

        legal_actions: list[Action] = []
        illegal_callsigns: list[str] = []

        # illegal == Aircraft not in Environment (doesn't exist or hasn't appeared yet)
        for action in actions:
            if (aircraft := self.environment.aircraft.get(action.callsign, None)) is not None:
                if action.sector is None:
                    action.sector = self.environment.airspace.expand_bandbox_sector(aircraft.current_sector)

                legal_actions.append(action)

            else:
                illegal_callsigns.append(action.callsign)

        # save Actions that cannot be issued and warn
        if len(illegal_callsigns) > 0:
            logger.warning(
                f"Cannot issue actions to {', '.join(illegal_callsigns)} (callsign not recognized)...",
                stacklevel=2,
            )

        # add clearances to the legal actions
        for action in legal_actions:
            action = add_phraseology(action=action, environment=self.environment)
        self._actions_to_issue.extend(legal_actions)

    def assign_aircraft_to_bandboxed_sector(self, sector_names: dict[str, list[str]]):
        """
        Assign aircraft in individual sectors to a bandboxed sector. Used when bandboxing sectors.
        Sets the aircraft.current_sector to the bandboxed version of that sector.

        Parameters
        ----------
        sector_names: dictionary[str, list[str]]
            Dictionary denoting the bandboxing. Key is the bandboxed sector name and the value is a list of
            the individual sectors that make up the bandboxed sector
        """
        # for each bandboxed sector in sector names, assign any aircraft in the
        # individual sectors to the new bandboxed sector
        for sec_name, individual_sectors in sector_names.items():
            # filter for the aircraft which are controlled by the individual sectors
            aircraft_in_individual_sectors = [
                ac for ac in self.environment.aircraft.values() if ac.current_sector in individual_sectors
            ]

            # update the aircraft's sector to the new bandboxed sector
            for ac in aircraft_in_individual_sectors:
                # assign to the hidden attribute (rather than the public current_sector)
                # as we don't want to automatically update the previous sector
                ac._current_sector = sec_name

    def assign_aircraft_to_individual_sector(self, bandboxed_sector_name: str):
        """
        Assign aircraft in a bandboxed sector to it's individual sectors. Used when splitting sectors.

        Aircraft will always be assigned to one of the individual sectors which make up the bandboxed sector.
        If aircraft is within an individual sector, it's current sector will be assigned to that sector,
        otherwise it will be assigned to the nearest sector of the constituent individual sectors.

        Parameters
        ----------
        bandboxed_sector_name: str
            Name of the bandboxed sector. Only aircraft which have this sector as their current sector
            will be re-assigned to individual sectors
        """
        # assign all aircraft in a bandboxed sector to the individual sector containing it
        # or else to the individual sector nearest to it
        airspace_sectors = self.environment.airspace.sectors
        individual_sector_names = self.environment.airspace.airspace_configuration[bandboxed_sector_name]
        individual_sector_dict = self.environment.airspace._individual_sectors

        # filter for the aircraft which are controlled by the bandboxed sector
        aircraft_in_bandboxed_sector = [
            ac for ac in self.environment.aircraft.values() if ac.current_sector == bandboxed_sector_name
        ]

        # update each aircraft's sector to an individual sector within the bandboxed sector
        for ac in aircraft_in_bandboxed_sector:
            # check if the aircraft is in the bandboxed sector (may have incommed before physically entering sector)
            if airspace_sectors[bandboxed_sector_name].contains(ac.pos3d()):
                # aircraft is in the bandboxed sector, loop over individual sectors
                # and assign to the individual sector it is in
                for single_sec_name in individual_sector_names:
                    single_sector = individual_sector_dict[single_sec_name]
                    if single_sector.contains(ac.pos3d()):
                        # aircraft is in this individual sector so assign the aircraft to
                        # this sector and exit the loop
                        # assign to the hidden attribute (rather than the public current_sector)
                        # as we don't want to automatically update the previous sector
                        ac._current_sector = single_sec_name
                        break
            else:
                # aircraft is not in the sector. Find the nearest sector of the
                # individual sectors in the bandboxed sector.
                closest_sec_index = np.argmin(
                    [individual_sector_dict[sector].distance(ac.pos2d()) for sector in individual_sector_names]
                )
                closest_sector = individual_sector_names[closest_sec_index]

                # assign to the hidden attribute (rather than the public current_sector)
                # as we don't want to automatically update the previous sector
                ac._current_sector = closest_sector

    def bandbox_sectors(self, sector_names: dict[str, list[str]]):
        """
        Combine 2 or more Sectors in the current Airspace into a single Sector.
        Updates the Airspace and all current and upcoming Aircraft.

        Parameters
        ----------
        sector_names: Dict[str, List[str]]
            <name for band-boxed sector>: <names of current sectors to band-box>
        """

        # check that proposed band-boxed sector names are new
        # and that the sectors to bandbox exist
        for bandboxed_name, bandboxed_sectors in sector_names.items():
            if bandboxed_name in self.environment.airspace.sectors:
                raise ValueError(f"{bandboxed_name} already exists as a Sector...")
            for sec_name in bandboxed_sectors:
                if sec_name not in self.environment.airspace.list_individual_sectors():
                    raise ValueError(f"{sec_name} is not a valid Sector name...")

        # reassign aircraft in the sectors to be combined to the bandboxed sector
        self.assign_aircraft_to_bandboxed_sector(sector_names)

        # update airspace
        self.environment.airspace.bandbox_sectors(sector_names)

    def split_sector(self, sector_name: str):
        """
        Split a bandboxed sector.

        Parameters
        ----------
        sector_name: str
            Name of Bandboxed sector to be split
        """
        # reassign aircraft in the split sector to the individual sectors
        self.assign_aircraft_to_individual_sector(sector_name)

        # update static data
        self.environment.airspace.split_sector(sector_name)

    def get_assigned_bay(self, sector_name: str, callsign: str) -> str | None:
        """
        Get the bay an aircraft in a sector is assigned to.

        Parameters
        ----------
        sector_name: str
            The sector for which the assigned bay is returned.
        callsign: str
            The callsign for which the assigned bay is returned.

        Returns
        -------
        str or None
            Name of bay an aircraft has been assigned to. None if not assigned to any of the bays.
        """
        aircraft = self.environment.aircraft[callsign]
        next_sector = self.environment.next_sector_of_aircraft(callsign)

        if (
            sector_name not in self.environment.airspace.airspace_configuration
            and (
                bandboxed_sector_name := self.environment.airspace.get_containing_bandboxed_sector(
                    sector_name.split(";")[0]
                )
            )
            is not None
        ):
            sector_name = bandboxed_sector_name

        if sector_name == aircraft.current_sector:
            return "INCOMM"
        if sector_name == aircraft.previous_sector:
            return "OUTCOMM"
        if sector_name == next_sector:
            return "PENDING"
        return None

    def get_bays_names(self) -> list[str]:
        """
        Return the names of the standard bays.

        Currently hard-coded to return ["PENDING", "INCOMM", "OUTCOMM"].

        Returns
        -------
        list of string
            List of names of the standard bays. Currently hard-coded to ["PENDING", "INCOMM", "OUTCOMM"]
        """
        return ["PENDING", "INCOMM", "OUTCOMM"]

    def replace_all_aircraft(self, new_aircraft: dict[str, TAircraft]):
        """
        Replace the complete set of aircraft in the environment.

        Parameters
        ----------
        new_aircraft: dict[str, Aircraft]
            Dictionary of Aircraft, keyed by callsign.
        """
        self.environment.aircraft = new_aircraft

        self.event_handler.reset_events()

    def replace_environment(
        self,
        new_environment: Environment[TAircraft, TWindField, TForecastWindField],
        airspace_settings: AirspaceSettingsType | None = None,
    ):
        """
        Replace the whole simulation environment.

        Parameters
        ----------
        new_environment: Environment
            The simulation environment to load in place of the current one.
        airspace_settings: AirspaceSettingsType
            Settings such as FL and lon/lat penumbra for the new airspace.
        """

        self.environment = new_environment
        # remove existing events from event_handler
        self.event_handler.reset_events()

        if airspace_settings is None:
            penumbra_fl = default_airspace_settings["penumbra_fl"]
            penumbra_lat = default_airspace_settings["penumbra_lat"]
        else:
            penumbra_fl = airspace_settings["penumbra_fl"]
            penumbra_lat = airspace_settings["penumbra_lat"]

        self.penumbra_lat = penumbra_lat
        self.penumbra_fl = penumbra_fl

        # ensure that the predictor has the correct set of fixes
        self.predictor.fixes = new_environment.airspace.fixes

        # tell the API that frontend needs to reload environment
        self.reload_environment = True

    def write_logs_to_buffer(self, sim_config: BaseModel, save_csv: bool | None = True) -> io.BytesIO:
        """
        Save the logs to a tar buffer in memory

        Parameters
        ----------
        sim_config: dict, optional
            Additional Simulator class parameters to be logged.
            The event logger save EnvironmentManager parameters automatically but attributes of
            the Simulator class need to be explicitly added to the save log.
        save_csv: bool
            A flag to determine if csv should be saved, default yes.

        Returns
        -------
        io.BytesIO
            The bytes buffer of which the logs is saved to as a tar archive
        """
        return self.event_logger.write_logs_to_buffer(sim_config, save_csv)

    def initialise_env_with_event_handler(self, jump_to_first_event: bool = True, log: bool = True):
        """
        Initialise the environment using the events in the event handler.
        If jump_to_first_event==False, then the events are processed up to the environment time.
        If jump_to_first_event==True, then the radar events with the earliest timestamp are processed
        and the environment time is set to this time

        'log' allows you to turn off logging if required. This is used for example for real
        world data where after initialising, the scenario manager does some heuristic updates
        before manually logging. If log=False, then in needs to be performed manually before the
        next evolve function is called.
        """
        jump_to_time = self.event_handler.radar_df.index.min() if jump_to_first_event else self.environment.datetime

        self.environment = self.event_handler.jump_to_time(env_manager=self, new_time=jump_to_time)
        self.remove_aircraft_outside_airspace_penumbra()

        if log:
            self.event_logger.log_environment(self.environment)
            self.event_logger.log_clearances(self.environment.time, self._actions_to_issue)

        # reset actions_to_issue_log
        self._actions_to_issue = []

    def rewind_to_time(self, new_time: pd.Timestamp):
        """
        Rewind to a specific time.

        Sets the environment back to a specific time according to the logs of events that occurred in the simulation.
        A new event handler is created from the logs and combined with the old event handler if appropriate,
        and the logs are reset to this time.

        Parameters
        ----------
        new_time: datetime
            The datetime to rewind to
        """
        logger.info(f"Rewinding! to {new_time}")
        old_time = self.environment.datetime

        # turn the event_log into the event handler
        # keep any callsign in the original event handler which occur
        # after old_time for aircraft that has never been simulated
        events_to_keep = self.event_handler.trim("<=", old_time).remove_simmed(self.event_logger)
        new_event_handler = self.event_logger.to_event_handler().add(events_to_keep)

        # keep "ignore" setting of original event handler
        new_event_handler.ignore = self.event_handler.ignore
        self.event_handler = new_event_handler

        # delete everything in the log after new_time
        self.event_logger = self.event_logger.trim(">", new_time)

        # update the environment state to the new time
        self.environment = self.event_handler.jump_to_time(env_manager=self, new_time=new_time)

        # update the "previous_log" attributes of the event logger so that future logging is correct
        self.event_logger.log_environment(self.environment, previous_only=True)

        # log sector at this new time (function checks if it's different to the logs before this time)
        self.event_logger._log_sectors_df(self.environment)

    def config(self) -> EnvironmentConfig:
        """
        Obtain the configuration this instance of an environment manager is running with.

        Returns
        -------
        EnvironmentConfig
            Object reflecting the current configuration of the environment manager.
        """
        return EnvironmentConfig(
            penumbra_flight_level=self.penumbra_fl,
            penumbra_latitude=self.penumbra_lat,
            predictor=PredictorConfig(
                predictor_type=self.predictor.__class__.__name__, fix_proximity=self.predictor.fix_proximity_threshold
            ),
        )
