from __future__ import annotations

import copy
import io
import json
import operator
import tarfile
import typing
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, RootModel
from typing_extensions import override

from bluebird_dt.core import Action, Aircraft, Environment
from bluebird_dt.events import EventHandler
from bluebird_dt.events.event_dtypes import EventDtypes
from bluebird_dt.utility.logging_utils import (
    save_df_to_csv_tar,
    save_df_to_parquet_tar,
    save_json_to_tar,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.simulator.simconfig import SaveConfig


class ClearanceLog(typing.TypedDict):
    datetime: pd.Timestamp
    callsign: str
    kind: str
    value: typing.Any
    agent: str | None
    text_clearance: str | None
    text_pilot_response: str | None
    voice_clearance: str | None
    voice_pilot_response: str | None
    sector: list[str] | None


class RadarLog(typing.TypedDict):
    datetime: pd.Timestamp
    callsign: str
    lat: float
    lon: float
    fl: float
    heading: float
    ufid: str
    speed_tas: float
    ground_speed: float
    ground_track_angle: float
    selected_fl: float


class SimRateUpdate(BaseModel):
    """
    An entry for logging the sim rate, normally issued when this information may be updated.

    Attributes
    ----------
    real_datetime: datetime
        The real datetime. Defaults to run datetime.now() when the class is instantiated.
    simulation_datetime: datetime
        The datetime of the environment.
    evolve_period: float
        The configured evolve_period of the runner
    tick_frequency: float
        The configured tick_frequency of the runner
    """

    real_datetime: datetime = Field(default_factory=lambda: datetime.now())
    simulation_datetime: datetime
    evolve_period: float
    tick_frequency: float


class SimStartStop(BaseModel):
    """
    An entry for logging the simulation runner starting and stopping.

    Attributes
    ----------
    real_datetime: datetime
        The real datetime. Defaults to run datetime.now() when the class is instantiated.
    simulation_datetime: datetime
        The datetime of the environment.
    event: Literal['clocks off', 'clocks on']
        Whether the runner has been set to play or pause.
    """

    real_datetime: datetime = Field(default_factory=lambda: datetime.now())
    simulation_datetime: datetime
    event: typing.Literal["clocks off", "clocks on"]


SimEvents = RootModel[list[SimRateUpdate | SimStartStop]]


class EventLogger:
    """
    Events to apply to the Environment.
    """

    _sim_events: list[SimRateUpdate | SimStartStop]
    clearances_log: list[ClearanceLog]
    radar_log: list[RadarLog]

    def __init__(self, log_name: str | None = None):
        """
        Construct a new EventLogger instance.

        Parameters
        ----------
        log_name: str
            Directory name to store logs. If None will be set to the datetime of instance creation

        Returns
        -------
        EventLogger
        """
        # TODO: use the columns from the event handler definition
        self.aircraft_internals_cols = [
            "callsign",
            "datetime",
            "rate_of_turn",
            "operation_params",
            "controllable",
            "simulated",
            "current_sector",
            "previous_sector",
            "percentile_rank_dict",
            "pilot_type",
            "pilot_action_queue",
            "predictor_params",
            "wake_vortex",
            "random_seed",
            "heading_changing_to",
            "next_fix_index",
            "vertical_speed",
            "route_current",
            "last_passed_filed_idx",
            "last_passed_current_idx",
            "squawk_ident_until",
        ]

        self.aircraft_intention_cols = [
            "fl",
            "mach",
            "cas",
            "vertical_speed",
            "heading",
            "on_route",
            "speed_action",
            "vertical_speed_action",
            "vertical_action",
            "lateral_action",
        ]

        self._sim_events = []
        self.sectors_cols = []
        self.incomm_cols = []

        # keep radar logs as lists of dicts until stored so append function is fast
        self.radar_log = []
        self.flight_log = {}

        self.clearances_log = []
        self.agent_plan_log = []
        self.metrics_log = []
        self.airspace_files_path_log = []  # may need deleting
        self.coordination_log = []
        self._previous_coordination = []
        self.sectors_log = []
        self.incomm_log = []
        self._previous_incomm_states = {}
        self.fixes_df = None
        self.config = {}
        self.wind_field_log = None
        self.forecast_log = None
        self.individual_sectors_log = None
        self._previous_ac_internals = []
        self.aircraft_internals_log = []

        if log_name is not None:
            self.log_name = log_name
        else:
            self.log_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    @override
    def __eq__(self, other: object) -> bool:
        """
        Equality comparison between event loggers.

        Parameters
        ----------
        other: EventLogger
            EventLogger to be compared to

        Returns
        -------
        bool
        """

        if not isinstance(other, EventLogger):
            return False

        for key in self.__dict__:
            val1 = self.__dict__[key]
            val2 = other.__dict__[key]

            # Special handling for pandas DataFrames
            if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
                if not val1.equals(val2):
                    return False

            # For non-DataFrame attributes, use regular comparison
            elif val1 != val2:
                return False

        return True

    def _log_wind(self, environment: Environment):
        """
        Log the wind field.
        For now, it is assumed that this will not change during the scenario.
        This static assumption may be relaxed in future.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        """
        # log the wind field and for now assume they don't change
        if self.wind_field_log is None:
            if environment.wind_field is None:
                self.wind_field_log = json.dumps({})
            else:
                self.wind_field_log = environment.wind_field.to_json()

    def _log_forecast(self, environment: Environment):
        """
        Log the forecasted wind field.
        For now, it is assumed that this will not change during the scenario.
        This static assumption may be relaxed in future.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        """
        # log the forecasted wind field and for now assume they don't change
        if self.forecast_log is None:
            if environment.forecast_wind_field is None:
                self.forecast_log = json.dumps({})
            else:
                self.forecast_log = environment.forecast_wind_field.to_json()

    def log_environment(self, environment: Environment, previous_only: bool = False):
        """
        Log current state of environment as a diff of relevant factors
        except clearances which are logged separately.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        previous_only: bool
            If True, don't update the actual logs.
            Used to update the hidden previous_log attributes, so that logging for the next time step is correct
            after a rewind.
        """
        self._log_coordination_df(environment, previous_only=previous_only)
        self._log_incomm(environment, previous_only=previous_only)  # probably shouldn't be logged here
        self._log_aircraft_internals(environment, previous_only=previous_only)  # probably shouldn't be logged here
        #        self._log_strips(environment, previous_only=previous_only)

        # don't log if set to only update the "previous_log" attributes
        if not previous_only:
            self._log_radar(environment)
            self._log_flight_plan(environment)
            self._log_sectors_df(environment)  # probably shouldn't be logged here
            self._log_wind(environment)
            self._log_forecast(environment)

            # for now assume fixes never change so log them only once
            if self.fixes_df is None:
                fix_data = [
                    {
                        "fix": name,
                        "lat": pos.lat,
                        "lon": pos.lon,
                        "visibility": environment.airspace.fixes.get_visibility(name),
                    }
                    for name, pos in environment.airspace.fixes.places.items()
                ]
                self.fixes_df = pd.DataFrame.from_records(fix_data)

            # log individual sectors and for now assume they don't change
            if self.individual_sectors_log is None:
                self.individual_sectors_log = {
                    name: sector.data() for name, sector in environment.airspace._individual_sectors.items()
                }

    def _log_aircraft_internals(self, environment: Environment, previous_only: bool = False):
        """
        Log any changes in the aircraft internals.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        previous_only: bool
            If True, don't update the actual logs.
            Used to update the hidden previous_log attributes, so that logging for the next time step is correct
            after a rewind.
        """

        columns = [
            col
            for col in self.aircraft_internals_cols
            if col not in ["datetime", "pilot_type", "pilot_action_queue", "route_current"]
        ]
        intent_action_cols = [
            "speed_action",
            "vertical_speed_action",
            "vertical_action",
            "lateral_action",
        ]
        intent_cols = [col for col in self.aircraft_intention_cols if col not in intent_action_cols]

        current_ac_internals = [
            {col: getattr(ac, col) for col in columns}  # aircraft columns
            # cleared intention columns
            | {f"cleared_{col}": getattr(ac.cleared_instructions, col) for col in intent_cols}
            # selected intention columns
            | {f"selected_{col}": getattr(ac.selected_instructions, col) for col in intent_cols}
            # stored actions in cleared instructions
            | {
                f"cleared_{col}": (
                    getattr(ac.cleared_instructions, col).to_json()
                    if getattr(ac.cleared_instructions, col) is not None
                    else None
                )
                for col in intent_action_cols
            }
            # stored actions in selected instructions
            | {
                f"selected_{col}": (
                    getattr(ac.selected_instructions, col).to_json()
                    if getattr(ac.selected_instructions, col) is not None
                    else None
                )
                for col in intent_action_cols
            }
            | {"route_current": None if ac.flight_plan is None else ac.flight_plan.route.current}
            for ac in environment.aircraft.values()
        ]

        # log the pilot type and action queue
        [
            log.update(
                {
                    "pilot_type": environment.aircraft[log["callsign"]].pilot.__class__.__name__,
                    "pilot_action_queue": [
                        json.dumps(queue_item)
                        for queue_item in environment.aircraft[log["callsign"]].pilot.action_queue
                    ],
                }
            )
            for log in current_ac_internals
        ]

        ac_internals_changes = [ac for ac in current_ac_internals if ac not in self._previous_ac_internals]

        the_datetime = environment.datetime

        # update previous_ac_internals
        self._previous_ac_internals = copy.deepcopy(current_ac_internals)

        # use current environment time for the logging
        [log.update({"datetime": the_datetime}) for log in ac_internals_changes]

        # update the logs unless flagged not to do so
        if not previous_only:
            # deepcopy as mutable objects are stored
            ac_internals_changes = copy.deepcopy(ac_internals_changes)
            self.aircraft_internals_log.extend(ac_internals_changes)

    def _log_radar(self, environment: Environment):
        """
        Log any changes aircraft position, speeds or heading as radar events.

        Parameters
        ----------
        environment: Environment
            The environment to be logged.
        """

        columns = [col for col in EventDtypes.radar_dtypes if col != "datetime"]
        current_radar = [{col: getattr(a, col) for col in columns} for a in environment.aircraft.values()]
        the_datetime = environment.datetime

        [log.update({"datetime": the_datetime}) for log in current_radar]

        self.radar_log.extend(current_radar)

    def _single_flight_plan_log(
        self, callsign: str, aircraft: Aircraft, the_datetime: pd.Timestamp
    ) -> dict[str, typing.Any]:
        """
        Transform a single aircraft's flight plan into a log.

        Parameters
        ----------
        callsign: str
            Callsign of aircraft for which the flight plan will be logged
        aircraft: Aircraft
            The aircraft for which the flight plan will be logged
        the_datetime: datetime
            Datetime of this log

        Returns
        -------
        dict
            A single flight plan log as a dictionary
        """
        flight_plan = aircraft.flight_plan
        assert flight_plan is not None, "Flight plan should not be None"

        return {
            "callsign": callsign,
            "datetime": the_datetime,
            "start_datetime": flight_plan.start_datetime,
            "end_datetime": flight_plan.end_datetime,
            "squawk": aircraft.squawk,
            "origin": flight_plan.origin,
            "dest": flight_plan.dest,
            "unexpanded_route": flight_plan.unexpanded_route,
            "route_filed": flight_plan.route.filed,
            "sector_crossing_seq": flight_plan.sector_crossing_seq,
            "actype": aircraft.aircraft_type,
            "milcivil": flight_plan.milcivil,
            "requested_flight_level": flight_plan.requested_flight_level,
            "filed_true_airspeed": flight_plan.filed_true_airspeed,
            "intention_code": flight_plan.intention_code,
            "ufid": aircraft.ufid,
            "assigned_squawk": flight_plan.assigned_squawk,
        }

    def is_same_flight_logs(self, old_flight_log: dict, new_flight_log: dict) -> bool:
        """
        Compare if two flight logs, represented by a dict, are identical.
        """
        return all(
            old_flight_log[col] == new_flight_log[col]
            or (
                not isinstance(old_flight_log[col], list)  # pd.isna fails on lists as returns multiple bools
                and pd.isna(old_flight_log[col])
                and not isinstance(new_flight_log[col], list)  # pd.isna fails on lists as returns multiple bools
                and pd.isna(new_flight_log[col])
            )
            for col in EventDtypes.flight_dtypes
            if col not in ["datetime", "start_datetime", "end_datetime", "callsign"]
        )

    def _log_flight_plan(self, environment: Environment) -> None:
        """
        Log any changes in the flight plans.

        Parameters
        ----------
        environment: Environment
            Environment containing the aircraft for which the flight plans changes are
            to be logged
        """
        #  create log of updated flight plan for any aircraft whose flight plan has changed
        #  TODO: if flight plan is set to None during the middle of a scenario, then use the aircraft params
        #  log to set the flight plan to None
        the_datetime = environment.datetime

        # only log flight_plans which are not None
        callsigns_to_log = [callsign for callsign, ac in environment.aircraft.items() if ac.flight_plan is not None]

        for callsign in callsigns_to_log:
            current_flight_plan = self._single_flight_plan_log(callsign, environment.aircraft[callsign], the_datetime)
            if callsign not in self.flight_log:
                self.flight_log[callsign] = [current_flight_plan]
            else:
                # add to new_logs only if the flight plan has changed
                last_flightplan = self.flight_log[callsign][-1]

                # append only if flight plans have changed
                if not self.is_same_flight_logs(last_flightplan, current_flight_plan):
                    # log the new flight plan
                    self.flight_log[callsign].append(current_flight_plan)

    def log_clearances(self, time_unix_s: int | float, action_list: list[Action]):
        """
        Log a list of actions (clearances) at a specific time.

        Parameters
        ----------
        time_unix_s: int or float
            Time of log in unix time (seconds since 01/01/1970)
        action_list: list of Actions
            The list of actions to be logged
        """

        new_actions = [
            {
                "datetime": pd.to_datetime(time_unix_s, unit="s"),
                "callsign": action.callsign,
                "kind": action.kind,
                "value": action.value,
                "agent": action.agent,
                "text_clearance": None if action.text_representation is None else action.text_representation.clearance,
                "text_pilot_response": (
                    None if action.text_representation is None else action.text_representation.pilot_response
                ),
                "voice_clearance": (
                    None if action.voice_representation is None else action.voice_representation.clearance
                ),
                "voice_pilot_response": (
                    None if action.voice_representation is None else action.voice_representation.pilot_response
                ),
                "sector": action.sector,
            }
            for action in action_list
        ]
        self.clearances_log.extend(new_actions)

    def _log_coordination_df(self, environment: Environment, previous_only: bool = False) -> None:
        """
        Log any changes in the coordinations.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        previous_only: bool
            If True, don't update the actual logs.
            Used to update the hidden previous_log attributes, so that logging for the next time step is correct
            after a rewind.
        """
        # update the logs unless flagged not to do so
        if not previous_only:
            # update log with any changes and sort for consistent log ordering
            new_coords = [
                coord
                for coord in environment.coordinations.values()
                if str(vars(coord)) not in self._previous_coordination
            ]

            for coord in new_coords:
                coord_data = coord.data()
                # keep the datetime as a Timestamp object
                coord_data["datetime"] = coord.datetime

                self.coordination_log.append(coord_data)

        # update previous_coordination
        # save as a string which is hashable and enables rapid comparisons
        self._previous_coordination = {str(vars(coord)) for coord in environment.coordinations.values()}

    def _log_sectors_df(self, environment: Environment):
        """
        Log any changes in the sector configuration.

        Parameters
        ----------
        environment: Environment
            Environment for which the sector configuration will be logged whenever it changes
        """
        sectors = sorted(environment.airspace.airspace_configuration.items())

        # if last logged sector list is different to current configuration,
        # then log the new configuration
        if len(self.sectors_log) == 0 or sectors != self.sectors_log[-1]["sectors_configuration"]:
            self.sectors_log.append({"datetime": environment.datetime, "sectors_configuration": sectors})

    def _log_incomm(self, environment: Environment, previous_only: bool = False):
        """
        Log any changes in which sectors are controlling each aircraft.

        Parameters
        ----------
        environment: Environment
            The environment to be logged
        previous_only: bool
            If True, don't update the actual logs.
            Used to update the hidden previous_log attributes, so that logging for the next time step is correct
            after a rewind.
        """

        new_incomm_states: dict[str, str] = {}

        # log any changes from previous incomm state
        for callsign, aircraft in environment.aircraft.items():
            sector_name = aircraft.current_sector
            new_incomm_states[callsign] = sector_name
            # update the logs unless flagged not to do so
            if sector_name != self._previous_incomm_states.get(callsign, None) and not previous_only:
                self.incomm_log.append(
                    {
                        "datetime": environment.datetime,
                        "callsign": callsign,
                        "sector_name": sector_name,
                    }
                )

        # update previous_incomm_states
        self._previous_incomm_states = new_incomm_states

    def log_sim_event(self, event: SimStartStop | SimRateUpdate):
        """
        Append a log event.

        Note that these events are not replayed, or loaded when running a replay, as they are isolated.

        Parameters
        ----------
        event: SimStartStop | SimRateUpdate
            The event to append, assumed to already have been validated.

        """
        self._sim_events.append(event)

    def _save_sim_events(self, tar: tarfile.TarFile):
        """
        Save the sim events as a json file called sim_events.json. See the pydantic model schema for the structure of
        the json.

        Parameters
        ----------
        tar: tarfile.TarFile
            The .tar file to save the events to.
        """
        if len(self._sim_events) > 0:
            save_json_to_tar(SimEvents(self._sim_events).model_dump_json(), tar, "sim_events")

    def radar_log_as_df(self) -> pd.DataFrame:
        """
        Transform radar log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The radar log as an EventHandler DataFrame
        """
        # transform radar log to dataframe
        if self.radar_log:
            radar_df = pd.DataFrame.from_records(self.radar_log)
        else:
            # else create empty dataframe with correct columns
            radar_df = pd.DataFrame(columns=EventDtypes.radar_dtypes)

        return radar_df

    def flight_log_as_df(self) -> pd.DataFrame:
        """
        Transform flight plan log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The flight plan log as an EventHandler DataFrame
        """
        # transform flight log to dataframe
        if self.flight_log:
            flight_df = pd.DataFrame.from_records([v for vals in self.flight_log.values() for v in vals]).astype(
                EventDtypes.flight_dtypes
            )
        else:
            # else create empty dataframe with correct columns
            flight_df = pd.DataFrame(columns=EventDtypes.flight_dtypes)

        return flight_df

    def clearance_log_as_df(self) -> pd.DataFrame:
        """
        Transform clearance log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The clearance log as an EventHandler DataFrame
        """
        # transform clearance log to dataframe
        if self.clearances_log:
            clearance_df = pd.DataFrame.from_records(self.clearances_log)
        else:
            # else create empty dataframe with correct columns
            clearance_df = pd.DataFrame(columns=EventDtypes.clearance_dtypes)

        return clearance_df

    def incomm_log_as_df(self) -> pd.DataFrame:
        """
        Transform incomm log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The incomm log as an EventHandler DataFrame
        """

        # transform incomm log to dataframe
        if self.incomm_log:
            incomm_df = pd.DataFrame.from_records(self.incomm_log)
        else:
            # else create empty dataframe with correct columns
            incomm_df = pd.DataFrame(columns=EventDtypes.incomm_dtypes)

        return incomm_df

    def sectors_log_as_df(self) -> pd.DataFrame:
        """
        Transform sectors log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The sector configuration log as an EventHandler DataFrame
        """
        # transform sectors log to dataframe
        if self.sectors_log:
            sectors_log = pd.DataFrame.from_records(self.sectors_log)
        else:
            # else create empty dataframe with correct columns
            sectors_log = pd.DataFrame(columns=EventDtypes.sectors_dtypes)

        return sectors_log

    def coordination_log_as_df(self) -> pd.DataFrame:
        """
        Transform coordination log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The coordination log as an EventHandler DataFrame
        """
        # transform coordination log to dataframe
        if self.coordination_log:
            coordination_df = pd.DataFrame.from_records(self.coordination_log)
        else:
            # else create empty dataframe with correct columns
            coordination_df = pd.DataFrame(columns=EventDtypes.coord_dtypes)

        return coordination_df

    def aircraft_internals_log_as_df(self) -> pd.DataFrame:
        """
        Transform aircraft internals log to EventHandler DataFrame.

        Returns
        -------
        pandas.DataFrame
            The aircraft_internals log as an EventHandler DataFrame
        """
        # transform aircraft_internals log to dataframe
        if self.aircraft_internals_log:
            ac_internals_df = pd.DataFrame.from_records(self.aircraft_internals_log)
        else:
            # else create empty dataframe with correct columns
            ac_internals_df = pd.DataFrame(columns=EventDtypes.aircraft_internals_dtypes)

        return ac_internals_df

    def _save_radar(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save radar log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where radar log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """

        # transform radar log to dataframe and save
        radar_df = self.radar_log_as_df()

        # only keep the last aircraft configuration for each time
        radar_df = radar_df.drop_duplicates(subset=["datetime", "callsign"], keep="last")

        # save csv for now to make human-readable, but parquet will be read back in
        if save_csv:
            save_df_to_csv_tar(radar_df, tar, "radar")

        save_df_to_parquet_tar(radar_df, tar, "radar")

    def _save_flight_plan(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the flight plan log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where flight plan log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        flight_df = self.flight_log_as_df()

        # save flight plans log
        if save_csv:
            save_df_to_csv_tar(flight_df, tar, "flight_plan")

        save_df_to_parquet_tar(flight_df, tar, "flight_plan")

    def _save_clearances(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the clearance log to parquet.

        Parameters
        ----------
        dir_name: str
            Directory where clearances log will be saved
        """
        # save clearance log
        clearances_df = self.clearance_log_as_df()

        # sort to make sure logging has well-defined ordering
        clearances_df = clearances_df.sort_values(["datetime", "callsign", "kind"])

        if save_csv:
            save_df_to_csv_tar(clearances_df, tar, "clearances")

        # convert mixed type values to a string to save as parquet
        clearances_df["value"] = clearances_df["value"].astype(str)
        save_df_to_parquet_tar(clearances_df, tar, "clearances")

    def _save_sectors(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the sectors log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where sectors log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        # save sector log (sector bandboxing)
        sectors_df = self.sectors_log_as_df()

        # only keep the last sectorisation for each time
        sectors_df = sectors_df.drop_duplicates(subset=["datetime"], keep="last")

        if save_csv:
            save_df_to_csv_tar(sectors_df, tar, "sectors")

        sectors_df["sectors_configuration"] = sectors_df["sectors_configuration"].astype(str)
        save_df_to_parquet_tar(sectors_df, tar, "sectors")

    def _save_incomm(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the incomm log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where incomn log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        # save incomm log
        incomm_df = self.incomm_log_as_df()

        if save_csv:
            save_df_to_csv_tar(incomm_df, tar, "incomm")

        save_df_to_parquet_tar(incomm_df, tar, "incomm")

    def _save_coordination(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the coordination log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where coordination log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        # save coordination log
        coordination_df = self.coordination_log_as_df()
        coordination_df = coordination_df.sort_values(["datetime", "callsign", "from_sector", "to_sector"])

        if save_csv:
            save_df_to_csv_tar(coordination_df, tar, "coordination")

        save_df_to_parquet_tar(coordination_df, tar, "coordination")

    def _save_aircraft_internals(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the aircraft_internals log to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where coordination log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        # transform radar log to dataframe and save
        ac_internals_df = self.aircraft_internals_log_as_df()

        # only keep the last aircraft configuration for each time
        ac_internals_df = ac_internals_df.drop_duplicates(subset=["datetime", "callsign"], keep="last")

        # change types to string
        ac_internals_df["pilot_action_queue"] = ac_internals_df["pilot_action_queue"].astype(str)
        ac_internals_df["percentile_rank_dict"] = ac_internals_df["percentile_rank_dict"].astype(str)
        ac_internals_df["operation_params"] = ac_internals_df["operation_params"].astype(str)
        ac_internals_df["predictor_params"] = ac_internals_df["predictor_params"].astype(str)

        # save csv for now to make human-readable, but parquet will be read back in
        if save_csv:
            save_df_to_csv_tar(ac_internals_df, tar, "ac_internals")

        save_df_to_parquet_tar(ac_internals_df, tar, "ac_internals")

    def _save_fixes(self, tar: tarfile.TarFile, save_csv: bool | None = True):
        """
        Save the fixes to parquet.

        Parameters
        ----------
        tar: TarFile
            TarFile where fixes log will be saved
        save_csv: bool
            A flag to determine if csv should be saved
        """
        # save airspace fixes to file
        # for now assume this is not mutable and was the same throughout the scenario
        if self.fixes_df is not None:  # none if scenario loaded and exited without running
            if save_csv:
                save_df_to_csv_tar(self.fixes_df, tar, "fixes")

            save_df_to_parquet_tar(self.fixes_df, tar, "fixes")

    def _save_config(self, tar: tarfile.TarFile, sim_config: SaveConfig[BaseModel]):
        """
        Save the configuration as a json.

        Parameters
        ----------
        tar: TarFile
            TarFile where environment manager configuration log will be saved
        sim_config: dict, optional
            Additional Simulator class parameters to be logged.
            The event logger save EnvironmentManager parameters automatically but attributes of
            the Simulator class need to be explicitly added to the config log.
        """
        # save the environment manager configuration to file
        # for now assume this is not mutable and was the same throughout the scenario
        save_json_to_tar(sim_config.model_dump_json(indent=4), tar, "config")

    def _save_wind_field(self, tar: tarfile.TarFile):
        """
        Save the wind field log to file.

        Parameters
        ----------
        tar: TarFile
            TarFile where wind field log will be saved
        """
        # save the wind field log to file
        # for now assume this is not mutable and was the same throughout the scenario
        json_str = self.wind_field_log if self.wind_field_log is not None else json.dumps({}, indent=4)
        save_json_to_tar(json_str, tar, "wind")

    def _save_forecast(self, tar: tarfile.TarFile):
        """
        Save the forecast log to file.

        Parameters
        ----------
        tar: TarFile
            TarFile where environment manager configuration log will be saved
        """
        # save the forecast log to file
        # for now assume this is not mutable and was the same throughout the scenario
        json_str = self.forecast_log if self.forecast_log is not None else json.dumps({}, indent=4)
        save_json_to_tar(json_str, tar, "forecast")

    def _save_individual_sectors(self, tar: tarfile.TarFile):
        """
        Save the individual sectors to file.

        Parameters
        ----------
        tar: TarFile
            TarFile where individual sectors log will be saved
        """
        # save individual sectors to file
        # we assume this is not mutable and was the same throughout the scenario
        json_str = json.dumps(self.individual_sectors_log, indent=4)
        save_json_to_tar(json_str, tar, "individual_sectors")

    def write_logs_to_buffer(self, sim_config: SaveConfig[BaseModel], save_csv: bool | None = True) -> io.BytesIO:
        """
        Save all the logs to a tar buffer in memory.

        Parameters
        ----------
        sim_config: dict, optional
            Additional Simulator class parameters to be logged.
            The event logger save EnvironmentManager parameters automatically but attributes of
            the Simulator class need to be explicitly added to the config log.
        save_csv: bool
            A flag to determine if csv should be saved, default yes.

        Returns
        -------
        io.BytesIO
            The bytes buffer of which the logs is saved to as a tar archive
        """
        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Json
            self._save_config(tar, sim_config)
            self._save_wind_field(tar)
            self._save_forecast(tar)
            self._save_individual_sectors(tar)

            # Dataframe
            self._save_radar(tar, save_csv)
            self._save_flight_plan(tar, save_csv)
            self._save_clearances(tar, save_csv)
            self._save_sectors(tar, save_csv)
            self._save_incomm(tar, save_csv)
            self._save_coordination(tar, save_csv)
            self._save_aircraft_internals(tar, save_csv)
            self._save_fixes(tar, save_csv)
            self._save_sim_events(tar)

        tar_buffer.seek(0)
        return tar_buffer

    def trim(self, comparison_function: str, comparison_datetime: pd.Timestamp) -> EventLogger:
        """
        Remove logs by comparing datetime to a reference datetime.
        Comparison function allows "<", "<=", ">", ">=".

        Any log which has a datetime satisfying the comparison function with the comparison datetime
        will be removed from the log.

        Parameters
        ----------
        comparison_function: str
            Function to use to compare the log datetime with a target datetime
        comparison_datetime: pandas.Timestamp
            Datetime to be compared to log in filter

        Returns
        -------
        EventLogger
            The EventLogger after removing any logs which satisfied the comparison function
        """
        match comparison_function:
            # operator: "<", "<=", ">", ">="
            case "<":
                comp = operator.lt
            case "<=":
                comp = operator.le
            case ">":
                comp = operator.gt
            case ">=":
                comp = operator.ge
            case _:
                raise ValueError('Comparison operator unknown, must be one of "<", "<=", ">", ">="')

        if False:
            # for now we don't trim the flight_plans as these can be at any time in the day
            df_flight_log = self.flight_log_as_df()
            self.flight_log = df_flight_log[comp(df_flight_log.start_datetime, comparison_datetime)].to_dict(
                orient="records"
            )

        self.radar_log = [log for log in self.radar_log if not comp(log["datetime"], comparison_datetime)]
        self.clearances_log = [log for log in self.clearances_log if not comp(log["datetime"], comparison_datetime)]
        # self.sectors_log = [log for log in self.sectors_log if not comp(log["datetime"], comparison_datetime)]
        # self.incomm_log = [log for log in self.incomm_log if not comp(log["datetime"], comparison_datetime)]
        # self.coordination_log = [log for log in self.coordination_log if not comp(log["datetime"], comparison_datetime)]
        # self.aircraft_internals_log = [
        #    log for log in self.aircraft_internals_log if not comp(log["datetime"], comparison_datetime)
        # ]
        #        self.strips_log = [log for log in self.strips_log if not comp(log["datetime"], comparison_datetime)]

        return self

    def to_event_handler(self) -> EventHandler:
        """
        Turn the events log into an event handler.

        Returns
        -------
        EventHandler
            The logged events as an EventHandler
        """
        return EventHandler(
            radar_dataframe=self.radar_log_as_df(),
            flight_df=self.flight_log_as_df(),
            clearances_df=self.clearance_log_as_df(),
            coordination_df=self.coordination_log_as_df(),
            sectors_df=self.sectors_log_as_df(),
            incomm_df=self.incomm_log_as_df(),
            aircraft_internals_df=self.aircraft_internals_log_as_df(),
            ac_attribute_update_df=None,  # taken into account via the ac_internals. Exists for convenience.
        )
