import ast
import calendar
import json
import os
import tarfile
import warnings
from datetime import datetime
from typing import Generic, Literal, TypeVar

import numpy as np
import pandas as pd
from pydantic import ValidationError
from typing_extensions import Self, override

from bluebird_dt.core import Aircraft, Airspace, Fixes, Pos2D, Sector, WindField
from bluebird_dt.events.event_handler import EventHandler, EventHandlerArgs
from bluebird_dt.logger import logger
from bluebird_dt.manager.environment_manager import EnvironmentManager
from bluebird_dt.predictor import Predictor, SimplePredictor
from bluebird_dt.scenario_manager.scenario_manager import (
    ScenarioManager,
    TAircraft,
    TEventHandler,
    TEventHandlerArgs,
    TForecastWindField,
    TSimulator,
    TWindField,
)
from bluebird_dt.scenario_manager.scenario_manager_configs import BluebirdSaveConfig, ReplayScenarioConfig, SaveConfig
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility.logging_utils import read_logs_from_tar
from bluebird_dt.utility.paths import LOG_DIR

TSaveConfig = TypeVar("TSaveConfig", bound=SaveConfig)


class ReplayerFromLogs(
    ScenarioManager[ReplayScenarioConfig], Generic[TAircraft, TEventHandler, TEventHandlerArgs, TSaveConfig]
):
    """
    Construct and manage a scenario from log files.
    """

    event_handler_ignore_flags = EventHandler.IgnoreFlags()
    _config: ReplayScenarioConfig
    typeof_environment_manager: type[EnvironmentManager[TAircraft, TWindField, TForecastWindField]] = EnvironmentManager
    typeof_aircraft: type[TAircraft] = Aircraft
    typeof_eventhandler: type[TEventHandler] = EventHandler
    typeof_eventhandler_args: type[TEventHandlerArgs] = EventHandlerArgs
    typeof_simulator: type[TSimulator] = Simulator
    typeof_saveconfig: type[TSaveConfig] = BluebirdSaveConfig

    def __init__(self, replay_dir_name: str, replay_buffer: tarfile.TarFile):
        """
        Create a ScenarioManager to load data from log files in memory

        Parameters
        ----------
        replay_dir_name: str
            Name of replay directory containing the log files to be used to create the simulation
        replay_buffer: TarFile | None, default is None
            A tarfile object passed by simulator class if the tar file is loaded in memory

        Returns
        -------
        ReplayerFromLogs scenario manager
            ScenarioManager to create scenarios from logs
        """
        self.replay_dir_name = replay_dir_name
        self.replay = replay_buffer

        file = replay_buffer.extractfile("config.json")
        try:
            original_sim_config: TSaveConfig | None = (
                self.typeof_saveconfig.model_validate_json(file.read()) if file else None
            )
            if original_sim_config is None:
                self._config = ReplayScenarioConfig(original_scenario_manager=None)
            else:
                match original_sim_config.scenario:
                    case ReplayScenarioConfig():
                        self._config = original_sim_config.scenario
                    case _:
                        self._config = ReplayScenarioConfig(original_scenario_manager=original_sim_config.scenario)

        except ValidationError:
            logger.warning("Failed to recognise the original scenario manager.")
            self._config = ReplayScenarioConfig(original_scenario_manager=None)

        # load all parquet and json files into a data dict
        # keyed by the filename base (e.g. "radar").
        self.replay_data = read_logs_from_tar(self.replay)
        self.replay_data_cleaned = False

    @classmethod
    def from_replay_dir_path(cls, replay_dir_name: str, replay_dir_path: str = LOG_DIR) -> Self:
        """
        Create a ScenarioManager to load data from reading log files

        Parameters
        ----------
        replay_dir_name: str
            Name of replay directory containing the log files to be used to create the simulation
        replay_dir_path: str, default is the default replay directory
            Path to the directory containing the specific replay directory

        Returns
        -------
        ReplayerFromLogs scenario manager
            ScenarioManager to create scenarios from logs
        """
        full_path = os.path.join(replay_dir_path, replay_dir_name + ".tar.gz")
        if not os.path.exists(full_path):
            raise ValueError(f"File {full_path} doesn't exist")
        return cls(replay_dir_name, tarfile.open(full_path, "r:gz"))  # noqa: SIM115

    @override
    def create_event_handler(self) -> TEventHandler:
        """
        Generate event_handler from the data from the log files

        Returns
        -------
        TEventHandler
        """
        # We have already loaded the log data into self.replay_data
        # just need to clean a few dataframe columns
        self.clean_dataframes()
        # The dict of dataframes should map into an EventHandlerArgs object.
        eh_args = self.typeof_eventhandler_args.from_data(self.replay_data, ignore=self.event_handler_ignore_flags)
        return eh_args.build()

    def clean_dataframes(self):
        """
        Ensure that various columns have correct types.
        """
        # only do this cleaning once
        if self.replay_data_cleaned:
            return
        if "clearances" in self.replay_data:
            self.replay_data["clearances"]["value"] = self.replay_data["clearances"].value.map(
                lambda v: ast.literal_eval(v) if "[" in v else v
            )
            self.replay_data["clearances"]["sector"] = self.replay_data["clearances"].sector.map(lambda v: list(v))
            # Backwards compatibility with old log files
            if "clearance" in self.replay_data["clearances"].columns:
                warnings.warn("The format of replays is deprecated.", DeprecationWarning, stacklevel=2)
                self.replay_data["clearances"]["text_clearance"] = self.replay_data["clearances"]["clearance"]
            if "pilot_response" in self.replay_data["clearances"].columns:
                warnings.warn("The format of replays is deprecated.", DeprecationWarning, stacklevel=2)
                self.replay_data["clearances"]["text_pilot_response"] = self.replay_data["clearances"]["pilot_response"]

        if "sectors" in self.replay_data:
            self.replay_data["sectors"]["sectors_configuration"] = self.replay_data[
                "sectors"
            ].sectors_configuration.map(ast.literal_eval)
        if "ac_internals" in self.replay_data:
            self.replay_data["ac_internals"]["pilot_action_queue"] = self.replay_data["ac_internals"][
                "pilot_action_queue"
            ].map(ast.literal_eval)
            self.replay_data["ac_internals"]["predictor_params"] = self.replay_data["ac_internals"][
                "predictor_params"
            ].map(
                lambda x: eval(x, {"np": np, "__builtins__": {}})  # Temporary hot fix to handle (np.float(x))
            )
            self.replay_data["ac_internals"]["percentile_rank_dict"] = self.replay_data["ac_internals"][
                "percentile_rank_dict"
            ].map(ast.literal_eval)
            self.replay_data["ac_internals"]["operation_params"] = self.replay_data["ac_internals"][
                "operation_params"
            ].map(ast.literal_eval)

            # Set all aircraft internals such that aircraft.simulated = False as this is a replay by default
            self.replay_data["ac_internals"]["simulated"] = False
        self.replay_data_cleaned = True

    def create_env_manager(
        self,
        predictor: Predictor,
        log_filename: str | None = None,
        vertical_buffer_distance: float | int | None = None,
        lateral_buffer_distance: float | int | None = None,
    ) -> EnvironmentManager[TAircraft, TWindField, TForecastWindField]:
        """
        Create an EnvironmentManager to replay from log files.

        Parameters
        ----------
        predictor: Predictor, optional
            Predictor to simulate aircraft trajectories
        log_filename: str or None
            Name of file logs will be saved to. If None, defaults to datetime logger created.
        vertical_buffer_distance: int or float, optional
            Distance to expand airspace vertical boundary by - UoM: FL
            If not None, will override the value read from logs
        lateral_buffer_distance: int or float, optional
            Distance to expand airspace lateral boundary by - UoM: NMI
            If not None, will override the value read from logs

        Returns
        ----------
        EnvironmentManager
            EnvironmentManager to replay from log files
        """

        logger.info(
            f"""
===================================================================
Loading Replay Scenario:- Sectors: {self.replay_dir_name}
===================================================================
        """
        )

        if predictor is None:
            predictor = SimplePredictor(dt=1.0, fix_proximity_threshold=2.0)

        fixes = self.replay_data["fixes"]
        fixes["positions"] = pd.Series(
            [Pos2D(lat, lon) for lat, lon in zip(fixes["lat"], fixes["lon"], strict=False)], index=fixes.index
        )
        fixes_dict = dict(zip(fixes["fix"], fixes["positions"], strict=False))
        visibility_dict = dict(zip(fixes["fix"], fixes["visibility"], strict=False))
        fixes = Fixes(fixes_dict, visibility_dict)

        # load sectors
        all_sectors = {}
        for sector_name, sector_json in self.replay_data["individual_sectors"].items():
            sector = Sector.from_json(json.dumps(sector_json))
            all_sectors[sector_name] = sector

        if "config" in self.replay_data and len(self.replay_data["config"]) > 0:
            config = self.typeof_saveconfig.model_validate(self.replay_data["config"])
            penumbra_lat = config.environment_manager.penumbra_latitude
            penumbra_fl = config.environment_manager.penumbra_flight_level
            self.projection_centre = (
                tuple(config.simulator.projection_centre) if config.simulator.projection_centre else None
            )
        else:
            raise FileNotFoundError("Failed to find config file, this log might be corrupt or not a BluebirdDT log.")

        # load wind field and forecast -- will be None if not present
        wind_field = WindField.from_dict(self.replay_data["wind"])
        forecast = WindField.from_dict(self.replay_data["forecast"])

        airspace = Airspace(sectors=all_sectors, fixes=fixes)

        # create event handler from the events list
        event_handler = self.create_event_handler()

        start_datetime = event_handler.radar_df.index.min()

        # override logged penumbra values if set as parameter. Give warning in this case
        if vertical_buffer_distance is not None:
            logger.debug(
                f"vertical_buffer_distance set in replay as {vertical_buffer_distance}."
                f"Overriding the logged value of {penumbra_fl}",
                stacklevel=2,
            )
            penumbra_fl = vertical_buffer_distance

        if lateral_buffer_distance is not None:
            logger.debug(
                f"lateral_buffer_distance set in replay ({lateral_buffer_distance})."
                f"Overriding logged value of {penumbra_lat}",
                stacklevel=2,
            )
            penumbra_lat = lateral_buffer_distance

        em = self.typeof_environment_manager(
            airspace=airspace,
            event_handler=event_handler,
            predictor=predictor,
            time=calendar.timegm(start_datetime.timetuple()),
            penumbra_fl=int(penumbra_fl),
            penumbra_lat=penumbra_lat,
            wind_field=wind_field,
            forecast_wind_field=forecast,
            log_filename=log_filename,
        )
        # initialise the environment using data in the event_handler
        logger.info("Initialising Environment using Event Handler")
        # initialise, but turn off logging until after updating the simulated status of the aircraft
        em.initialise_env_with_event_handler(log=False)
        logger.info("Completed Initialising Environment using Event Handler")

        # now log the environment
        em.event_logger.log_environment(em.environment)
        em.event_logger.log_clearances(em.environment.time, em._actions_to_issue)
        return em

    def get_projection_centre(self) -> tuple[float, float]:
        """
        Get the projection centre from the config log file.

        Returns
        -------
        Tuple
            (projection longitude, projection latitude)
        """
        # load environment manager configuration
        config = self.replay_data["config"]
        # return the projection centre as a tuple. Json presents it as a list.
        return tuple(config["simulator"]["projection_centre"])

    @override
    def update(
        self, env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField]
    ) -> EnvironmentManager[TAircraft, TWindField, TForecastWindField]:
        """
        Optionally update the environment or coordinations.

        Intended to allow scenario managers the option to dynamically update the environment depending on
        the state at any time


        Parameters
        ----------
        env_manager: EnvironmentManager
            An environment manager containing the environment and coordinations

        Returns
        -------
        EnvironmentManager
        """
        # Replay scenario manager does not alter the scenario outside the event handler
        return env_manager

    def __del__(self):
        """
        Destructor to close replay tar file properly
        """
        if hasattr(self, "replay") and self.replay:
            self.replay.close()

    @override
    def config(self) -> ReplayScenarioConfig:
        return self._config

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
    ) -> TSimulator:
        """Set up a replay scenario

        Parameters
        ----------
        scenario_name: str
            The scenario name - in this case the name of the log to replay
        replay_buffer: TarFile | None, default is None
                A tarfile object passed by simulator class if the tar file is loaded in memory
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
            Aircraft Trajectory prediction used to evolve Aircraft. If None, then LinearPredictor will be created.

        Returns
        -------
        TSimulator
            A fully configured simulator instance
        """

        scenario_manager = cls.from_replay_dir_path(replay_dir_name=scenario_name, replay_dir_path=LOG_DIR)

        sim = scenario_manager.to_simulator(
            scenario_name=scenario_name,
            category="Replay",
            use_wind=use_wind,
            use_forecast=use_forecast,
            autosave=autosave,
            attach_context_to_logger=attach_context_to_logger,
            save_log_to_file=save_log_to_file,
            log_filename=log_filename,
            predictor=predictor,
        )

        sim.scenario_manager.replay.close()

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
        simulated_sectors: list[str] | Literal["ALL"] = "ALL",
    ) -> TSimulator:
        """
        Create a Simulator instance for Replay scenarios.

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

        return self.typeof_simulator(
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
