from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from bluebird_dt.core import WindField
from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.events.event_handler import EventHandler, EventHandlerArgs
from bluebird_dt.events.event_logger import EventLogger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.simulator.simulator import Simulator

# Define generic types here - will be used for scenario managers
# that may have other scenario managers inheriting from them.
TConfig = TypeVar("TConfig", bound=BaseModel)
TAircraft = TypeVar("TAircraft", bound=Aircraft)
TWindField = TypeVar("TWindField", bound=WindField)
TForecastWindField = TypeVar("TForecastWindField", bound=WindField)
TEventLogger = TypeVar("TEventLogger", bound=EventLogger)
TEventHandler = TypeVar("TEventHandler", bound=EventHandler[Aircraft])
TEventHandlerArgs = TypeVar("TEventHandlerArgs", bound=EventHandlerArgs)
TSimulator = TypeVar("TSimulator", bound=Simulator)


class ScenarioManager(ABC, Generic[TConfig]):
    """
    Scenario Manager
    """

    @abstractmethod
    def create_event_handler(self) -> EventHandler:
        """
        Generate an event handler for the scenario

        Returns
        -------
        EventHandler
        """
        return EventHandler()

    def update(self, env_manager: EnvironmentManager) -> EnvironmentManager:
        """
        Optionally update the environment or coordinations.

        Intended to allow scenario managers the option to dynamically update the environment depending on
        the state at any time.


        Parameters
        ----------
        env_manager: EnvironmentManager
            An environment manager containing the environment and coordinations

        Returns
        -------
        EnvironmentManager
        """
        return env_manager

    @abstractmethod
    def config(self) -> TConfig:
        """
        Obtain the configuration this instance of the specific scenario manager.

        Returns
        -------
        TConfig
            Object reflecting the current configuration of the specific scenario manager, of the specific
            type of the scenario manager.
        """
        pass

    def close(self):
        pass
