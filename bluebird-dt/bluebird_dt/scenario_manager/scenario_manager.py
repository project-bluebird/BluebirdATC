from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from bluebird_dt.events.event_handler import EventHandler
from bluebird_dt.manager import EnvironmentManager

TConfig = TypeVar("TConfig", bound=BaseModel)


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
