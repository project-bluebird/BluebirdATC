import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime

import aiofiles
from pydantic import BaseModel

from bluebird_dt.logger import logger
from bluebird_dt.simulator.simconfig import SaveConfig
from bluebird_dt.utility.paths import LOG_DIR


@dataclass(slots=True, frozen=True)
class SaveData:
    log_buffer: bytes
    log_name: str
    save_config: SaveConfig
    end_save: bool = False


class SaveStatus(BaseModel):
    is_saving: bool = False
    last_save_task_success: bool | None = None
    last_save_task_save_simtime: datetime | None = None

    def set_saving(self):
        self.is_saving = True

    def set_success(self, simtime: datetime | None):
        self.is_saving = False
        self.last_save_task_success = True
        self.last_save_task_save_simtime = simtime

    def set_fail(self, simtime: datetime | None):
        self.is_saving = False
        self.last_save_task_success = False
        self.last_save_task_save_simtime = simtime


@dataclass
class Saver:
    """
    A helper class to handle saving logic for simulator
    """

    save_config: SaveConfig
    save_status: SaveStatus = field(default_factory=SaveStatus)
    current_save_task: asyncio.Task | None = None
    next_save_data: SaveData | None = None

    async def dispatch(self, save_data: SaveData, force: bool = False) -> bool:
        """
        Queue or immediately execute a save operation. When force is true, any pending save request is discarded
        and this method execute and await the new save operation before returning its result.

        Parameters
        ----------
        save_data: SaveData
            A populated SaveData object.
        force: bool
            If True, perform the save immediately and wait for completion.
            If False, queue the save request for asynchronous processing.

        Returns
        -------
        bool
            If the queue is successful. If force, if the save task is successful.
        """
        if force:
            # Force current task to finish and await a new task
            self.next_save_data = None
            if self.current_save_task is not None and not self.current_save_task.done():
                logger.info("saver force dispatch - awaiting previous save task to finish")
                await self.current_save_task
            await self.async_save(save_data)
            self.update()
            logger.info("saver force dispatch - save task finished")
            return self.save_status.last_save_task_success or False

        # Put and override new save data in the next_save_data
        logger.info("saver dispatch - putting savedata to queue")
        self.next_save_data = save_data
        self.update()
        return True

    def update(self):
        """
        Update the saver queue.

        Description
        -----------
        This method manages execution of asynchronous save operations, ensuring that only one save task runs at a time.
        If no save is currently active and pending save data exists, a new save task is started immediately.
        When the active save task completes, a new save is started using latest pending save data or the queue is reset
        to idle state by clearing current_save_task to None if there is no pending save data.
        """
        if self.current_save_task is None and self.next_save_data is not None:
            logger.info("saver update - moving next_save_data to current_save_task")
            self.current_save_task = asyncio.create_task(self.async_save(self.next_save_data))
            self.next_save_data = None

        if self.current_save_task is not None and self.current_save_task.done():
            if self.next_save_data is not None:
                logger.info("saver update - moving next_save_data to current_save_task")
                self.current_save_task = asyncio.create_task(self.async_save(self.next_save_data))
                self.next_save_data = None
            else:
                logger.info("saver update - setting current_save_task to None as all save tasks finishes")
                self.current_save_task = None

    async def close(self):
        """
        Closes the saver queue by ensureing all save tasks are finished
        """
        logger.info("closing saver")
        if self.next_save_data is not None:
            await self.dispatch(self.next_save_data, force=True)
        elif self.current_save_task is not None:
            logger.info("awaiting current save task to finish")
            await self.current_save_task
        self.current_save_task = None
        self.next_save_data = None

    def should_save(self, autosave: bool, current_simtime: datetime, current_realtime: datetime) -> bool:
        """
        Determine if a savedata object should be generated

        Parameters
        ----------
        autosave: bool
            The autosave mode.
        current_simtime: datetime
            The simulator datetime from the environment.
        current_realtime: datetime
            The real datetime from the host system clock in UTC.

        Returns
        -------
        bool
        """
        return not (
            # Check if autosave is enabled
            autosave
            and self.save_config.autosave_interval is not None
            # Check if not enough time has passed with both simtime and realtime for pausing/fast time etc.
            and self.save_config.save_simtime is not None
            and self.save_config.save_realtime is not None
            and (
                current_simtime - self.save_config.save_simtime < self.save_config.autosave_interval
                or current_realtime - self.save_config.save_realtime < self.save_config.autosave_interval
            )
        )

    def should_chunk(self, current_simtime: datetime, current_realtime: datetime) -> bool:
        """
        Determine if should chunk

        Parameters
        ----------
        current_simtime: datetime
            The simulator datetime from the environment.
        current_realtime: datetime
            The real datetime from the host system clock in UTC.

        Returns
        -------
        bool
        """
        if (
            # Check if chunking is enabled
            self.save_config.save_chunk_interval is not None
            and self.save_config.save_chunk_id is not None
            # Check if enough time has passed with both simtime and realtime for pausing/fast time etc.
            and self.save_config.chunk_start_simtime is not None
            and self.save_config.chunk_start_realtime is not None
            and current_simtime - self.save_config.chunk_start_simtime >= self.save_config.save_chunk_interval
            and current_realtime - self.save_config.chunk_start_realtime >= self.save_config.save_chunk_interval
        ):
            if (
                # check if there is no current_save_task and that last async save is completed and successful
                (
                    self.current_save_task is None
                    or (self.current_save_task is not None and self.current_save_task.done())
                )
                and self.save_status.last_save_task_success is True
                # check if the last async save task simtime is the same as the just finished current_save_task simtime,
                # this guarantee that the chunking is only executed if there is no missing data in between
                and self.save_status.last_save_task_save_simtime is not None
                and self.save_config.save_simtime is not None
                and self.save_status.last_save_task_save_simtime == self.save_config.save_simtime
            ):
                return True
            logger.warning("skipping chunking logger and handler")
        return False

    def save(self, save_data: SaveData):
        """
        Process and save save_data synchronously. Raise exception if failed.

        Parameters
        ----------
        save_data: SaveData
            A SaveData object that contains necessary information for saving
        """
        self.save_status.set_saving()
        log_path = os.path.join(
            LOG_DIR,
            save_data.log_name
            + (
                f"/{save_data.save_config.save_chunk_id}.tar.gz"
                if save_data.save_config.save_chunk_id is not None
                else ".tar.gz"
            ),
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "wb") as tar:
            tar.write(save_data.log_buffer)
        logger.info(f"sync save task - log saved to {log_path}")
        self.save_status.set_success(save_data.save_config.save_simtime)

    async def async_save(self, save_data: SaveData) -> None:
        """
        Process and save save_data asynchronously.

        Parameters
        ----------
        save_data: SaveData
            A SaveData object that contains necessary information for saving
        """
        self.save_status.set_saving()
        try:
            log_path = os.path.join(
                LOG_DIR,
                save_data.log_name
                + (
                    f"/{save_data.save_config.save_chunk_id}.tar.gz"
                    if save_data.save_config.save_chunk_id is not None
                    else ".tar.gz"
                ),
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            async with aiofiles.open(log_path, "wb") as tar:
                await tar.write(save_data.log_buffer)
            logger.info(f"async save task - log saved to {log_path}")
            self.save_status.set_success(save_data.save_config.save_simtime)
        except Exception as e:
            logger.error(f"async save task - log failed to saved to {log_path} with exception {e}")
            self.save_status.set_fail(save_data.save_config.save_simtime)
