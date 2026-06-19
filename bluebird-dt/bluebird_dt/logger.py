"""
Logging in BluebirdATC is performed using the standard python logging API. Basic usage has been documented here but
details in the library also apply.

# BluebirdATC as a dependency
BluebirdATC will emit all logs to the logger exposed as part of this module and with the name 'bluebird_dt', but by
default will not be displayed (although the simulator class might save them to file if being used).
Libraries interested in capturing the logs emitted by BluebirdATC should implement handlers and add them to this logger.

>>> import logging
>>> from bluebird_dt import logger
>>> bluebird_dt_logger = logging.getlogger("bluebird_dt")
>>> bluebird_dt_logger.setLevel(logging.DEBUG)
>>> stream_handler = logging.StreamHandler()
>>> stream_handler.setFormatter(logger.CustomFormatter())
>>> bluebird_dt_logger.addHandler(stream_handler)

# Attaching context to the logger

Adding context to the logger is intended to be used within BluebirdATC and documented for the purposes of developers of
BluebirdATC and not its dependents.

Adding context to the logger can be done with the ContextFilter filter, see its respective documentation.

>>> logging_context = ContextFilter({"scenario_name": self.scenario_name, "scenario_category": self.category})
>>> logger.addFilter(logging_context)

"""

import logging
import sys
from typing import ClassVar

from typing_extensions import override


class CustomFormatter(logging.Formatter):
    purple: ClassVar[str] = "\x1b[35;20m"
    green: ClassVar[str] = "\x1b[32;20m"
    yellow: ClassVar[str] = "\x1b[33;20m"
    red: ClassVar[str] = "\x1b[31;20m"
    bold_red: ClassVar[str] = "\x1b[31;1m"
    reset: ClassVar[str] = "\x1b[0m"
    fmt: ClassVar[str] = "[%(filename)s:%(lineno)d] [%(scenario_category)s:%(scenario_name)s] %(timestamp)s %(message)s"
    optional_attributes: ClassVar[list[str]] = [
        "scenario_name",
        "scenario_category",
        "timestamp",
    ]
    LABELS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "DEBUG:   ",
        logging.INFO: "INFO:     ",
        logging.WARNING: "WARNING:  ",
        logging.ERROR: "ERROR:    ",
        logging.CRITICAL: "CRITICAL: ",
    }

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: purple + "DEBUG:   " + reset,
        logging.INFO: green + "INFO:     " + reset,
        logging.WARNING: yellow + "WARNING:  " + reset,
        logging.ERROR: red + "ERROR:    " + reset,
        logging.CRITICAL: bold_red + "CRITICAL: " + reset,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno) if sys.stderr.isatty() else self.LABELS.get(record.levelno)
        formatter = logging.Formatter(log_fmt + self.fmt)

        for optional_attribute in self.optional_attributes:
            if optional_attribute not in record.__dict__:
                setattr(record, optional_attribute, "")

        return formatter.format(record)


class ContextFilter(logging.Filter):
    """
    The context filter is a Filter that appends context to the logger and doesn't do any filtering, but is named as such
    because it is registered as a filter.

    Once registered, all logs emitted with the logger this filter has been added to will consider this context through a
    reference to the original object. Future modifications to the object will be automatically applied.

    Adding new attributes which want to be displayed need modifications in the CustomFormatter type, adding the key to
    the optional attributes, unless the context will always be present, and to the fmt string.
    """

    __context: dict[str, str]

    def __init__(self, context: dict[str, str], name: str = "") -> None:
        super().__init__(name)
        self.__context = context

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        for key, val in self.__context.items():
            setattr(record, key, val)
        return True

    def set(self, key: str, val: str):
        """
        Modify or create the context of a key. If the key has already been defined, it gets replaced.
        Change gets applied on all logs referencing the respective context filter.

        Parameters
        ----------
        key: str
            The key of the entry.
        value: str
            The value of the entry.
        """
        self.__context[key] = val

    def remove(self, key: str) -> str | None:
        """
        Remove the context of a key.
        Change gets applied on all logs referencing the respective context filter.

        Returns
        -------
        The value if the entry already exists, else None.
        """
        if key in self.__context:
            return self.__context.pop(key)

        return None


logger = logging.getLogger(__name__)
