"""Structured logging setup."""

import logging
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: int = logging.INFO, *, json: bool = False) -> None:
    """
    Configure structlog once per process.

    Parameters
    ----------
    level : int, default: logging.INFO
        Minimum level emitted.
    json : bool, default: False
        Emit JSON lines instead of the console format.
    """
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """
    Return a logger bound to a module name.

    Parameters
    ----------
    name : str
        Usually ``__name__``.

    Returns
    -------
    FilteringBoundLogger
        Logger carrying ``module=name`` on every event.
    """
    return cast(FilteringBoundLogger, structlog.get_logger(module=name))
