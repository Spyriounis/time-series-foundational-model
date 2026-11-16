"""Small parsing helpers shared by the concrete fetchers."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from tsfm.constants import TIME_COLUMN
from tsfm.data.schemas import EVENTS_SCHEMA


def optional_float(value: str | float | None, missing: str | None = None) -> float | None:
    """
    Convert a possibly missing observation to a float.

    Parameters
    ----------
    value : str, float or None
        Raw observation, often a decimal string.
    missing : str or None, default: None
        Marker the source uses for a missing value.

    Returns
    -------
    float or None
        The number, or None if the observation is missing.
    """
    return None if value is None or value == missing else float(value)


def clip(table: pa.Table, start: datetime, end: datetime) -> pa.Table:
    """
    Keep the rows with ``start <= time < end``.

    Parameters
    ----------
    table : pa.Table
        Rows with a time column.
    start : datetime
        Inclusive lower bound, UTC.
    end : datetime
        Exclusive upper bound, UTC.

    Returns
    -------
    pa.Table
        The rows inside the bounds.
    """
    times = table[TIME_COLUMN]
    inside = pc.and_(pc.greater_equal(times, pa.scalar(start)), pc.less(times, pa.scalar(end)))
    return table.filter(inside)


def events_table(times: Sequence[datetime], event_ids: Sequence[str], currencies: Sequence[str]) -> pa.Table:
    """
    Build an events table from parallel sequences.

    Parameters
    ----------
    times : Sequence[datetime]
        Scheduled times, UTC.
    event_ids : Sequence[str]
        Event names.
    currencies : Sequence[str]
        Currency each event concerns.

    Returns
    -------
    pa.Table
        Rows in ``EVENTS_SCHEMA``.
    """
    columns: dict[str, list[Any]] = {
        TIME_COLUMN: list(times),
        "event_id": list(event_ids),
        "currency": list(currencies),
    }
    return pa.table(columns, schema=EVENTS_SCHEMA)
