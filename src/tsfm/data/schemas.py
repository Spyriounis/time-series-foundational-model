"""Arrow schemas of the stored datasets."""

from typing import Any

import pyarrow as pa

from tsfm.constants import ESTIMATED_COLUMN, FIRST_SEEN_COLUMN, PUBLISHED_COLUMN, TIME_COLUMN
from tsfm.enums import Dataset

TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")


def _field(name: str, kind: pa.DataType, nullable: bool = True) -> "pa.Field[Any]":
    """
    Build a schema field whose static type does not depend on its data type.

    Parameters
    ----------
    name : str
        Column name.
    kind : pa.DataType
        Arrow type of the column.
    nullable : bool, default: True
        Whether the column may hold nulls.

    Returns
    -------
    pa.Field
        The field.
    """
    return pa.field(name, kind, nullable=nullable)


BARS_SCHEMA = pa.schema(
    [
        _field(TIME_COLUMN, TIMESTAMP_TYPE, nullable=False),  # bar open
        _field("open", pa.float64()),
        _field("high", pa.float64()),
        _field("low", pa.float64()),
        _field("close", pa.float64()),
        _field("spread", pa.float32()),  # ask close minus bid close
        _field("volume", pa.float64()),
    ]
)

# What a series parser returns, before its publication times are added.
_OBSERVATION_FIELDS = [
    _field(TIME_COLUMN, TIMESTAMP_TYPE, nullable=False),  # reference period, not publication
    _field("value", pa.float64()),
]
OBSERVATIONS_SCHEMA = pa.schema(_OBSERVATION_FIELDS)

# One row per reference period, holding its first release.
SERIES_SCHEMA = pa.schema(
    [
        *_OBSERVATION_FIELDS,
        _field(PUBLISHED_COLUMN, TIMESTAMP_TYPE, nullable=False),
        _field(ESTIMATED_COLUMN, pa.bool_(), nullable=False),
        _field(FIRST_SEEN_COLUMN, TIMESTAMP_TYPE, nullable=False),
    ]
)

EVENTS_SCHEMA = pa.schema(
    [
        _field(TIME_COLUMN, TIMESTAMP_TYPE, nullable=False),  # scheduled time
        _field("event_id", pa.string(), nullable=False),
        _field("currency", pa.string()),
    ]
)

BOOKS_SCHEMA = pa.schema(
    [
        _field(TIME_COLUMN, TIMESTAMP_TYPE, nullable=False),  # snapshot time
        _field("price", pa.float64()),
        _field("bucket_width", pa.float64()),
        _field("bucket_price", pa.list_(pa.float64())),
        _field("long_pct", pa.list_(pa.float32())),
        _field("short_pct", pa.list_(pa.float32())),
    ]
)

SYNTHETIC_SCHEMA = pa.schema(
    [
        _field("task_id", pa.int64(), nullable=False),
        _field("row", pa.int16(), nullable=False),
        _field("kind", pa.string(), nullable=False),
        _field("values", pa.list_(pa.float32()), nullable=False),
    ]
)

SCHEMAS: dict[Dataset, pa.Schema] = {
    Dataset.BARS: BARS_SCHEMA,
    Dataset.SERIES: SERIES_SCHEMA,
    Dataset.EVENTS: EVENTS_SCHEMA,
    Dataset.BOOKS: BOOKS_SCHEMA,
    Dataset.SYNTHETIC: SYNTHETIC_SCHEMA,
}


def conform(table: pa.Table, dataset: Dataset) -> pa.Table:
    """
    Select and cast a table's columns to a dataset's schema.

    Parameters
    ----------
    table : pa.Table
        Table holding at least the schema's columns.
    dataset : Dataset
        Target dataset.

    Returns
    -------
    pa.Table
        Table with exactly the schema's columns, order and types.

    Raises
    ------
    KeyError
        If a schema column is absent.
    """
    schema = SCHEMAS[dataset]
    return table.select(schema.names).cast(schema)


def empty_table(dataset: Dataset) -> pa.Table:
    """
    Build an empty table of a dataset's schema.

    Parameters
    ----------
    dataset : Dataset
        Target dataset.

    Returns
    -------
    pa.Table
        Zero-row table.
    """
    return SCHEMAS[dataset].empty_table()
