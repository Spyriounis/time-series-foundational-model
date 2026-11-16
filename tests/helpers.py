from datetime import UTC, datetime

import pyarrow as pa

from tsfm.constants import TIME_COLUMN
from tsfm.data.schemas import BARS_SCHEMA


def utc(year: int, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0) -> datetime:
    """Shorthand for a UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def bars(times: list[datetime], close: float = 1.0) -> pa.Table:
    """A bars table with constant prices at the given times."""
    n = len(times)
    columns = {name: [close] * n for name in ("open", "high", "low", "close", "spread", "volume")}
    return pa.table({TIME_COLUMN: times, **columns}, schema=BARS_SCHEMA)
