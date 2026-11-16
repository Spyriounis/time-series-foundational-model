"""UTC datetime helpers."""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    """
    Return the current time in UTC.

    Returns
    -------
    datetime
        Timezone-aware now.
    """
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """
    Convert a datetime to UTC, reading a naive one as UTC.

    Parameters
    ----------
    value : datetime
        Naive or aware datetime.

    Returns
    -------
    datetime
        The same instant with ``tzinfo=UTC``.
    """
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def parse_utc(text: str) -> datetime:
    """
    Parse an ISO 8601 timestamp into UTC.

    Parameters
    ----------
    text : str
        Timestamp such as ``2024-01-02T03:04:05.000000000Z`` or ``2024-01-02``.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime, truncated to microseconds.
    """
    return ensure_utc(datetime.fromisoformat(text))


def format_utc(value: datetime) -> str:
    """
    Format a datetime as an RFC 3339 timestamp in UTC.

    Parameters
    ----------
    value : datetime
        Naive or aware datetime.

    Returns
    -------
    str
        Timestamp to the second, such as ``2024-01-02T03:04:05Z``.
    """
    return ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_date(value: datetime) -> str:
    """
    Format a datetime as its UTC calendar date.

    Parameters
    ----------
    value : datetime
        Naive or aware datetime.

    Returns
    -------
    str
        Date such as ``2024-01-02``.
    """
    return ensure_utc(value).date().isoformat()


def day_end(day: date, timezone: str) -> datetime:
    """
    Return the instant a calendar day ends in a time zone.

    Parameters
    ----------
    day : date
        Calendar day.
    timezone : str
        IANA time zone name.

    Returns
    -------
    datetime
        Midnight that starts the next day there, in UTC, daylight saving accounted for.
    """
    return datetime.combine(day + timedelta(days=1), time(), ZoneInfo(timezone)).astimezone(UTC)
