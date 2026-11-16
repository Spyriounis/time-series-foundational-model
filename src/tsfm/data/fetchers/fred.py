"""FRED fetchers: macro and market series, and the economic release calendar."""

import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from itertools import batched
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa

from tsfm.constants import (
    FRED_BASE_URL,
    FRED_DELAY_QUANTILE,
    FRED_INITIAL_RELEASE_OUTPUT,
    FRED_MISSING_VALUE,
    FRED_PAGE_SIZE,
    FRED_RELEASE_CURRENCY,
    FRED_RELEASE_PAGE_SIZE,
    FRED_RELEASE_TIMEZONE,
    FRED_RELEASES,
    FRED_REQUESTS_PER_SECOND,
    FRED_SERIES,
    FRED_VINTAGE_PAGE_SIZE,
    FRED_VINTAGES_PER_REQUEST,
    PUBLICATION_TIMEZONE,
    PUBLISHED_COLUMN,
    TIME_COLUMN,
)
from tsfm.data.fetchers.base import APIDataFetcher, SeriesFetcher, observations_table, with_publication
from tsfm.data.fetchers.parsing import clip, events_table, optional_float
from tsfm.data.storage import DataKey
from tsfm.enums import Dataset, PullMode, Source
from tsfm.utils.http import HttpClient, Params
from tsfm.utils.time import day_end, format_date, parse_utc

_OBSERVATIONS_URL = f"{FRED_BASE_URL}/series/observations"
_VINTAGE_DATES_URL = f"{FRED_BASE_URL}/series/vintagedates"
_RELEASE_DATES_URL = f"{FRED_BASE_URL}/release/dates"


class FredSeriesFetcher(SeriesFetcher):
    """
    Pull the series in ``FRED_SERIES``.

    Examples
    --------
    >>> FredSeriesFetcher(Settings()).pull(store, only=["DFF"])  # doctest: +SKIP
    """

    source = Source.FRED
    requests_per_second = FRED_REQUESTS_PER_SECOND

    def series_ids(self) -> list[str]:
        """
        List the configured FRED series.

        Returns
        -------
        list[str]
            FRED series ids.
        """
        return list(FRED_SERIES)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Download the first release of each observation, with its publication time.

        Values first released inside FRED's archive of past releases (ALFRED) carry the
        recorded date. Older values come from the current release, with a publication time
        estimated from the series' archived delays.

        Parameters
        ----------
        series_id : str
            FRED series id.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        pa.Table
            Rows in ``OBSERVATIONS_SCHEMA`` with the ``published`` and ``estimated`` columns.

        Raises
        ------
        ValueError
            If values before the archive need an estimate and no archived release exists.
        """
        query: Params = {"series_id": series_id, "api_key": self.settings.secret("fred_api_key"), "file_type": "json"}
        archive_query: Params = {**query, "limit": FRED_VINTAGE_PAGE_SIZE}
        archive = _fetch_all(self.client, _VINTAGE_DATES_URL, archive_query, "vintage_dates")
        first = format_date(start)
        releases: dict[str, dict[str, Any]] = {}
        # A value is published on or after its date, so earlier archive dates hold none of its first releases.
        for chunk in batched([day for day in archive if day >= first], FRED_VINTAGES_PER_REQUEST):
            params: Params = {
                **query,
                "observation_start": first,
                "realtime_start": chunk[0],
                "realtime_end": chunk[-1],
                "output_type": FRED_INITIAL_RELEASE_OUTPUT,
                "limit": FRED_PAGE_SIZE,
            }
            for record in _fetch_all(self.client, _OBSERVATIONS_URL, params, "observations"):
                releases.setdefault(record["date"], record)
        records = [{**record, PUBLISHED_COLUMN: _day_end(record["realtime_start"])} for record in releases.values()]
        recorded = len(records)
        if archive and first < archive[0]:
            delay = _archived_delay(list(releases.values()), series_id)
            params = {**query, "observation_start": first, "observation_end": archive[0], "limit": FRED_PAGE_SIZE}
            for record in _fetch_all(self.client, _OBSERVATIONS_URL, params, "observations"):
                if record["date"] < archive[0] and record["date"] not in releases:
                    published = day_end(date.fromisoformat(record["date"]) + delay, PUBLICATION_TIMEZONE)
                    records.append({**record, PUBLISHED_COLUMN: published})
        estimated = [index >= recorded for index in range(len(records))]
        table = with_publication(_parse_observations(records), [r[PUBLISHED_COLUMN] for r in records], estimated)
        return clip(table.sort_by(TIME_COLUMN), start, end)


class FredReleaseCalendarFetcher(APIDataFetcher):
    """
    Pull past and scheduled dates of the releases in ``FRED_RELEASES``.

    Examples
    --------
    >>> FredReleaseCalendarFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    source = Source.FRED
    dataset = Dataset.EVENTS
    requests_per_second = FRED_REQUESTS_PER_SECOND
    # A rescheduled release must lose its old date.
    mode = PullMode.REPLACE

    def list_keys(self) -> list[DataKey]:
        """
        List the single calendar table.

        Returns
        -------
        list[DataKey]
            One table address.
        """
        return [DataKey(self.dataset, (self.source,))]

    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Download the whole calendar, one release at a time.

        The bounds are not applied: scheduled dates lie beyond ``end``.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Unused.
        end : datetime
            Unused.

        Yields
        ------
        pa.Table
            Every date of one release, in ``EVENTS_SCHEMA``.
        """
        for event_id, (release_id, local_time) in FRED_RELEASES.items():
            params: Params = {
                "release_id": release_id,
                "api_key": self.settings.secret("fred_api_key"),
                "file_type": "json",
                # Without this flag FRED omits dates that are scheduled but not yet published.
                "include_release_dates_with_no_data": "true",
                "limit": FRED_RELEASE_PAGE_SIZE,
                "sort_order": "asc",
            }
            release_dates = _fetch_all(self.client, _RELEASE_DATES_URL, params, "release_dates")
            yield _parse_release_dates(release_dates, event_id, local_time)


def _fetch_all(client: HttpClient, url: str, params: Params, field: str) -> list[Any]:
    """
    Collect every record of an offset-paged FRED endpoint.

    Parameters
    ----------
    client : HttpClient
        Client to request with.
    url : str
        Endpoint.
    params : Params
        Query without the offset.
    field : str
        Key of the record list in the response.

    Returns
    -------
    list[Any]
        Records of all pages, in order: dicts, or date strings for archive dates.
    """
    records: list[Any] = []
    while True:
        payload = client.get_json(url, {**params, "offset": len(records)})
        page = payload[field]
        records += page
        if not page or len(records) >= payload["count"]:
            return records


def _parse_observations(observations: list[dict[str, Any]]) -> pa.Table:
    """
    Convert FRED observations to a series.

    Parameters
    ----------
    observations : list[dict[str, Any]]
        Records with a ``date`` and a string ``value``.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``; FRED's missing marker becomes null.
    """
    times = [parse_utc(observation["date"]) for observation in observations]
    values = [optional_float(observation["value"], FRED_MISSING_VALUE) for observation in observations]
    return observations_table(times, values)


def _day_end(day: str) -> datetime:
    """
    Return when a FRED publication date counts as public.

    Parameters
    ----------
    day : str
        Date, ``YYYY-MM-DD``.

    Returns
    -------
    datetime
        The end of that day in ``PUBLICATION_TIMEZONE``, UTC.
    """
    return day_end(date.fromisoformat(day), PUBLICATION_TIMEZONE)


def _archived_delay(releases: list[dict[str, Any]], series_id: str) -> timedelta:
    """
    Estimate a series' publication delay from its archived first releases.

    Parameters
    ----------
    releases : list[dict[str, Any]]
        First releases, each with a ``date`` and the ``realtime_start`` it was published on.
    series_id : str
        Series id, for the error message.

    Returns
    -------
    timedelta
        The ``FRED_DELAY_QUANTILE`` quantile of the delays, in whole days.

    Raises
    ------
    ValueError
        If there is no archived release to estimate from.
    """
    delays = sorted((date.fromisoformat(r["realtime_start"]) - date.fromisoformat(r["date"])).days for r in releases)
    if not delays:
        raise ValueError(f"{series_id}: no archived release to estimate the publication delay of older values")
    return timedelta(days=delays[math.ceil(FRED_DELAY_QUANTILE * (len(delays) - 1))])


def _release_time(day: str, local_time: str) -> datetime:
    """
    Convert a release date and its local clock time to UTC.

    Parameters
    ----------
    day : str
        Release date, ``YYYY-MM-DD``.
    local_time : str
        Clock time in ``FRED_RELEASE_TIMEZONE``, ``HH:MM``.

    Returns
    -------
    datetime
        The release instant in UTC, daylight saving accounted for.
    """
    local = datetime.combine(date.fromisoformat(day), time.fromisoformat(local_time), ZoneInfo(FRED_RELEASE_TIMEZONE))
    return local.astimezone(UTC)


def _parse_release_dates(release_dates: list[dict[str, Any]], event_id: str, local_time: str) -> pa.Table:
    """
    Convert the dates of one release to events.

    Parameters
    ----------
    release_dates : list[dict[str, Any]]
        Records with a ``date``.
    event_id : str
        Event name of the release.
    local_time : str
        Clock time of the release in ``FRED_RELEASE_TIMEZONE``, ``HH:MM``.

    Returns
    -------
    pa.Table
        Rows in ``EVENTS_SCHEMA``.
    """
    times = [_release_time(record["date"], local_time) for record in release_dates]
    return events_table(times, [event_id] * len(times), [FRED_RELEASE_CURRENCY] * len(times))
