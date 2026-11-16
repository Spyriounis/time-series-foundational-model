"""Abstract fetcher interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Sequence
from datetime import datetime, timedelta
from functools import cached_property
from itertools import product
from typing import ClassVar, cast

import pyarrow as pa

from tsfm.constants import (
    ESTIMATED_COLUMN,
    FIRST_SEEN_COLUMN,
    GRANULARITIES,
    HISTORY_START,
    PUBLICATION_TIMEZONE,
    PUBLISHED_COLUMN,
    PULL_MAX_CONSECUTIVE_FAILURES,
    TIME_COLUMN,
)
from tsfm.data.schemas import OBSERVATIONS_SCHEMA, TIMESTAMP_TYPE
from tsfm.data.storage import DataKey, ParquetStore
from tsfm.enums import Dataset, Granularity, PullMode, Source
from tsfm.log import get_logger
from tsfm.settings import MissingSecretError, Settings
from tsfm.utils.http import HttpClient
from tsfm.utils.time import day_end, ensure_utc, utc_now

logger = get_logger(__name__)


class PullError(RuntimeError):
    """
    Some tables of a pull failed; the others were stored.

    Parameters
    ----------
    rows : int
        Rows written before and between the failures.
    failed : list[DataKey]
        Tables that failed.
    """

    def __init__(self, rows: int, failed: list[DataKey]) -> None:
        super().__init__(f"{len(failed)} table(s) failed: {[key.partitions for key in failed]}")
        self.rows = rows
        self.failed = failed


class APIDataFetcher(ABC):
    """
    Pull one source into the store.

    A subclass names its tables (`list_keys`) and downloads one of them in pages
    (`fetch`); `pull` adds resuming, storage and logging.

    Parameters
    ----------
    settings : Settings
        Credentials and data location.
    client : HttpClient or None, default: None
        Client override; tests pass one built on a mock transport.

    Attributes
    ----------
    source : Source
        Origin written to the store's ``source`` partition.
    dataset : Dataset
        Kind of table the fetcher produces.
    requests_per_second : float
        Rate limit of the default client.
    mode : PullMode
        How a stored table is brought up to date.
    """

    source: ClassVar[Source]
    dataset: ClassVar[Dataset]
    requests_per_second: ClassVar[float] = 1.0
    mode: ClassVar[PullMode] = PullMode.RESUME

    def __init__(self, settings: Settings, client: HttpClient | None = None) -> None:
        self.settings = settings
        self._client = client

    @cached_property
    def client(self) -> HttpClient:
        """
        Return the HTTP client, built on first use.

        Returns
        -------
        HttpClient
            Rate-limited client carrying the source's headers.
        """
        return self._client or HttpClient(self.requests_per_second, self.headers())

    def headers(self) -> dict[str, str]:
        """
        Return headers sent with every request, e.g. authorisation.

        Returns
        -------
        dict[str, str]
            Empty unless a subclass overrides it.
        """
        return {}

    @abstractmethod
    def list_keys(self) -> list[DataKey]:
        """
        List every table this fetcher can pull.

        Returns
        -------
        list[DataKey]
            Table addresses.
        """

    @abstractmethod
    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Download one table for ``start <= time < end``, page by page.

        Parameters
        ----------
        key : DataKey
            Table address, one of `list_keys`.
        start : datetime
            Inclusive lower bound, UTC.
        end : datetime
            Exclusive upper bound, UTC.

        Yields
        ------
        pa.Table
            A page of rows in the dataset's schema.
        """

    def pull(
        self,
        store: ParquetStore,
        start: datetime | None = None,
        end: datetime | None = None,
        only: Collection[str] | None = None,
    ) -> int:
        """
        Pull every selected table into the store, carrying on past a table that fails.

        Parameters
        ----------
        store : ParquetStore
            Destination.
        start : datetime or None, default: None
            Lower bound; defaults to the start of history.
        end : datetime or None, default: None
            Upper bound; defaults to now.
        only : Collection[str] or None, default: None
            Instrument or series names to keep; ignored by single-table fetchers.

        Returns
        -------
        int
            Rows written across all tables.

        Raises
        ------
        PullError
            If any table failed, after the others were attempted.
        MissingSecretError
            If the source's credential is unset.
        """
        begin = ensure_utc(start) if start else HISTORY_START
        finish = ensure_utc(end) if end else utc_now()
        keys = [key for key in self.list_keys() if only is None or len(key.partitions) < 2 or key.partitions[1] in only]
        logger.info("pull_started", source=self.source, dataset=self.dataset, tables=len(keys))
        rows, streak = 0, 0
        failed: list[DataKey] = []
        for key in keys:
            try:
                rows += self._pull_key(store, key, begin, finish)
                streak = 0
            except MissingSecretError:
                raise
            except Exception as error:  # one bad table must not end an unattended pull
                logger.error("table_failed", partitions=key.partitions, error=f"{type(error).__name__}: {error}")
                failed.append(key)
                streak += 1
                if streak == PULL_MAX_CONSECUTIVE_FAILURES:
                    logger.error("pull_abandoned", source=self.source, untried=len(keys) - keys.index(key) - 1)
                    break
        if failed:
            raise PullError(rows, failed)
        return rows

    def _pull_key(self, store: ParquetStore, key: DataKey, start: datetime, end: datetime) -> int:
        """
        Bring one table up to date in the fetcher's mode.

        Parameters
        ----------
        store : ParquetStore
            Destination.
        key : DataKey
            Table address.
        start : datetime
            Lower bound.
        end : datetime
            Upper bound.

        Returns
        -------
        int
            Rows written.
        """
        if self.mode is PullMode.REPLACE:
            pages = list(self.fetch(key, start, end))
            # An empty answer is more likely a fault than a source that lost all its rows.
            rows = store.replace(key, pa.concat_tables(pages)) if sum(page.num_rows for page in pages) else 0
        else:
            rows = self._resume(store, key, start, end)
        logger.info("table_pulled", source=self.source, partitions=key.partitions, rows=rows)
        return rows

    def _resume(self, store: ParquetStore, key: DataKey, start: datetime, end: datetime) -> int:
        """
        Pull what a table lacks: history before its recorded lower bound, then rows after its newest.

        The stored span stays gap-free, so the newest row is always a safe point to resume from.

        Parameters
        ----------
        store : ParquetStore
            Destination.
        key : DataKey
            Table address.
        start : datetime
            Lower bound.
        end : datetime
            Upper bound.

        Returns
        -------
        int
            Rows written.
        """
        origin, newest = store.pulled_from(key), store.last_timestamp(key)
        if origin is None or newest is None:
            store.mark_pulled_from(key, start)
            return self._pull_span(store, key, start, end)
        rows = 0
        if start < origin:
            rows += self._pull_span(store, key, start, origin)
            # Recorded only once complete: an interrupted backfill is repeated, never left with a gap.
            store.mark_pulled_from(key, start)
        return rows + self._pull_span(store, key, newest, end)

    def _pull_span(self, store: ParquetStore, key: DataKey, start: datetime, end: datetime) -> int:
        """
        Fetch one time span of a table and merge it in page by page.

        Parameters
        ----------
        store : ParquetStore
            Destination.
        key : DataKey
            Table address.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        int
            Rows written; zero for an empty span.
        """
        if start >= end:
            return 0
        write = store.insert if self.mode is PullMode.APPEND else store.upsert
        return sum(write(key, page) for page in self.fetch(key, start, end))


class CandleFetcher(APIDataFetcher):
    """
    Pull OHLC bars for a set of instruments and bar sizes.

    Parameters
    ----------
    settings : Settings
        Credentials and data location.
    client : HttpClient or None, default: None
        Client override.
    granularities : Sequence[Granularity], default: GRANULARITIES
        Bar sizes to pull.
    """

    dataset = Dataset.BARS
    mode = PullMode.RESUME

    def __init__(
        self,
        settings: Settings,
        client: HttpClient | None = None,
        granularities: Sequence[Granularity] = GRANULARITIES,
    ) -> None:
        super().__init__(settings, client)
        self.granularities = tuple(granularities)

    @abstractmethod
    def instruments(self) -> list[str]:
        """
        List the instruments to pull.

        Returns
        -------
        list[str]
            Instrument names as stored.
        """

    @abstractmethod
    def fetch_candles(
        self, instrument: str, granularity: Granularity, start: datetime, end: datetime
    ) -> Iterator[pa.Table]:
        """
        Download bars of one instrument and bar size, page by page.

        Parameters
        ----------
        instrument : str
            Instrument name.
        granularity : Granularity
            Bar size.
        start : datetime
            Inclusive lower bound on bar open.
        end : datetime
            Exclusive upper bound on bar open.

        Yields
        ------
        pa.Table
            A page of completed bars in ``BARS_SCHEMA``.
        """

    def list_keys(self) -> list[DataKey]:
        """
        List one table per instrument and bar size.

        Returns
        -------
        list[DataKey]
            Table addresses.
        """
        pairs = product(self.instruments(), self.granularities)
        return [DataKey(self.dataset, (self.source, instrument, granularity)) for instrument, granularity in pairs]

    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Download the bars addressed by a key.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Yields
        ------
        pa.Table
            A page of bars.
        """
        _, instrument, granularity = key.partitions
        yield from self.fetch_candles(instrument, Granularity(granularity), start, end)


class SeriesFetcher(APIDataFetcher):
    """
    Pull scalar time series, one table per series, each value with its publication time.

    A stored value is never overwritten, so it keeps its first release.

    Attributes
    ----------
    publication_delay : timedelta
        Days from a value's reference date until it is public, for sources that do not
        record publication; the value counts as public at the end of that day.
    """

    dataset = Dataset.SERIES
    mode = PullMode.APPEND
    publication_delay: ClassVar[timedelta]

    @abstractmethod
    def series_ids(self) -> list[str]:
        """
        List the series to pull.

        Returns
        -------
        list[str]
            Series names as stored.
        """

    @abstractmethod
    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Download one series.

        Parameters
        ----------
        series_id : str
            Series name, one of `series_ids`.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        pa.Table
            Rows in ``OBSERVATIONS_SCHEMA``, plus the ``published`` and ``estimated``
            columns if the source records publication times.
        """

    def list_keys(self) -> list[DataKey]:
        """
        List one table per series.

        Returns
        -------
        list[DataKey]
            Table addresses.
        """
        return [DataKey(self.dataset, (self.source, series_id)) for series_id in self.series_ids()]

    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Download the series addressed by a key.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Yields
        ------
        pa.Table
            The whole series as one page, in ``SERIES_SCHEMA``.
        """
        observations = self.fetch_series(key.partitions[1], start, end)
        if PUBLISHED_COLUMN not in observations.column_names:
            times = cast("list[datetime]", observations[TIME_COLUMN].to_pylist())
            days = (moment.date() + self.publication_delay for moment in times)
            published = [day_end(day, PUBLICATION_TIMEZONE) for day in days]
            observations = with_publication(observations, published, [True] * len(published))
        seen = pa.array([utc_now()] * observations.num_rows, TIMESTAMP_TYPE)
        yield observations.append_column(FIRST_SEEN_COLUMN, seen)


class FileFetcher(SeriesFetcher):
    """
    Pull series published as one downloadable file.

    The file is downloaded once per fetcher and may hold several series.

    Attributes
    ----------
    url : str
        Location of the file.
    """

    url: ClassVar[str]

    @abstractmethod
    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract every series from the file.

        Parameters
        ----------
        content : bytes
            Raw file body.

        Returns
        -------
        dict[str, pa.Table]
            Series name to rows in ``OBSERVATIONS_SCHEMA``.
        """

    @cached_property
    def _series(self) -> dict[str, pa.Table]:
        """
        Download and parse the file once.

        Returns
        -------
        dict[str, pa.Table]
            Series name to rows.
        """
        return self.parse(self.client.get(self.url).content)

    def series_ids(self) -> list[str]:
        """
        List the series found in the file, which downloads it.

        Returns
        -------
        list[str]
            Series names.
        """
        return list(self._series)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Return one series from the downloaded file.

        The file always holds full history, so the bounds are not applied.

        Parameters
        ----------
        series_id : str
            Series name.
        start : datetime
            Unused.
        end : datetime
            Unused.

        Returns
        -------
        pa.Table
            Rows in ``OBSERVATIONS_SCHEMA``.
        """
        return self._series[series_id]


def observations_table(times: Sequence[datetime], values: Sequence[float | None]) -> pa.Table:
    """
    Build a table of observations from parallel sequences.

    Parameters
    ----------
    times : Sequence[datetime]
        Reference times, UTC.
    values : Sequence[float or None]
        Observations; None marks a missing value.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``.
    """
    return pa.table({TIME_COLUMN: list(times), "value": list(values)}, schema=OBSERVATIONS_SCHEMA)


def with_publication(observations: pa.Table, published: Sequence[datetime], estimated: Sequence[bool]) -> pa.Table:
    """
    Add publication times to observations.

    Parameters
    ----------
    observations : pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``.
    published : Sequence[datetime]
        When each value became public, UTC.
    estimated : Sequence[bool]
        Whether each time comes from a fixed delay rather than a record.

    Returns
    -------
    pa.Table
        The observations with the ``published`` and ``estimated`` columns.
    """
    return observations.append_column(PUBLISHED_COLUMN, pa.array(published, TIMESTAMP_TYPE)).append_column(
        ESTIMATED_COLUMN, pa.array(estimated, pa.bool_())
    )
