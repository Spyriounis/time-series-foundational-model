"""OANDA v20 fetchers: candles and client order and position books."""

from collections.abc import Iterator, Sequence
from datetime import datetime
from functools import cached_property
from http import HTTPStatus
from itertools import batched, product
from typing import Any

import httpx
import pyarrow as pa

from tsfm.constants import (
    G10_CURRENCIES,
    GRANULARITIES,
    HISTORY_START,
    OANDA_ALIGNMENT_TIMEZONE,
    OANDA_BOOK_BATCH_SNAPSHOTS,
    OANDA_BOOK_ENDPOINTS,
    OANDA_BOOK_HISTORY_START,
    OANDA_BOOK_INTERVAL,
    OANDA_BOOK_PRICE_BAND,
    OANDA_CURRENCY_TYPE,
    OANDA_DAILY_ALIGNMENT_HOUR,
    OANDA_HELD_OUT_PAIRS,
    OANDA_HOSTS,
    OANDA_MAX_CANDLES,
    OANDA_METAL_TYPE,
    OANDA_METALS,
    OANDA_PRICE_COMPONENTS,
    OANDA_REQUESTS_PER_SECOND,
    OANDA_SMOOTH,
    TIME_COLUMN,
)
from tsfm.data.fetchers.base import APIDataFetcher, CandleFetcher
from tsfm.data.fetchers.parsing import clip
from tsfm.data.schemas import BARS_SCHEMA, BOOKS_SCHEMA
from tsfm.data.storage import DataKey
from tsfm.enums import BookKind, Dataset, Granularity, InstrumentSet, PullMode, Source
from tsfm.settings import Settings
from tsfm.utils.http import HttpClient, Params
from tsfm.utils.time import format_utc, parse_utc


class _OandaFetcher(APIDataFetcher):
    """Hold what every OANDA fetcher shares: host, authorisation and the instrument list."""

    source = Source.OANDA
    requests_per_second = OANDA_REQUESTS_PER_SECOND

    @property
    def host(self) -> str:
        """
        Return the API host of the configured environment.

        Returns
        -------
        str
            Base URL, without a trailing slash.
        """
        return OANDA_HOSTS[self.settings.oanda_environment]

    def headers(self) -> dict[str, str]:
        """
        Return the bearer-token header.

        Returns
        -------
        dict[str, str]
            Authorisation header.
        """
        return {"Authorization": f"Bearer {self.settings.secret('oanda_api_token')}"}

    @cached_property
    def instrument_types(self) -> dict[str, str]:
        """
        Return the account's tradeable instruments, requested once.

        Returns
        -------
        dict[str, str]
            Instrument name to type, e.g. ``{"EUR_USD": "CURRENCY"}``.
        """
        url = f"{self.host}/accounts/{self.settings.secret('oanda_account_id')}/instruments"
        return _parse_instruments(self.client.get_json(url))


class OandaCandleFetcher(_OandaFetcher, CandleFetcher):
    """
    Pull mid-price bars with the closing bid-ask spread.

    Parameters
    ----------
    settings : Settings
        Credentials and data location.
    client : HttpClient or None, default: None
        Client override.
    granularities : Sequence[Granularity], default: GRANULARITIES
        Bar sizes to pull.
    instrument_set : InstrumentSet, default: InstrumentSet.TARGETS
        Which group of instruments to pull.

    Examples
    --------
    >>> fetcher = OandaCandleFetcher(Settings(), granularities=[Granularity.H1])  # doctest: +SKIP
    >>> fetcher.pull(store, only=["EUR_USD"])  # doctest: +SKIP
    """

    def __init__(
        self,
        settings: Settings,
        client: HttpClient | None = None,
        granularities: Sequence[Granularity] = GRANULARITIES,
        *,
        instrument_set: InstrumentSet = InstrumentSet.TARGETS,
    ) -> None:
        super().__init__(settings, client, granularities)
        self.instrument_set = instrument_set

    def instruments(self) -> list[str]:
        """
        List the instruments of the fetcher's group that the account offers.

        Returns
        -------
        list[str]
            Sorted instrument names.
        """
        return _select_instruments(self.instrument_types, self.instrument_set)

    def fetch_candles(
        self, instrument: str, granularity: Granularity, start: datetime, end: datetime
    ) -> Iterator[pa.Table]:
        """
        Download completed bars, paging forward from ``start``.

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
        url = f"{self.host}/instruments/{instrument}/candles"
        cursor = start
        while cursor < end:
            page = _parse_candles(self.client.get_json(url, _candle_params(granularity, cursor)))
            if page.num_rows == 0:
                return
            kept = clip(page, start, end)
            if kept.num_rows:
                yield kept
            last: datetime = page[TIME_COLUMN][-1].as_py()
            # A page ending before the cursor made no progress; asking again would loop.
            if last >= end or last < cursor:
                return
            cursor = last + granularity.timedelta


class OandaBookFetcher(_OandaFetcher):
    """
    Pull daily snapshots of OANDA clients' order and position books.

    Examples
    --------
    >>> OandaBookFetcher(Settings()).pull(store, start=datetime(2024, 1, 1))  # doctest: +SKIP
    """

    dataset = Dataset.BOOKS
    mode = PullMode.RESUME

    def list_keys(self) -> list[DataKey]:
        """
        List one table per book kind of every G10 pair that has books.

        Returns
        -------
        list[DataKey]
            Table addresses.
        """
        instruments = filter(self._has_books, _select_instruments(self.instrument_types, InstrumentSet.TARGETS))
        return [
            DataKey(self.dataset, (self.source, instrument, kind))
            for instrument, kind in product(instruments, BookKind)
        ]

    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Download the snapshots of one book, a batch of snapshot times at a time.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Inclusive lower bound, raised to ``OANDA_BOOK_HISTORY_START``.
        end : datetime
            Exclusive upper bound.

        Yields
        ------
        pa.Table
            The snapshots found in one batch, in ``BOOKS_SCHEMA``.
        """
        _, instrument, kind = key.partitions
        endpoint = OANDA_BOOK_ENDPOINTS[BookKind(kind)]
        url = f"{self.host}/instruments/{instrument}/{endpoint}"
        first = max(start, OANDA_BOOK_HISTORY_START)
        for times in batched(_snapshot_times(first, end), OANDA_BOOK_BATCH_SNAPSHOTS):
            payloads = (self._fetch_snapshot(url, time) for time in times)
            books = (_parse_book(payload[endpoint]) for payload in payloads if payload is not None)
            rows = [book for book in books if book is not None]
            if rows:
                yield pa.Table.from_pylist(rows, schema=BOOKS_SCHEMA)

    def _has_books(self, instrument: str) -> bool:
        """
        Ask for the latest order book to learn whether an instrument has books.

        Parameters
        ----------
        instrument : str
            Instrument name.

        Returns
        -------
        bool
            False if OANDA publishes no books for the instrument.

        Raises
        ------
        httpx.HTTPStatusError
            On any error status other than 400.
        """
        try:
            self.client.get(f"{self.host}/instruments/{instrument}/{OANDA_BOOK_ENDPOINTS[BookKind.ORDER]}")
        except httpx.HTTPStatusError as error:
            # OANDA answers 400 "not a valid instrument" for a pair without books.
            if error.response.status_code != HTTPStatus.BAD_REQUEST:
                raise
            return False
        return True

    def _fetch_snapshot(self, url: str, time: datetime) -> dict[str, Any] | None:
        """
        Request the book at one snapshot time.

        Parameters
        ----------
        url : str
            Book endpoint of one instrument.
        time : datetime
            Snapshot time.

        Returns
        -------
        dict[str, Any] or None
            Response body, or None if no snapshot exists at that time.

        Raises
        ------
        httpx.HTTPStatusError
            On any error status other than 404.
        """
        try:
            payload: dict[str, Any] = self.client.get_json(url, {"time": format_utc(time)})
        except httpx.HTTPStatusError as error:
            # Snapshots are missing now and then, which must not end the pull.
            if error.response.status_code != HTTPStatus.NOT_FOUND:
                raise
            return None
        return payload


def _parse_instruments(payload: dict[str, Any]) -> dict[str, str]:
    """
    Map each instrument of an account to its type.

    Parameters
    ----------
    payload : dict[str, Any]
        Body of the account instruments endpoint.

    Returns
    -------
    dict[str, str]
        Instrument name to type.
    """
    return {instrument["name"]: instrument["type"] for instrument in payload["instruments"]}


def _is_g10_pair(name: str) -> bool:
    """
    Tell whether both legs of a currency pair are G10 currencies.

    Parameters
    ----------
    name : str
        Pair name such as ``EUR_USD``.

    Returns
    -------
    bool
        True if both legs are in ``G10_CURRENCIES``.
    """
    return set(name.split("_")) <= G10_CURRENCIES


def _select_instruments(types: dict[str, str], instrument_set: InstrumentSet) -> list[str]:
    """
    Pick the instruments of a group that the account offers.

    Parameters
    ----------
    types : dict[str, str]
        Instrument name to type.
    instrument_set : InstrumentSet
        The G10 currency pairs; the non-currency instruments, with metals limited to
        ``OANDA_METALS``; or the pairs in ``OANDA_HELD_OUT_PAIRS``.

    Returns
    -------
    list[str]
        Sorted instrument names.
    """
    if instrument_set is InstrumentSet.TARGETS:
        return sorted(name for name, kind in types.items() if kind == OANDA_CURRENCY_TYPE and _is_g10_pair(name))
    if instrument_set is InstrumentSet.HELD_OUT:
        return sorted(name for name in OANDA_HELD_OUT_PAIRS if name in types)
    covariates = (name for name, kind in types.items() if kind != OANDA_CURRENCY_TYPE)
    return sorted(name for name in covariates if types[name] != OANDA_METAL_TYPE or name in OANDA_METALS)


def _candle_params(granularity: Granularity, start: datetime) -> Params:
    """
    Build the query of one candle page.

    Parameters
    ----------
    granularity : Granularity
        Bar size.
    start : datetime
        Open of the first bar wanted.

    Returns
    -------
    Params
        Query parameters.
    """
    # `to` is left out on purpose: OANDA rejects it together with `count`.
    return {
        "granularity": granularity.value,
        "price": OANDA_PRICE_COMPONENTS,
        "from": format_utc(start),
        "count": OANDA_MAX_CANDLES,
        "smooth": OANDA_SMOOTH,
        "dailyAlignment": OANDA_DAILY_ALIGNMENT_HOUR,
        "alignmentTimezone": OANDA_ALIGNMENT_TIMEZONE,
    }


def _parse_candles(payload: dict[str, Any]) -> pa.Table:
    """
    Convert a candle page to bars, dropping the bar still forming.

    Parameters
    ----------
    payload : dict[str, Any]
        Body of the candles endpoint, with mid, bid and ask prices.

    Returns
    -------
    pa.Table
        Completed bars in ``BARS_SCHEMA``.
    """
    candles = [candle for candle in payload["candles"] if candle["complete"]]
    columns: dict[str, list[Any]] = {
        TIME_COLUMN: [parse_utc(candle["time"]) for candle in candles],
        "open": [float(candle["mid"]["o"]) for candle in candles],
        "high": [float(candle["mid"]["h"]) for candle in candles],
        "low": [float(candle["mid"]["l"]) for candle in candles],
        "close": [float(candle["mid"]["c"]) for candle in candles],
        "spread": [float(candle["ask"]["c"]) - float(candle["bid"]["c"]) for candle in candles],
        "volume": [float(candle["volume"]) for candle in candles],
    }
    return pa.table(columns, schema=BARS_SCHEMA)


def _snapshot_times(start: datetime, end: datetime) -> Iterator[datetime]:
    """
    Step through the snapshot grid inside ``[start, end)``.

    Parameters
    ----------
    start : datetime
        Inclusive lower bound.
    end : datetime
        Exclusive upper bound.

    Yields
    ------
    datetime
        Snapshot times, ``OANDA_BOOK_INTERVAL`` apart.
    """
    # Snapshots sit on a grid from midnight UTC, so an off-grid start is rounded up to it.
    time = start + (HISTORY_START - start) % OANDA_BOOK_INTERVAL
    while time < end:
        yield time
        time += OANDA_BOOK_INTERVAL


def _parse_book(book: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one book snapshot to a row, keeping the buckets near the current price.

    Parameters
    ----------
    book : dict[str, Any]
        The ``orderBook`` or ``positionBook`` object of a response.

    Returns
    -------
    dict[str, Any] or None
        Row in ``BOOKS_SCHEMA``, with one list entry per kept bucket; None for the
        occasional snapshot OANDA publishes without a price.
    """
    if not book["price"]:
        return None
    price = float(book["price"])
    band = price * OANDA_BOOK_PRICE_BAND
    buckets = [bucket for bucket in book["buckets"] if abs(float(bucket["price"]) - price) <= band]
    return {
        TIME_COLUMN: parse_utc(book["time"]),
        "price": price,
        "bucket_width": float(book["bucketWidth"]),
        "bucket_price": [float(bucket["price"]) for bucket in buckets],
        "long_pct": [float(bucket["longCountPercent"]) for bucket in buckets],
        "short_pct": [float(bucket["shortCountPercent"]) for bucket in buckets],
    }
