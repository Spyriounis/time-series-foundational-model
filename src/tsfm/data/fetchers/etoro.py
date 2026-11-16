"""Fetcher for the eToro public API: bid-side candles of ETFs, stocks and crypto."""

from collections.abc import Iterator
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any
from uuid import uuid4

import pyarrow as pa

from tsfm.constants import (
    ETORO_BASE_URL,
    ETORO_HISTORY_START,
    ETORO_INSTRUMENTS_PAGE_SIZE,
    ETORO_INTERVALS,
    ETORO_MAX_CANDLES,
    ETORO_MAX_WINDOW_PERIODS,
    ETORO_PRICE_SIDE,
    ETORO_REQUESTS_PER_SECOND,
    ETORO_SYMBOLS,
    TIME_COLUMN,
)
from tsfm.data.fetchers.base import CandleFetcher
from tsfm.data.fetchers.parsing import clip, optional_float
from tsfm.data.schemas import BARS_SCHEMA
from tsfm.enums import Granularity, Source
from tsfm.log import get_logger
from tsfm.utils.http import Params
from tsfm.utils.time import format_utc, parse_utc, utc_now

logger = get_logger(__name__)

_INSTRUMENTS_URL = f"{ETORO_BASE_URL}/api/v2/market-data/instruments"
_CANDLES_URL = f"{ETORO_BASE_URL}/api/v1/data/instruments/{{instrument_id}}/candles"


class EtoroCandleFetcher(CandleFetcher):
    """
    Pull bid-side bars for the symbols in ``ETORO_SYMBOLS``.

    Examples
    --------
    >>> fetcher = EtoroCandleFetcher(Settings(), granularities=[Granularity.D])  # doctest: +SKIP
    >>> fetcher.pull(store, only=["SPY"])  # doctest: +SKIP
    """

    source = Source.ETORO
    requests_per_second = ETORO_REQUESTS_PER_SECOND

    def headers(self) -> dict[str, str]:
        """
        Return the two key headers.

        Returns
        -------
        dict[str, str]
            API key and user key.
        """
        return {
            "x-api-key": self.settings.secret("etoro_api_key"),
            "x-user-key": self.settings.secret("etoro_user_key"),
        }

    def instruments(self) -> list[str]:
        """
        List the configured symbols.

        Returns
        -------
        list[str]
            Symbols as eToro spells them.
        """
        return list(ETORO_SYMBOLS)

    def fetch_candles(
        self, instrument: str, granularity: Granularity, start: datetime, end: datetime
    ) -> Iterator[pa.Table]:
        """
        Download bars window by window, oldest window first.

        Parameters
        ----------
        instrument : str
            Symbol.
        granularity : Granularity
            Bar size.
        start : datetime
            Inclusive lower bound on bar open, raised to ``ETORO_HISTORY_START``.
        end : datetime
            Exclusive upper bound on bar open.

        Yields
        ------
        pa.Table
            The bars of one window in ``BARS_SCHEMA``; nothing for a symbol eToro does not know.
        """
        instrument_id = self._instrument_ids.get(instrument)
        if instrument_id is None:
            return
        url = _CANDLES_URL.format(instrument_id=instrument_id)
        span = granularity.timedelta * ETORO_MAX_WINDOW_PERIODS
        # eToro does not mark the bar still forming, so bars opened less than one period ago are left out.
        end = min(end, utc_now() - granularity.timedelta)
        for window_start, window_end in _windows(max(start, ETORO_HISTORY_START), end, span):
            page = self._fetch_window(url, _candle_params(granularity, window_start, window_end))
            window = clip(page, window_start, window_end)
            if window.num_rows:
                yield window

    @cached_property
    def _instrument_ids(self) -> dict[str, int]:
        """
        Resolve the configured symbols to eToro instrument ids, once.

        Returns
        -------
        dict[str, int]
            Symbol to instrument id, without the symbols eToro does not know.
        """
        ids: dict[str, int] = {}
        params: dict[str, str | int] = {"symbols": ",".join(ETORO_SYMBOLS), "pageSize": ETORO_INSTRUMENTS_PAGE_SIZE}
        while True:
            payload = self._get(_INSTRUMENTS_URL, params)
            ids |= _parse_instrument_ids(payload)
            pagination = payload["pagination"]
            if not payload["results"] or not pagination["hasNext"]:
                break
            params["pageToken"] = pagination["nextPageToken"]
        unknown = sorted(set(ETORO_SYMBOLS) - set(ids))
        if unknown:
            logger.warning("etoro_unknown_symbols", symbols=unknown)
        return ids

    def _fetch_window(self, url: str, params: dict[str, str | int]) -> pa.Table:
        """
        Download one whole time window, following the cursor to its last page.

        eToro pages from the newest bar backwards. Storing a window only once it is
        complete keeps the store free of gaps below its newest bar, which is where
        a pull resumes.

        Parameters
        ----------
        url : str
            Candle endpoint of one instrument.
        params : dict[str, str | int]
            Query of the window's first page.

        Returns
        -------
        pa.Table
            The window's bars, oldest first.
        """
        pages: list[pa.Table] = []
        while True:
            payload = self._get(url, params)
            pages.append(_parse_candles(payload))
            pagination = payload["pagination"]
            if not payload["results"] or not pagination["hasNext"]:
                return pa.concat_tables(pages).sort_by(TIME_COLUMN)
            params = {**params, "cursor": pagination["nextCursor"]}

    def _get(self, url: str, params: Params) -> dict[str, Any]:
        """
        Send a request under its own request id, which eToro requires.

        Parameters
        ----------
        url : str
            Absolute URL.
        params : Params
            Query parameters.

        Returns
        -------
        dict[str, Any]
            Decoded JSON body.
        """
        # A function, so that a retry carries a new id.
        payload: dict[str, Any] = self.client.get_json(url, params, lambda: {"x-request-id": str(uuid4())})
        return payload


def _parse_instrument_ids(payload: dict[str, Any]) -> dict[str, int]:
    """
    Map symbols to instrument ids.

    Parameters
    ----------
    payload : dict[str, Any]
        One page of the instruments endpoint.

    Returns
    -------
    dict[str, int]
        Symbol to instrument id.
    """
    return {result["symbol"]: int(result["instrumentId"]) for result in payload["results"]}


def _windows(start: datetime, end: datetime, span: timedelta) -> Iterator[tuple[datetime, datetime]]:
    """
    Cut ``[start, end)`` into consecutive windows no longer than ``span``.

    Parameters
    ----------
    start : datetime
        Inclusive lower bound.
    end : datetime
        Exclusive upper bound.
    span : timedelta
        Longest window allowed.

    Yields
    ------
    tuple[datetime, datetime]
        Window bounds, lower inclusive and upper exclusive.
    """
    while start < end:
        stop = min(start + span, end)
        yield start, stop
        start = stop


def _candle_params(granularity: Granularity, start: datetime, end: datetime) -> dict[str, str | int]:
    """
    Build the query of a window's first page.

    Parameters
    ----------
    granularity : Granularity
        Bar size.
    start : datetime
        Inclusive lower bound.
    end : datetime
        Exclusive upper bound.

    Returns
    -------
    dict[str, str | int]
        Query parameters.
    """
    return {
        "interval": ETORO_INTERVALS[granularity],
        "from": format_utc(start),
        "to": format_utc(end),
        "limit": ETORO_MAX_CANDLES,
        "side": ETORO_PRICE_SIDE,
    }


def _parse_candles(payload: dict[str, Any]) -> pa.Table:
    """
    Convert a candle page to bars.

    Parameters
    ----------
    payload : dict[str, Any]
        Body of the candles endpoint; prices and volume are nullable decimal strings.

    Returns
    -------
    pa.Table
        Bars in ``BARS_SCHEMA``, in the order received.
    """
    candles = payload["results"]
    columns: dict[str, list[Any]] = {
        TIME_COLUMN: [parse_utc(candle["time"]) for candle in candles],
        **{name: [optional_float(candle[name]) for candle in candles] for name in ("open", "high", "low", "close")},
        # Only the bid side is requested, so there is no spread to store.
        "spread": [None] * len(candles),
        "volume": [optional_float(candle["volume"]) for candle in candles],
    }
    return pa.table(columns, schema=BARS_SCHEMA)
