import httpx

from helpers import utc
from tsfm.constants import ETORO_MAX_CANDLES, ETORO_MAX_WINDOW_PERIODS, ETORO_SYMBOLS, TIME_COLUMN
from tsfm.data.fetchers.etoro import EtoroCandleFetcher, _parse_candles, _windows
from tsfm.enums import Granularity

INSTRUMENT_PAGES = {
    None: {
        "results": [{"instrumentId": 1001, "symbol": "AAPL", "type": "Stocks"}],
        "pagination": {"hasNext": True, "nextPageToken": "page-2", "pageSize": 1},
    },
    "page-2": {
        "results": [{"instrumentId": 1002, "symbol": "MSFT", "type": "Stocks"}],
        "pagination": {"hasNext": False, "nextPageToken": None, "pageSize": 1},
    },
}
CANDLE_PAGES = {
    None: {
        "results": [
            {"time": "2024-01-03T00:00:00Z", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "9"}
        ],
        "pagination": {"limit": 1, "hasNext": True, "nextCursor": "older"},
    },
    "older": {
        "results": [
            {"time": "2024-01-02T00:00:00Z", "open": "1", "high": "2", "low": "1", "close": "1.5", "volume": None}
        ],
        "pagination": {"limit": 1, "hasNext": False, "nextCursor": None},
    },
}


def server(requests: list[httpx.Request]):
    """Answer instrument and candle requests from the paged fixtures above."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        if request.url.path.endswith("/instruments"):
            return httpx.Response(200, json=INSTRUMENT_PAGES[params.get("pageToken")])
        return httpx.Response(200, json=CANDLE_PAGES[params.get("cursor")])

    return handler


def test_key_headers(settings):
    """eToro needs both keys on every request."""
    assert EtoroCandleFetcher(settings).headers() == {"x-api-key": "api-key", "x-user-key": "user-key"}


def test_window_is_stored_whole_and_oldest_first(settings, make_client):
    """eToro pages newest first; a page stored on its own would leave a gap that a resumed pull never refills."""
    requests: list[httpx.Request] = []
    fetcher = EtoroCandleFetcher(settings, make_client(server(requests)))
    tables = list(fetcher.fetch_candles("AAPL", Granularity.D, utc(2024, 1, 1), utc(2024, 1, 4)))
    assert [table[TIME_COLUMN].to_pylist() for table in tables] == [[utc(2024, 1, 2), utc(2024, 1, 3)]]
    lookups, candles = requests[:2], requests[2:]
    assert [request.url.params.get("pageToken") for request in lookups] == [None, "page-2"]
    assert lookups[0].url.params["symbols"] == ",".join(ETORO_SYMBOLS)
    assert candles[0].url.path == "/api/v1/data/instruments/1001/candles"
    assert dict(candles[0].url.params) == {
        "interval": "1d",
        "from": "2024-01-01T00:00:00Z",
        "to": "2024-01-04T00:00:00Z",
        "limit": str(ETORO_MAX_CANDLES),
        "side": "bid",
    }
    assert [request.url.params.get("cursor") for request in candles] == [None, "older"]


def test_each_request_has_a_fresh_request_id(settings, make_client):
    """eToro deduplicates on `x-request-id`, so a reused id would drop a request."""
    requests: list[httpx.Request] = []
    fetcher = EtoroCandleFetcher(settings, make_client(server(requests)))
    list(fetcher.fetch_candles("AAPL", Granularity.D, utc(2024, 1, 1), utc(2024, 1, 4)))
    ids = [request.headers["x-request-id"] for request in requests]
    assert len(ids) == len(set(ids)) == 4


def test_unknown_symbol_is_skipped(settings, make_client):
    """A symbol eToro does not list must not stop the pull of the others."""
    requests: list[httpx.Request] = []
    fetcher = EtoroCandleFetcher(settings, make_client(server(requests)))
    assert list(fetcher.fetch_candles("SPY", Granularity.D, utc(2024), utc(2025))) == []
    assert all(request.url.path.endswith("/instruments") for request in requests)


def test_parse_reads_decimal_strings_and_nulls():
    """Prices arrive as strings and volume may be null; neither may crash or become zero."""
    table = _parse_candles(CANDLE_PAGES["older"])
    assert table["close"].to_pylist() == [1.5]
    assert table["volume"].to_pylist() == [None]
    assert table["spread"].to_pylist() == [None]


def test_empty_results_end_the_window(settings, make_client):
    """An instrument with no data answers 200 with no results, which must end the loop."""
    empty = {"results": [], "pagination": {"hasNext": True, "nextCursor": "again"}}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        is_lookup = request.url.path.endswith("/instruments")
        return httpx.Response(200, json=INSTRUMENT_PAGES["page-2"] if is_lookup else empty)

    fetcher = EtoroCandleFetcher(settings, make_client(handler))
    assert list(fetcher.fetch_candles("MSFT", Granularity.D, utc(2024, 1, 1), utc(2024, 1, 4))) == []
    assert len(requests) == 2


def test_no_request_before_the_start_of_history(settings, make_client):
    """Windows before eToro's first bar are thousands of empty requests on a full pull."""
    requests: list[httpx.Request] = []
    fetcher = EtoroCandleFetcher(settings, make_client(server(requests)))
    list(fetcher.fetch_candles("AAPL", Granularity.D, utc(2000), utc(2012, 1, 3)))
    assert requests[2].url.params["from"] == "2012-01-01T00:00:00Z"


def test_windows_respect_the_period_cap():
    """A window is the unit a pull stores and repeats, so a long history is cut into bounded windows."""
    span = Granularity.D.timedelta * ETORO_MAX_WINDOW_PERIODS
    windows = list(_windows(utc(2000), utc(2030), span))
    assert windows == [(utc(2000), utc(2000) + span), (utc(2000) + span, utc(2030))]
