import httpx
import pytest

from helpers import utc
from tsfm.constants import OANDA_MAX_CANDLES, TIME_COLUMN
from tsfm.data.fetchers.oanda import OandaBookFetcher, OandaCandleFetcher, _parse_candles
from tsfm.data.storage import DataKey
from tsfm.enums import BookKind, Dataset, Granularity, InstrumentSet

HOST = "https://api-fxpractice.oanda.com/v3"
INSTRUMENTS = {
    "instruments": [
        {"name": "USD_MXN", "type": "CURRENCY"},
        {"name": "EUR_USD", "type": "CURRENCY"},
        {"name": "XAU_USD", "type": "METAL"},
        {"name": "XAU_EUR", "type": "METAL"},
        {"name": "SPX500_USD", "type": "CFD"},
    ]
}


def candle(day: int, *, complete: bool = True) -> dict:
    """One daily candle of January 2024 with a 2-pip closing spread."""
    mid = {"o": "1.1000", "h": "1.2000", "l": "1.0000", "c": "1.1500"}
    bid, ask = {**mid, "c": "1.1499"}, {**mid, "c": "1.1501"}
    time = f"2024-01-{day:02d}T00:00:00.000000000Z"
    return {"complete": complete, "volume": 7, "time": time, "mid": mid, "bid": bid, "ask": ask}


def candle_server(pages: list[list[dict]], requests: list[httpx.Request]):
    """Answer the instrument list, then one candle page per call, then empty pages."""
    remaining = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/instruments"):
            return httpx.Response(200, json=INSTRUMENTS)
        return httpx.Response(200, json={"candles": next(remaining, [])})

    return handler


def test_auth_header_carries_the_token(settings):
    """OANDA rejects every call without the bearer token."""
    assert OandaCandleFetcher(settings).headers() == {"Authorization": "Bearer token"}


def test_targets_are_g10_pairs_only(settings, make_client):
    """An emerging-market pair or a metal must not become a forecast target."""
    requests: list[httpx.Request] = []
    fetcher = OandaCandleFetcher(settings, make_client(candle_server([], requests)))
    assert fetcher.instruments() == ["EUR_USD"]
    assert fetcher.instruments() == ["EUR_USD"]
    assert [request.url.path for request in requests] == ["/v3/accounts/account/instruments"]


def test_covariates_are_the_non_currency_instruments(settings, make_client):
    """Currency pairs are targets, and gold in euros is gold in dollars times a target, so neither is a covariate."""
    client = make_client(candle_server([], []))
    fetcher = OandaCandleFetcher(settings, client, instrument_set=InstrumentSet.COVARIATES)
    assert fetcher.instruments() == ["SPX500_USD", "XAU_USD"]


def test_held_out_pairs_are_the_configured_ones_on_offer(settings, make_client):
    """The zero-shot test uses fixed non-G10 pairs; one the account lacks is skipped, not requested."""
    client = make_client(candle_server([], []))
    fetcher = OandaCandleFetcher(settings, client, instrument_set=InstrumentSet.HELD_OUT)
    assert fetcher.instruments() == ["USD_MXN"]


def test_first_candle_request_shape(settings, make_client):
    """A wrong query is a 400 on the first live pull; `to` with `count` is one such error."""
    requests: list[httpx.Request] = []
    fetcher = OandaCandleFetcher(settings, make_client(candle_server([[candle(1)]], requests)))
    list(fetcher.fetch_candles("EUR_USD", Granularity.D, utc(2024, 1, 1), utc(2024, 1, 2)))
    assert str(requests[0].url).startswith(f"{HOST}/instruments/EUR_USD/candles")
    params = dict(requests[0].url.params)
    assert "to" not in params
    assert params == {
        "granularity": "D",
        "price": "MBA",
        "from": "2024-01-01T00:00:00Z",
        "count": str(OANDA_MAX_CANDLES),
        "smooth": "false",
        "dailyAlignment": "17",
        "alignmentTimezone": "America/New_York",
    }


def test_parse_drops_the_forming_candle():
    """The bar still forming changes until it closes, so storing it would corrupt history."""
    table = _parse_candles({"candles": [candle(1), candle(2, complete=False)]})
    assert table[TIME_COLUMN].to_pylist() == [utc(2024, 1, 1)]
    assert table["close"].to_pylist() == [1.15]
    assert table["spread"][0].as_py() == pytest.approx(0.0002, rel=1e-3)
    assert table["volume"].to_pylist() == [7.0]


def test_paging_advances_and_stops_at_end(settings, make_client):
    """Each page must start one bar after the last, and bars at or after `end` must not be stored."""
    requests: list[httpx.Request] = []
    pages = [[candle(1), candle(2)], [candle(3), candle(4)], [candle(5)]]
    fetcher = OandaCandleFetcher(settings, make_client(candle_server(pages, requests)))
    tables = list(fetcher.fetch_candles("EUR_USD", Granularity.D, utc(2024, 1, 1), utc(2024, 1, 4)))
    assert [table[TIME_COLUMN].to_pylist() for table in tables] == [
        [utc(2024, 1, 1), utc(2024, 1, 2)],
        [utc(2024, 1, 3)],
    ]
    assert [request.url.params["from"] for request in requests] == ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"]


def test_paging_stops_on_an_empty_page(settings, make_client):
    """History that ends before `end` must end the loop instead of polling forever."""
    requests: list[httpx.Request] = []
    fetcher = OandaCandleFetcher(settings, make_client(candle_server([[candle(1)]], requests)))
    tables = list(fetcher.fetch_candles("EUR_USD", Granularity.D, utc(2024, 1, 1), utc(2025)))
    assert sum(table.num_rows for table in tables) == 1
    assert len(requests) == 2


def test_pull_writes_readable_bars(settings, make_client, store):
    """The whole path, from HTTP response to Parquet, must hold together for the main dataset."""
    handler = candle_server([[candle(1), candle(2)]], [])
    fetcher = OandaCandleFetcher(settings, make_client(handler), granularities=[Granularity.D])
    assert fetcher.pull(store, start=utc(2024), end=utc(2025)) == 2
    stored = store.read(DataKey(Dataset.BARS, ("oanda", "EUR_USD", "D")))
    assert stored[TIME_COLUMN].to_pylist() == [utc(2024, 1, 1), utc(2024, 1, 2)]
    assert stored["open"].to_pylist() == [1.1, 1.1]


def book_handler(requests: list[httpx.Request]):
    """Answer the instrument list, a 404 on 2 January, no price on 3 January and an order book otherwise."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/instruments"):
            return httpx.Response(200, json=INSTRUMENTS)
        requests.append(request)
        time = request.url.params.get("time", "2024-01-09T00:00:00Z")
        if time == "2024-01-02T00:00:00Z":
            return httpx.Response(404)
        price = "" if time == "2024-01-03T00:00:00Z" else "1.105"
        buckets = [
            {"price": "0.0", "longCountPercent": "0.1", "shortCountPercent": "0.1"},
            {"price": "1.1", "longCountPercent": "0.25", "shortCountPercent": "0.5"},
            {"price": "5555555555.0", "longCountPercent": "0.1", "shortCountPercent": "0.1"},
        ]
        book = {"instrument": "EUR_USD", "time": time, "price": price, "bucketWidth": "0.0005", "buckets": buckets}
        return httpx.Response(200, json={"orderBook": book})

    return handler


def test_book_keys_cover_both_kinds_of_g10_pairs(settings, make_client):
    """Books are pulled for the forecast targets only, one table per book kind."""
    keys = OandaBookFetcher(settings, make_client(book_handler([]))).list_keys()
    assert [key.partitions for key in keys] == [("oanda", "EUR_USD", "order"), ("oanda", "EUR_USD", "position")]


def test_pairs_without_books_are_not_listed(settings, make_client):
    """OANDA answers 400 for a pair without books; pulling it would abort the pairs that have them."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/instruments"):
            return httpx.Response(200, json=INSTRUMENTS)
        return httpx.Response(400, json={"errorMessage": "EUR_USD is not a valid instrument."})

    assert OandaBookFetcher(settings, make_client(handler)).list_keys() == []


def test_book_skips_unusable_snapshots_and_far_buckets(settings, make_client):
    """A missing or priceless snapshot must not abort the pull, and levels far from the price are noise."""
    requests: list[httpx.Request] = []
    fetcher = OandaBookFetcher(settings, make_client(book_handler(requests)))
    key = DataKey(Dataset.BOOKS, ("oanda", "EUR_USD", BookKind.ORDER))
    (table,) = fetcher.fetch(key, utc(2024, 1, 1, 12), utc(2024, 1, 5))
    assert [request.url.path for request in requests] == ["/v3/instruments/EUR_USD/orderBook"] * 3
    assert [request.url.params["time"][:10] for request in requests] == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert table.to_pylist() == [
        {
            TIME_COLUMN: utc(2024, 1, 4),
            "price": 1.105,
            "bucket_width": 0.0005,
            "bucket_price": [1.1],
            "long_pct": [0.25],
            "short_pct": [0.5],
        }
    ]


def test_book_raises_on_other_errors(settings, make_client):
    """Only a 404 is expected; a 401 must surface instead of looking like an empty history."""
    fetcher = OandaBookFetcher(settings, make_client(lambda request: httpx.Response(401)))
    key = DataKey(Dataset.BOOKS, ("oanda", "EUR_USD", BookKind.POSITION))
    with pytest.raises(httpx.HTTPStatusError):
        list(fetcher.fetch(key, utc(2024, 1, 1), utc(2024, 1, 2)))
