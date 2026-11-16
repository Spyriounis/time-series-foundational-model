from collections.abc import Iterator
from datetime import date, datetime, timedelta

import httpx
import pyarrow as pa
import pytest

from helpers import bars, utc
from tsfm.data.fetchers.base import CandleFetcher, PullError, SeriesFetcher, observations_table
from tsfm.data.schemas import SERIES_SCHEMA
from tsfm.data.storage import DataKey, ParquetStore
from tsfm.enums import Dataset, Granularity, PullMode, Source
from tsfm.settings import MissingSecretError, Settings
from tsfm.utils.http import RetryableStatusError
from tsfm.utils.time import day_end, parse_utc


class _FakeCandles(CandleFetcher):
    """Two daily bars per call, recording the bounds it was asked for."""

    source = Source.OANDA

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, granularities=[Granularity.D])
        self.calls: list[tuple[datetime, datetime]] = []
        self.broken: set[str] = set()

    def instruments(self) -> list[str]:
        return ["EUR_USD", "USD_JPY"]

    def fetch_candles(self, instrument, granularity, start, end) -> Iterator[pa.Table]:
        self.calls.append((start, end))
        if instrument in self.broken:
            raise KeyError("candles")
        yield bars([day for day in (utc(2024, 1, 1), utc(2024, 1, 2)) if start <= day < end])


class _FakeSeries(SeriesFetcher):
    """One series whose only observation takes whatever value the source currently reports."""

    source = Source.FRED
    publication_delay = timedelta(days=30)
    value = 1.0

    def series_ids(self) -> list[str]:
        return ["GDP"]

    def fetch_series(self, series_id, start, end) -> pa.Table:
        return observations_table([utc(2024)], [self.value])


def test_parse_utc_accepts_nanoseconds_and_dates():
    """OANDA sends nanosecond stamps and FRED sends bare dates; both must land on UTC."""
    assert parse_utc("2024-01-02T03:04:05.000000000Z") == utc(2024, 1, 2, 3, 4).replace(second=5)
    assert parse_utc("2024-01-02") == utc(2024, 1, 2)


def test_missing_secret_fails_fast():
    """An unset key must stop the pull with its name, not send an empty credential."""
    with pytest.raises(MissingSecretError, match="FRED_API_KEY"):
        # `_env_file` is a runtime-only pydantic-settings argument.
        Settings(_env_file=None).secret("fred_api_key")  # pyright: ignore[reportCallIssue]


def test_client_retries_transient_status(make_client):
    """A 503 followed by a 200 is one successful call from the fetcher's point of view."""
    statuses = iter([503, 429, 200])
    client = make_client(lambda request: httpx.Response(next(statuses), json={"ok": True}))
    assert client.get_json("https://example.com") == {"ok": True}


def test_client_gives_up_after_max_attempts(make_client):
    """A permanently failing server must surface an error instead of looping forever."""
    client = make_client(lambda request: httpx.Response(503))
    with pytest.raises(RetryableStatusError):
        client.get("https://example.com")


def test_client_does_not_retry_client_errors(make_client):
    """A 401 means a bad key; retrying would only delay the message."""
    calls = []
    client = make_client(lambda request: calls.append(1) or httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://example.com")
    assert len(calls) == 1


def test_pull_resumes_from_last_stored_row(store: ParquetStore, settings: Settings):
    """A second pull asks only for data after what is stored, and re-sent rows do not duplicate."""
    fetcher = _FakeCandles(settings)
    assert fetcher.pull(store, start=utc(2020), end=utc(2025)) == 4
    fetcher.pull(store, start=utc(2020), end=utc(2025))
    assert fetcher.calls[0][0] == utc(2020)
    assert fetcher.calls[-1][0] == utc(2024, 1, 2)
    assert store.read(DataKey(Dataset.BARS, ("oanda", "EUR_USD", "D"))).num_rows == 2


def test_pull_backfills_history_before_an_earlier_partial_pull(store: ParquetStore, settings: Settings):
    """A one-day trial pull must not make the full pull believe everything before that day is stored."""
    fetcher = _FakeCandles(settings)
    fetcher.pull(store, start=utc(2024, 1, 2), end=utc(2024, 1, 3), only=["EUR_USD"])
    fetcher.pull(store, start=utc(2020), end=utc(2025), only=["EUR_USD"])
    assert fetcher.calls[1:] == [(utc(2020), utc(2024, 1, 2)), (utc(2024, 1, 2), utc(2025))]
    assert store.read(DataKey(Dataset.BARS, ("oanda", "EUR_USD", "D"))).num_rows == 2
    fetcher.pull(store, start=utc(2020), end=utc(2025), only=["EUR_USD"])
    assert len(fetcher.calls) == 4


def test_pull_only_selects_by_name(store: ParquetStore, settings: Settings):
    """`--only EUR_USD` must not touch other instruments, and `--only D` must not match the bar size."""
    fetcher = _FakeCandles(settings)
    assert fetcher.pull(store, only=["USD_JPY"]) == 2
    assert store.layout.files(DataKey(Dataset.BARS, ("oanda", "EUR_USD", "D"))) == []
    assert fetcher.pull(store, only=["D", "oanda"]) == 0


def test_pull_continues_past_a_failing_table(store: ParquetStore, settings: Settings):
    """An unattended pull must store the healthy tables and still report the broken one."""
    fetcher = _FakeCandles(settings)
    fetcher.broken = {"EUR_USD"}
    with pytest.raises(PullError) as raised:
        fetcher.pull(store, start=utc(2020), end=utc(2025))
    assert [key.partitions[1] for key in raised.value.failed] == ["EUR_USD"]
    assert raised.value.rows == 2
    assert store.read(DataKey(Dataset.BARS, ("oanda", "USD_JPY", "D"))).num_rows == 2


def test_series_keep_their_first_release(store: ParquetStore, settings: Settings):
    """A revision must not overwrite the value first published, or a backtest would see numbers not yet known."""
    fetcher = _FakeSeries(settings)
    assert fetcher.mode is PullMode.APPEND
    key = DataKey(Dataset.SERIES, ("fred", "GDP"))
    fetcher.pull(store, start=utc(2020), end=utc(2025))
    first = store.read(key)
    fetcher.value = 2.0
    assert fetcher.pull(store, start=utc(2020), end=utc(2025)) == 0
    assert store.read(key).equals(first)


def test_series_publication_is_estimated_from_the_source_delay(store: ParquetStore, settings: Settings):
    """Without a record, a value counts as public at the end of its date plus the delay, New York time."""
    _FakeSeries(settings).pull(store, start=utc(2020), end=utc(2025))
    table = store.read(DataKey(Dataset.SERIES, ("fred", "GDP")))
    assert table.schema.equals(SERIES_SCHEMA)
    (row,) = table.to_pylist()
    # 1 January plus 30 days ends at midnight in New York, 05:00 UTC in winter.
    assert (row["published"], row["estimated"]) == (utc(2024, 2, 1, 5), True)


def test_day_end_follows_daylight_saving():
    """New York midnight is 05:00 UTC in winter and 04:00 in summer; a fixed offset would publish an hour early."""
    assert day_end(date(2024, 1, 31), "America/New_York") == utc(2024, 2, 1, 5)
    assert day_end(date(2024, 7, 1), "America/New_York") == utc(2024, 7, 2, 4)


def test_retry_headers_are_rebuilt_per_attempt(make_client):
    """eToro deduplicates on the request id, so a retry that reuses it may be dropped."""
    seen = []
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-request-id"])
        return httpx.Response(next(statuses), json={})

    counter = iter("ab")
    make_client(handler).get("https://example.com", headers=lambda: {"x-request-id": next(counter)})
    assert seen == ["a", "b"]


def test_http_error_does_not_leak_the_query(make_client):
    """FRED and EIA carry the API key in the query string, which httpx prints in its own error text."""
    client = make_client(lambda request: httpx.Response(400, text="series does not exist"))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        client.get("https://api.example.com/data", {"api_key": "secret"})
    assert "secret" not in str(raised.value)
    assert "400 from https://api.example.com/data: series does not exist" in str(raised.value)
