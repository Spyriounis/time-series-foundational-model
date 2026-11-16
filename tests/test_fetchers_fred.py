import httpx
import pytest

from helpers import utc
from tsfm.constants import FRED_PAGE_SIZE, FRED_RELEASES, TIME_COLUMN
from tsfm.data.fetchers.fred import (
    FredReleaseCalendarFetcher,
    FredSeriesFetcher,
    _archived_delay,
    _parse_observations,
    _release_time,
)
from tsfm.data.storage import DataKey
from tsfm.enums import Dataset, Source

OBSERVATIONS = [
    {"date": "2024-01-01", "value": "5.33"},
    {"date": "2024-01-02", "value": "."},
    {"date": "2024-01-03", "value": "5.31"},
]
# FRED's archive of past releases: the dates it holds, each value's first release, and today's values.
ARCHIVE = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
FIRST_RELEASES = [
    {"date": "2024-01-01", "value": "5.33", "realtime_start": "2024-01-02"},
    {"date": "2024-01-02", "value": ".", "realtime_start": "2024-01-03"},
    {"date": "2024-01-03", "value": "5.31", "realtime_start": "2024-01-05"},
]
CURRENT = [{"date": "2023-12-29", "value": "5.30"}, {"date": "2024-01-01", "value": "5.35"}]


def fred_server(requests: list[httpx.Request]):
    """Answer archive dates, first releases within a real-time window, and current values, honouring `offset`."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        if request.url.path.endswith("/vintagedates"):
            records, field = ARCHIVE, "vintage_dates"
        elif "output_type" in params:
            window = (params["realtime_start"], params["realtime_end"])
            records = [r for r in FIRST_RELEASES if window[0] <= r["realtime_start"] <= window[1]]
            records, field = [r for r in records if r["date"] >= params["observation_start"]], "observations"
        else:
            bounds = (params["observation_start"], params["observation_end"])
            records, field = [r for r in CURRENT if bounds[0] <= r["date"] <= bounds[1]], "observations"
        return httpx.Response(200, json={"count": len(records), field: records[int(params["offset"]) :]})

    return handler


def paged(records: list[dict], field: str, requests: list[httpx.Request], page_size: int = 2):
    """Serve records `page_size` at a time, honouring `offset`."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["offset"])
        page = records[offset : offset + page_size]
        return httpx.Response(200, json={"count": len(records), "offset": offset, field: page})

    return handler


def test_first_releases_come_from_archive_chunks(settings, make_client, monkeypatch):
    """FRED refuses more than 2,000 archive dates per request; each value must keep its first release, not today's."""
    monkeypatch.setattr("tsfm.data.fetchers.fred.FRED_VINTAGES_PER_REQUEST", 2)
    requests: list[httpx.Request] = []
    table = FredSeriesFetcher(settings, make_client(fred_server(requests))).fetch_series(
        "DFF", utc(2024, 1, 1), utc(2024, 1, 4)
    )
    chunks = [request.url.params for request in requests if "output_type" in request.url.params]
    assert [(chunk["realtime_start"], chunk["realtime_end"]) for chunk in chunks] == [
        ("2024-01-02", "2024-01-03"),
        ("2024-01-04", "2024-01-05"),
    ]
    assert (chunks[0]["output_type"], chunks[0]["observation_start"]) == ("4", "2024-01-01")
    assert (chunks[0]["series_id"], chunks[0]["api_key"], chunks[0]["limit"]) == (
        "DFF",
        "fred-key",
        str(FRED_PAGE_SIZE),
    )
    assert table[TIME_COLUMN].to_pylist() == [utc(2024, 1, 1), utc(2024, 1, 2), utc(2024, 1, 3)]
    # 5.33 was first published; 5.35 is today's revision of the same day.
    assert table["value"].to_pylist() == [5.33, None, 5.31]
    # A value counts as public at the end of its archive date in New York.
    assert table["published"].to_pylist() == [utc(2024, 1, 3, 5), utc(2024, 1, 4, 5), utc(2024, 1, 6, 5)]
    assert table["estimated"].to_pylist() == [False, False, False]


def test_values_before_the_archive_get_an_estimated_publication(settings, make_client):
    """History older than the archive has no recorded date; its delay is a high quantile of the recorded ones."""
    requests: list[httpx.Request] = []
    table = FredSeriesFetcher(settings, make_client(fred_server(requests))).fetch_series(
        "DFF", utc(2023, 12, 1), utc(2024, 2, 1)
    )
    first = table.to_pylist()[0]
    # Recorded delays are 1, 1 and 2 days, so 29 December counts as published at the end of 31 December.
    assert (first[TIME_COLUMN], first["value"]) == (utc(2023, 12, 29), 5.30)
    assert (first["published"], first["estimated"]) == (utc(2024, 1, 1, 5), True)
    assert table["estimated"].to_pylist() == [True, False, False, False]


def test_a_resumed_pull_reads_only_recent_archive_dates(settings, make_client):
    """A value cannot be released before its date, so a pull resuming from a date skips the older archive."""
    requests: list[httpx.Request] = []
    FredSeriesFetcher(settings, make_client(fred_server(requests))).fetch_series("DFF", utc(2024, 1, 4), utc(2025))
    assert [request.url.path for request in requests] == ["/fred/series/vintagedates", "/fred/series/observations"]
    assert requests[1].url.params["realtime_start"] == "2024-01-04"


def test_estimate_needs_an_archived_release():
    """Without any recorded release there is nothing to base a delay on; guessing zero would leak."""
    with pytest.raises(ValueError, match="DFF"):
        _archived_delay([], "DFF")


def test_series_stops_on_an_empty_page(settings, make_client):
    """A count that overstates the data must not turn into an endless loop."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"count": 10, "vintage_dates": [], "observations": []})

    assert FredSeriesFetcher(settings, make_client(handler)).fetch_series("DFF", utc(2024), utc(2025)).num_rows == 0
    assert len(requests) == 1


def test_missing_observation_becomes_null():
    """FRED writes a missing value as "."; storing it as zero would fake a rate cut."""
    assert _parse_observations(OBSERVATIONS)["value"].to_pylist() == [5.33, None, 5.31]


def test_release_time_follows_daylight_saving():
    """08:30 in New York is 13:30 UTC in winter but 12:30 in summer; a fixed offset would misplace events."""
    assert _release_time("2024-01-11", "08:30") == utc(2024, 1, 11, 13, 30)
    assert _release_time("2024-07-11", "08:30") == utc(2024, 7, 11, 12, 30)


def test_calendar_keeps_scheduled_dates_beyond_end(settings, make_client):
    """Future release dates are known covariates, so `end` must not cut them off."""
    requests: list[httpx.Request] = []
    dates = [{"release_id": 10, "date": "2024-01-11"}, {"release_id": 10, "date": "2030-01-11"}]
    fetcher = FredReleaseCalendarFetcher(settings, make_client(paged(dates, "release_dates", requests, page_size=1)))
    (key,) = fetcher.list_keys()
    assert key == DataKey(Dataset.EVENTS, (Source.FRED,))
    tables = list(fetcher.fetch(key, utc(2024), utc(2025)))
    assert len(tables) == len(FRED_RELEASES)
    assert tables[0].to_pylist() == [
        {TIME_COLUMN: utc(2024, 1, 11, 13, 30), "event_id": "US_CPI", "currency": "USD"},
        {TIME_COLUMN: utc(2030, 1, 11, 13, 30), "event_id": "US_CPI", "currency": "USD"},
    ]
    assert requests[0].url.path == "/fred/release/dates"
    first = requests[0].url.params
    assert (first["release_id"], first["api_key"], first["sort_order"]) == ("10", "fred-key", "asc")
    assert first["include_release_dates_with_no_data"] == "true"
    assert [request.url.params["offset"] for request in requests[:2]] == ["0", "1"]
