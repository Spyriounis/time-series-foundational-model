import httpx
import pytest

from helpers import utc
from tsfm.constants import CFTC_PAGE_SIZE, EIA_PAGE_SIZE, PORTWATCH_PAGE_SIZE, TIME_COLUMN
from tsfm.data.fetchers.bis import BisPolicyRateFetcher
from tsfm.data.fetchers.cftc import CftcPositioningFetcher
from tsfm.data.fetchers.eia import EiaSeriesFetcher
from tsfm.data.fetchers.portwatch import PortWatchFetcher

JAN_2, JAN_9 = utc(2024, 1, 2), utc(2024, 1, 9)


BIS_CSV = """FREQ,REF_AREA,COMPILATION,TITLE,TIME_PERIOD,OBS_VALUE,OBS_STATUS
D,XM,"From 2024: deposit facility rate, fixed rate; before: main refinancing rate",Policy rates,2024-01-02,4.5,A
D,XM,"From 2024: deposit facility rate, fixed rate; before: main refinancing rate",Policy rates,2024-01-06,NaN,M
D,XM,"From 2024: deposit facility rate, fixed rate; before: main refinancing rate",Policy rates,2024-01-09,4.5,A
"""


def test_bis_parses_and_clips(settings, make_client):
    """Free-text columns hold commas, weekends are "NaN", and the API's end date is inclusive."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=BIS_CSV)

    fetcher = BisPolicyRateFetcher(settings, make_client(handler))
    table = fetcher.fetch_series("policy_rate_eur", JAN_2, JAN_9)
    assert requests[0].url.path == "/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.XM"
    assert dict(requests[0].url.params) == {"format": "csv", "startPeriod": "2024-01-02", "endPeriod": "2024-01-09"}
    assert table.to_pylist() == [{TIME_COLUMN: JAN_2, "value": 4.5}, {TIME_COLUMN: utc(2024, 1, 6), "value": None}]


def cftc_report(day: str, long: str, short: str) -> dict[str, str]:
    """One weekly report row with leveraged-fund positions and open interest."""
    return {
        "report_date_as_yyyy_mm_dd": f"{day}T00:00:00.000",
        "lev_money_positions_long": long,
        "lev_money_positions_short": short,
        "open_interest_all": "1000",
    }


def cftc_server(requests: list[httpx.Request]):
    """Serve two reports one per page, then an empty page."""
    reports = [cftc_report("2024-01-02", "300", "100"), cftc_report("2024-01-09", "50", "250")]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["$offset"])
        return httpx.Response(200, json=reports[offset : offset + 1])

    return handler


def test_cftc_request_and_paging(settings, make_client):
    """The Socrata filter selects one contract since `start`, and paging ends on the empty page."""
    requests: list[httpx.Request] = []
    fetcher = CftcPositioningFetcher(settings, make_client(cftc_server(requests)))
    table = fetcher.fetch_series("cot_eur_leveraged_net", JAN_2, utc(2025))
    assert dict(requests[0].url.params) == {
        "$where": "cftc_contract_market_code='099741' AND report_date_as_yyyy_mm_dd >= '2024-01-02T00:00:00'",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": str(CFTC_PAGE_SIZE),
        "$offset": "0",
    }
    assert [request.url.params["$offset"] for request in requests] == ["0", "1", "2"]
    assert table.to_pylist() == [{TIME_COLUMN: JAN_2, "value": 200.0}, {TIME_COLUMN: JAN_9, "value": -200.0}]


def test_cftc_series_of_a_currency_share_one_download(settings, make_client):
    """Four series come from the same report rows; refetching them would quadruple the load on the API."""
    requests: list[httpx.Request] = []
    fetcher = CftcPositioningFetcher(settings, make_client(cftc_server(requests)))
    fetcher.fetch_series("cot_eur_leveraged_net", JAN_2, utc(2025))
    downloads = len(requests)
    interest = fetcher.fetch_series("cot_eur_open_interest", JAN_2, utc(2025))
    dealers = fetcher.fetch_series("cot_eur_dealer_net", JAN_2, utc(2025))
    assert len(requests) == downloads
    assert interest["value"].to_pylist() == [1000.0, 1000.0]
    assert dealers["value"].to_pylist() == [None, None]


def test_cftc_names_four_series_per_currency(settings):
    """Each currency yields three net-position series and open interest, under stable names."""
    eur = [name for name in CftcPositioningFetcher(settings).series_ids() if name.startswith("cot_eur_")]
    assert eur == ["cot_eur_dealer_net", "cot_eur_asset_manager_net", "cot_eur_leveraged_net", "cot_eur_open_interest"]


def test_eia_request_and_paging(settings, make_client):
    """EIA's bracketed query style is easy to get wrong, and `total` arrives as a string."""
    requests: list[httpx.Request] = []
    records = [
        {"period": "2024-01-05", "series": "WCESTUS1", "value": "430000"},
        {"period": "2024-01-12", "series": "WCESTUS1", "value": None},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json={"response": {"total": "2", "data": records[offset : offset + 1]}})

    table = EiaSeriesFetcher(settings, make_client(handler)).fetch_series("WCESTUS1", utc(2024), utc(2025))
    assert requests[0].url.path == "/v2/petroleum/stoc/wstk/data/"
    assert dict(requests[0].url.params) == {
        "api_key": "eia-key",
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WCESTUS1",
        "start": "2024-01-01",
        "end": "2025-01-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": str(EIA_PAGE_SIZE),
        "offset": "0",
    }
    assert [request.url.params["offset"] for request in requests] == ["0", "1"]
    assert table["value"].to_pylist() == [430000.0, None]


def test_portwatch_pages_while_the_limit_is_exceeded(settings, make_client):
    """ArcGIS truncates silently; `exceededTransferLimit` is the only sign that more rows exist."""
    requests: list[httpx.Request] = []
    days = [utc(2024, 1, 1), JAN_2, JAN_9]
    features = [{"attributes": {"date": f"{day:%Y-%m-%d}", "n_total": 50 + i}} for i, day in enumerate(days)]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["resultOffset"])
        page = features[offset : offset + 2]
        return httpx.Response(200, json={"features": page, "exceededTransferLimit": offset + 2 < len(features)})

    table = PortWatchFetcher(settings, make_client(handler)).fetch_series("portwatch_chokepoint1", JAN_2, utc(2025))
    assert dict(requests[0].url.params) == {
        "where": "portid='chokepoint1'",
        "outFields": "date,n_total",
        "orderByFields": "date",
        "resultRecordCount": str(PORTWATCH_PAGE_SIZE),
        "f": "json",
        "resultOffset": "0",
    }
    assert [request.url.params["resultOffset"] for request in requests] == ["0", "2"]
    assert table.to_pylist() == [{TIME_COLUMN: JAN_2, "value": 51.0}, {TIME_COLUMN: JAN_9, "value": 52.0}]


def test_portwatch_reports_an_arcgis_error(settings, make_client):
    """ArcGIS answers 200 with an error body, e.g. during PortWatch's weekly update; a bare KeyError hid the cause."""
    body = {"error": {"code": 499, "message": "Token Required"}}
    fetcher = PortWatchFetcher(settings, make_client(lambda request: httpx.Response(200, json=body)))
    with pytest.raises(RuntimeError, match="499: Token Required"):
        fetcher.fetch_series("portwatch_chokepoint1", JAN_2, utc(2025))
