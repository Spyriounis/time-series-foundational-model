import io
import json

import httpx
import pandas as pd

from helpers import utc
from tsfm.constants import GPR_DATE_COLUMN, GPR_VALUE_COLUMN, TIME_COLUMN
from tsfm.data.fetchers.indices import (
    CryptoFearGreedFetcher,
    EconomicPolicyUncertaintyFetcher,
    GlobalPolicyUncertaintyFetcher,
    NewsSentimentFetcher,
    SupplyChainPressureFetcher,
    _dated_series,
)
from tsfm.data.storage import DataKey
from tsfm.enums import Dataset


def workbook(frame: pd.DataFrame, sheet: str = "Sheet1", *, header: bool = True) -> bytes:
    """An in-memory `.xlsx` file holding one sheet."""
    buffer = io.BytesIO()
    frame.to_excel(buffer, sheet_name=sheet, index=False, header=header)
    return buffer.getvalue()


def rows(table) -> list[tuple]:
    """A series table as (time, value) pairs."""
    return list(zip(table[TIME_COLUMN].to_pylist(), table["value"].to_pylist(), strict=True))


def test_dated_series_handles_the_gpr_frame():
    """xlrd cannot write an `.xls`, so the GPR parser is checked at the frame it hands over."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    frame = pd.DataFrame({"DAY": [20240101, 20240102], GPR_DATE_COLUMN: dates, GPR_VALUE_COLUMN: [101.5, None]})
    assert rows(_dated_series(frame, GPR_DATE_COLUMN, GPR_VALUE_COLUMN)) == [
        (utc(2024, 1, 1), 101.5),
        (utc(2024, 1, 2), None),
    ]


def test_epu_drops_the_footnote_rows(settings):
    """The US file ends with source notes in the Year column, which are not observations."""
    frame = pd.DataFrame(
        {
            "Year": [2024, 2024, "Source: Baker, Bloom and Davis"],
            "Month": [1, 2, None],
            "News_Based_Policy_Uncert_Index": [110.0, 120.0, None],
        }
    )
    series = EconomicPolicyUncertaintyFetcher(settings).parse(workbook(frame))
    assert rows(series["epu_us"]) == [(utc(2024, 1, 1), 110.0), (utc(2024, 2, 1), 120.0)]


def test_global_epu_reads_its_own_column(settings):
    """The global file shares the layout but names its value column differently."""
    frame = pd.DataFrame({"Year": [2024], "Month": [3], "GEPU_current": [250.0], "GEPU_ppp": [240.0]})
    series = GlobalPolicyUncertaintyFetcher(settings).parse(workbook(frame))
    assert rows(series["epu_global"]) == [(utc(2024, 3, 1), 250.0)]


def test_news_sentiment_reads_the_data_sheet(settings):
    """The SF Fed workbook keeps the series on a sheet named Data, not on the first sheet."""
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "News Sentiment": [-0.12]})
    series = NewsSentimentFetcher(settings).parse(workbook(frame, sheet="Data"))
    assert rows(series["news_sentiment"]) == [(utc(2024, 1, 1), -0.12)]


def test_gscpi_skips_rows_without_a_date(settings):
    """The NY Fed sheet has title rows of unknown number; only rows that start with a date are data."""
    frame = pd.DataFrame(
        [
            ["Global Supply Chain Pressure Index", None],
            ["Date", "GSCPI"],
            [pd.Timestamp("2024-01-31"), 0.5],
            [pd.Timestamp("2024-02-29"), -0.25],
        ]
    )
    series = SupplyChainPressureFetcher(settings).parse(workbook(frame, header=False))
    assert rows(series["gscpi"]) == [(utc(2024, 1, 31), 0.5), (utc(2024, 2, 29), -0.25)]


def test_file_fetcher_pulls_the_crypto_index_once(settings, make_client, store):
    """A file fetcher must download its file a single time and store what the parser found."""
    requests: list[httpx.Request] = []
    body = json.dumps({"data": [{"value": "45", "timestamp": "1704067200"}]}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body)

    fetcher = CryptoFearGreedFetcher(settings, make_client(handler))
    assert fetcher.pull(store) == 1
    assert len(requests) == 1
    assert requests[0].url.host == "api.alternative.me"
    assert rows(store.read(DataKey(Dataset.SERIES, ("indices", "crypto_fear_greed")))) == [(utc(2024, 1, 1), 45.0)]
