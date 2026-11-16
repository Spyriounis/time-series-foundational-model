"""Fetchers for indices published as one file each."""

import io
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow as pa

from tsfm.constants import (
    CRYPTO_FEAR_GREED_PUBLICATION_DELAY,
    CRYPTO_FEAR_GREED_SERIES_ID,
    CRYPTO_FEAR_GREED_URL,
    EPU_GLOBAL_PUBLICATION_DELAY,
    EPU_GLOBAL_SERIES_ID,
    EPU_GLOBAL_URL,
    EPU_GLOBAL_VALUE_COLUMN,
    EPU_MONTH_COLUMN,
    EPU_US_PUBLICATION_DELAY,
    EPU_US_SERIES_ID,
    EPU_US_URL,
    EPU_US_VALUE_COLUMN,
    EPU_YEAR_COLUMN,
    GPR_DAILY_URL,
    GPR_DATE_COLUMN,
    GPR_PUBLICATION_DELAY,
    GPR_SERIES_ID,
    GPR_VALUE_COLUMN,
    GSCPI_PUBLICATION_DELAY,
    GSCPI_SERIES_ID,
    GSCPI_SHEET,
    GSCPI_URL,
    INDICES_REQUESTS_PER_SECOND,
    SF_FED_SENTIMENT_DATE_COLUMN,
    SF_FED_SENTIMENT_PUBLICATION_DELAY,
    SF_FED_SENTIMENT_SERIES_ID,
    SF_FED_SENTIMENT_SHEET,
    SF_FED_SENTIMENT_URL,
    SF_FED_SENTIMENT_VALUE_COLUMN,
    TIME_COLUMN,
)
from tsfm.data.fetchers.base import FileFetcher, observations_table
from tsfm.data.schemas import OBSERVATIONS_SCHEMA
from tsfm.enums import Source


class _IndexFetcher(FileFetcher):
    """Hold what every index file fetcher shares: source and rate limit."""

    source = Source.INDICES
    requests_per_second = INDICES_REQUESTS_PER_SECOND


class GeopoliticalRiskFetcher(_IndexFetcher):
    """
    Pull the daily Geopolitical Risk index of Caldara and Iacoviello.

    Examples
    --------
    >>> GeopoliticalRiskFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = GPR_DAILY_URL
    publication_delay = GPR_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the daily index from the ``.xls`` file.

        Parameters
        ----------
        content : bytes
            Raw workbook.

        Returns
        -------
        dict[str, pa.Table]
            ``gpr_daily`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        return {GPR_SERIES_ID: _dated_series(_read_excel(content), GPR_DATE_COLUMN, GPR_VALUE_COLUMN)}


class EconomicPolicyUncertaintyFetcher(_IndexFetcher):
    """
    Pull the monthly US news-based Economic Policy Uncertainty index.

    Examples
    --------
    >>> EconomicPolicyUncertaintyFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = EPU_US_URL
    publication_delay = EPU_US_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the US index from the ``.xlsx`` file.

        Parameters
        ----------
        content : bytes
            Raw workbook.

        Returns
        -------
        dict[str, pa.Table]
            ``epu_us`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        return {EPU_US_SERIES_ID: _monthly_series(_read_excel(content), EPU_US_VALUE_COLUMN)}


class GlobalPolicyUncertaintyFetcher(_IndexFetcher):
    """
    Pull the monthly Global Economic Policy Uncertainty index.

    Examples
    --------
    >>> GlobalPolicyUncertaintyFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = EPU_GLOBAL_URL
    publication_delay = EPU_GLOBAL_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the global index from the ``.xlsx`` file.

        Parameters
        ----------
        content : bytes
            Raw workbook.

        Returns
        -------
        dict[str, pa.Table]
            ``epu_global`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        return {EPU_GLOBAL_SERIES_ID: _monthly_series(_read_excel(content), EPU_GLOBAL_VALUE_COLUMN)}


class NewsSentimentFetcher(_IndexFetcher):
    """
    Pull the San Francisco Fed's Daily News Sentiment index.

    Examples
    --------
    >>> NewsSentimentFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = SF_FED_SENTIMENT_URL
    publication_delay = SF_FED_SENTIMENT_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the index from the ``Data`` sheet of the ``.xlsx`` file.

        Parameters
        ----------
        content : bytes
            Raw workbook.

        Returns
        -------
        dict[str, pa.Table]
            ``news_sentiment`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        frame = _read_excel(content, sheet=SF_FED_SENTIMENT_SHEET)
        series = _dated_series(frame, SF_FED_SENTIMENT_DATE_COLUMN, SF_FED_SENTIMENT_VALUE_COLUMN)
        return {SF_FED_SENTIMENT_SERIES_ID: series}


class CryptoFearGreedFetcher(_IndexFetcher):
    """
    Pull the daily Crypto Fear and Greed index of alternative.me.

    Examples
    --------
    >>> CryptoFearGreedFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = CRYPTO_FEAR_GREED_URL
    publication_delay = CRYPTO_FEAR_GREED_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the index from the JSON body.

        Parameters
        ----------
        content : bytes
            Raw JSON, with values as strings and times in epoch seconds.

        Returns
        -------
        dict[str, pa.Table]
            ``crypto_fear_greed`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        records = json.loads(content)["data"]
        times = [datetime.fromtimestamp(int(record["timestamp"]), UTC) for record in records]
        values = [float(record["value"]) for record in records]
        return {CRYPTO_FEAR_GREED_SERIES_ID: observations_table(times, values)}


class SupplyChainPressureFetcher(_IndexFetcher):
    """
    Pull the New York Fed's monthly Global Supply Chain Pressure Index.

    Examples
    --------
    >>> SupplyChainPressureFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    url = GSCPI_URL
    publication_delay = GSCPI_PUBLICATION_DELAY

    def parse(self, content: bytes) -> dict[str, pa.Table]:
        """
        Extract the index from the first two columns of the ``.xlsx`` sheet.

        Parameters
        ----------
        content : bytes
            Raw workbook.

        Returns
        -------
        dict[str, pa.Table]
            ``gscpi`` rows in ``OBSERVATIONS_SCHEMA``.
        """
        # Read without a header: title rows above the data fail date parsing and drop out.
        frame = _read_excel(content, sheet=GSCPI_SHEET, header=None)
        return {GSCPI_SERIES_ID: _dated_series(frame, frame.columns[0], frame.columns[1])}


INDEX_FETCHERS: tuple[type[FileFetcher], ...] = (
    GeopoliticalRiskFetcher,
    EconomicPolicyUncertaintyFetcher,
    GlobalPolicyUncertaintyFetcher,
    NewsSentimentFetcher,
    CryptoFearGreedFetcher,
    SupplyChainPressureFetcher,
)


def _read_excel(content: bytes, sheet: str | int = 0, header: int | None = 0) -> pd.DataFrame:
    """
    Read one sheet of an ``.xls`` or ``.xlsx`` workbook.

    Parameters
    ----------
    content : bytes
        Raw workbook; pandas detects the format.
    sheet : str or int, default: 0
        Sheet name or position.
    header : int or None, default: 0
        Row holding the column names, or None if the sheet has none.

    Returns
    -------
    pd.DataFrame
        The sheet.
    """
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=header)


def _dated_series(frame: pd.DataFrame, date_column: str | int, value_column: str | int) -> pa.Table:
    """
    Convert a date column and a value column to a series.

    Parameters
    ----------
    frame : pd.DataFrame
        Sheet holding both columns.
    date_column : str or int
        Label of the column of dates.
    value_column : str or int
        Label of the column of observations.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``; rows without a parseable date are dropped and
        non-numeric values become null.
    """
    times = pd.to_datetime(frame[date_column], errors="coerce", utc=True, format="mixed")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    dated = times.notna()
    columns: dict[str, Any] = {
        TIME_COLUMN: pa.Array.from_pandas(times[dated]),
        "value": pa.Array.from_pandas(values[dated]),
    }
    return pa.table(columns).cast(OBSERVATIONS_SCHEMA)


def _monthly_series(frame: pd.DataFrame, value_column: str) -> pa.Table:
    """
    Convert year and month columns and a value column to a series.

    Parameters
    ----------
    frame : pd.DataFrame
        Sheet with the ``Year`` and ``Month`` columns of the policy-uncertainty files.
    value_column : str
        Label of the column of observations.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``, dated the first of each month; footnote rows
        without a numeric year are dropped.
    """
    year = pd.to_numeric(frame[EPU_YEAR_COLUMN], errors="coerce")
    months = frame[year.notna()]
    parts = {"year": year[year.notna()], "month": months[EPU_MONTH_COLUMN], "day": 1}
    dated = pd.DataFrame({"date": pd.to_datetime(pd.DataFrame(parts)), "value": months[value_column]})
    return _dated_series(dated, "date", "value")
