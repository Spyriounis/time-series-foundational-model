"""EIA fetcher: weekly US petroleum stocks."""

from datetime import datetime
from typing import Any

import pyarrow as pa

from tsfm.constants import (
    EIA_FREQUENCY,
    EIA_PAGE_SIZE,
    EIA_PUBLICATION_DELAY,
    EIA_REQUESTS_PER_SECOND,
    EIA_SERIES,
    EIA_STOCKS_URL,
)
from tsfm.data.fetchers.base import SeriesFetcher, observations_table
from tsfm.data.fetchers.parsing import clip, optional_float
from tsfm.enums import Source
from tsfm.utils.time import format_date, parse_utc


class EiaSeriesFetcher(SeriesFetcher):
    """
    Pull the stock series in ``EIA_SERIES``.

    Examples
    --------
    >>> EiaSeriesFetcher(Settings()).pull(store, only=["WCESTUS1"])  # doctest: +SKIP
    """

    source = Source.EIA
    requests_per_second = EIA_REQUESTS_PER_SECOND
    publication_delay = EIA_PUBLICATION_DELAY

    def series_ids(self) -> list[str]:
        """
        List the configured EIA series.

        Returns
        -------
        list[str]
            EIA series ids.
        """
        return list(EIA_SERIES)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Download one weekly series, page by page.

        Parameters
        ----------
        series_id : str
            EIA series id.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        pa.Table
            Rows in ``OBSERVATIONS_SCHEMA``.
        """
        params = self._params(series_id, start, end)
        records: list[dict[str, Any]] = []
        while True:
            response = self.client.get_json(EIA_STOCKS_URL, {**params, "offset": len(records)})["response"]
            records += response["data"]
            if not response["data"] or len(records) >= int(response["total"]):
                break
        # The API's end date may be inclusive; ours is not.
        return clip(_parse_records(records), start, end)

    def _params(self, series_id: str, start: datetime, end: datetime) -> dict[str, str | int]:
        """
        Build the query of one series without the offset.

        Parameters
        ----------
        series_id : str
            EIA series id.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        dict[str, str | int]
            Query parameters in the API's bracketed style.
        """
        return {
            "api_key": self.settings.secret("eia_api_key"),
            "frequency": EIA_FREQUENCY,
            "data[0]": "value",
            "facets[series][]": series_id,
            "start": format_date(start),
            "end": format_date(end),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": EIA_PAGE_SIZE,
        }


def _parse_records(records: list[dict[str, Any]]) -> pa.Table:
    """
    Convert EIA records to a series.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records with a ``period`` date and a nullable ``value``.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``.
    """
    times = [parse_utc(record["period"]) for record in records]
    values = [optional_float(record["value"]) for record in records]
    return observations_table(times, values)
