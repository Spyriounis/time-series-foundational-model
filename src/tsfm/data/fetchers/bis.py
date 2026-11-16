"""BIS fetcher: daily central-bank policy rates."""

import csv
import io
from datetime import datetime

import pyarrow as pa

from tsfm.constants import (
    BIS_DATE_COLUMN,
    BIS_FREQUENCY,
    BIS_MISSING_VALUE,
    BIS_POLICY_RATE_AREAS,
    BIS_POLICY_RATES_URL,
    BIS_PUBLICATION_DELAY,
    BIS_REQUESTS_PER_SECOND,
    BIS_SERIES_PREFIX,
    BIS_VALUE_COLUMN,
)
from tsfm.data.fetchers.base import SeriesFetcher, observations_table
from tsfm.data.fetchers.parsing import clip, optional_float
from tsfm.enums import Source
from tsfm.utils.http import Params
from tsfm.utils.time import format_date, parse_utc

# Stored series name -> BIS reference area.
_AREAS: dict[str, str] = {
    f"{BIS_SERIES_PREFIX}_{currency.lower()}": area for currency, area in BIS_POLICY_RATE_AREAS.items()
}


class BisPolicyRateFetcher(SeriesFetcher):
    """
    Pull one daily policy-rate series per currency, named ``policy_rate_{currency}``.

    Examples
    --------
    >>> BisPolicyRateFetcher(Settings()).pull(store, only=["policy_rate_usd"])  # doctest: +SKIP
    """

    source = Source.BIS
    requests_per_second = BIS_REQUESTS_PER_SECOND
    publication_delay = BIS_PUBLICATION_DELAY

    def series_ids(self) -> list[str]:
        """
        List the policy-rate series.

        Returns
        -------
        list[str]
            Series names.
        """
        return list(_AREAS)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Download one policy-rate series.

        Parameters
        ----------
        series_id : str
            Series name.
        start : datetime
            Inclusive lower bound.
        end : datetime
            Exclusive upper bound.

        Returns
        -------
        pa.Table
            Rows in ``OBSERVATIONS_SCHEMA``.
        """
        url = f"{BIS_POLICY_RATES_URL}/{BIS_FREQUENCY}.{_AREAS[series_id]}"
        params: Params = {"format": "csv", "startPeriod": format_date(start), "endPeriod": format_date(end)}
        # The API's end date is inclusive; ours is not.
        return clip(_parse_csv(self.client.get(url, params).text), start, end)


def _parse_csv(text: str) -> pa.Table:
    """
    Convert a BIS CSV export to a series.

    Parameters
    ----------
    text : str
        CSV with one row per day, among other columns a date and a value.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``; days without a fixing become null.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    times = [parse_utc(row[BIS_DATE_COLUMN]) for row in rows]
    values = [optional_float(row[BIS_VALUE_COLUMN], BIS_MISSING_VALUE) for row in rows]
    return observations_table(times, values)
