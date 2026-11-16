"""IMF PortWatch fetcher: daily ship transits through maritime chokepoints."""

from datetime import datetime
from typing import Any

import pyarrow as pa

from tsfm.constants import (
    PORTWATCH_CHOKEPOINT_IDS,
    PORTWATCH_CHOKEPOINTS_URL,
    PORTWATCH_DATE_FIELD,
    PORTWATCH_ID_FIELD,
    PORTWATCH_PAGE_SIZE,
    PORTWATCH_PUBLICATION_DELAY,
    PORTWATCH_REQUESTS_PER_SECOND,
    PORTWATCH_SERIES_PREFIX,
    PORTWATCH_VALUE_FIELD,
)
from tsfm.data.fetchers.base import SeriesFetcher, observations_table
from tsfm.data.fetchers.parsing import clip, optional_float
from tsfm.enums import Source
from tsfm.utils.time import parse_utc

# Stored series name -> PortWatch chokepoint id.
_CHOKEPOINTS: dict[str, str] = {f"{PORTWATCH_SERIES_PREFIX}_{port}": port for port in PORTWATCH_CHOKEPOINT_IDS}


class PortWatchFetcher(SeriesFetcher):
    """
    Pull the daily transit count of each chokepoint, named ``portwatch_{chokepoint_id}``.

    Examples
    --------
    >>> PortWatchFetcher(Settings()).pull(store, only=["portwatch_chokepoint1"])  # doctest: +SKIP
    """

    source = Source.PORTWATCH
    requests_per_second = PORTWATCH_REQUESTS_PER_SECOND
    publication_delay = PORTWATCH_PUBLICATION_DELAY

    def series_ids(self) -> list[str]:
        """
        List one series per chokepoint.

        Returns
        -------
        list[str]
            Series names.
        """
        return list(_CHOKEPOINTS)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Download one chokepoint's history and cut it to the bounds.

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

        Raises
        ------
        RuntimeError
            If ArcGIS answers with an error, e.g. while PortWatch withdraws the data for its
            weekly update.
        """
        params: dict[str, str | int] = {
            "where": f"{PORTWATCH_ID_FIELD}='{_CHOKEPOINTS[series_id]}'",
            "outFields": f"{PORTWATCH_DATE_FIELD},{PORTWATCH_VALUE_FIELD}",
            "orderByFields": PORTWATCH_DATE_FIELD,
            "resultRecordCount": PORTWATCH_PAGE_SIZE,
            "f": "json",
        }
        features: list[dict[str, Any]] = []
        while True:
            payload = self.client.get_json(PORTWATCH_CHOKEPOINTS_URL, {**params, "resultOffset": len(features)})
            # ArcGIS reports errors in the body of a 200 response.
            if "error" in payload:
                error = payload["error"]
                raise RuntimeError(f"PortWatch answered {error.get('code')}: {error.get('message')}")
            features += payload["features"]
            if not payload["features"] or not payload.get("exceededTransferLimit"):
                break
        return clip(_parse_features(features), start, end)


def _parse_features(features: list[dict[str, Any]]) -> pa.Table:
    """
    Convert ArcGIS features to a series.

    Parameters
    ----------
    features : list[dict[str, Any]]
        Features whose attributes hold a ``YYYY-MM-DD`` date and a count.

    Returns
    -------
    pa.Table
        Rows in ``OBSERVATIONS_SCHEMA``.
    """
    attributes = [feature["attributes"] for feature in features]
    times = [parse_utc(row[PORTWATCH_DATE_FIELD]) for row in attributes]
    values = [optional_float(row[PORTWATCH_VALUE_FIELD]) for row in attributes]
    return observations_table(times, values)
