"""Central-bank rate-decision calendar, read from a CSV packaged with the library."""

import csv
import io
from collections.abc import Iterator
from datetime import datetime
from importlib import resources

import pyarrow as pa

from tsfm.constants import CENTRAL_BANK_EVENT_PREFIX, CENTRAL_BANK_MEETINGS_RESOURCE, RESOURCES_PACKAGE
from tsfm.data.fetchers.base import APIDataFetcher
from tsfm.data.fetchers.parsing import events_table
from tsfm.data.storage import DataKey
from tsfm.enums import Dataset, PullMode, Source
from tsfm.utils.time import parse_utc


class CentralBankMeetingFetcher(APIDataFetcher):
    """
    Load past and scheduled G10 monetary-policy meetings into the store.

    Examples
    --------
    >>> CentralBankMeetingFetcher(Settings()).pull(store)  # doctest: +SKIP
    """

    source = Source.CENTRAL_BANKS
    dataset = Dataset.EVENTS
    mode = PullMode.REPLACE

    def list_keys(self) -> list[DataKey]:
        """
        List the single meetings table.

        Returns
        -------
        list[DataKey]
            One table address.
        """
        return [DataKey(self.dataset, (self.source,))]

    def fetch(self, key: DataKey, start: datetime, end: datetime) -> Iterator[pa.Table]:
        """
        Read every meeting from the packaged CSV, with no request.

        The bounds are not applied: scheduled meetings lie beyond ``end``.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Unused.
        end : datetime
            Unused.

        Yields
        ------
        pa.Table
            All meetings, in ``EVENTS_SCHEMA``.
        """
        resource = resources.files(RESOURCES_PACKAGE).joinpath(CENTRAL_BANK_MEETINGS_RESOURCE)
        yield _parse_meetings(resource.read_text(encoding="utf-8"))


def _parse_meetings(text: str) -> pa.Table:
    """
    Convert the meetings CSV to events.

    Parameters
    ----------
    text : str
        CSV with columns ``date``, ``time_utc``, ``bank``, ``currency`` and ``source_url``.

    Returns
    -------
    pa.Table
        Rows in ``EVENTS_SCHEMA``; a meeting without an announcement time is set at midnight UTC.
    """
    meetings = list(csv.DictReader(io.StringIO(text)))
    times = [parse_utc(f"{meeting['date']}T{meeting['time_utc'] or '00:00'}") for meeting in meetings]
    event_ids = [f"{CENTRAL_BANK_EVENT_PREFIX}_{meeting['bank']}" for meeting in meetings]
    return events_table(times, event_ids, [meeting["currency"] for meeting in meetings])
