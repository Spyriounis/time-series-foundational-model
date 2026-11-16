"""CFTC fetcher: weekly futures positioning from the Traders in Financial Futures report."""

from datetime import datetime

import pyarrow as pa

from tsfm.constants import (
    CFTC_CODE_FIELD,
    CFTC_CONTRACT_CODES,
    CFTC_DATE_FIELD,
    CFTC_OPEN_INTEREST_FIELD,
    CFTC_PAGE_SIZE,
    CFTC_PUBLICATION_DELAY,
    CFTC_REQUESTS_PER_SECOND,
    CFTC_SERIES_PREFIX,
    CFTC_TFF_URL,
    CFTC_TRADER_GROUPS,
)
from tsfm.data.fetchers.base import SeriesFetcher, observations_table
from tsfm.data.fetchers.parsing import clip, optional_float
from tsfm.enums import Source
from tsfm.settings import Settings
from tsfm.utils.http import HttpClient
from tsfm.utils.time import parse_utc

type Report = dict[str, str]


def _series_index() -> dict[str, tuple[str, str | None]]:
    """
    Name every series and say where its numbers come from.

    Returns
    -------
    dict[str, tuple[str, str or None]]
        Series name to currency and trader group; the group is None for open interest.
    """
    index: dict[str, tuple[str, str | None]] = {}
    for currency in CFTC_CONTRACT_CODES:
        stem = f"{CFTC_SERIES_PREFIX}_{currency.lower()}"
        index |= {f"{stem}_{group}_net": (currency, group) for group in CFTC_TRADER_GROUPS}
        index[f"{stem}_open_interest"] = (currency, None)
    return index


_SERIES = _series_index()


class CftcPositioningFetcher(SeriesFetcher):
    """
    Pull net positions per trader group, and open interest, of each currency future.

    Series are named ``cot_{currency}_{group}_net`` and ``cot_{currency}_open_interest``.

    Parameters
    ----------
    settings : Settings
        Credentials and data location.
    client : HttpClient or None, default: None
        Client override.

    Examples
    --------
    >>> CftcPositioningFetcher(Settings()).pull(store, only=["cot_eur_leveraged_net"])  # doctest: +SKIP
    """

    source = Source.CFTC
    requests_per_second = CFTC_REQUESTS_PER_SECOND
    publication_delay = CFTC_PUBLICATION_DELAY

    def __init__(self, settings: Settings, client: HttpClient | None = None) -> None:
        super().__init__(settings, client)
        self._reports: dict[tuple[str, datetime], list[Report]] = {}

    def series_ids(self) -> list[str]:
        """
        List the positioning series.

        Returns
        -------
        list[str]
            Series names.
        """
        return list(_SERIES)

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pa.Table:
        """
        Build one series from its currency's weekly reports.

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
        currency, group = _SERIES[series_id]
        reports = self._currency_reports(currency, start)
        times = [parse_utc(report[CFTC_DATE_FIELD]) for report in reports]
        values = [_open_interest(report) if group is None else _net_position(report, group) for report in reports]
        return clip(observations_table(times, values), start, end)

    def _currency_reports(self, currency: str, start: datetime) -> list[Report]:
        """
        Return a currency's reports since ``start``, downloading them once.

        Parameters
        ----------
        currency : str
            Currency of the futures contract.
        start : datetime
            Inclusive lower bound on the report date.

        Returns
        -------
        list[Report]
            Weekly reports, oldest first.
        """
        # One report row feeds all of a currency's series, which a pull requests one after another.
        if (currency, start) not in self._reports:
            self._reports[currency, start] = self._fetch_reports(currency, start)
        return self._reports[currency, start]

    def _fetch_reports(self, currency: str, start: datetime) -> list[Report]:
        """
        Download a currency's reports since ``start``, page by page.

        Parameters
        ----------
        currency : str
            Currency of the futures contract.
        start : datetime
            Inclusive lower bound on the report date.

        Returns
        -------
        list[Report]
            Weekly reports, oldest first.
        """
        params: dict[str, str | int] = {
            "$where": _where(CFTC_CONTRACT_CODES[currency], start),
            "$order": CFTC_DATE_FIELD,
            "$limit": CFTC_PAGE_SIZE,
        }
        reports: list[Report] = []
        while True:
            page: list[Report] = self.client.get_json(CFTC_TFF_URL, {**params, "$offset": len(reports)})
            if not page:
                return reports
            reports += page


def _where(contract_code: str, start: datetime) -> str:
    """
    Build the Socrata filter for one contract since a date.

    Parameters
    ----------
    contract_code : str
        CFTC contract market code.
    start : datetime
        Inclusive lower bound on the report date.

    Returns
    -------
    str
        SoQL ``$where`` clause.
    """
    return f"{CFTC_CODE_FIELD}='{contract_code}' AND {CFTC_DATE_FIELD} >= '{start:%Y-%m-%dT%H:%M:%S}'"


def _net_position(report: Report, group: str) -> float | None:
    """
    Compute a trader group's long minus short contracts.

    Parameters
    ----------
    report : Report
        One weekly report row.
    group : str
        Key of ``CFTC_TRADER_GROUPS``.

    Returns
    -------
    float or None
        Net contracts, or None if either side is absent.
    """
    long_field, short_field = CFTC_TRADER_GROUPS[group]
    long, short = optional_float(report.get(long_field)), optional_float(report.get(short_field))
    return None if long is None or short is None else long - short


def _open_interest(report: Report) -> float | None:
    """
    Read the total open interest of a report.

    Parameters
    ----------
    report : Report
        One weekly report row.

    Returns
    -------
    float or None
        Open contracts, or None if absent.
    """
    return optional_float(report.get(CFTC_OPEN_INTEREST_FIELD))
