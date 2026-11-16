"""Names the command line uses for groups of fetchers."""

from collections.abc import Callable, Sequence

from tsfm.constants import GRANULARITIES
from tsfm.data.fetchers.base import APIDataFetcher
from tsfm.data.fetchers.bis import BisPolicyRateFetcher
from tsfm.data.fetchers.cftc import CftcPositioningFetcher
from tsfm.data.fetchers.eia import EiaSeriesFetcher
from tsfm.data.fetchers.etoro import EtoroCandleFetcher
from tsfm.data.fetchers.events import CentralBankMeetingFetcher
from tsfm.data.fetchers.fred import FredReleaseCalendarFetcher, FredSeriesFetcher
from tsfm.data.fetchers.indices import INDEX_FETCHERS
from tsfm.data.fetchers.oanda import OandaBookFetcher, OandaCandleFetcher
from tsfm.data.fetchers.portwatch import PortWatchFetcher
from tsfm.enums import Granularity, InstrumentSet, PullTarget
from tsfm.settings import Settings

type Factory = Callable[[Settings, Sequence[Granularity]], list[APIDataFetcher]]

# Only the candle fetchers use the bar sizes; the other factories ignore them.
_FACTORIES: dict[PullTarget, Factory] = {
    PullTarget.OANDA_FX: lambda settings, sizes: [OandaCandleFetcher(settings, granularities=sizes)],
    PullTarget.OANDA_OTHER: lambda settings, sizes: [
        OandaCandleFetcher(settings, granularities=sizes, instrument_set=InstrumentSet.COVARIATES)
    ],
    PullTarget.OANDA_HELD_OUT: lambda settings, sizes: [
        OandaCandleFetcher(settings, granularities=sizes, instrument_set=InstrumentSet.HELD_OUT)
    ],
    PullTarget.OANDA_BOOKS: lambda settings, _: [OandaBookFetcher(settings)],
    PullTarget.ETORO: lambda settings, sizes: [EtoroCandleFetcher(settings, granularities=sizes)],
    PullTarget.FRED: lambda settings, _: [FredSeriesFetcher(settings)],
    PullTarget.FRED_CALENDAR: lambda settings, _: [FredReleaseCalendarFetcher(settings)],
    PullTarget.RATES: lambda settings, _: [BisPolicyRateFetcher(settings)],
    PullTarget.CFTC: lambda settings, _: [CftcPositioningFetcher(settings)],
    PullTarget.EIA: lambda settings, _: [EiaSeriesFetcher(settings)],
    PullTarget.PORTWATCH: lambda settings, _: [PortWatchFetcher(settings)],
    PullTarget.INDICES: lambda settings, _: [fetcher(settings) for fetcher in INDEX_FETCHERS],
    PullTarget.CENTRAL_BANKS: lambda settings, _: [CentralBankMeetingFetcher(settings)],
}


def build_fetchers(
    target: PullTarget, settings: Settings, granularities: Sequence[Granularity] = GRANULARITIES
) -> list[APIDataFetcher]:
    """
    Build the fetchers behind a pull target.

    Parameters
    ----------
    target : PullTarget
        Name given on the command line.
    settings : Settings
        Credentials and data location.
    granularities : Sequence[Granularity], default: GRANULARITIES
        Bar sizes, used by the candle targets only.

    Returns
    -------
    list[APIDataFetcher]
        One fetcher, or several for a target that groups sources.

    Examples
    --------
    >>> for fetcher in build_fetchers(PullTarget.INDICES, Settings()):  # doctest: +SKIP
    ...     fetcher.pull(store)
    """
    return _FACTORIES[target](settings, granularities)
