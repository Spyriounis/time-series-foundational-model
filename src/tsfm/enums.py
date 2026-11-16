"""Enumerations shared across the package."""

from datetime import timedelta
from enum import StrEnum


class Source(StrEnum):
    """Origin of a stored dataset."""

    OANDA = "oanda"
    ETORO = "etoro"
    FRED = "fred"
    BIS = "bis"
    CFTC = "cftc"
    EIA = "eia"
    PORTWATCH = "portwatch"
    INDICES = "indices"
    CENTRAL_BANKS = "central_banks"


class PullMode(StrEnum):
    """How a fetcher brings a stored table up to date."""

    RESUME = "resume"  # fetch only what is newer than the stored rows
    APPEND = "append"  # as RESUME, but never overwrite a stored row, so a value keeps its first release
    REPLACE = "replace"  # fetch everything again and drop rows the source no longer has


class Dataset(StrEnum):
    """Kind of table held in the store, one schema each."""

    BARS = "bars"
    SERIES = "series"
    EVENTS = "events"
    BOOKS = "books"
    SYNTHETIC = "synthetic"


class Granularity(StrEnum):
    """Bar size, named as OANDA names it."""

    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D = "D"

    @property
    def timedelta(self) -> timedelta:
        """
        Return the nominal bar duration.

        Returns
        -------
        timedelta
            Length of one bar.
        """
        return _GRANULARITY_DURATION[self]


_GRANULARITY_DURATION: dict[Granularity, timedelta] = {
    Granularity.M5: timedelta(minutes=5),
    Granularity.M15: timedelta(minutes=15),
    Granularity.H1: timedelta(hours=1),
    Granularity.H4: timedelta(hours=4),
    Granularity.D: timedelta(days=1),
}


class OandaEnvironment(StrEnum):
    """OANDA account type, which selects the API host."""

    PRACTICE = "practice"
    LIVE = "live"


class BookKind(StrEnum):
    """OANDA client book type."""

    ORDER = "order"
    POSITION = "position"


class InstrumentSet(StrEnum):
    """Group of OANDA instruments a candle fetcher pulls."""

    TARGETS = "targets"  # the G10 currency pairs
    COVARIATES = "covariates"  # indices, bonds, commodities, metals
    HELD_OUT = "held-out"  # non-G10 pairs never trained on, for the zero-shot test


class PullTarget(StrEnum):
    """Group of fetchers the command line can pull by name."""

    OANDA_FX = "oanda-fx"
    OANDA_OTHER = "oanda-other"
    OANDA_HELD_OUT = "oanda-held-out"
    OANDA_BOOKS = "oanda-books"
    ETORO = "etoro"
    FRED = "fred"
    FRED_CALENDAR = "fred-calendar"
    RATES = "rates"
    CFTC = "cftc"
    EIA = "eia"
    PORTWATCH = "portwatch"
    INDICES = "indices"
    CENTRAL_BANKS = "central-banks"


class Split(StrEnum):
    """Partition of the synthetic corpus."""

    TRAIN = "train"
    VAL = "val"


class RowKind(StrEnum):
    """What a row of a synthetic task may be used as."""

    CONTINUOUS = "continuous"
    EVENT_FLAG = "event_flag"


class GeneratorFamily(StrEnum):
    """Synthetic process family, also the sampling stratum and validation slice."""

    TREND_SEASONAL = "trend_seasonal"
    KERNEL = "kernel"
    GARCH = "garch"
    JUMP = "jump"
    REGIME = "regime"
    FACTOR = "factor"
    LEAD_LAG = "lead_lag"
    COINTEGRATION = "cointegration"
    TRIANGLE = "triangle"
    EVENT = "event"


class TrendKind(StrEnum):
    """Shape of the deterministic trend of a synthetic series."""

    LINEAR = "linear"
    PIECEWISE = "piecewise"
    EXPONENTIAL = "exponential"


class Nonlinearity(StrEnum):
    """Pointwise transform a lead-lag follower applies to its leader."""

    IDENTITY = "identity"
    TANH = "tanh"
    SQUARE_SIGN = "square_sign"
    ABS = "abs"
