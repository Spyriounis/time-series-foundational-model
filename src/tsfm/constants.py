"""Constants for the whole package."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from tsfm.enums import BookKind, Dataset, GeneratorFamily, Granularity, OandaEnvironment

# --- Storage -----------------------------------------------------------------

DEFAULT_DATA_DIR = Path.home() / "tsfm-data"
PARQUET_COMPRESSION: Final = "zstd"
PARQUET_COMPRESSION_LEVEL = 9
PARQUET_FILE_NAME = "data.parquet"
PARTIAL_SUFFIX = ".partial"
# Per-table files next to the Parquet files: a write lock, and the lower bound already pulled.
LOCK_FILE_NAME = ".lock"
PULLED_FROM_FILE_NAME = ".pulled_from"
TIME_COLUMN = "time"
# Series columns beside time and value: when the value became public, whether that time is
# estimated from a fixed delay, and when our pulls first stored it.
PUBLISHED_COLUMN = "published"
ESTIMATED_COLUMN = "estimated"
FIRST_SEEN_COLUMN = "first_seen"

# Hive partition columns of each dataset, in directory order.
PARTITION_COLUMNS: dict[Dataset, tuple[str, ...]] = {
    Dataset.BARS: ("source", "instrument", "granularity"),
    Dataset.SERIES: ("source", "series_id"),
    Dataset.EVENTS: ("source",),
    Dataset.BOOKS: ("source", "instrument", "kind"),
    Dataset.SYNTHETIC: ("split", "family"),
}
# Datasets large enough to split into one file per calendar year.
YEARLY_DATASETS: frozenset[Dataset] = frozenset({Dataset.BARS, Dataset.BOOKS})
# Columns that identify a row; an upsert keeps the newest row per key.
KEY_COLUMNS: dict[Dataset, tuple[str, ...]] = {
    Dataset.BARS: (TIME_COLUMN,),
    Dataset.SERIES: (TIME_COLUMN,),
    Dataset.EVENTS: (TIME_COLUMN, "event_id"),
    Dataset.BOOKS: (TIME_COLUMN,),
}

# --- Scope -------------------------------------------------------------------

G10_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "NOK", "SEK"})
GRANULARITIES: tuple[Granularity, ...] = tuple(Granularity)
# Default lower bound of a pull: earlier than the first FX bar (2002), later than the start of some macro series.
HISTORY_START = datetime(2000, 1, 1, tzinfo=UTC)
# Date cuts of the evaluation: training scores steps before the validation start, validation before the
# test start, the test everything after. Everything is pulled; the cuts are applied when tasks are assembled.
VALIDATION_START = datetime(2026, 1, 1, tzinfo=UTC)
TEST_START = datetime(2026, 7, 1, tzinfo=UTC)
# Scored steps this close before a cut are dropped, so no window straddles two periods.
SPLIT_EMBARGO = timedelta(weeks=1)
# A series value counts as public from the end of its publication day here, the latest clock of our sources.
PUBLICATION_TIMEZONE = "America/New_York"

# --- HTTP --------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS = 30.0
# Waits of 2, 4, ... 120 s: about four minutes in total, enough to ride out a dropped connection.
HTTP_MAX_ATTEMPTS = 8
HTTP_BACKOFF_SECONDS = 2.0
HTTP_BACKOFF_MAX_SECONDS = 120.0
HTTP_RETRY_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
# Tables failing in a row before a fetcher gives up; the cause is then the source, not the table.
PULL_MAX_CONSECUTIVE_FAILURES = 3
# Characters of an error body quoted in the exception.
HTTP_ERROR_BODY_CHARS = 200
HTTP_USER_AGENT = "tsfm-data-pipeline (https://github.com/Spyriounis/time-series-foundational-model)"

# --- OANDA v20 ---------------------------------------------------------------

OANDA_HOSTS: dict[OandaEnvironment, str] = {
    OandaEnvironment.PRACTICE: "https://api-fxpractice.oanda.com/v3",
    OandaEnvironment.LIVE: "https://api-fxtrade.oanda.com/v3",
}
OANDA_MAX_CANDLES = 5000
OANDA_REQUESTS_PER_SECOND = 20.0  # documented cap is 120
OANDA_PRICE_COMPONENTS = "MBA"
OANDA_CURRENCY_TYPE = "CURRENCY"
OANDA_METAL_TYPE = "METAL"
# Metals kept as covariates; the other metal pairs are one of these times an FX rate.
OANDA_METALS: frozenset[str] = frozenset({"XAU_USD", "XAG_USD", "XAU_XAG"})
# Non-G10 pairs for the zero-shot test, one per region; never used in training. Pegs (HKD, DKK) barely move.
OANDA_HELD_OUT_PAIRS: tuple[str, ...] = ("USD_MXN", "USD_PLN", "USD_SGD", "USD_ZAR")
OANDA_SMOOTH = "false"
# Daily and 4-hour bars are cut at the New York close, OANDA's default and the FX market convention.
OANDA_DAILY_ALIGNMENT_HOUR = 17
OANDA_ALIGNMENT_TIMEZONE = "America/New_York"
# OANDA publishes a snapshot every 20 minutes; one a day, at midnight UTC, is kept.
OANDA_BOOK_INTERVAL = timedelta(days=1)
OANDA_BOOK_HISTORY_START = datetime(2017, 1, 18, tzinfo=UTC)
# Snapshots requested between two writes to the store.
OANDA_BOOK_BATCH_SNAPSHOTS = 30
# Buckets further than this fraction from the current price are dropped; most are stale or absurd levels.
OANDA_BOOK_PRICE_BAND = 0.05
# Book kind -> name of both the endpoint and the key wrapping its response.
OANDA_BOOK_ENDPOINTS: dict[BookKind, str] = {
    BookKind.ORDER: "orderBook",
    BookKind.POSITION: "positionBook",
}

# --- eToro -------------------------------------------------------------------

ETORO_BASE_URL = "https://public-api.etoro.com"
ETORO_MAX_CANDLES = 2000
# Periods per request window. A window is stored whole, so this bounds what an interrupted pull repeats.
ETORO_MAX_WINDOW_PERIODS = 10_080
# Earlier than eToro's first intraday bar (2012 for the oldest stocks); daily bars start in 2018.
ETORO_HISTORY_START = datetime(2012, 1, 1, tzinfo=UTC)
# The only side requested, so eToro bars carry no spread.
ETORO_PRICE_SIDE = "bid"
ETORO_REQUESTS_PER_SECOND = 1.5  # documented cap is 120 per minute
ETORO_INSTRUMENTS_PAGE_SIZE = 100
ETORO_INTERVALS: dict[Granularity, str] = {
    Granularity.M5: "5m",
    Granularity.M15: "15m",
    Granularity.H1: "1h",
    Granularity.H4: "4h",
    Granularity.D: "1d",
}
# Symbols as eToro spells them.
ETORO_SYMBOLS: tuple[str, ...] = (
    # Broad, regional and sector equity ETFs
    *("SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWA", "EWC", "FXI", "XLF", "XLE", "XLK"),
    # Rates, credit, commodities and volatility ETFs
    *("TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "USO", "UUP", "VXX"),
    # Large caps
    *("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM"),
    # Crypto
    *("BTC", "ETH", "SOL", "XRP"),
)

# --- FRED --------------------------------------------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_REQUESTS_PER_SECOND = 1.5
FRED_PAGE_SIZE = 100_000
FRED_VINTAGE_PAGE_SIZE = 10_000
# Archive dates per first-release request. FRED allows 2,000, but series re-estimated every week
# (NFCI) time out above a few hundred.
FRED_VINTAGES_PER_REQUEST = 100
FRED_INITIAL_RELEASE_OUTPUT = 4  # `output_type` of the observations endpoint: first releases only
# Before a series' archive starts, its publication delay is this quantile of the archived ones, in whole days.
FRED_DELAY_QUANTILE = 0.95
FRED_RELEASE_PAGE_SIZE = 1000
FRED_MISSING_VALUE = "."
FRED_SERIES: tuple[str, ...] = (
    # Policy and money-market rates
    *("DFF", "SOFR"),
    # Treasury curve, breakevens
    *("DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10YIE", "T5YIFR"),
    # Risk: volatility, dollar, credit spreads, financial conditions
    # FRED keeps only three years of the two ICE spreads; the Moody's spreads go back to the 1980s.
    *("VIXCLS", "DTWEXBGS", "BAMLC0A0CM", "BAMLH0A0HYM2", "BAA10Y", "AAA10Y", "NFCI", "STLFSI4"),
    # Oil
    *("DCOILWTICO", "DCOILBRENTEU"),
    # Inflation, labour, activity
    *("CPIAUCSL", "CPILFESL", "PCEPI", "UNRATE", "PAYEMS", "ICSA", "INDPRO", "RSAFS", "UMCSENT", "GDPC1"),
)
# Event id -> (FRED release id, local release time).
FRED_RELEASES: dict[str, tuple[int, str]] = {
    "US_CPI": (10, "08:30"),
    "US_EMPLOYMENT": (50, "08:30"),
    "US_GDP": (53, "08:30"),
    "US_PCE": (54, "08:30"),
    "US_PPI": (46, "08:30"),
    "US_RETAIL_SALES": (9, "08:30"),
    "US_INDUSTRIAL_PRODUCTION": (13, "09:15"),
}
FRED_RELEASE_TIMEZONE = "America/New_York"
FRED_RELEASE_CURRENCY = "USD"

# --- BIS (central-bank policy rates) ------------------------------------------

BIS_POLICY_RATES_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0"
BIS_REQUESTS_PER_SECOND = 1.0
BIS_FREQUENCY = "D"
# Currency -> BIS reference area; the euro area is XM.
BIS_POLICY_RATE_AREAS: dict[str, str] = {
    "USD": "US",
    "EUR": "XM",
    "JPY": "JP",
    "GBP": "GB",
    "CHF": "CH",
    "AUD": "AU",
    "CAD": "CA",
    "NZD": "NZ",
    "NOK": "NO",
    "SEK": "SE",
}
BIS_DATE_COLUMN = "TIME_PERIOD"
BIS_VALUE_COLUMN = "OBS_VALUE"
# Written on days without a fixing, e.g. weekends.
BIS_MISSING_VALUE = "NaN"
BIS_SERIES_PREFIX = "policy_rate"
# The rate in force on a day is announced on or before it; the BIS copy lags by a week, the news does not.
BIS_PUBLICATION_DELAY = timedelta(0)

# --- CFTC (Traders in Financial Futures, futures only) ------------------------

CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CFTC_REQUESTS_PER_SECOND = 2.0
CFTC_PAGE_SIZE = 50_000
# Currency -> contract market code; NOK and SEK futures are not in the report.
CFTC_CONTRACT_CODES: dict[str, str] = {
    "EUR": "099741",
    "JPY": "097741",
    "GBP": "096742",
    "CHF": "092741",
    "AUD": "232741",
    "CAD": "090741",
    "NZD": "112741",
    "USD": "098662",
}
# Trader group -> (long field, short field).
CFTC_TRADER_GROUPS: dict[str, tuple[str, str]] = {
    "dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
    "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
    "leveraged": ("lev_money_positions_long", "lev_money_positions_short"),
}
CFTC_DATE_FIELD = "report_date_as_yyyy_mm_dd"
CFTC_CODE_FIELD = "cftc_contract_market_code"
CFTC_OPEN_INTEREST_FIELD = "open_interest_all"
CFTC_SERIES_PREFIX = "cot"
# Tuesday positions are released on Friday, or on Monday after a holiday. Government shutdowns delayed
# some reports by weeks (2013, 2018-19, 2025); those weeks are published too early here.
CFTC_PUBLICATION_DELAY = timedelta(days=6)

# --- EIA (weekly petroleum stocks) --------------------------------------------

EIA_STOCKS_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
EIA_REQUESTS_PER_SECOND = 1.0
EIA_PAGE_SIZE = 5000
EIA_FREQUENCY = "weekly"
# Crude excluding the strategic reserve, gasoline, distillate, strategic reserve.
EIA_SERIES: tuple[str, ...] = ("WCESTUS1", "WGTSTUS1", "WDISTUS1", "WCSSTUS1")
# The week to Friday is released the next Wednesday, or Thursday after a holiday.
EIA_PUBLICATION_DELAY = timedelta(days=6)

# --- IMF PortWatch ------------------------------------------------------------

PORTWATCH_CHOKEPOINTS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
)
PORTWATCH_REQUESTS_PER_SECOND = 1.0
PORTWATCH_PAGE_SIZE = 1000
PORTWATCH_CHOKEPOINT_IDS: tuple[str, ...] = tuple(f"chokepoint{i}" for i in range(1, 29))
PORTWATCH_VALUE_FIELD = "n_total"
PORTWATCH_ID_FIELD = "portid"
PORTWATCH_DATE_FIELD = "date"
PORTWATCH_SERIES_PREFIX = "portwatch"
# Released weekly on Tuesdays; how far behind the data run is unverified (the service was down when checked).
PORTWATCH_PUBLICATION_DELAY = timedelta(days=15)

# --- Published indices (file downloads) ---------------------------------------

GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_US_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
EPU_GLOBAL_URL = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
SF_FED_SENTIMENT_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
CRYPTO_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
INDICES_REQUESTS_PER_SECOND = 1.0
# Series name, then the sheet and columns holding it in each file.
GPR_SERIES_ID = "gpr_daily"
GPR_DATE_COLUMN = "date"
GPR_VALUE_COLUMN = "GPRD"
EPU_YEAR_COLUMN = "Year"
EPU_MONTH_COLUMN = "Month"
EPU_US_SERIES_ID = "epu_us"
EPU_US_VALUE_COLUMN = "News_Based_Policy_Uncert_Index"
EPU_GLOBAL_SERIES_ID = "epu_global"
EPU_GLOBAL_VALUE_COLUMN = "GEPU_current"
SF_FED_SENTIMENT_SERIES_ID = "news_sentiment"
SF_FED_SENTIMENT_SHEET = "Data"
SF_FED_SENTIMENT_DATE_COLUMN = "date"
SF_FED_SENTIMENT_VALUE_COLUMN = "News Sentiment"
CRYPTO_FEAR_GREED_SERIES_ID = "crypto_fear_greed"
GSCPI_SERIES_ID = "gscpi"
GSCPI_SHEET: str | int = 0
# Delay from each index's date until it is public, rounded up from what the files showed on 2026-11-16.
GPR_PUBLICATION_DELAY = timedelta(days=1)  # daily, current to the day
EPU_US_PUBLICATION_DELAY = timedelta(days=45)  # dated the 1st, released early the next month
EPU_GLOBAL_PUBLICATION_DELAY = timedelta(days=190)  # released months late, irregularly
SF_FED_SENTIMENT_PUBLICATION_DELAY = timedelta(days=8)  # daily, updated at least weekly
CRYPTO_FEAR_GREED_PUBLICATION_DELAY = timedelta(0)  # stamped at midnight UTC when published
GSCPI_PUBLICATION_DELAY = timedelta(days=10)  # dated the month end, released on the fourth business day

# --- Central-bank meetings ----------------------------------------------------

RESOURCES_PACKAGE = "tsfm.data.resources"
CENTRAL_BANK_MEETINGS_RESOURCE = "central_bank_meetings.csv"
CENTRAL_BANK_EVENT_PREFIX = "RATE_DECISION"

# --- Synthetic corpus ---------------------------------------------------------

SYNTHETIC_DEFAULT_TASKS = 100_000
SYNTHETIC_VAL_TASKS_PER_FAMILY = 500
SYNTHETIC_SHARD_TASKS = 1000
SYNTHETIC_MIN_LENGTH = 1024
SYNTHETIC_MAX_LENGTH = 8192
SYNTHETIC_MIXUP_PROBABILITY = 0.2
# Mixup partners come from the same group, so a mix of prices stays positive.
SYNTHETIC_MIXUP_GROUPS: tuple[tuple[GeneratorFamily, ...], ...] = (
    (GeneratorFamily.TREND_SEASONAL, GeneratorFamily.KERNEL),
    (GeneratorFamily.GARCH, GeneratorFamily.JUMP, GeneratorFamily.REGIME),
)
# Floor of a divisor that may be zero.
EPSILON = 1e-12
SYNTHETIC_SEEDS: dict[str, int] = {"train": 0, "val": 1}
# Sampling weight of each family in the training split; they need not sum to one.
SYNTHETIC_FAMILY_WEIGHTS: dict[GeneratorFamily, float] = {
    GeneratorFamily.TREND_SEASONAL: 1.0,
    GeneratorFamily.KERNEL: 1.5,
    GeneratorFamily.GARCH: 2.0,
    GeneratorFamily.JUMP: 1.0,
    GeneratorFamily.REGIME: 1.0,
    GeneratorFamily.FACTOR: 1.0,
    GeneratorFamily.LEAD_LAG: 1.0,
    GeneratorFamily.COINTEGRATION: 1.0,
    GeneratorFamily.TRIANGLE: 2.0,
    GeneratorFamily.EVENT: 1.0,
}
# Bars per day and per week on the 5-minute FX grid, the periods of the intraday volatility profile.
BARS_PER_DAY = 288
BARS_PER_WEEK = 1440
