# time-series-foundational-model

A from-scratch foundation model for zero-shot probabilistic forecasting of foreign-exchange rates, small enough to train on a laptop. One network forecasts any G10 currency pair at any bar size from 5 minutes to daily, conditioned on cross-asset, macro and event covariates. It forecasts how far a rate is likely to move, which is predictable, rather than which way, which is close to a coin toss. The project covers the full pipeline:
- literature review;
- real and synthetic data;
- model and training code;
- evaluation on held-out FX data (from July 2026, and four currencies never trained on) against volatility models and other foundation models;
- open weights on Hugging Face;
- a Python library API.

## Status

**Stage 2: data pipeline.** Fetchers, the local store, the synthetic corpus and the CLI are implemented, tested offline, and each source has passed a small live pull. Model and training code come next.

## Data pipeline

```
src/tsfm/
  constants.py, enums.py, settings.py     every constant, enum and setting
  utils/                                  HTTP client (rate limit, retries), UTC helpers
  data/schemas.py                         Arrow schemas of the stored tables
  data/storage/                           Parquet store (upserts) and DuckDB catalogue
  data/fetchers/                          APIDataFetcher and one subclass per source
  data/synthetic/                         SyntheticDataGenerator and one subclass per process family
  cli/                                    the `tsfm` command
```

### Sources

| Target | Source | Data | Credential |
|---|---|---|---|
| `oanda-fx` | OANDA v20 | The 32 G10 currency pairs: mid OHLC, closing spread, volume | `OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID` |
| `oanda-other` | OANDA v20 | Index, bond and commodity CFDs, gold and silver | same |
| `oanda-held-out` | OANDA v20 | USD against MXN, PLN, SGD and ZAR, kept out of training for the zero-shot test | same |
| `oanda-books` | OANDA v20 | Daily client order and position books, within 5% of the price | same |
| `etoro` | eToro public API | Stocks, ETFs, crypto | `ETORO_API_KEY`, `ETORO_USER_KEY` |
| `fred`, `fred-calendar` | FRED | US rates, yields, inflation, labour, risk gauges; release calendar | `FRED_API_KEY` |
| `rates` | BIS | Policy rates of the ten currencies | none |
| `cftc` | CFTC | Weekly futures positioning per currency | none |
| `indices` | File downloads | Geopolitical risk, policy uncertainty, news sentiment, crypto fear and greed, supply-chain pressure | none |
| `portwatch` | IMF PortWatch | Daily chokepoint transits | none |
| `eia` | EIA | Weekly US petroleum stocks | `EIA_API_KEY` |
| `central-banks` | Packaged file | Policy-decision dates of nine central banks (no RBNZ yet) | none |

Bars are pulled at 5-minute, 15-minute, 1-hour, 4-hour and daily sizes. Timestamps are UTC and refer to the period observed. Every series value also stores when it was published, so a backtest can use only what was known at the time:
- `published`: from FRED's archive of past releases where it exists. Otherwise it is the reference date plus a fixed delay from the source's release schedule, rounded up, counted to the end of that day in New York.
- `estimated`: true when `published` comes from such a delay.
- `first_seen`: when our pulls first stored the value, a check on the estimates from now on.

A series value is stored once and never overwritten, so it keeps its first release; later revisions are ignored.

### Usage

```bash
cp .env.template .env                                # fill in the keys you have
make pull-oanda-fx ARGS="--only EUR_USD"             # one pair first, to check access and size
make pull-all                                        # everything; resumes where it stopped
make synth                                           # synthetic training and validation splits
make catalog                                         # row counts and time spans of the store
make help                                            # all targets
make test-live                                       # one small pull per source, into a temporary store
```

Every pull can be interrupted and repeated. Bars, books and series resume from the newest stored row and backfill if an earlier start is asked for; series rows already stored are kept as they are; calendars are replaced. A table that fails is logged and the pull carries on. The store defaults to `~/tsfm-data` (`TSFM_DATA_DIR`) and is plain hive-partitioned Parquet:

```
bars/source=…/instrument=…/granularity=…/year=…/data.parquet
series/source=…/series_id=…/data.parquet
events/source=…/data.parquet
books/source=…/instrument=…/kind=…/year=…/data.parquet
synthetic/split=…/family=…/shard-NNNNN.parquet
```

```python
from tsfm.data.storage import Catalog, DataLayout
from tsfm.enums import Dataset
from tsfm.settings import Settings

with Catalog(DataLayout(Settings().tsfm_data_dir)) as catalog:
    hourly = catalog.relation(Dataset.BARS).filter("instrument = 'EUR_USD' AND granularity = 'H1'").df()
```

### Synthetic corpus

Ten process families, each a seeded generator: trend and seasonality, Gaussian-process kernel mixes, GARCH with intraday volatility sessions, jump diffusion, Markov regimes, latent factors, lead-lag copies, cointegrated groups, scheduled events with known-future flags, and currency triangles (cross rates built as ratios of latent currency values, so `A/C = A/B × B/C` holds up to float32 rounding). A task is a pure function of `(seed, split, task id)`.

### What the sources hold

Checked against the live services on 2026-11-11:
- **OANDA bars:** from 2002–2004, depending on the pair. Daily and 4-hour bars are cut at 17:00 New York, so their UTC time shifts by an hour with daylight saving.
- **OANDA books:** 14 of the 32 pairs, from 2017-01-18.
- **eToro:** intraday bars from about 2014–2016, daily bars only from 2018 (stocks, ETFs) or 2021 (crypto). Bid prices only; volume is often missing.
- **FRED:** the two ICE credit spreads hold three years; the Moody's spreads cover the rest. The archive of past releases starts late for many series (jobless claims 2009, the financial conditions index 2011, the 10-year yield 2005), so older values get estimated publication times and today's revised values. The first full pull reads the archive in chunks of 100 dates, about half an hour; later pulls take a request or two per series.
- **CFTC:** from 2006; no NOK or SEK futures. **EIA:** from 1982 (pulls start at 2000 unless `--start` says otherwise). **PortWatch:** from 2019; unavailable during its weekly update on Tuesdays (and on 2026-11-16 all day).
- **Publication delays** (checked 2026-11-16): CFTC 6 days, covering holiday weeks but not the government shutdowns of 2013, 2018–19 and 2025; EIA 6 days; global policy uncertainty about 6 months; PortWatch 15 days, unverified.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pre-commit install
make lint test
```

CI runs linting (ruff), docstring validation (numpydoc), type checking (mypy) and tests inside the Docker image:

```bash
docker build -t tsfm:latest .
docker run --rm tsfm:latest uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
