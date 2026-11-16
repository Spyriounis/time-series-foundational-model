"""Live checks: one small pull per source into a temporary store. Run with `make test-live`."""

import operator
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from helpers import utc
from tsfm.constants import KEY_COLUMNS, PUBLISHED_COLUMN
from tsfm.data.fetchers.registry import build_fetchers
from tsfm.data.schemas import SCHEMAS
from tsfm.data.storage import DataKey, DataLayout, ParquetStore
from tsfm.enums import Dataset, PullTarget
from tsfm.settings import Settings

pytestmark = pytest.mark.live

# A closed trading week, so a repeated pull sees the same bars.
WEEK = (utc(2024, 1, 8), utc(2024, 1, 15))
OANDA = ("oanda_api_token", "oanda_account_id")
ETORO = ("etoro_api_key", "etoro_user_key")

# target, secrets needed, `only` filter, (start, end)
CASES = [
    (PullTarget.OANDA_FX, OANDA, ["EUR_USD"], WEEK),
    (PullTarget.OANDA_OTHER, OANDA, ["XAU_USD"], WEEK),
    (PullTarget.OANDA_HELD_OUT, OANDA, ["USD_MXN"], WEEK),
    (PullTarget.OANDA_BOOKS, OANDA, ["EUR_USD"], (utc(2024, 1, 8), utc(2024, 1, 8, 2))),
    (PullTarget.ETORO, ETORO, ["SPY"], WEEK),
    # Weekly claims: archived first releases since 2009, estimated publication before.
    (PullTarget.FRED, ("fred_api_key",), ["ICSA"], (None, None)),
    (PullTarget.FRED_CALENDAR, ("fred_api_key",), None, (None, None)),
    (PullTarget.RATES, (), ["policy_rate_usd"], (None, None)),
    (PullTarget.CFTC, (), ["cot_eur_leveraged_net"], (None, None)),
    (PullTarget.EIA, ("eia_api_key",), ["WCESTUS1"], (None, None)),
    (PullTarget.PORTWATCH, (), ["portwatch_chokepoint1"], (None, None)),
    (PullTarget.INDICES, (), None, (None, None)),
    (PullTarget.CENTRAL_BANKS, (), None, (None, None)),
]


@pytest.fixture
def live_settings(tmp_path: Path) -> Settings:
    """The developer's credentials, with the store redirected to a temporary directory."""
    return Settings(tsfm_data_dir=tmp_path)


@pytest.mark.parametrize(("target", "secrets", "only", "window"), CASES, ids=[case[0].value for case in CASES])
def test_live_pull(
    target: PullTarget,
    secrets: tuple[str, ...],
    only: list[str] | None,
    window: tuple[datetime | None, datetime | None],
    live_settings: Settings,
) -> None:
    """Each source must answer, parse into its schema, and be stable under a repeated pull."""
    missing = [name for name in secrets if getattr(live_settings, name) is None]
    if missing:
        pytest.skip(f"not set: {', '.join(missing)}")
    store = ParquetStore(DataLayout(live_settings.tsfm_data_dir))
    start, end = window

    for fetcher in build_fetchers(target, live_settings):
        fetcher.pull(store, start, end, only)
        keys = [key for key in fetcher.list_keys() if only is None or set(key.partitions) & set(only)]
        assert keys
        first = {key: _checked(store, key) for key in keys}
        fetcher.pull(store, start, end, only)
        assert {key: store.read(key).num_rows for key in keys} == first


def _checked(store: ParquetStore, key: DataKey) -> int:
    """Validate one stored table and return its row count."""
    table = store.read(key)
    assert table.num_rows > 0, key
    assert table.schema.equals(SCHEMAS[key.dataset]), key
    times = cast("list[datetime]", table["time"].to_pylist())
    assert times == sorted(times), key
    rows = table.select(list(KEY_COLUMNS[key.dataset])).to_pylist()
    assert len({tuple(row.values()) for row in rows}) == len(rows), key
    if key.dataset is Dataset.SERIES:
        # No value is public before its reference period begins.
        published = cast("list[datetime]", table[PUBLISHED_COLUMN].to_pylist())
        assert all(map(operator.gt, published, times)), key
    return int(table.num_rows)
