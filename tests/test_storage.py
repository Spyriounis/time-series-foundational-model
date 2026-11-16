from pathlib import Path

import pyarrow as pa
import pytest

from helpers import bars, utc
from tsfm.data.fetchers.parsing import events_table
from tsfm.data.schemas import SYNTHETIC_SCHEMA
from tsfm.data.storage import Catalog, DataKey, DataLayout, ParquetStore
from tsfm.enums import Dataset

BARS_KEY = DataKey(Dataset.BARS, ("oanda", "EUR_USD", "M5"))


def test_layout_is_hive_partitioned():
    """DuckDB recovers partition values from `column=value` directories, so the path shape is a contract."""
    path = DataLayout(Path("/data")).file(BARS_KEY, year=2024)
    assert path.as_posix() == "/data/bars/source=oanda/instrument=EUR_USD/granularity=M5/year=2024/data.parquet"


@pytest.mark.parametrize("partitions", [("oanda", "EUR_USD"), ("oanda", "EUR/USD", "M5"), ("oanda", "", "M5")])
def test_key_rejects_bad_partitions(partitions):
    """A wrong count or a path separator would silently write outside the layout."""
    with pytest.raises(ValueError, match="artition"):
        DataKey(Dataset.BARS, partitions)


def test_replace_drops_rows_the_source_no_longer_has(store: ParquetStore):
    """A rescheduled release must not leave its old date behind, which an upsert would."""
    key = DataKey(Dataset.EVENTS, ("fred",))
    store.upsert(key, events_table([utc(2025, 10, 15)], ["US_CPI"], ["USD"]))
    store.replace(key, events_table([utc(2025, 10, 24)], ["US_CPI"], ["USD"]))
    assert store.read(key)["time"].to_pylist() == [utc(2025, 10, 24)]


def test_insert_never_changes_a_stored_row(store: ParquetStore):
    """Series keep their first release; an insert that let a revision through would undo that."""
    store.upsert(BARS_KEY, bars([utc(2024, 1, 1)], close=1.0))
    assert store.insert(BARS_KEY, bars([utc(2024, 1, 1), utc(2024, 1, 2)], close=2.0)) == 1
    assert store.read(BARS_KEY)["close"].to_pylist() == [1.0, 2.0]


def test_clear_forgets_the_pulled_span(store: ParquetStore):
    """A cleared table that still claimed to be pulled from 2000 would never be refilled."""
    store.upsert(BARS_KEY, bars([utc(2024)]))
    store.mark_pulled_from(BARS_KEY, utc(2000))
    assert store.pulled_from(BARS_KEY) == utc(2000)
    store.clear(BARS_KEY)
    assert store.pulled_from(BARS_KEY) is None
    assert store.read(BARS_KEY).num_rows == 0


def test_yearly_dataset_requires_year():
    """Bars are split per year; forgetting the year must not fall back to one giant file."""
    with pytest.raises(ValueError, match="yearly"):
        DataLayout(Path("/data")).file(BARS_KEY)


def test_upsert_round_trips_and_splits_by_year(store: ParquetStore):
    """Rows spanning New Year land in two files yet read back as one ordered table."""
    times = [utc(2023, 12, 31, 23, 55), utc(2024, 1, 1, 0, 0)]
    assert store.upsert(BARS_KEY, bars(times)) == 2
    assert len(store.layout.files(BARS_KEY)) == 2
    assert store.read(BARS_KEY)["time"].to_pylist() == times


def test_upsert_replaces_rows_with_the_same_key(store: ParquetStore):
    """A resumed pull re-sends its boundary bar; the newer copy must win and nothing may duplicate."""
    store.upsert(BARS_KEY, bars([utc(2024, 1, 1), utc(2024, 1, 2)], close=1.0))
    store.upsert(BARS_KEY, bars([utc(2024, 1, 2), utc(2024, 1, 3)], close=2.0))
    assert store.read(BARS_KEY)["close"].to_pylist() == [1.0, 2.0, 2.0]


def test_last_timestamp_is_the_resume_point(store: ParquetStore):
    """Pulls resume from the newest stored row, and from nothing when the table is absent."""
    assert store.last_timestamp(BARS_KEY) is None
    store.upsert(BARS_KEY, bars([utc(2023, 6, 1), utc(2024, 2, 1)]))
    assert store.last_timestamp(BARS_KEY) == utc(2024, 2, 1)


def test_read_filters_half_open_interval(store: ParquetStore):
    """The interval is `[start, end)` everywhere, so adjacent reads never overlap."""
    store.upsert(BARS_KEY, bars([utc(2024, 1, day) for day in (1, 2, 3)]))
    assert store.read(BARS_KEY, utc(2024, 1, 2), utc(2024, 1, 3))["time"].to_pylist() == [utc(2024, 1, 2)]


def test_empty_upsert_writes_nothing(store: ParquetStore):
    """An empty page (a weekend, say) must not create an empty file that breaks resume."""
    assert store.upsert(BARS_KEY, bars([])) == 0
    assert store.layout.files(BARS_KEY) == []


def test_catalog_summarises_partitions(store: ParquetStore):
    """The catalogue reads the Parquet folders in place and exposes partition values as columns."""
    store.upsert(BARS_KEY, bars([utc(2023, 12, 31), utc(2024, 1, 1)]))
    store.upsert(DataKey(Dataset.BARS, ("oanda", "USD_JPY", "H1")), bars([utc(2024, 1, 1)]))
    with Catalog(store.layout) as catalog:
        summary = catalog.summary(Dataset.BARS).to_pylist()
    assert [(row["instrument"], row["granularity"], row["rows"]) for row in summary] == [
        ("EUR_USD", "M5", 2),
        ("USD_JPY", "H1", 1),
    ]


def test_catalog_summary_is_empty_without_data(store: ParquetStore):
    """Summarising before the first pull is a normal first-run action, not an error."""
    with Catalog(store.layout) as catalog:
        assert catalog.summary(Dataset.SERIES).num_rows == 0


def test_synthetic_shards_are_counted_by_task(store: ParquetStore):
    """Synthetic rows have no time column, so the catalogue counts tasks instead of a time span."""
    key = DataKey(Dataset.SYNTHETIC, ("train", "garch"))
    table = pa.table(
        {"task_id": [0, 0, 1], "row": [0, 1, 0], "kind": ["continuous"] * 3, "values": [[1.0, 2.0]] * 3},
        schema=SYNTHETIC_SCHEMA,
    )
    store.write_shard(key, 0, table)
    with Catalog(store.layout) as catalog:
        (row,) = catalog.summary(Dataset.SYNTHETIC).to_pylist()
    assert (row["split"], row["family"], row["tasks"], row["rows"]) == ("train", "garch", 2, 3)
