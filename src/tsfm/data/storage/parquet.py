"""Parquet reads and idempotent writes."""

import fcntl
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tsfm.constants import (
    KEY_COLUMNS,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARTIAL_SUFFIX,
    TIME_COLUMN,
    YEARLY_DATASETS,
)
from tsfm.data.schemas import conform, empty_table
from tsfm.data.storage.layout import DataKey, DataLayout
from tsfm.utils.time import ensure_utc, parse_utc

_ROW_INDEX = "_row"


class ParquetStore:
    """
    Read and write the store's tables.

    Writes merge on the dataset's key columns, so repeating or resuming a pull never duplicates rows.

    Parameters
    ----------
    layout : DataLayout
        Path layout of the store.

    Examples
    --------
    >>> store = ParquetStore(DataLayout(Path("~/tsfm-data").expanduser()))  # doctest: +SKIP
    >>> key = DataKey(Dataset.SERIES, ("fred", "DFF"))  # doctest: +SKIP
    >>> store.upsert(key, table)  # doctest: +SKIP
    >>> store.last_timestamp(key)  # doctest: +SKIP
    """

    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout

    def upsert(self, key: DataKey, table: pa.Table) -> int:
        """
        Merge rows into a table, replacing rows that share a key.

        Parameters
        ----------
        key : DataKey
            Table address.
        table : pa.Table
            New rows holding at least the dataset's schema columns.

        Returns
        -------
        int
            Number of rows received.
        """
        self._merge_into(key, table, keep_stored=False)
        return int(table.num_rows)

    def insert(self, key: DataKey, table: pa.Table) -> int:
        """
        Add the rows whose key is not stored yet; stored rows are never changed.

        Parameters
        ----------
        key : DataKey
            Table address.
        table : pa.Table
            New rows holding at least the dataset's schema columns.

        Returns
        -------
        int
            Number of rows added.
        """
        return self._merge_into(key, table, keep_stored=True)

    def _merge_into(self, key: DataKey, table: pa.Table, *, keep_stored: bool) -> int:
        """
        Merge rows into a table under its lock.

        Parameters
        ----------
        key : DataKey
            Table address.
        table : pa.Table
            New rows.
        keep_stored : bool
            On a key clash, keep the stored row if True, the new row otherwise.

        Returns
        -------
        int
            Growth of the table in rows.
        """
        table = conform(table, key.dataset)
        if table.num_rows == 0:
            return 0
        added = 0
        with _locked(self.layout.lock(key)):
            for year, part in _split_by_year(table, key).items():
                path = self.layout.file(key, year)
                existing = _read_file(path) if path.exists() else empty_table(key.dataset)
                pair = (part, existing) if keep_stored else (existing, part)
                merged = _merge(*pair, KEY_COLUMNS[key.dataset])
                _write_file(path, merged)
                added += merged.num_rows - existing.num_rows
        return added

    def replace(self, key: DataKey, table: pa.Table) -> int:
        """
        Make a table hold exactly the given rows, dropping stored rows absent from them.

        Parameters
        ----------
        key : DataKey
            Table address.
        table : pa.Table
            The whole new content.

        Returns
        -------
        int
            Number of rows received.
        """
        table = conform(table, key.dataset)
        parts = _split_by_year(table, key)
        with _locked(self.layout.lock(key)):
            written = set[Path]()
            for year, part in parts.items():
                path = self.layout.file(key, year)
                _write_file(path, _merge(empty_table(key.dataset), part, KEY_COLUMNS[key.dataset]))
                written.add(path)
            for stale in set(self.layout.files(key)) - written:
                stale.unlink()
        return int(table.num_rows)

    def read(self, key: DataKey, start: datetime | None = None, end: datetime | None = None) -> pa.Table:
        """
        Read a table, optionally restricted to ``start <= time < end``.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime or None, default: None
            Inclusive lower bound.
        end : datetime or None, default: None
            Exclusive upper bound.

        Returns
        -------
        pa.Table
            Rows in time order; empty if nothing is stored.
        """
        files = self.layout.files(key)
        if not files:
            return empty_table(key.dataset)
        table = pa.concat_tables([_read_file(path) for path in files])
        if start is not None:
            table = table.filter(pc.greater_equal(table[TIME_COLUMN], pa.scalar(ensure_utc(start))))
        if end is not None:
            table = table.filter(pc.less(table[TIME_COLUMN], pa.scalar(ensure_utc(end))))
        return table

    def last_timestamp(self, key: DataKey) -> datetime | None:
        """
        Return the newest stored timestamp, the point a pull resumes from.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        datetime or None
            Latest ``time`` value, or None if nothing is stored.
        """
        files = self.layout.files(key)
        if not files:
            return None
        times = pq.ParquetFile(files[-1]).read(columns=[TIME_COLUMN])[TIME_COLUMN]
        latest: datetime | None = pc.max(times).as_py()
        return latest

    def pulled_from(self, key: DataKey) -> datetime | None:
        """
        Return the lower bound a table has been pulled from.

        The table is complete between this time and its newest row.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        datetime or None
            Recorded lower bound, or None if no pull recorded one.
        """
        path = self.layout.pulled_from(key)
        return parse_utc(path.read_text().strip()) if path.exists() else None

    def mark_pulled_from(self, key: DataKey, start: datetime) -> None:
        """
        Record the lower bound a table has been pulled from.

        Parameters
        ----------
        key : DataKey
            Table address.
        start : datetime
            Lower bound.
        """
        path = self.layout.pulled_from(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ensure_utc(start).isoformat())

    def clear(self, key: DataKey) -> None:
        """
        Delete every file of a table.

        Parameters
        ----------
        key : DataKey
            Table address.
        """
        for path in [*self.layout.files(key), self.layout.pulled_from(key)]:
            path.unlink(missing_ok=True)

    def write_shard(self, key: DataKey, index: int, table: pa.Table) -> Path:
        """
        Write one shard of the synthetic corpus, replacing any previous one.

        Parameters
        ----------
        key : DataKey
            Synthetic table address.
        index : int
            Shard number.
        table : pa.Table
            Rows of the shard.

        Returns
        -------
        Path
            File written.
        """
        path = self.layout.shard(key, index)
        _write_file(path, conform(table, key.dataset))
        return path


def _split_by_year(table: pa.Table, key: DataKey) -> dict[int | None, pa.Table]:
    """
    Group rows by calendar year for yearly datasets.

    Parameters
    ----------
    table : pa.Table
        Rows to split.
    key : DataKey
        Table address.

    Returns
    -------
    dict[int or None, pa.Table]
        Rows per year, or all rows under None for a non-yearly dataset.
    """
    if key.dataset not in YEARLY_DATASETS:
        return {None: table}
    years = pc.year(table[TIME_COLUMN])
    return {year.as_py(): table.filter(pc.equal(years, year)) for year in pc.unique(years)}


def _merge(base: pa.Table, winners: pa.Table, keys: tuple[str, ...]) -> pa.Table:
    """
    Concatenate two tables, keeping the last row per key, in time order.

    Parameters
    ----------
    base : pa.Table
        Rows that lose on a key clash.
    winners : pa.Table
        Rows that win on a key clash.
    keys : tuple[str, ...]
        Columns identifying a row.

    Returns
    -------
    pa.Table
        Deduplicated, sorted rows.
    """
    combined = pa.concat_tables([base, winners])
    indexed = combined.select(list(keys)).append_column(_ROW_INDEX, pa.array(range(combined.num_rows), pa.int64()))
    last = indexed.group_by(list(keys)).aggregate([(_ROW_INDEX, "max")])
    return combined.take(last[f"{_ROW_INDEX}_max"]).sort_by(TIME_COLUMN)


def _read_file(path: Path) -> pa.Table:
    """
    Read one Parquet file without inferring partition columns from its path.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    pa.Table
        File contents.
    """
    return pq.ParquetFile(path).read()


def _write_file(path: Path, table: pa.Table) -> None:
    """
    Write a table atomically, so an interrupted pull leaves the old file intact.

    Parameters
    ----------
    path : Path
        Destination file.
    table : pa.Table
        Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=PARTIAL_SUFFIX)
    os.close(handle)
    partial = Path(name)
    try:
        pq.write_table(table, partial, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


@contextmanager
def _locked(path: Path) -> Generator[None]:
    """
    Hold an exclusive lock on a file, so two pulls cannot rewrite one table at once.

    Parameters
    ----------
    path : Path
        Lock file, created if missing.

    Yields
    ------
    None
        While the lock is held.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
