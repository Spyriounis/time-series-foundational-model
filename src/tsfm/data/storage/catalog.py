"""DuckDB queries over the Parquet store, read in place."""

from types import TracebackType
from typing import Self

import duckdb
import pyarrow as pa

from tsfm.constants import PARTITION_COLUMNS, TIME_COLUMN
from tsfm.data.storage.layout import DataLayout
from tsfm.enums import Dataset

_TIME_AGGREGATES = f"count(*) AS rows, min({TIME_COLUMN}) AS first, max({TIME_COLUMN}) AS last"
_TASK_AGGREGATES = "count(DISTINCT task_id) AS tasks, count(*) AS rows"


class Catalog:
    """
    Query the store with DuckDB, with no database file.

    Parameters
    ----------
    layout : DataLayout
        Path layout of the store.

    Examples
    --------
    >>> with Catalog(DataLayout(Path("~/tsfm-data").expanduser())) as catalog:  # doctest: +SKIP
    ...     summary = catalog.summary(Dataset.BARS).to_pandas()
    ...     bars = catalog.relation(Dataset.BARS).filter("instrument = 'EUR_USD'").df()
    """

    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout
        self._connection = duckdb.connect()

    def __enter__(self) -> Self:
        """
        Enter the context.

        Returns
        -------
        Self
            This catalogue.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """
        Close the connection on leaving the context.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            Exception type, if one was raised.
        exc : BaseException or None
            Exception, if one was raised.
        tb : TracebackType or None
            Traceback, if one was raised.
        """
        self.close()

    def close(self) -> None:
        """Close the DuckDB connection; relations made from it stop working."""
        self._connection.close()

    def has_data(self, dataset: Dataset) -> bool:
        """
        Tell whether any file of a dataset exists.

        Parameters
        ----------
        dataset : Dataset
            Kind of table.

        Returns
        -------
        bool
            True if at least one Parquet file is stored.
        """
        return any((self.layout.root / dataset.value).rglob("*.parquet"))

    def relation(self, dataset: Dataset) -> duckdb.DuckDBPyRelation:
        """
        Return a lazy relation over every file of a dataset.

        Partition values appear as columns.

        Parameters
        ----------
        dataset : Dataset
            Kind of table.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation to filter, join or aggregate.
        """
        return self._connection.read_parquet(self.layout.glob(dataset), hive_partitioning=True)

    def summary(self, dataset: Dataset) -> pa.Table:
        """
        Count rows and report the time span per partition.

        Parameters
        ----------
        dataset : Dataset
            Kind of table.

        Returns
        -------
        pa.Table
            One row per partition; empty if nothing is stored.
        """
        if not self.has_data(dataset):
            return pa.table({})
        columns = ", ".join(PARTITION_COLUMNS[dataset])
        aggregates = _TASK_AGGREGATES if dataset is Dataset.SYNTHETIC else _TIME_AGGREGATES
        summary = self.relation(dataset).aggregate(f"{columns}, {aggregates}", columns).order(columns)
        table: pa.Table = summary.to_arrow_table()
        return table
