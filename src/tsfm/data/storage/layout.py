"""Directory layout of the store."""

from dataclasses import dataclass
from pathlib import Path

from tsfm.constants import (
    LOCK_FILE_NAME,
    PARQUET_FILE_NAME,
    PARTITION_COLUMNS,
    PULLED_FROM_FILE_NAME,
    YEARLY_DATASETS,
)
from tsfm.enums import Dataset


@dataclass(frozen=True)
class DataKey:
    """
    Address of one stored table.

    Attributes
    ----------
    dataset : Dataset
        Kind of table.
    partitions : tuple[str, ...]
        Values of the dataset's partition columns, in ``PARTITION_COLUMNS`` order.
    """

    dataset: Dataset
    partitions: tuple[str, ...]

    def __post_init__(self) -> None:
        """
        Validate the partition values against the dataset's columns.

        Raises
        ------
        ValueError
            If the count is wrong or a value cannot be a directory name.
        """
        columns = PARTITION_COLUMNS[self.dataset]
        if len(self.partitions) != len(columns):
            raise ValueError(f"{self.dataset} needs partitions {columns}, got {self.partitions}")
        if any(not value or "/" in value or "=" in value for value in self.partitions):
            raise ValueError(f"Partition values must be non-empty without '/' or '=': {self.partitions}")


class DataLayout:
    """
    Map keys to hive-partitioned paths under a root directory.

    Parameters
    ----------
    root : Path
        Root of the store.

    Examples
    --------
    >>> layout = DataLayout(Path("/data"))
    >>> key = DataKey(Dataset.BARS, ("oanda", "EUR_USD", "M5"))
    >>> layout.file(key, year=2024).as_posix()
    '/data/bars/source=oanda/instrument=EUR_USD/granularity=M5/year=2024/data.parquet'
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def directory(self, key: DataKey) -> Path:
        """
        Return the directory holding a key's files.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        Path
            Directory path, not necessarily existing.
        """
        parts = (
            f"{column}={value}" for column, value in zip(PARTITION_COLUMNS[key.dataset], key.partitions, strict=True)
        )
        return self.root.joinpath(key.dataset.value, *parts)

    def file(self, key: DataKey, year: int | None = None) -> Path:
        """
        Return the Parquet file of a key, or of one of its years.

        Parameters
        ----------
        key : DataKey
            Table address.
        year : int or None, default: None
            Calendar year, required for yearly datasets.

        Returns
        -------
        Path
            File path.

        Raises
        ------
        ValueError
            If ``year`` is given for a non-yearly dataset or missing for a yearly one.
        """
        if (year is not None) != (key.dataset in YEARLY_DATASETS):
            raise ValueError(f"{key.dataset} is {'yearly' if year is None else 'not yearly'}")
        directory = self.directory(key)
        return directory / PARQUET_FILE_NAME if year is None else directory / f"year={year}" / PARQUET_FILE_NAME

    def lock(self, key: DataKey) -> Path:
        """
        Return the file writers of a table lock.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        Path
            File path.
        """
        return self.directory(key) / LOCK_FILE_NAME

    def pulled_from(self, key: DataKey) -> Path:
        """
        Return the file recording the lower bound a table was pulled from.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        Path
            File path.
        """
        return self.directory(key) / PULLED_FROM_FILE_NAME

    def shard(self, key: DataKey, index: int) -> Path:
        """
        Return the file of one shard of a synthetic split and family.

        Parameters
        ----------
        key : DataKey
            Synthetic table address.
        index : int
            Shard number.

        Returns
        -------
        Path
            File path.
        """
        return self.directory(key) / f"shard-{index:05d}.parquet"

    def files(self, key: DataKey) -> list[Path]:
        """
        List a key's existing Parquet files in time order.

        Parameters
        ----------
        key : DataKey
            Table address.

        Returns
        -------
        list[Path]
            Existing files, sorted.
        """
        return sorted(self.directory(key).rglob("*.parquet"))

    def glob(self, dataset: Dataset) -> str:
        """
        Return the glob matching every file of a dataset.

        Parameters
        ----------
        dataset : Dataset
            Kind of table.

        Returns
        -------
        str
            Pattern for DuckDB's ``read_parquet``.
        """
        return (self.root / dataset.value / "**" / "*.parquet").as_posix()
