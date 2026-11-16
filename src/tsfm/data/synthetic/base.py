"""Abstract generator interface and the task it produces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from tsfm.enums import GeneratorFamily, RowKind

type FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SyntheticTask:
    """
    One synthetic forecasting task: rows sharing a time axis.

    Attributes
    ----------
    values : FloatArray
        Raw levels, shape ``(rows, length)``; NaN marks a missing value.
    kinds : tuple[RowKind, ...]
        What each row may be used as; an event flag is known-future and exactly 0/1.
    family : GeneratorFamily
        Process family that produced the task.
    """

    values: FloatArray
    kinds: tuple[RowKind, ...]
    family: GeneratorFamily

    def __post_init__(self) -> None:
        """
        Check that the array and the row kinds agree.

        Raises
        ------
        ValueError
            If ``values`` is not 2-D float32 with one kind per row.
        """
        if self.values.ndim != 2 or self.values.dtype != np.float32:
            raise ValueError(f"values must be 2-D float32, got {self.values.dtype} {self.values.shape}")
        if len(self.kinds) != self.values.shape[0]:
            raise ValueError(f"{self.values.shape[0]} rows but {len(self.kinds)} kinds")


class SyntheticDataGenerator(ABC):
    """
    Sample tasks from one process family.

    A generator holds only its prior; all randomness comes from the generator
    passed to `sample`, so a seed reproduces a task exactly.

    Attributes
    ----------
    family : GeneratorFamily
        Process family, used as sampling stratum and validation slice.
    """

    family: ClassVar[GeneratorFamily]

    @abstractmethod
    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            Task with ``length`` columns.
        """

    def _task(self, rows: FloatArray | NDArray[np.float64], kinds: tuple[RowKind, ...] | None = None) -> SyntheticTask:
        """
        Wrap sampled rows as a task of this family.

        Parameters
        ----------
        rows : FloatArray or NDArray[np.float64]
            Values, shape ``(length,)`` or ``(rows, length)``.
        kinds : tuple[RowKind, ...] or None, default: None
            Row kinds; every row is continuous if omitted.

        Returns
        -------
        SyntheticTask
            Float32 task.
        """
        values = np.atleast_2d(rows).astype(np.float32)
        return SyntheticTask(values, kinds or (RowKind.CONTINUOUS,) * values.shape[0], self.family)
