"""TSMixup: convex combinations of single-row series."""

from collections.abc import Sequence

import numpy as np

from tsfm.constants import EPSILON
from tsfm.data.synthetic.base import SyntheticTask
from tsfm.data.synthetic.priors import MixupPrior
from tsfm.enums import RowKind


def mixup(tasks: Sequence[SyntheticTask], rng: np.random.Generator, prior: MixupPrior | None = None) -> SyntheticTask:
    """
    Mix single-row tasks with Dirichlet weights.

    Each series is divided by its mean absolute value first, so that no
    component dominates by scale; the mix is returned at the first one's scale.

    Parameters
    ----------
    tasks : Sequence[SyntheticTask]
        Two or more continuous single-row tasks of equal length.
    rng : np.random.Generator
        Source of randomness.
    prior : MixupPrior or None, default: None
        Dirichlet concentration; the default prior if omitted.

    Returns
    -------
    SyntheticTask
        Single-row task carrying the family of the first component.

    Raises
    ------
    ValueError
        If fewer than two tasks are given or one is not a continuous single row of the common length.
    """
    shapes = {task.values.shape for task in tasks}
    if len(tasks) < 2 or len(shapes) != 1 or any(task.kinds != (RowKind.CONTINUOUS,) for task in tasks):
        raise ValueError("mixup needs two or more continuous single-row tasks of equal length")
    series = np.concatenate([task.values for task in tasks]).astype(np.float64)
    scales = np.maximum(np.abs(series).mean(axis=1, keepdims=True), EPSILON)
    weights = rng.dirichlet(np.full(len(tasks), (prior or MixupPrior()).concentration))
    mixed = scales[0] * (weights @ (series / scales))
    return SyntheticTask(mixed[None].astype(np.float32), (RowKind.CONTINUOUS,), tasks[0].family)
