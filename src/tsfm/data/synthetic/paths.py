"""Sampling helpers shared by the generators, and the return-process interface."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from tsfm.constants import EPSILON
from tsfm.data.synthetic.base import SyntheticDataGenerator
from tsfm.data.synthetic.priors import InnovationPrior, IntRange, LevelPrior, PathPrior, Range

type Array = NDArray[np.float64]


def log_uniform(rng: np.random.Generator, bounds: Range, shape: tuple[int, ...] = ()) -> Array:
    """
    Sample uniformly in log space, so every order of magnitude is equally likely.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    bounds : Range
        Positive lower and upper bound.
    shape : tuple[int, ...], default: ()
        Output shape; a scalar by default.

    Returns
    -------
    Array
        Samples in ``[low, high]``.
    """
    return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), shape))


def randint(rng: np.random.Generator, bounds: IntRange) -> int:
    """
    Sample an integer uniformly from a closed interval.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    bounds : IntRange
        Lowest and highest value, both attainable.

    Returns
    -------
    int
        Sample.
    """
    return int(rng.integers(bounds[0], bounds[1] + 1))


def pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
    """
    Choose one option uniformly, keeping its Python type.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    options : Sequence[T]
        Non-empty candidates.

    Returns
    -------
    T
        Chosen option.
    """
    return options[int(rng.integers(len(options)))]


def innovations(rng: np.random.Generator, prior: InnovationPrior, shape: tuple[int, ...]) -> Array:
    """
    Draw i.i.d. unit-variance shocks, Gaussian or Student-t.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    prior : InnovationPrior
        Chance of Gaussian shocks and range of the degrees of freedom.
    shape : tuple[int, ...]
        Output shape; one distribution is drawn for the whole array.

    Returns
    -------
    Array
        Shocks.
    """
    if rng.random() < prior.gaussian_probability:
        return rng.standard_normal(shape)
    dof = rng.uniform(*prior.dof)
    return np.asarray(rng.standard_t(dof, shape) * np.sqrt((dof - 2.0) / dof), dtype=np.float64)


def standardise(series: Array) -> Array:
    """
    Centre and scale to unit standard deviation along time.

    Parameters
    ----------
    series : Array
        Values, time on the last axis.

    Returns
    -------
    Array
        Standardised values; a constant series maps to zeros.
    """
    centred = series - series.mean(axis=-1, keepdims=True)
    return centred / np.maximum(series.std(axis=-1, keepdims=True), EPSILON)


def ar1(noise: Array, coefficient: float) -> Array:
    """
    Filter white noise into a stationary unit-variance AR(1) path.

    A discretely sampled Ornstein-Uhlenbeck process is exactly this recursion.

    Parameters
    ----------
    noise : Array
        Unit-variance shocks, shape ``(length,)``.
    coefficient : float
        Autoregressive coefficient in ``[0, 1)``.

    Returns
    -------
    Array
        Path with the stationary distribution from its first step.
    """
    gain = np.sqrt(1.0 - coefficient**2)
    shocks = gain * noise
    shocks[0] = noise[0]
    filtered = cast("Array", lfilter([1.0], [1.0, -coefficient], shocks))
    return np.asarray(filtered, dtype=np.float64)


def bound_range(paths: Array, max_range: float, *, jointly: bool = False) -> Array:
    """
    Shrink paths about their start so that no excursion exceeds a bound.

    Shrinking only lowers the volatility, so the path keeps its shape and a
    later ``exp`` cannot overflow.

    Parameters
    ----------
    paths : Array
        Log-paths, time on the last axis.
    max_range : float
        Largest allowed distance from the first value.
    jointly : bool, default: False
        Shrink all rows by one factor, which preserves linear relations between them.

    Returns
    -------
    Array
        Bounded paths; paths already inside the bound are unchanged.
    """
    start = paths[..., :1]
    excursion = np.abs(paths - start).max(axis=None if jointly else -1, keepdims=True)
    return start + (paths - start) * np.minimum(1.0, max_range / np.maximum(excursion, EPSILON))


def price_path(returns: Array, rng: np.random.Generator, prior: PathPrior) -> Array:
    """
    Cumulate unit-scale log-returns into strictly positive prices.

    Each row gets its own volatility, drift and initial level.

    Parameters
    ----------
    returns : Array
        Log-returns of roughly unit variance, shape ``(length,)`` or ``(rows, length)``.
    rng : np.random.Generator
        Source of randomness.
    prior : PathPrior
        Ranges of the level, volatility and drift, and the log-range bound.

    Returns
    -------
    Array
        Prices, same shape as ``returns``.
    """
    column = (*returns.shape[:-1], 1)
    volatility = log_uniform(rng, prior.volatility, column)
    drift = rng.uniform(*prior.drift, column)
    log_path = np.cumsum(volatility * (returns + drift), axis=-1)
    return log_uniform(rng, prior.level, column) * np.exp(bound_range(log_path, prior.max_log_range))


def level_series(signal: Array, rng: np.random.Generator, prior: LevelPrior) -> Array:
    """
    Give a unit-scale signal a random offset and scale.

    Parameters
    ----------
    signal : Array
        Signal of order one, shape ``(length,)``.
    rng : np.random.Generator
        Source of randomness.
    prior : LevelPrior
        Ranges of the scale and offset.

    Returns
    -------
    Array
        Level-like series, which may be negative.
    """
    return log_uniform(rng, prior.scale) * (signal + rng.uniform(*prior.offset))


class ReturnGenerator(SyntheticDataGenerator):
    """
    Generator of a single price whose log-returns are available on their own.

    The multivariate families compose these return processes instead of
    reimplementing them.
    """

    @abstractmethod
    def returns(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw log-returns of roughly unit variance.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        Array
            Log-returns, shape ``(length,)``.
        """
