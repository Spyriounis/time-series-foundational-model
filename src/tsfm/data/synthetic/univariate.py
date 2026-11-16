"""Trend + seasonality + level shifts + noise: the classical decomposition family."""

import numpy as np

from tsfm.constants import EPSILON
from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.paths import Array, innovations, level_series, log_uniform, pick, randint
from tsfm.data.synthetic.priors import TrendSeasonalPrior
from tsfm.enums import GeneratorFamily, TrendKind


class TrendSeasonalGenerator(SyntheticDataGenerator):
    """
    Sample a trend plus seasonal components, level shifts and noise.

    Parameters
    ----------
    prior : TrendSeasonalPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = TrendSeasonalGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape
    (1, 256)
    """

    family = GeneratorFamily.TREND_SEASONAL

    def __init__(self, prior: TrendSeasonalPrior | None = None) -> None:
        self.prior = prior or TrendSeasonalPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one single-row task.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            One level-like row.
        """
        prior = self.prior
        steps = np.arange(length, dtype=np.float64)
        signal = _trend(rng, steps / length, prior) + _shifts(rng, steps, prior)
        for _ in range(randint(rng, prior.seasons)):
            signal += _season(rng, steps, prior)
        noise = log_uniform(rng, prior.noise_ratio) * signal.std() * innovations(rng, prior.innovations, (length,))
        return self._task(level_series(signal + noise, rng, prior.level))


def _trend(rng: np.random.Generator, time: Array, prior: TrendSeasonalPrior) -> Array:
    """
    Draw a linear, piecewise-linear or exponential trend.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    time : Array
        Time rescaled to ``[0, 1)``, shape ``(length,)``.
    prior : TrendSeasonalPrior
        Trend kinds and ranges.

    Returns
    -------
    Array
        Trend whose total change is drawn from ``prior.trend``.
    """
    kind = pick(rng, prior.trend_kinds)
    if kind is TrendKind.LINEAR:
        shape = time
    elif kind is TrendKind.PIECEWISE:
        shape = _piecewise(rng, time, prior)
    else:
        rate = rng.uniform(*prior.growth)
        shape = np.expm1(rate * time) / np.expm1(rate)
    return rng.uniform(*prior.trend) * shape


def _piecewise(rng: np.random.Generator, time: Array, prior: TrendSeasonalPrior) -> Array:
    """
    Draw a continuous piecewise-linear shape with random breakpoints and slopes.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    time : Array
        Time rescaled to ``[0, 1)``, shape ``(length,)``.
    prior : TrendSeasonalPrior
        Range of the number of breakpoints.

    Returns
    -------
    Array
        Shape with unit peak magnitude.
    """
    breakpoints = np.sort(rng.uniform(size=randint(rng, prior.breakpoints)))
    slopes = rng.standard_normal(breakpoints.size + 1)
    shape = np.cumsum(slopes[np.searchsorted(breakpoints, time)])
    return np.asarray(shape / max(np.abs(shape).max(), EPSILON), dtype=np.float64)


def _season(rng: np.random.Generator, steps: Array, prior: TrendSeasonalPrior) -> Array:
    """
    Draw one periodic component with a random waveform, phase and amplitude.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    steps : Array
        Step indices, shape ``(length,)``.
    prior : TrendSeasonalPrior
        Periods, harmonics, amplitude and modulation ranges.

    Returns
    -------
    Array
        Seasonal component, optionally with a slowly varying amplitude.
    """
    candidate = rng.random() < prior.candidate_probability
    period = pick(rng, prior.periods) if candidate else log_uniform(rng, prior.random_period)
    harmonics = np.arange(1, randint(rng, prior.harmonics) + 1)[:, None]
    weights = rng.standard_normal(harmonics.shape) / harmonics
    phases = rng.uniform(0.0, 2.0 * np.pi, harmonics.shape)
    wave = (weights * np.sin(2.0 * np.pi * harmonics * steps / period + phases)).sum(axis=0)
    wave *= log_uniform(rng, prior.amplitude) / max(np.abs(wave).max(), EPSILON)
    if rng.random() < prior.modulation_probability:
        cycles, phase = rng.uniform(*prior.modulation_cycles), rng.uniform(0.0, 2.0 * np.pi)
        wave *= 1.0 + rng.uniform(*prior.modulation_depth) * np.sin(2.0 * np.pi * cycles * steps / steps.size + phase)
    return np.asarray(wave, dtype=np.float64)


def _shifts(rng: np.random.Generator, steps: Array, prior: TrendSeasonalPrior) -> Array:
    """
    Draw level shifts at random times: steps that persist or spikes that decay.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    steps : Array
        Step indices, shape ``(length,)``.
    prior : TrendSeasonalPrior
        Number, size and recovery ranges of the shifts.

    Returns
    -------
    Array
        Sum of the shifts; zeros if none is drawn.
    """
    total = np.zeros_like(steps)
    for _ in range(randint(rng, prior.shifts)):
        elapsed = steps - rng.integers(steps.size)
        size = rng.choice((-1.0, 1.0)) * rng.uniform(*prior.shift_size)
        recovers = rng.random() < prior.recovery_probability
        decay = np.exp(-np.maximum(elapsed, 0.0) / log_uniform(rng, prior.recovery_time)) if recovers else 1.0
        total += size * (elapsed >= 0) * decay
    return total
