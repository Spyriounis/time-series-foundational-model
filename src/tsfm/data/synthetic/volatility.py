"""GARCH(1,1) returns and the intraday and weekly volatility profile of FX sessions."""

import numpy as np

from tsfm.constants import BARS_PER_DAY, BARS_PER_WEEK
from tsfm.data.synthetic.base import SyntheticTask
from tsfm.data.synthetic.paths import Array, ReturnGenerator, innovations, pick, price_path, randint
from tsfm.data.synthetic.priors import GarchPrior, SessionPrior
from tsfm.enums import GeneratorFamily

_DAYS_PER_WEEK = BARS_PER_WEEK // BARS_PER_DAY


class GarchGenerator(ReturnGenerator):
    """
    Sample a price with volatility clustering and a session volatility profile.

    Parameters
    ----------
    prior : GarchPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = GarchGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape
    (1, 256)
    """

    family = GeneratorFamily.GARCH

    def __init__(self, prior: GarchPrior | None = None) -> None:
        self.prior = prior or GarchPrior()

    def returns(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw GARCH(1,1) log-returns with unit unconditional variance.

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
        persistence = rng.uniform(*self.prior.persistence)
        shock = rng.uniform(*self.prior.shock)
        shocks = innovations(rng, self.prior.innovations, (length,))
        return shocks * np.sqrt(_garch_variance(shocks, 1.0 - persistence, shock, persistence - shock))

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
            One price row.
        """
        returns = self.returns(rng, length) * session_profile(rng, length, self.prior.session)
        return self._task(price_path(returns, rng, self.prior.path))


def _garch_variance(shocks: Array, constant: float, shock: float, memory: float) -> Array:
    """
    Run the GARCH(1,1) variance recursion from its unconditional level.

    The recursion feeds each variance into the next, so it is a scalar loop.

    Parameters
    ----------
    shocks : Array
        Unit-variance innovations, shape ``(length,)``.
    constant : float
        Constant term of the recursion.
    shock : float
        Coefficient of the previous squared return.
    memory : float
        Coefficient of the previous variance.

    Returns
    -------
    Array
        Conditional variance of each step, shape ``(length,)``.
    """
    variance = constant / (1.0 - shock - memory)
    variances: list[float] = []
    for squared_shock in (shocks**2).tolist():
        variances.append(variance)
        variance = constant + (shock * squared_shock + memory) * variance
    return np.array(variances)


def session_profile(rng: np.random.Generator, length: int, prior: SessionPrior) -> Array:
    """
    Draw a deterministic volatility multiplier that repeats daily and weekly.

    The bars-per-day divisor puts the same session shape on 5-minute, 15-minute,
    hourly and 4-hourly grids; a flat profile stands for daily bars.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    length : int
        Number of time steps.
    prior : SessionPrior
        Bar sizes and ranges of the session peaks and weekday effects.

    Returns
    -------
    Array
        Multiplier with unit mean square, shape ``(length,)``.
    """
    if rng.random() < prior.flat_probability:
        return np.ones(length)
    bars = BARS_PER_DAY // pick(rng, prior.divisors)
    steps = np.arange(length) + rng.integers(bars * _DAYS_PER_WEEK)
    weekday = rng.uniform(*prior.weekday, _DAYS_PER_WEEK)[(steps // bars) % _DAYS_PER_WEEK]
    profile = weekday * _intraday_shape(rng, (steps % bars) / bars, prior)
    return np.asarray(profile / np.sqrt(np.mean(profile**2)), dtype=np.float64)


def _intraday_shape(rng: np.random.Generator, day_fraction: Array, prior: SessionPrior) -> Array:
    """
    Draw a quiet level plus bumps at session opens; two or three bumps give U and W shapes.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    day_fraction : Array
        Time of day in ``[0, 1)``, shape ``(length,)``.
    prior : SessionPrior
        Ranges of the number, height and width of the peaks.

    Returns
    -------
    Array
        Positive multiplier, shape ``(length,)``.
    """
    peaks = randint(rng, prior.peaks)
    heights = rng.uniform(*prior.peak_height, peaks)
    widths = rng.uniform(*prior.peak_width, peaks)
    distance = np.abs(day_fraction[:, None] - rng.uniform(size=peaks))
    distance = np.minimum(distance, 1.0 - distance)  # midnight wraps
    return np.asarray(1.0 + (heights * np.exp(-0.5 * (distance / widths) ** 2)).sum(axis=1), dtype=np.float64)
