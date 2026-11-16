"""Multivariate dependence: shared factors, lead-lag copies and cointegrated prices."""

from collections.abc import Callable

import numpy as np

from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.drivers import driver_returns
from tsfm.data.synthetic.paths import (
    Array,
    ar1,
    bound_range,
    innovations,
    level_series,
    log_uniform,
    pick,
    price_path,
    randint,
    standardise,
)
from tsfm.data.synthetic.priors import CointegrationPrior, FactorPrior, LeadLagPrior
from tsfm.enums import GeneratorFamily, Nonlinearity

_NONLINEARITIES: dict[Nonlinearity, Callable[[Array], Array]] = {
    Nonlinearity.IDENTITY: lambda x: x,
    Nonlinearity.TANH: np.tanh,
    Nonlinearity.SQUARE_SIGN: lambda x: np.sign(x) * x**2,
    Nonlinearity.ABS: np.abs,
}


class FactorGenerator(SyntheticDataGenerator):
    """
    Sample prices whose returns load on a few shared latent factors.

    Parameters
    ----------
    prior : FactorPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = FactorGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape[1]
    256
    """

    family = GeneratorFamily.FACTOR

    def __init__(self, prior: FactorPrior | None = None) -> None:
        self.prior = prior or FactorPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task of correlated prices.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            Two or more price rows.
        """
        prior = self.prior
        rows = randint(rng, prior.rows)
        factors = driver_returns(rng, length, randint(rng, prior.factors), prior.drivers)
        loadings = rng.standard_normal((rows, factors.shape[0]))
        own = log_uniform(rng, prior.idiosyncratic, (rows, 1))
        returns = loadings @ factors + own * innovations(rng, prior.innovations, (rows, length))
        # Unit-variance rows, so that the path prior alone sets each row's volatility.
        returns /= np.sqrt((loadings**2).sum(axis=1, keepdims=True) + own**2)
        return self._task(price_path(returns, rng, prior.path))


class LeadLagGenerator(SyntheticDataGenerator):
    """
    Sample a base price and noisy, transformed copies of it at different lags.

    Every row reads one source at its own delay, so rows with a short delay lead
    the others, the base included.

    Parameters
    ----------
    prior : LeadLagPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = LeadLagGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape[1]
    256
    """

    family = GeneratorFamily.LEAD_LAG

    def __init__(self, prior: LeadLagPrior | None = None) -> None:
        self.prior = prior or LeadLagPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task; row 0 is the base price.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            The base price and its followers.
        """
        prior = self.prior
        max_lag = min(prior.max_lag, length // 4)
        lags = self._lags(rng, 1 + randint(rng, prior.followers), max_lag)
        source = standardise(driver_returns(rng, length + max_lag, 1, prior.drivers)[0])
        delayed = [source[max_lag - lag : max_lag - lag + length] for lag in lags]
        base = price_path(delayed[0], rng, prior.path)
        return self._task(np.stack([base, *(self._follower(rng, returns) for returns in delayed[1:])]))

    def _lags(self, rng: np.random.Generator, rows: int, max_lag: int) -> list[int]:
        """
        Draw a distinct delay per row, favouring short ones.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        rows : int
            Number of rows, at most ``max_lag + 1``.
        max_lag : int
            Largest delay in steps.

        Returns
        -------
        list[int]
            Delay of each row in ``[0, max_lag]``.
        """
        weights = 1.0 / np.arange(1, max_lag + 2)
        lags: list[int] = rng.choice(max_lag + 1, size=rows, replace=False, p=weights / weights.sum()).tolist()
        return lags

    def _follower(self, rng: np.random.Generator, returns: Array) -> Array:
        """
        Build one follower from the delayed returns or from their cumulated level.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        returns : Array
            Delayed unit-variance base returns, shape ``(length,)``.

        Returns
        -------
        Array
            A price (return copy) or a level-like series (level copy).
        """
        prior = self.prior
        copies_level = rng.random() < prior.level_probability
        signal = standardise(np.cumsum(returns)) if copies_level else returns
        transformed = standardise(_NONLINEARITIES[pick(rng, prior.nonlinearities)](signal))
        noise = log_uniform(rng, prior.noise_ratio) * innovations(rng, prior.innovations, returns.shape)
        if copies_level:
            return level_series(transformed + noise, rng, prior.level)
        return price_path(standardise(transformed + noise), rng, prior.path)


class CointegrationGenerator(SyntheticDataGenerator):
    """
    Sample prices that share one stochastic trend and differ by mean-reverting spreads.

    Parameters
    ----------
    prior : CointegrationPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = CointegrationGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape[1]
    256
    """

    family = GeneratorFamily.COINTEGRATION

    def __init__(self, prior: CointegrationPrior | None = None) -> None:
        self.prior = prior or CointegrationPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task of cointegrated prices.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            Two or more price rows.
        """
        prior, path = self.prior, self.prior.path
        rows = randint(rng, prior.rows)
        returns = driver_returns(rng, length, 1, prior.drivers)[0] + rng.uniform(*path.drift)
        loadings = rng.uniform(*prior.loading, (rows, 1))
        spreads = np.stack([self._spread(rng, length) for _ in range(rows)])
        log_prices = log_uniform(rng, path.volatility) * (loadings * np.cumsum(returns) + spreads)
        # One shrink factor for all rows keeps the cointegrating relations intact.
        log_prices = bound_range(log_prices, path.max_log_range, jointly=True)
        return self._task(log_uniform(rng, path.level, (rows, 1)) * np.exp(log_prices))

    def _spread(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw a stationary Ornstein-Uhlenbeck spread in units of the trend's one-step volatility.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        Array
            Spread, shape ``(length,)``.
        """
        coefficient = 0.5 ** (1.0 / float(log_uniform(rng, self.prior.half_life)))
        return log_uniform(rng, self.prior.spread) * ar1(rng.standard_normal(length), coefficient)
