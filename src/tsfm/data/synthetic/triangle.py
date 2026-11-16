"""FX cross rates as ratios of latent currency values, so triangles close exactly."""

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.drivers import driver_returns
from tsfm.data.synthetic.paths import Array, bound_range, log_uniform, randint
from tsfm.data.synthetic.priors import TrianglePrior
from tsfm.enums import GeneratorFamily

_TRIANGLE = [(0, 1), (1, 2), (0, 2)]


class TriangleGenerator(SyntheticDataGenerator):
    """
    Sample cross rates between currencies whose latent values are correlated.

    With three or more rates the task contains a closed triangle, for which
    rate(A/C) = rate(A/B) * rate(B/C) up to the orientation of each quote.

    Parameters
    ----------
    prior : TrianglePrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = TriangleGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape[1]
    256
    """

    family = GeneratorFamily.TRIANGLE

    def __init__(self, prior: TrianglePrior | None = None) -> None:
        self.prior = prior or TrianglePrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task of cross rates.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            Two or more rate rows in random order and orientation.
        """
        values = self._log_values(rng, length)
        pairs = self._pairs(rng, values.shape[0])
        return self._task(np.exp(values[pairs[:, 0]] - values[pairs[:, 1]]))

    def _log_values(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw the log-value path of each currency: a common factor plus its own returns.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        Array
            Log-values, shape ``(currencies, length)``.
        """
        prior = self.prior
        currencies = randint(rng, prior.currencies)
        factor, *own = driver_returns(rng, length, currencies + 1, prior.drivers)
        returns = rng.uniform(*prior.loading, (currencies, 1)) * factor + np.stack(own)
        volatility = log_uniform(rng, prior.volatility) * rng.uniform(*prior.relative_volatility, (currencies, 1))
        paths = bound_range(np.cumsum(volatility * returns, axis=1), prior.max_log_range, jointly=True)
        return np.log(log_uniform(rng, prior.value, (currencies, 1))) + paths

    def _pairs(self, rng: np.random.Generator, currencies: int) -> NDArray[np.int64]:
        """
        Choose which cross rates to emit, a closed triangle first.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        currencies : int
            Number of currencies, at least three.

        Returns
        -------
        NDArray[np.int64]
            Base and quote currency of each rate, shape ``(pairs, 2)``.
        """
        others = [pair for pair in combinations(range(currencies), 2) if pair not in _TRIANGLE]
        pool = np.array(_TRIANGLE + [others[index] for index in rng.permutation(len(others))])
        count = min(randint(rng, self.prior.pairs), len(pool))
        # Relabelling the currencies makes the triangle a random one.
        pairs = rng.permutation(currencies)[pool[:count]]
        flipped = rng.random(count) < 0.5
        pairs[flipped] = pairs[flipped, ::-1]
        return rng.permutation(pairs)
