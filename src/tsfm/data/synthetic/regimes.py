"""Markov-switching returns: hidden sticky regimes, each with its own drift and volatility."""

import numpy as np
from numpy.typing import NDArray

from tsfm.data.synthetic.base import SyntheticTask
from tsfm.data.synthetic.paths import Array, ReturnGenerator, innovations, log_uniform, price_path, randint
from tsfm.data.synthetic.priors import RegimePrior
from tsfm.enums import GeneratorFamily


class RegimeGenerator(ReturnGenerator):
    """
    Sample a price whose drift and volatility switch with a hidden Markov chain.

    Parameters
    ----------
    prior : RegimePrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = RegimeGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape
    (1, 256)
    """

    family = GeneratorFamily.REGIME

    def __init__(self, prior: RegimePrior | None = None) -> None:
        self.prior = prior or RegimePrior()

    def returns(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw regime-switching log-returns of roughly unit variance.

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
        prior = self.prior
        regimes = randint(rng, prior.regimes)
        states = _markov_states(rng, length, log_uniform(rng, prior.exit_probability, (regimes,)))
        drift = rng.uniform(*prior.drift, regimes)
        volatility = log_uniform(rng, prior.volatility, (regimes,))
        volatility /= np.sqrt(np.mean(volatility**2))
        return drift[states] + volatility[states] * innovations(rng, prior.innovations, (length,))

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
        return self._task(price_path(self.returns(rng, length), rng, self.prior.path))


def _markov_states(rng: np.random.Generator, length: int, exit_probability: Array) -> NDArray[np.int64]:
    """
    Simulate a Markov chain one sojourn at a time.

    A stay in a regime lasts a geometric number of steps, so the loop runs once
    per regime change instead of once per step.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    length : int
        Number of time steps.
    exit_probability : Array
        Chance per step of leaving each regime, shape ``(regimes,)``; the next
        regime is uniform over the others.

    Returns
    -------
    NDArray[np.int64]
        Regime index of each step, shape ``(length,)``.
    """
    regimes = exit_probability.size
    states = np.empty(length, dtype=np.int64)
    state, start = int(rng.integers(regimes)), 0
    while start < length:
        stop = start + int(rng.geometric(exit_probability[state]))
        states[start:stop] = state
        state = (state + int(rng.integers(1, regimes))) % regimes
        start = stop
    return states
