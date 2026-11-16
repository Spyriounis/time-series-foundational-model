"""Merton jump diffusion: a diffusion plus compound-Poisson jumps in the log-price."""

import numpy as np

from tsfm.data.synthetic.base import SyntheticTask
from tsfm.data.synthetic.paths import Array, ReturnGenerator, innovations, log_uniform, price_path
from tsfm.data.synthetic.priors import JumpPrior
from tsfm.enums import GeneratorFamily


class JumpGenerator(ReturnGenerator):
    """
    Sample a price with rare jumps several times larger than its diffusion moves.

    Parameters
    ----------
    prior : JumpPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = JumpGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape
    (1, 256)
    """

    family = GeneratorFamily.JUMP

    def __init__(self, prior: JumpPrior | None = None) -> None:
        self.prior = prior or JumpPrior()

    def returns(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw jump-diffusion log-returns, scaled to unit variance.

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
        intensity = log_uniform(rng, prior.intensity)
        mean, scale = rng.uniform(*prior.jump_mean), rng.uniform(*prior.jump_scale)
        counts = rng.poisson(intensity, length)
        # A sum of `counts` Gaussian jumps is Gaussian with both moments scaled by the count.
        jumps = counts * mean + np.sqrt(counts) * scale * rng.standard_normal(length)
        diffusion = innovations(rng, prior.innovations, (length,))
        return np.asarray((diffusion + jumps) / np.sqrt(1.0 + intensity * (mean**2 + scale**2)), dtype=np.float64)

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
