"""KernelSynth-style draws from Gaussian processes with composed kernels."""

from collections.abc import Callable
from typing import cast

import numpy as np
from sklearn.gaussian_process.kernels import (  # pyright: ignore[reportMissingTypeStubs]  # no stub package exists
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.paths import Array, level_series, log_uniform, pick, randint
from tsfm.data.synthetic.priors import KernelPrior
from tsfm.enums import GeneratorFamily


class KernelGenerator(SyntheticDataGenerator):
    """
    Sample a Gaussian-process path whose kernel is a random sum and product of bank kernels.

    Parameters
    ----------
    prior : KernelPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = KernelGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.values.shape
    (1, 256)
    """

    family = GeneratorFamily.KERNEL

    def __init__(self, prior: KernelPrior | None = None) -> None:
        self.prior = prior or KernelPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one single-row task.

        The process is drawn on at most ``prior.grid`` points and interpolated
        linearly, because a draw costs the cube of the number of points.

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

        Raises
        ------
        RuntimeError
            If no kernel in ``prior.attempts`` tries gives a positive-definite covariance.
        """
        prior = self.prior
        grid = np.linspace(0.0, 1.0, min(length, prior.grid))
        for _ in range(prior.attempts):
            draw = _gp_draw(rng, _compose(rng, prior), grid, prior.jitter)
            if draw is not None:
                path = np.interp(np.linspace(0.0, 1.0, length), grid, draw)
                return self._task(level_series(path, rng, prior.level))
        raise RuntimeError(f"No positive-definite kernel in {prior.attempts} attempts")


def _bank_kernel(rng: np.random.Generator, prior: KernelPrior) -> Kernel:
    """
    Draw one kernel of the bank with random hyperparameters.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    prior : KernelPrior
        Hyperparameter ranges.

    Returns
    -------
    Kernel
        Linear, RBF, periodic, rational-quadratic, white or constant kernel.
    """
    length_scale = float(log_uniform(rng, prior.length_scale))
    bank: tuple[Callable[[], Kernel], ...] = (
        lambda: DotProduct(sigma_0=rng.uniform(*prior.intercept)),
        lambda: RBF(length_scale=length_scale),
        lambda: ExpSineSquared(length_scale=length_scale, periodicity=float(log_uniform(rng, prior.periodicity))),
        lambda: RationalQuadratic(length_scale=length_scale, alpha=float(log_uniform(rng, prior.mixture))),
        lambda: WhiteKernel(noise_level=float(log_uniform(rng, prior.noise))),
        lambda: ConstantKernel(constant_value=float(log_uniform(rng, prior.constant))),
    )
    return pick(rng, bank)()


def _compose(rng: np.random.Generator, prior: KernelPrior) -> Kernel:
    """
    Combine bank kernels with random sums and products.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    prior : KernelPrior
        Number of kernels and the chance of a product.

    Returns
    -------
    Kernel
        Composite kernel.
    """
    kernel = _bank_kernel(rng, prior)
    for _ in range(randint(rng, prior.kernels) - 1):
        other = _bank_kernel(rng, prior)
        kernel = kernel * other if rng.random() < prior.product_probability else kernel + other
    return kernel


def _gp_draw(rng: np.random.Generator, kernel: Kernel, grid: Array, jitter: float) -> Array | None:
    """
    Sample the zero-mean process at the grid points, scaled to unit prior variance.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    kernel : Kernel
        Covariance function.
    grid : Array
        Input points, shape ``(points,)``.
    jitter : float
        Diagonal added for numerical stability, relative to the mean variance.

    Returns
    -------
    Array or None
        Sample, shape ``(points,)``; None if the covariance is not positive definite.
    """
    covariance = cast("Array", kernel(grid[:, None]))
    variance = float(np.mean(np.diag(covariance)))
    try:
        factor = np.linalg.cholesky(covariance + jitter * variance * np.eye(grid.size))
    except np.linalg.LinAlgError:
        return None
    return np.asarray(factor @ rng.standard_normal(grid.size) / np.sqrt(variance), dtype=np.float64)
