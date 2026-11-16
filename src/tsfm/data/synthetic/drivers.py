"""Return processes that drive the multivariate families, composed from the univariate ones."""

import numpy as np

from tsfm.data.synthetic.jumps import JumpGenerator
from tsfm.data.synthetic.paths import Array, ReturnGenerator
from tsfm.data.synthetic.priors import DriverPrior
from tsfm.data.synthetic.regimes import RegimeGenerator
from tsfm.data.synthetic.volatility import GarchGenerator, session_profile


def driver_returns(rng: np.random.Generator, length: int, rows: int, prior: DriverPrior) -> Array:
    """
    Draw independent unit-scale return rows that share one session volatility profile.

    Each row is a GARCH, jump-diffusion or regime-switching process, so every
    multivariate family inherits heavy tails and volatility clustering.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    length : int
        Number of time steps.
    rows : int
        Number of return rows.
    prior : DriverPrior
        Priors and probabilities of the three processes, and the session profile.

    Returns
    -------
    Array
        Log-returns, shape ``(rows, length)``.
    """
    processes: tuple[ReturnGenerator, ...] = (
        GarchGenerator(prior.garch),
        JumpGenerator(prior.jump),
        RegimeGenerator(prior.regime),
    )
    weights = np.array(prior.weights) / sum(prior.weights)
    chosen = rng.choice(len(processes), size=rows, p=weights)
    returns = np.stack([processes[index].returns(rng, length) for index in map(int, chosen)])
    return returns * session_profile(rng, length, prior.session)
