"""Scheduled events: known-future 0/1 flags with volatility and level effects on a price."""

import numpy as np

from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.drivers import driver_returns
from tsfm.data.synthetic.paths import Array, log_uniform, price_path, randint
from tsfm.data.synthetic.priors import EventPrior
from tsfm.enums import GeneratorFamily, RowKind


class EventGenerator(SyntheticDataGenerator):
    """
    Sample a price that reacts to one or two calendars of scheduled events.

    Parameters
    ----------
    prior : EventPrior or None, default: None
        Parameter ranges; the default prior if omitted.

    Examples
    --------
    >>> task = EventGenerator().sample(np.random.default_rng(0), length=256)
    >>> task.kinds[0], task.kinds[1]
    (<RowKind.CONTINUOUS: 'continuous'>, <RowKind.EVENT_FLAG: 'event_flag'>)
    """

    family = GeneratorFamily.EVENT

    def __init__(self, prior: EventPrior | None = None) -> None:
        self.prior = prior or EventPrior()

    def sample(self, rng: np.random.Generator, length: int) -> SyntheticTask:
        """
        Draw one task; row 0 is the price and the other rows are event flags.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        SyntheticTask
            A price row followed by 0/1 flag rows.
        """
        returns = driver_returns(rng, length, 1, self.prior.drivers)[0]
        flags = [self._schedule(rng, length) for _ in range(randint(rng, self.prior.flags))]
        for flag in flags:
            returns = returns * self._volatility_scale(rng, flag) + self._jumps(rng, flag)
        price = price_path(returns, rng, self.prior.path)
        return self._task(np.stack([price, *flags]), (RowKind.CONTINUOUS, *[RowKind.EVENT_FLAG] * len(flags)))

    def _schedule(self, rng: np.random.Generator, length: int) -> Array:
        """
        Draw events at a regular spacing, each displaced by a small jitter.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        length : int
            Number of time steps.

        Returns
        -------
        Array
            Flag that is 1 at event steps and 0 elsewhere, shape ``(length,)``.
        """
        spacing = min(int(log_uniform(rng, self.prior.spacing)), length // 2)
        scheduled = np.arange(rng.integers(spacing), length, spacing)
        jitter = rng.uniform(*self.prior.jitter) * spacing
        times = np.rint(scheduled + rng.uniform(-jitter, jitter, scheduled.size)).astype(np.int64)
        flag = np.zeros(length)
        flag[np.clip(times, 0, length - 1)] = 1.0
        return flag

    def _volatility_scale(self, rng: np.random.Generator, flag: Array) -> Array:
        """
        Draw the volatility multiplier: a boost decaying after events, optionally a lull before.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        flag : Array
            Event flag, shape ``(length,)``.

        Returns
        -------
        Array
            Positive multiplier, shape ``(length,)``.
        """
        prior = self.prior
        boost, decay = log_uniform(rng, prior.boost), log_uniform(rng, prior.decay)
        scale = 1.0 + (boost - 1.0) * np.exp(-_steps_since(flag) / decay)
        if rng.random() < prior.damp_probability:
            until = _steps_since(flag[::-1])[::-1]
            lull = (until > 0) & (until <= randint(rng, prior.damp_window))
            scale = np.where(lull, rng.uniform(*prior.damp), scale)
        return np.asarray(scale, dtype=np.float64)

    def _jumps(self, rng: np.random.Generator, flag: Array) -> Array:
        """
        Draw random-signed level jumps at the events of one calendar, or none.

        Parameters
        ----------
        rng : np.random.Generator
            Source of randomness.
        flag : Array
            Event flag, shape ``(length,)``.

        Returns
        -------
        Array
            Return added at each step, zero away from events, shape ``(length,)``.
        """
        if rng.random() >= self.prior.jump_probability:
            return np.zeros_like(flag)
        return flag * rng.uniform(*self.prior.jump_scale) * rng.standard_normal(flag.size)


def _steps_since(flag: Array) -> Array:
    """
    Count the steps since the most recent event.

    Parameters
    ----------
    flag : Array
        Event flag, shape ``(length,)``.

    Returns
    -------
    Array
        Zero at an event, growing after it, and infinite before the first one.
    """
    steps = np.arange(flag.size)
    last = np.maximum.accumulate(np.where(flag > 0, steps, -1))
    return np.where(last >= 0, steps - last, np.inf)
