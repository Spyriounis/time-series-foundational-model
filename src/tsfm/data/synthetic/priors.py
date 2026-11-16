"""
Priors of the synthetic generators: every parameter range and hyperparameter.

A ``Range`` is a closed interval ``(low, high)``. Each attribute says whether it
is sampled uniformly or log-uniformly; `research/synthetic_data.md` gives the
process equations the parameters enter.
"""

from dataclasses import dataclass

from tsfm.enums import Nonlinearity, TrendKind

type Range = tuple[float, float]
type IntRange = tuple[int, int]


@dataclass(frozen=True)
class InnovationPrior:
    """
    Distribution of unit-variance shocks.

    Attributes
    ----------
    gaussian_probability : float
        Chance that the shocks are Gaussian instead of Student-t.
    dof : Range
        Student-t degrees of freedom, uniform; above 2 so the variance exists.
    """

    gaussian_probability: float = 0.3
    dof: Range = (3.0, 10.0)


@dataclass(frozen=True)
class PathPrior:
    """
    Turn unit-scale log-returns into a price path.

    Attributes
    ----------
    level : Range
        Initial price, log-uniform.
    volatility : Range
        Standard deviation of one log-return, log-uniform (5-minute to daily bars).
    drift : Range
        Mean log-return in units of the volatility, uniform.
    max_log_range : float
        Largest excursion of the log-price from its start; longer paths are shrunk to it.
    """

    level: Range = (1e-3, 1e4)
    volatility: Range = (1e-5, 2e-2)
    drift: Range = (-0.02, 0.02)
    max_log_range: float = 4.0


@dataclass(frozen=True)
class LevelPrior:
    """
    Place a unit-scale signal at a random scale and offset.

    Attributes
    ----------
    scale : Range
        Multiplier of the signal, log-uniform.
    offset : Range
        Constant added to the signal before scaling, uniform.
    """

    scale: Range = (1e-3, 1e4)
    offset: Range = (-10.0, 10.0)


@dataclass(frozen=True)
class SessionPrior:
    """
    Deterministic intraday and day-of-week volatility profile.

    Attributes
    ----------
    flat_probability : float
        Chance of no profile, as for daily bars.
    divisors : tuple[int, ...]
        Bar sizes in 5-minute bars; ``BARS_PER_DAY`` divided by one gives the bars per day.
    peaks : IntRange
        Number of session peaks in a day; two or three give the U and W shapes.
    peak_height : Range
        Volatility added at a peak relative to the quiet level, uniform.
    peak_width : Range
        Width of a peak as a fraction of the day, uniform.
    weekday : Range
        Multiplier of each trading day of the week, uniform.
    """

    flat_probability: float = 0.25
    divisors: tuple[int, ...] = (1, 3, 12, 48)
    peaks: IntRange = (1, 3)
    peak_height: Range = (0.5, 3.0)
    peak_width: Range = (0.03, 0.15)
    weekday: Range = (0.7, 1.3)


@dataclass(frozen=True)
class TrendSeasonalPrior:
    """
    Prior of the trend + seasonality + shifts + noise family.

    Attributes
    ----------
    trend_kinds : tuple[TrendKind, ...]
        Trend shapes, equally likely.
    trend : Range
        Total change of the trend over the series, uniform.
    breakpoints : IntRange
        Number of slope changes of a piecewise-linear trend.
    growth : Range
        Rate of an exponential trend over the whole series, uniform.
    seasons : IntRange
        Number of seasonal components.
    periods : tuple[int, ...]
        Market periods in steps (hour, 2 hours, day and week of 5-minute bars).
    candidate_probability : float
        Chance that a period comes from ``periods`` instead of ``random_period``.
    random_period : Range
        Any other period in steps, log-uniform.
    amplitude : Range
        Peak of a seasonal component, log-uniform.
    harmonics : IntRange
        Number of harmonics shaping one seasonal waveform.
    modulation_probability : float
        Chance that a component's amplitude varies slowly.
    modulation_depth : Range
        Relative depth of the amplitude modulation, uniform.
    modulation_cycles : Range
        Modulation cycles over the series, uniform.
    shifts : IntRange
        Number of level shifts or spikes.
    shift_size : Range
        Magnitude of a shift, uniform; the sign is random.
    recovery_probability : float
        Chance that a shift decays back (a spike) instead of persisting (a step).
    recovery_time : Range
        Decay time of a spike in steps, log-uniform.
    noise_ratio : Range
        Noise standard deviation over the signal's, log-uniform.
    innovations : InnovationPrior
        Noise distribution.
    level : LevelPrior
        Output scale and offset.
    """

    trend_kinds: tuple[TrendKind, ...] = tuple(TrendKind)
    trend: Range = (-3.0, 3.0)
    breakpoints: IntRange = (1, 4)
    growth: Range = (0.5, 4.0)
    seasons: IntRange = (1, 3)
    periods: tuple[int, ...] = (12, 24, 288, 1440)
    candidate_probability: float = 0.7
    random_period: Range = (4.0, 2048.0)
    amplitude: Range = (0.1, 1.0)
    harmonics: IntRange = (1, 3)
    modulation_probability: float = 0.3
    modulation_depth: Range = (0.2, 0.8)
    modulation_cycles: Range = (0.5, 4.0)
    shifts: IntRange = (0, 3)
    shift_size: Range = (0.5, 3.0)
    recovery_probability: float = 0.5
    recovery_time: Range = (2.0, 200.0)
    noise_ratio: Range = (0.01, 1.0)
    innovations: InnovationPrior = InnovationPrior()
    level: LevelPrior = LevelPrior()


@dataclass(frozen=True)
class KernelPrior:
    """
    Prior of the Gaussian-process family; inputs lie on the unit interval.

    Attributes
    ----------
    kernels : IntRange
        Number of bank kernels composed.
    product_probability : float
        Chance that two kernels are multiplied instead of added.
    grid : int
        Most points the process is drawn on, since a draw costs the cube of that number.
    length_scale : Range
        Length scale of the RBF, periodic and rational-quadratic kernels, log-uniform.
    periodicity : Range
        Period of the periodic kernel, log-uniform.
    mixture : Range
        Scale-mixture parameter of the rational-quadratic kernel, log-uniform.
    noise : Range
        Variance of the white kernel, log-uniform.
    constant : Range
        Value of the constant kernel, log-uniform.
    intercept : Range
        Offset ``sigma_0`` of the linear kernel, uniform.
    jitter : float
        Diagonal added to the covariance, relative to its mean variance.
    attempts : int
        Kernels tried before giving up when a covariance is not positive definite.
    level : LevelPrior
        Output scale and offset.
    """

    kernels: IntRange = (1, 5)
    product_probability: float = 0.5
    grid: int = 512
    length_scale: Range = (0.01, 1.0)
    periodicity: Range = (0.02, 0.5)
    mixture: Range = (0.1, 10.0)
    noise: Range = (1e-3, 0.1)
    constant: Range = (0.1, 2.0)
    intercept: Range = (0.0, 1.0)
    jitter: float = 1e-6
    attempts: int = 8
    level: LevelPrior = LevelPrior()


@dataclass(frozen=True)
class GarchPrior:
    """
    Prior of the GARCH(1,1) family; the unconditional variance is one.

    Attributes
    ----------
    persistence : Range
        Sum of the shock and variance coefficients, uniform; near one for long memory.
    shock : Range
        Coefficient of the squared return, uniform.
    innovations : InnovationPrior
        Shock distribution.
    session : SessionPrior
        Intraday and weekly volatility profile.
    path : PathPrior
        Price level, volatility and drift.
    """

    persistence: Range = (0.90, 0.999)
    shock: Range = (0.02, 0.15)
    innovations: InnovationPrior = InnovationPrior()
    session: SessionPrior = SessionPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class JumpPrior:
    """
    Prior of the Merton jump-diffusion family; the diffusion volatility is one.

    Attributes
    ----------
    intensity : Range
        Expected jumps per step, log-uniform.
    jump_scale : Range
        Standard deviation of a jump in diffusion volatilities, uniform.
    jump_mean : Range
        Mean of a jump in diffusion volatilities, uniform.
    innovations : InnovationPrior
        Diffusion shock distribution.
    path : PathPrior
        Price level, volatility and drift.
    """

    intensity: Range = (1e-3, 5e-2)
    jump_scale: Range = (3.0, 15.0)
    jump_mean: Range = (-1.0, 1.0)
    innovations: InnovationPrior = InnovationPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class RegimePrior:
    """
    Prior of the Markov-switching family.

    Attributes
    ----------
    regimes : IntRange
        Number of hidden regimes.
    exit_probability : Range
        Chance per step of leaving a regime, log-uniform; small values make regimes sticky.
    drift : Range
        Mean return of a regime in units of the volatility, uniform.
    volatility : Range
        Relative volatility of a regime, log-uniform.
    innovations : InnovationPrior
        Shock distribution.
    path : PathPrior
        Price level, volatility and drift.
    """

    regimes: IntRange = (2, 4)
    exit_probability: Range = (5e-4, 2e-2)
    drift: Range = (-0.1, 0.1)
    volatility: Range = (0.3, 3.0)
    innovations: InnovationPrior = InnovationPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class DriverPrior:
    """
    Mixture of univariate return processes that drive the multivariate families.

    Attributes
    ----------
    garch : GarchPrior
        Prior of the GARCH driver.
    jump : JumpPrior
        Prior of the jump-diffusion driver.
    regime : RegimePrior
        Prior of the regime-switching driver.
    weights : tuple[float, float, float]
        Probabilities of the GARCH, jump and regime drivers.
    session : SessionPrior
        Volatility profile shared by all rows of a task.
    """

    garch: GarchPrior = GarchPrior()
    jump: JumpPrior = JumpPrior()
    regime: RegimePrior = RegimePrior()
    weights: tuple[float, float, float] = (0.6, 0.2, 0.2)
    session: SessionPrior = SessionPrior()


@dataclass(frozen=True)
class FactorPrior:
    """
    Prior of the latent-factor family.

    Attributes
    ----------
    rows : IntRange
        Number of price rows.
    factors : IntRange
        Number of shared latent factors.
    idiosyncratic : Range
        Standard deviation of a row's own noise against unit-variance loadings, log-uniform.
    drivers : DriverPrior
        Processes of the factor returns.
    innovations : InnovationPrior
        Idiosyncratic noise distribution.
    path : PathPrior
        Price level, volatility and drift of each row.
    """

    rows: IntRange = (2, 8)
    factors: IntRange = (1, 3)
    idiosyncratic: Range = (0.2, 2.0)
    drivers: DriverPrior = DriverPrior()
    innovations: InnovationPrior = InnovationPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class LeadLagPrior:
    """
    Prior of the lead-lag family.

    Attributes
    ----------
    followers : IntRange
        Number of rows besides the base path.
    max_lag : int
        Largest lag in steps; lags are distinct, with short ones more likely.
    level_probability : float
        Chance that a follower copies the base level instead of its returns.
    nonlinearities : tuple[Nonlinearity, ...]
        Transforms of the copied signal, equally likely.
    noise_ratio : Range
        Noise standard deviation over the copied signal's, log-uniform.
    drivers : DriverPrior
        Process of the base returns.
    innovations : InnovationPrior
        Noise distribution.
    path : PathPrior
        Price level, volatility and drift of the base and of return followers.
    level : LevelPrior
        Scale and offset of level followers.
    """

    followers: IntRange = (1, 4)
    max_lag: int = 64
    level_probability: float = 0.4
    nonlinearities: tuple[Nonlinearity, ...] = tuple(Nonlinearity)
    noise_ratio: Range = (0.05, 1.0)
    drivers: DriverPrior = DriverPrior()
    innovations: InnovationPrior = InnovationPrior()
    path: PathPrior = PathPrior()
    level: LevelPrior = LevelPrior()


@dataclass(frozen=True)
class CointegrationPrior:
    """
    Prior of the cointegration family.

    Attributes
    ----------
    rows : IntRange
        Number of prices sharing the trend.
    loading : Range
        Exposure of a log-price to the common trend, uniform.
    half_life : Range
        Half-life of a spread in steps, log-uniform.
    spread : Range
        Stationary standard deviation of a spread in one-step trend volatilities, log-uniform.
    drivers : DriverPrior
        Process of the trend returns.
    path : PathPrior
        Price levels, trend volatility and drift.
    """

    rows: IntRange = (2, 5)
    loading: Range = (0.5, 2.0)
    half_life: Range = (4.0, 512.0)
    spread: Range = (1.0, 10.0)
    drivers: DriverPrior = DriverPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class TrianglePrior:
    """
    Prior of the FX cross-rate family.

    Attributes
    ----------
    currencies : IntRange
        Number of latent currency values.
    pairs : IntRange
        Number of cross rates emitted, at most every pair of currencies.
    loading : Range
        Exposure of a currency to the common factor, uniform.
    volatility : Range
        Standard deviation of one currency log-return, log-uniform.
    relative_volatility : Range
        Multiplier of that volatility per currency, uniform.
    value : Range
        Initial value of a currency, log-uniform; rates are ratios of two.
    max_log_range : float
        Largest excursion of a log-value from its start.
    drivers : DriverPrior
        Processes of the factor and currency returns.
    """

    currencies: IntRange = (3, 5)
    pairs: IntRange = (2, 6)
    loading: Range = (-1.0, 1.0)
    volatility: Range = (1e-5, 2e-2)
    relative_volatility: Range = (0.5, 2.0)
    value: Range = (1e-2, 1e2)
    max_log_range: float = 3.0
    drivers: DriverPrior = DriverPrior()


@dataclass(frozen=True)
class EventPrior:
    """
    Prior of the scheduled-event family.

    Attributes
    ----------
    flags : IntRange
        Number of event calendars, one flag row each.
    spacing : Range
        Steps between scheduled events, log-uniform; at most half the series.
    jitter : Range
        Largest displacement of an event as a fraction of the spacing, uniform.
    boost : Range
        Volatility multiplier at an event, log-uniform.
    decay : Range
        Decay time of the boost in steps, log-uniform.
    damp_probability : float
        Chance that volatility is damped just before events.
    damp : Range
        Volatility multiplier in the pre-event window, uniform.
    damp_window : IntRange
        Length of the pre-event window in steps.
    jump_probability : float
        Chance that a calendar's events move the level.
    jump_scale : Range
        Standard deviation of an event jump in volatilities, uniform.
    drivers : DriverPrior
        Process of the baseline returns.
    path : PathPrior
        Price level, volatility and drift.
    """

    flags: IntRange = (1, 2)
    spacing: Range = (48.0, 1024.0)
    jitter: Range = (0.0, 0.2)
    boost: Range = (1.5, 6.0)
    decay: Range = (2.0, 32.0)
    damp_probability: float = 0.5
    damp: Range = (0.4, 0.9)
    damp_window: IntRange = (4, 32)
    jump_probability: float = 0.7
    jump_scale: Range = (3.0, 15.0)
    drivers: DriverPrior = DriverPrior()
    path: PathPrior = PathPrior()


@dataclass(frozen=True)
class MixupPrior:
    """
    Prior of the TSMixup augmentation.

    Attributes
    ----------
    components : IntRange
        Number of series mixed.
    concentration : float
        Dirichlet concentration of the mixing weights.
    """

    components: IntRange = (2, 3)
    concentration: float = 1.5
