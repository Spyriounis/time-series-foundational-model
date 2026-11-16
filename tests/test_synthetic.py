from itertools import combinations, product

import numpy as np
import pyarrow.parquet as pq
import pytest

from tsfm.data.schemas import SYNTHETIC_SCHEMA
from tsfm.data.storage import DataKey, ParquetStore
from tsfm.data.synthetic import GENERATORS, CorpusBuilder, SyntheticTask, corpus, mixup
from tsfm.data.synthetic.events import EventGenerator
from tsfm.data.synthetic.multivariate import CointegrationGenerator, LeadLagGenerator
from tsfm.data.synthetic.priors import LeadLagPrior, PathPrior
from tsfm.data.synthetic.triangle import TriangleGenerator
from tsfm.data.synthetic.volatility import GarchGenerator
from tsfm.enums import Dataset, GeneratorFamily, Nonlinearity, RowKind, Split

FAMILIES = list(GeneratorFamily)


def sample(family: GeneratorFamily, seed: int, length: int = 512) -> SyntheticTask:
    """One task of a family from a fresh seeded stream."""
    return GENERATORS[family]().sample(np.random.default_rng(seed), length)


def test_registry_covers_every_family():
    """The corpus builder looks generators up by family, so a missing or mislabelled entry loses a stratum."""
    assert set(GENERATORS) == set(GeneratorFamily)
    assert all(generator.family is family for family, generator in GENERATORS.items())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("length", [64, 4096])
def test_task_is_finite_float32_of_the_requested_length(family, length):
    """The model's scaler cannot recover from an overflow or a wrong shape, at either end of the length range."""
    for seed in range(3):
        task = sample(family, seed, length)
        assert task.family is family
        assert task.values.dtype == np.float32
        assert task.values.shape == (len(task.kinds), length)
        assert np.isfinite(task.values).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_seed_determines_the_task(family):
    """The corpus is reproducible only if a generator draws from nothing but the stream it is handed."""
    np.testing.assert_array_equal(sample(family, 0).values, sample(family, 0).values)
    first, second = sample(family, 0).values, sample(family, 1).values
    assert first.shape != second.shape or not np.array_equal(first, second)


@pytest.mark.parametrize("family", [GeneratorFamily.GARCH, GeneratorFamily.JUMP, GeneratorFamily.REGIME])
def test_prices_are_strictly_positive(family):
    """Log-returns are a candidate target transform, and they need positive prices."""
    assert all((sample(family, seed).values > 0).all() for seed in range(10))


def test_triangle_closes():
    """Cross-rate consistency is the structure this family teaches; it must hold exactly, not on average."""
    closed = 0
    for seed in range(20):
        logs = np.log(sample(GeneratorFamily.TRIANGLE, seed).values.astype(np.float64))
        if len(logs) < 3:
            continue
        # Row order and quote orientation are random, so search the triples and signs.
        residuals = [
            np.abs(logs[i] + a * logs[j] + b * logs[k]).max()
            for i, j, k in combinations(range(len(logs)), 3)
            for a, b in product((-1, 1), repeat=2)
        ]
        assert min(residuals) < 1e-5
        closed += 1
    assert closed > 0


def test_triangle_emits_distinct_pairs():
    """Two rows quoting the same pair would be a trivial copy task instead of a cross-rate one."""
    pairs = TriangleGenerator()._pairs(np.random.default_rng(0), currencies=5)
    assert len({frozenset(pair) for pair in pairs.tolist()}) == len(pairs)


def test_garch_returns_have_fat_tails_and_clustered_volatility():
    """These two stylised facts of market returns are the reason the family exists."""
    kurtosis, autocorrelation = [], []
    for seed in range(20):
        returns = GarchGenerator().returns(np.random.default_rng(seed), 4096)
        centred = returns - returns.mean()
        kurtosis.append(np.mean(centred**4) / np.mean(centred**2) ** 2 - 3.0)
        autocorrelation.append(np.corrcoef(centred[1:] ** 2, centred[:-1] ** 2)[0, 1])
    assert np.mean(kurtosis) > 0.5
    assert np.mean(autocorrelation) > 0.05


def test_event_flags_are_binary_and_follow_the_price():
    """The model treats flag rows as known-future covariates and row 0 as the target, so the layout is a contract."""
    for seed in range(10):
        task = sample(GeneratorFamily.EVENT, seed)
        assert task.kinds[0] is RowKind.CONTINUOUS
        assert set(task.kinds[1:]) == {RowKind.EVENT_FLAG}
        assert set(np.unique(task.values[1:])) == {0.0, 1.0}
        assert (task.values[0] > 0).all()


def test_events_raise_volatility():
    """A flag with no effect on the price would teach the model to ignore known-future covariates."""
    ratios = []
    for seed in range(30):
        price, flag = EventGenerator().sample(np.random.default_rng(seed), 4096).values[:2].astype(np.float64)
        moves = np.abs(np.diff(np.log(price)))
        ratios.append(moves[flag[1:] > 0].mean() / moves.mean())
    assert np.median(ratios) > 1.5


def test_cointegration_spread_is_tighter_than_the_levels():
    """Without a stationary spread the rows are just correlated walks and there is no error correction to learn."""
    ratios = []
    for seed in range(20):
        logs = np.log(CointegrationGenerator().sample(np.random.default_rng(seed), 4096).values.astype(np.float64))
        slope, intercept = np.polyfit(logs[0], logs[1], 1)
        ratios.append(np.std(logs[1] - slope * logs[0] - intercept) / np.std(logs[1]))
    assert np.median(ratios) < 0.3


def test_lead_lag_peaks_at_a_nonzero_lag():
    """If the copies were contemporaneous, no row would carry information about another's future."""
    prior = LeadLagPrior(level_probability=0.0, nonlinearities=(Nonlinearity.IDENTITY,), noise_ratio=(0.1, 0.1))
    lags = np.arange(-prior.max_lag, prior.max_lag + 1)
    for seed in range(5):
        values = LeadLagGenerator(prior).sample(np.random.default_rng(seed), 2048).values.astype(np.float64)
        base, follower = np.diff(np.log(values[:2]), axis=1)
        window = slice(prior.max_lag, -prior.max_lag)
        correlation = [np.corrcoef(base[window], np.roll(follower, -lag)[window])[0, 1] for lag in lags]
        assert lags[int(np.argmax(np.abs(correlation)))] != 0
        assert np.max(np.abs(correlation)) > 0.5


def test_path_prior_bounds_the_log_range():
    """The bound is what keeps `exp` from overflowing float32 on long, volatile paths."""
    prior = GarchGenerator().prior
    wild = type(prior)(path=PathPrior(volatility=(1.0, 1.0), max_log_range=2.0))
    logs = np.log(GarchGenerator(wild).sample(np.random.default_rng(0), 4096).values.astype(np.float64))
    assert np.abs(logs - logs[:, :1]).max() <= 2.0 + 1e-4


def test_mixup_is_a_convex_combination():
    """A mix outside its components' envelope would mean the weights are not a point on the simplex."""
    tasks = [sample(family, seed=3, length=256) for family in (GeneratorFamily.GARCH, GeneratorFamily.KERNEL)]
    tasks.append(sample(GeneratorFamily.TREND_SEASONAL, seed=4, length=256))
    mixed = mixup(tasks, np.random.default_rng(0))
    scaled = np.stack([task.values[0] / np.abs(task.values[0]).mean() for task in tasks]).astype(np.float64)
    ours = mixed.values[0] / np.abs(tasks[0].values[0]).mean()
    tolerance = 1e-4 * np.abs(scaled).max()
    assert mixed.family is GeneratorFamily.GARCH
    assert (ours >= scaled.min(axis=0) - tolerance).all()
    assert (ours <= scaled.max(axis=0) + tolerance).all()
    # The weights that reproduce the mix are non-negative and sum to one.
    weights, *_ = np.linalg.lstsq(scaled.T, ours, rcond=None)
    assert weights.min() > 0
    assert weights.sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("family", [GeneratorFamily.FACTOR, GeneratorFamily.EVENT])
def test_mixup_rejects_tasks_it_cannot_mix(family):
    """Mixing multi-row or flag tasks would silently destroy their structure."""
    with pytest.raises(ValueError, match="single-row"):
        mixup([sample(family, 0), sample(family, 1)], np.random.default_rng(0))


# Tasks per shard in the shrunken corpus.
SHARD_TASKS = 4


@pytest.fixture
def small_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink lengths and shards so a corpus builds in well under a second."""
    monkeypatch.setattr(corpus, "SYNTHETIC_MIN_LENGTH", 64)
    monkeypatch.setattr(corpus, "SYNTHETIC_MAX_LENGTH", 128)
    monkeypatch.setattr(corpus, "SYNTHETIC_SHARD_TASKS", SHARD_TASKS)
    monkeypatch.setattr(corpus, "SYNTHETIC_MIXUP_PROBABILITY", 0.5)


def read_split(store: ParquetStore, split: Split) -> dict[tuple[str, int, int], list[float]]:
    """Every stored row of a split, keyed by (family, task_id, row), after checking each shard."""
    rows = {}
    for family in GeneratorFamily:
        for path in store.layout.files(DataKey(Dataset.SYNTHETIC, (split.value, family.value))):
            table = pq.ParquetFile(path).read()
            assert table.schema.equals(SYNTHETIC_SCHEMA)
            assert len(set(table["task_id"].to_pylist())) <= SHARD_TASKS
            for record in table.to_pylist():
                rows[family.value, record["task_id"], record["row"]] = record["values"]
    return rows


@pytest.mark.usefixtures("small_corpus")
def test_train_corpus_is_sharded_complete_and_deterministic(store: ParquetStore, tmp_path_factory):
    """Training reads these shards blindly: every task must be there once, and a rebuild must not change it."""
    assert CorpusBuilder(store, seed=0).build(Split.TRAIN, 40) == 40
    rows = read_split(store, Split.TRAIN)
    assert sorted({task_id for _, task_id, _ in rows}) == list(range(40))
    assert all(64 <= len(values) <= 128 for values in rows.values())

    other = ParquetStore(type(store.layout)(tmp_path_factory.mktemp("again")))
    CorpusBuilder(other, seed=0).build(Split.TRAIN, 40)
    assert read_split(other, Split.TRAIN) == rows
    CorpusBuilder(other, seed=1).build(Split.TRAIN, 40)
    assert read_split(other, Split.TRAIN) != rows


@pytest.mark.usefixtures("small_corpus")
def test_val_corpus_is_balanced_and_disjoint_from_train(store: ParquetStore):
    """Per-family validation loss needs equal slices, and it means nothing if a slice repeats training data."""
    builder = CorpusBuilder(store, seed=0)
    assert builder.build(Split.VAL, 3) == 3 * len(GeneratorFamily)
    builder.build(Split.TRAIN, 30)
    val, train = read_split(store, Split.VAL), read_split(store, Split.TRAIN)
    for family in GeneratorFamily:
        assert len({task_id for name, task_id, _ in val if name == family.value}) == 3
    # Flag rows are excluded: 0/1 calendars can coincide without any leakage.
    continuous = [{tuple(v) for v in rows.values() if set(v) != {0.0, 1.0}} for rows in (val, train)]
    assert continuous[0].isdisjoint(continuous[1])


@pytest.mark.usefixtures("small_corpus")
def test_mixed_price_tasks_stay_positive(store: ParquetStore):
    """The loader takes log-returns of price families, so a mix with a level series would give NaN."""
    builder = CorpusBuilder(store, seed=0)
    prices = [GeneratorFamily.GARCH, GeneratorFamily.JUMP, GeneratorFamily.REGIME, GeneratorFamily.KERNEL]
    tasks = [builder.sample(Split.TRAIN, task_id, prices) for task_id in range(300)]
    assert all(task.values.min() > 0 for task in tasks if task.family is not GeneratorFamily.KERNEL)


@pytest.mark.usefixtures("small_corpus")
def test_rebuild_replaces_the_whole_split(store: ParquetStore):
    """Task ids depend on the families built, so leftovers of another build would repeat ids."""
    builder = CorpusBuilder(store, seed=0)
    builder.build(Split.VAL, 2)
    builder.build(Split.VAL, 2, families=[GeneratorFamily.GARCH])
    assert {family for family, _, _ in read_split(store, Split.VAL)} == {"garch"}


@pytest.mark.usefixtures("small_corpus")
def test_builder_regenerates_a_stored_task(store: ParquetStore):
    """A task must be a pure function of (seed, split, task_id), so one can be debugged without the corpus."""
    builder = CorpusBuilder(store, seed=7)
    builder.build(Split.TRAIN, 12, families=[GeneratorFamily.GARCH, GeneratorFamily.TRIANGLE])
    rows = read_split(store, Split.TRAIN)
    assert {family for family, _, _ in rows} <= {"garch", "triangle"}
    for (_, task_id, row), values in rows.items():
        task = builder.sample(Split.TRAIN, task_id, [GeneratorFamily.GARCH, GeneratorFamily.TRIANGLE])
        assert task.values[row].tolist() == values
