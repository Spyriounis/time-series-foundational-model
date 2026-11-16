"""Generator registry and the builder that writes the synthetic corpus to the store."""

from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from tsfm.constants import (
    SYNTHETIC_FAMILY_WEIGHTS,
    SYNTHETIC_MAX_LENGTH,
    SYNTHETIC_MIN_LENGTH,
    SYNTHETIC_MIXUP_GROUPS,
    SYNTHETIC_MIXUP_PROBABILITY,
    SYNTHETIC_SHARD_TASKS,
)
from tsfm.data.schemas import SYNTHETIC_SCHEMA
from tsfm.data.storage import DataKey, ParquetStore
from tsfm.data.synthetic.augment import mixup
from tsfm.data.synthetic.base import SyntheticDataGenerator, SyntheticTask
from tsfm.data.synthetic.events import EventGenerator
from tsfm.data.synthetic.jumps import JumpGenerator
from tsfm.data.synthetic.kernels import KernelGenerator
from tsfm.data.synthetic.multivariate import CointegrationGenerator, FactorGenerator, LeadLagGenerator
from tsfm.data.synthetic.paths import log_uniform, pick, randint
from tsfm.data.synthetic.priors import MixupPrior
from tsfm.data.synthetic.regimes import RegimeGenerator
from tsfm.data.synthetic.triangle import TriangleGenerator
from tsfm.data.synthetic.univariate import TrendSeasonalGenerator
from tsfm.data.synthetic.volatility import GarchGenerator
from tsfm.enums import Dataset, GeneratorFamily, Split
from tsfm.log import get_logger

log = get_logger(__name__)

GENERATORS: dict[GeneratorFamily, type[SyntheticDataGenerator]] = {
    GeneratorFamily.TREND_SEASONAL: TrendSeasonalGenerator,
    GeneratorFamily.KERNEL: KernelGenerator,
    GeneratorFamily.GARCH: GarchGenerator,
    GeneratorFamily.JUMP: JumpGenerator,
    GeneratorFamily.REGIME: RegimeGenerator,
    GeneratorFamily.FACTOR: FactorGenerator,
    GeneratorFamily.LEAD_LAG: LeadLagGenerator,
    GeneratorFamily.COINTEGRATION: CointegrationGenerator,
    GeneratorFamily.TRIANGLE: TriangleGenerator,
    GeneratorFamily.EVENT: EventGenerator,
}
# Family -> families it may be mixed with; the single-row families only.
MIXUP_PARTNERS: dict[GeneratorFamily, tuple[GeneratorFamily, ...]] = {
    family: group for group in SYNTHETIC_MIXUP_GROUPS for family in group
}


class CorpusBuilder:
    """
    Generate the synthetic corpus and write it as Parquet shards.

    A task is a pure function of ``(seed, split, task_id)``: its random stream is
    the child ``(split, task_id)`` of the seed, so splits never share a stream
    and any task can be regenerated alone.

    Parameters
    ----------
    store : ParquetStore
        Destination store.
    seed : int
        Root seed of the corpus.
    mixing : MixupPrior or None, default: None
        Prior of the mixup augmentation; the default prior if omitted.

    Examples
    --------
    >>> builder = CorpusBuilder(store, seed=0)  # doctest: +SKIP
    >>> builder.build(Split.TRAIN, n_tasks=10_000)  # doctest: +SKIP
    10000
    """

    def __init__(self, store: ParquetStore, seed: int, mixing: MixupPrior | None = None) -> None:
        self.store = store
        self.seed = seed
        self.mixing = mixing or MixupPrior()
        self.generators = {family: generator() for family, generator in GENERATORS.items()}

    def build(self, split: Split, n_tasks: int, families: Sequence[GeneratorFamily] | None = None) -> int:
        """
        Generate a split and write it in place of the stored one, a shard at a time.

        Parameters
        ----------
        split : Split
            Training tasks draw their family by ``SYNTHETIC_FAMILY_WEIGHTS`` and
            may be mixed; validation tasks are balanced and never mixed.
        n_tasks : int
            Number of tasks: in total for training, per family for validation.
        families : Sequence[GeneratorFamily] or None, default: None
            Families to generate; all if omitted.

        Returns
        -------
        int
            Number of tasks written.
        """
        chosen = tuple(families or GeneratorFamily)
        total = n_tasks * len(chosen) if split is Split.VAL else n_tasks
        task_ids: dict[GeneratorFamily, list[int]] = {family: [] for family in chosen}
        for task_id in range(total):
            task_ids[self._family(split, task_id, self._rng(split, task_id), chosen)].append(task_id)
        # Task ids depend on the families chosen, so shards of an earlier build must not survive.
        for family in GeneratorFamily:
            self.store.clear(DataKey(Dataset.SYNTHETIC, (split.value, family.value)))
        for family, ids in task_ids.items():
            self._write_family(split, family, ids, chosen)
        log.info("corpus_built", split=split.value, tasks=total, seed=self.seed)
        return total

    def sample(self, split: Split, task_id: int, families: Sequence[GeneratorFamily] | None = None) -> SyntheticTask:
        """
        Generate one task of a split.

        Parameters
        ----------
        split : Split
            Split the task belongs to.
        task_id : int
            Index of the task within the split.
        families : Sequence[GeneratorFamily] or None, default: None
            Families the split was built with; all if omitted.

        Returns
        -------
        SyntheticTask
            The same task on every call.
        """
        rng = self._rng(split, task_id)
        family = self._family(split, task_id, rng, tuple(families or GeneratorFamily))
        length = round(float(log_uniform(rng, (SYNTHETIC_MIN_LENGTH, SYNTHETIC_MAX_LENGTH))))
        task = self.generators[family].sample(rng, length)
        mixes = split is Split.TRAIN and family in MIXUP_PARTNERS and rng.random() < SYNTHETIC_MIXUP_PROBABILITY
        return self._mix(task, rng) if mixes else task

    def _rng(self, split: Split, task_id: int) -> np.random.Generator:
        """
        Return the random stream of one task.

        Parameters
        ----------
        split : Split
            Split the task belongs to.
        task_id : int
            Index of the task within the split.

        Returns
        -------
        np.random.Generator
            Generator seeded by the child ``(split, task_id)`` of the root seed.
        """
        spawn_key = (list(Split).index(split), task_id)
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=spawn_key))

    def _family(
        self, split: Split, task_id: int, rng: np.random.Generator, families: tuple[GeneratorFamily, ...]
    ) -> GeneratorFamily:
        """
        Assign a task to a family: round-robin for validation, weighted draw for training.

        Parameters
        ----------
        split : Split
            Split the task belongs to.
        task_id : int
            Index of the task within the split.
        rng : np.random.Generator
            The task's stream; a training task spends its first draw here.
        families : tuple[GeneratorFamily, ...]
            Candidate families.

        Returns
        -------
        GeneratorFamily
            Family of the task.
        """
        if split is Split.VAL:
            return families[task_id % len(families)]
        weights = np.array([SYNTHETIC_FAMILY_WEIGHTS[family] for family in families])
        return families[int(rng.choice(len(families), p=weights / weights.sum()))]

    def _mix(self, task: SyntheticTask, rng: np.random.Generator) -> SyntheticTask:
        """
        Mix a single-row task with one or two fresh ones of its group and length.

        Parameters
        ----------
        task : SyntheticTask
            First component, which names the family of the mix.
        rng : np.random.Generator
            The task's stream.

        Returns
        -------
        SyntheticTask
            Mixed task.
        """
        length = task.values.shape[1]
        partners = [
            self.generators[pick(rng, MIXUP_PARTNERS[task.family])].sample(rng, length)
            for _ in range(randint(rng, self.mixing.components) - 1)
        ]
        return mixup([task, *partners], rng, self.mixing)

    def _write_family(
        self, split: Split, family: GeneratorFamily, task_ids: list[int], families: tuple[GeneratorFamily, ...]
    ) -> None:
        """
        Generate and write the shards of one family, holding one shard in memory.

        Parameters
        ----------
        split : Split
            Split being built.
        family : GeneratorFamily
            Family of every listed task.
        task_ids : list[int]
            Tasks assigned to the family, in increasing order.
        families : tuple[GeneratorFamily, ...]
            Families the split is built with.
        """
        key = DataKey(Dataset.SYNTHETIC, (split.value, family.value))
        for shard, start in enumerate(range(0, len(task_ids), SYNTHETIC_SHARD_TASKS)):
            ids = task_ids[start : start + SYNTHETIC_SHARD_TASKS]
            table = _shard_table(ids, [self.sample(split, task_id, families) for task_id in ids])
            self.store.write_shard(key, shard, table)
            log.info("shard_written", split=split.value, family=family.value, shard=shard, tasks=len(ids))


def _shard_table(task_ids: Sequence[int], tasks: Sequence[SyntheticTask]) -> pa.Table:
    """
    Lay tasks out as one table row per task row.

    Parameters
    ----------
    task_ids : Sequence[int]
        Identifier of each task.
    tasks : Sequence[SyntheticTask]
        Tasks, aligned with ``task_ids``.

    Returns
    -------
    pa.Table
        Table of ``SYNTHETIC_SCHEMA``.
    """
    columns: dict[str, list[object]] = {name: [] for name in SYNTHETIC_SCHEMA.names}
    for task_id, task in zip(task_ids, tasks, strict=True):
        for row, (kind, values) in enumerate(zip(task.kinds, task.values, strict=True)):
            columns["task_id"].append(task_id)
            columns["row"].append(row)
            columns["kind"].append(kind.value)
            columns["values"].append(values)
    return pa.table(columns, schema=SYNTHETIC_SCHEMA)
