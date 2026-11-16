"""Commands that generate the synthetic corpus."""

from typing import Annotated

import typer

from tsfm.constants import (
    SYNTHETIC_DEFAULT_TASKS,
    SYNTHETIC_FAMILY_WEIGHTS,
    SYNTHETIC_SEEDS,
    SYNTHETIC_VAL_TASKS_PER_FAMILY,
)
from tsfm.data.storage import DataLayout, ParquetStore
from tsfm.data.synthetic.corpus import CorpusBuilder
from tsfm.enums import GeneratorFamily, Split
from tsfm.log import get_logger
from tsfm.settings import Settings

logger = get_logger(__name__)
app = typer.Typer(no_args_is_help=True, help="Generate the synthetic corpus.")

_DEFAULT_TASKS: dict[Split, int] = {Split.TRAIN: SYNTHETIC_DEFAULT_TASKS, Split.VAL: SYNTHETIC_VAL_TASKS_PER_FAMILY}


@app.command()
def generate(
    split: Split,
    n_tasks: Annotated[int | None, typer.Option(min=1, help="Tasks in total (train) or per family (val).")] = None,
    seed: Annotated[int | None, typer.Option(help="Corpus seed. Default: the split's fixed seed.")] = None,
    family: Annotated[
        list[GeneratorFamily] | None, typer.Option(help="Restrict to these families. Repeatable.")
    ] = None,
) -> None:
    """
    Write one split of the synthetic corpus to the store, replacing the stored split.

    Parameters
    ----------
    split : Split
        Train (weighted mixture) or val (equal slices per family).
    n_tasks : int or None, default: None
        Tasks in total for train, per family for val.
    seed : int or None, default: None
        Corpus seed; the split's constant if omitted.
    family : list[GeneratorFamily] or None, default: None
        Families to generate; all if omitted.
    """
    store = ParquetStore(DataLayout(Settings().tsfm_data_dir))
    builder = CorpusBuilder(store, SYNTHETIC_SEEDS[split] if seed is None else seed)
    tasks = builder.build(split, _DEFAULT_TASKS[split] if n_tasks is None else n_tasks, family)
    logger.info("corpus_written", split=split, tasks=tasks, root=str(store.layout.root))


@app.command()
def families() -> None:
    """List the generator families and their sampling weights in the training split."""
    for name, weight in SYNTHETIC_FAMILY_WEIGHTS.items():
        typer.echo(f"{name.value:<16}{weight:g}")
