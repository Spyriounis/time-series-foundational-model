"""Commands that pull real data and inspect the store."""

from datetime import datetime
from typing import Annotated

import typer

from tsfm.data.fetchers.base import PullError
from tsfm.data.fetchers.registry import build_fetchers
from tsfm.data.storage import Catalog, DataLayout, ParquetStore
from tsfm.enums import Dataset, Granularity, PullTarget
from tsfm.log import get_logger
from tsfm.settings import MissingSecretError, Settings

logger = get_logger(__name__)
app = typer.Typer(no_args_is_help=True, help="Pull real data into the local store.")

Start = Annotated[datetime | None, typer.Option(help="Lower bound, UTC. Default: start of history.")]
End = Annotated[datetime | None, typer.Option(help="Upper bound, UTC. Default: now.")]
Only = Annotated[list[str] | None, typer.Option(help="Restrict to these instruments or series. Repeatable.")]
Granularities = Annotated[list[Granularity] | None, typer.Option("--granularity", help="Bar sizes. Repeatable.")]


@app.command()
def pull(
    target: PullTarget, start: Start = None, end: End = None, only: Only = None, granularity: Granularities = None
) -> None:
    """
    Pull one target, resuming from what is already stored.

    Parameters
    ----------
    target : PullTarget
        Source to pull.
    start : datetime or None, default: None
        Lower bound.
    end : datetime or None, default: None
        Upper bound.
    only : list[str] or None, default: None
        Instruments or series to keep.
    granularity : list[Granularity] or None, default: None
        Bar sizes; all if omitted.

    Raises
    ------
    typer.Exit
        With code 1 if any table failed.
    """
    settings = Settings()
    if not _pull_target(target, settings, _store(settings), start, end, only, granularity):
        raise typer.Exit(code=1)


@app.command()
def pull_all(start: Start = None, end: End = None, granularity: Granularities = None) -> None:
    """
    Pull every target, skipping those without credentials and continuing past failures.

    Parameters
    ----------
    start : datetime or None, default: None
        Lower bound.
    end : datetime or None, default: None
        Upper bound.
    granularity : list[Granularity] or None, default: None
        Bar sizes; all if omitted.

    Raises
    ------
    typer.Exit
        With code 1 if any table failed.
    """
    settings = Settings()
    store = _store(settings)
    incomplete: list[str] = []
    for target in PullTarget:
        try:
            if not _pull_target(target, settings, store, start, end, None, granularity):
                incomplete.append(target.value)
        except MissingSecretError as error:
            logger.warning("target_skipped", target=target, reason=str(error))
    if incomplete:
        logger.error("pull_incomplete", targets=incomplete)
        raise typer.Exit(code=1)


@app.command()
def catalog(
    dataset: Annotated[Dataset | None, typer.Argument(help="Dataset to summarise; all if omitted.")] = None,
) -> None:
    """
    Print row counts and time spans of what is stored.

    Parameters
    ----------
    dataset : Dataset or None, default: None
        Dataset to summarise; all if omitted.
    """
    with Catalog(DataLayout(Settings().tsfm_data_dir)) as reader:
        for item in [dataset] if dataset else list(Dataset):
            summary = reader.summary(item)
            typer.echo(f"\n# {item.value}: {summary.num_rows} tables")
            if summary.num_rows:
                typer.echo(summary.to_pandas().to_string(index=False))


def _store(settings: Settings) -> ParquetStore:
    """
    Open the store at the configured data directory.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    ParquetStore
        Store rooted at ``settings.tsfm_data_dir``.
    """
    return ParquetStore(DataLayout(settings.tsfm_data_dir))


def _pull_target(
    target: PullTarget,
    settings: Settings,
    store: ParquetStore,
    start: datetime | None,
    end: datetime | None,
    only: list[str] | None,
    granularities: list[Granularity] | None,
) -> bool:
    """
    Run every fetcher of one target, carrying on past one that fails.

    Parameters
    ----------
    target : PullTarget
        Source to pull.
    settings : Settings
        Runtime settings.
    store : ParquetStore
        Destination.
    start : datetime or None
        Lower bound.
    end : datetime or None
        Upper bound.
    only : list[str] or None
        Instruments or series to keep.
    granularities : list[Granularity] or None
        Bar sizes; all if None.

    Returns
    -------
    bool
        True if every table was pulled.
    """
    fetchers = build_fetchers(target, settings, granularities) if granularities else build_fetchers(target, settings)
    rows, complete = 0, True
    for fetcher in fetchers:
        try:
            rows += fetcher.pull(store, start, end, only)
        except MissingSecretError:
            raise
        except PullError as error:
            rows, complete = rows + error.rows, False
        except Exception as error:  # the other fetchers of the target must still run
            logger.error("fetcher_failed", source=fetcher.source, error=f"{type(error).__name__}: {error}")
            complete = False
    logger.info("target_pulled", target=target, rows=rows, complete=complete)
    return complete
