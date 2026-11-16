"""Root of the ``tsfm`` command."""

import logging
from typing import Annotated

import typer

from tsfm.cli.pull import app as data_app
from tsfm.cli.synth import app as synth_app
from tsfm.log import configure_logging

app = typer.Typer(no_args_is_help=True, help="Time-series foundation model tooling.")
app.add_typer(data_app, name="data")
app.add_typer(synth_app, name="synth")


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every stored page.")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Emit JSON lines.")] = False,
) -> None:
    """
    Configure logging before any command runs.

    Parameters
    ----------
    verbose : bool, default: False
        Lower the level to debug.
    json_logs : bool, default: False
        Emit JSON lines instead of the console format.
    """
    configure_logging(logging.DEBUG if verbose else logging.INFO, json=json_logs)
