import pytest
from typer.testing import CliRunner

from tsfm.cli import pull
from tsfm.cli.app import app
from tsfm.data.storage import Catalog, DataLayout
from tsfm.enums import Dataset, GeneratorFamily, PullTarget
from tsfm.settings import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Point the CLI at a temporary store and away from the developer's `.env`."""
    monkeypatch.setenv("TSFM_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        if name != "tsfm_data_dir":
            monkeypatch.delenv(name.upper(), raising=False)
    return tmp_path


@pytest.mark.parametrize("command", [["--help"], ["data", "--help"], ["data", "pull", "--help"], ["synth", "--help"]])
def test_help_renders(command):
    """Every command group must import and build its options without touching the network."""
    assert runner.invoke(app, command).exit_code == 0


def test_pull_without_credentials_names_the_missing_key():
    """The first run happens before `.env` is filled; the error must say which key to set."""
    result = runner.invoke(app, ["data", "pull", "fred"])
    assert result.exit_code != 0
    assert "FRED_API_KEY" in str(result.exception)


def test_pull_all_skips_sources_without_credentials(data_dir, monkeypatch):
    """Most keys are optional; a missing one must not fail the run or stop the sources that need none."""
    monkeypatch.setattr(pull, "PullTarget", [PullTarget.FRED, PullTarget.CENTRAL_BANKS])
    assert runner.invoke(app, ["data", "pull-all"]).exit_code == 0
    with Catalog(DataLayout(data_dir)) as catalog:
        assert catalog.summary(Dataset.EVENTS).num_rows == 1


def test_central_bank_pull_needs_no_network(data_dir):
    """The meeting calendar ships with the package, so this pull works offline end to end."""
    assert runner.invoke(app, ["data", "pull", "central-banks"]).exit_code == 0
    with Catalog(DataLayout(data_dir)) as catalog:
        assert catalog.summary(Dataset.EVENTS).num_rows == 1


def test_synth_generate_writes_a_readable_corpus(data_dir):
    """`make synth-val` must leave shards that the catalogue can count."""
    result = runner.invoke(
        app, ["synth", "generate", "val", "--n-tasks", "2", "--family", "garch", "--family", "triangle"]
    )
    assert result.exit_code == 0, result.output
    with Catalog(DataLayout(data_dir)) as catalog:
        summary = catalog.summary(Dataset.SYNTHETIC).to_pylist()
    assert {(row["family"], row["tasks"]) for row in summary} == {("garch", 2), ("triangle", 2)}


def test_catalog_runs_on_an_empty_store():
    """Inspecting the store before the first pull is a normal first action."""
    result = runner.invoke(app, ["data", "catalog"])
    assert result.exit_code == 0
    assert all(dataset.value in result.output for dataset in Dataset)


def test_families_lists_every_family():
    """The listing is how a user discovers valid `--family` values."""
    output = runner.invoke(app, ["synth", "families"]).output
    assert all(family.value in output for family in GeneratorFamily)
