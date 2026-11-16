from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tsfm.data.storage import DataLayout, ParquetStore
from tsfm.settings import Settings
from tsfm.utils.http import HttpClient

type Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    """A store rooted in a temporary directory."""
    return ParquetStore(DataLayout(tmp_path))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with dummy credentials, isolated from the developer's `.env`."""
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]  # a runtime-only pydantic-settings argument
        tsfm_data_dir=tmp_path,
        oanda_api_token="token",
        oanda_account_id="account",
        etoro_api_key="api-key",
        etoro_user_key="user-key",
        fred_api_key="fred-key",
        eia_api_key="eia-key",
    )


@pytest.fixture
def make_client() -> Callable[[Handler], HttpClient]:
    """Build an `HttpClient` that answers from a handler instead of the network, without sleeping."""

    def factory(handler: Handler) -> HttpClient:
        return HttpClient(requests_per_second=1e6, transport=httpx.MockTransport(handler), sleep=lambda _: None)

    return factory
