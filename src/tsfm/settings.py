"""Runtime settings read from the environment and ``.env``."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tsfm.constants import DEFAULT_DATA_DIR
from tsfm.enums import OandaEnvironment


class MissingSecretError(RuntimeError):
    """A source was used without its credential being set."""


class Settings(BaseSettings):
    """
    Data location and API credentials.

    Each field is read from the environment variable of the same name in upper case.

    Attributes
    ----------
    tsfm_data_dir : Path
        Root of the local store.
    oanda_api_token : SecretStr or None
        OANDA v20 personal access token.
    oanda_account_id : SecretStr or None
        OANDA account whose instrument list defines the universe.
    oanda_environment : OandaEnvironment
        Practice or live host.
    etoro_api_key : SecretStr or None
        eToro public API key.
    etoro_user_key : SecretStr or None
        eToro user key paired with the API key.
    fred_api_key : SecretStr or None
        FRED API key.
    eia_api_key : SecretStr or None
        EIA open-data API key.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tsfm_data_dir: Path = DEFAULT_DATA_DIR
    oanda_api_token: SecretStr | None = None
    oanda_account_id: SecretStr | None = None
    oanda_environment: OandaEnvironment = OandaEnvironment.PRACTICE
    etoro_api_key: SecretStr | None = None
    etoro_user_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    eia_api_key: SecretStr | None = None

    @field_validator("tsfm_data_dir")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        """
        Expand a leading ``~``, which ``.env`` files do not.

        Parameters
        ----------
        value : Path
            Configured data directory.

        Returns
        -------
        Path
            Path with the home directory expanded.
        """
        return value.expanduser()

    def secret(self, name: str) -> str:
        """
        Return a credential, failing fast if it is unset.

        Parameters
        ----------
        name : str
            Field name, e.g. ``"fred_api_key"``.

        Returns
        -------
        str
            The secret value.

        Raises
        ------
        MissingSecretError
            If the credential is unset or empty.
        """
        value: SecretStr | None = getattr(self, name)
        if value is None or not value.get_secret_value():
            raise MissingSecretError(f"Set {name.upper()} in the environment or .env")
        return value.get_secret_value()
