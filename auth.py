"""Small helpers for reading settings and checking login credentials."""

import hmac
import os
import tomllib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"


def get_setting(setting_name, default_value=""):
    """Read a setting from the environment or the private local secrets file."""
    value = os.getenv(setting_name, "").strip()
    if value:
        return value

    if LOCAL_SECRETS_PATH.exists():
        with LOCAL_SECRETS_PATH.open("rb") as secrets_file:
            local_settings = tomllib.load(secrets_file)
        value = str(local_settings.get(setting_name, "")).strip()
        if value:
            return value

    return default_value


def credentials_match(entered_username, entered_password, saved_username, saved_password):
    """Return True when the entered username and password both match."""
    username_matches = hmac.compare_digest(
        entered_username.encode("utf-8"), saved_username.encode("utf-8")
    )
    password_matches = hmac.compare_digest(
        entered_password.encode("utf-8"), saved_password.encode("utf-8")
    )
    return username_matches and password_matches


def api_key_matches(entered_key, saved_key):
    """Check that an API request contains the configured shared key."""
    return hmac.compare_digest(entered_key.encode("utf-8"), saved_key.encode("utf-8"))
