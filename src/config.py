from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


class ConfigError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        api_key = (
            os.getenv("ACTIONPILOT_API_KEY", "").strip()
            or os.getenv("DEEPSEEK_API_KEY", "").strip()
        )
        if not api_key:
            raise ConfigError("Missing ACTIONPILOT_API_KEY or DEEPSEEK_API_KEY environment variable.")

        timeout_raw = os.getenv("ACTIONPILOT_TIMEOUT_SECONDS", "60").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ConfigError("ACTIONPILOT_TIMEOUT_SECONDS must be a number.") from exc

        if timeout_seconds <= 0:
            raise ConfigError("ACTIONPILOT_TIMEOUT_SECONDS must be greater than 0.")

        return cls(
            api_key=api_key,
            base_url=os.getenv("ACTIONPILOT_BASE_URL", DEFAULT_BASE_URL).strip(),
            model=os.getenv("ACTIONPILOT_MODEL", DEFAULT_MODEL).strip(),
            timeout_seconds=timeout_seconds,
        )
