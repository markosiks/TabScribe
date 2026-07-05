from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .protocol import SchedulerProfile


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yml"


class AppMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Chrome Tab Transcription Backend"
    version: str = "0.1.0"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)


class PersistenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_audio: bool = False


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppMetadata = Field(default_factory=AppMetadata)
    mode: str = "local-first"
    default_profile: SchedulerProfile = SchedulerProfile.balanced
    server: ServerConfig = Field(default_factory=ServerConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)


def load_settings(config_path: Path | str | None = None) -> Settings:
    selected_path = Path(
        os.getenv("CTTS_CONFIG_FILE") or config_path or DEFAULT_CONFIG_PATH
    )
    raw_config = _load_yaml_mapping(selected_path)
    return Settings.model_validate(_with_environment_overrides(raw_config))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")

    return loaded


def _with_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)

    env_overrides: dict[str, tuple[tuple[str, ...], Callable[[str], Any]]] = {
        "CTTS_HOST": (("server", "host"), str),
        "CTTS_PORT": (("server", "port"), int),
        "CTTS_DEFAULT_PROFILE": (("default_profile",), str),
        "CTTS_MODE": (("mode",), str),
        "CTTS_RAW_AUDIO_PERSISTENCE": (("persistence", "raw_audio"), _parse_bool),
        "CTTS_APP_VERSION": (("app", "version"), str),
    }

    for env_name, (path, parser) in env_overrides.items():
        value = os.getenv(env_name)
        if value is None:
            continue
        _set_nested(updated, path, parser(value))

    return updated


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        next_value = current.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override non-mapping configuration key: {key}")
        current = next_value
    current[path[-1]] = value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")
