"""Settings: one place for the env-tunable knobs (KEV_*)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


@dataclass
class Settings:
    api_key: str | None = None            # KEV_API_KEY (unset = open server)
    date_facts: bool = False              # KEV_DATE_FACTS ("1")
    default_model: str = "jev-latest"     # KEV_DEFAULT_MODEL
    alias_models: tuple = ("kev-latest", "jev-latest")
    models: dict = field(default_factory=dict)   # filled by the CLI, not env

    # serving limits (admission.py)
    max_state_tokens: int | None = None   # KEV_MAX_STATE_TOKENS; None = derive from the backbone, else 65536
    truncate_states: bool = False         # KEV_TRUNCATE_STATES ("1"): read the first max_state_tokens instead of a 422

    # sampling / calibration
    temperature: float | None = None      # KEV_TEMPERATURE; None = the checkpoint's fitted value

    # kernels
    fused: bool | None = None             # KEV_FUSED: "1"/"0" force, unset = auto (use fused kernels when available)

    # batching / caching (engine)
    max_batch: int = 64                   # KEV_MAX_BATCH
    prefix_cache: int = 4                 # KEV_PREFIX_CACHE: states kept (0 disables)
    prefix_min_tokens: int | None = None  # KEV_PREFIX_MIN_TOKENS; None = the backend's prefix_min_tokens
    prefix_max_tokens: int = 65536        # KEV_PREFIX_MAX_TOKENS: state tokens the cache holds in all

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env = {
            "api_key": os.environ.get("KEV_API_KEY") or None,
            "date_facts": _env_flag("KEV_DATE_FACTS"),
            "default_model": os.environ.get("KEV_DEFAULT_MODEL", "jev-latest"),
            "max_state_tokens": int(os.environ["KEV_MAX_STATE_TOKENS"]) if os.environ.get("KEV_MAX_STATE_TOKENS") else None,
            "truncate_states": _env_flag("KEV_TRUNCATE_STATES"),
            "temperature": float(os.environ["KEV_TEMPERATURE"]) if os.environ.get("KEV_TEMPERATURE") else None,
            "fused": (os.environ["KEV_FUSED"] == "1") if os.environ.get("KEV_FUSED") in ("0", "1") else None,
            "max_batch": int(os.environ.get("KEV_MAX_BATCH", "64")),
            "prefix_cache": int(os.environ.get("KEV_PREFIX_CACHE", "4")),
            "prefix_min_tokens": int(os.environ["KEV_PREFIX_MIN_TOKENS"]) if os.environ.get("KEV_PREFIX_MIN_TOKENS") else None,
            "prefix_max_tokens": int(os.environ.get("KEV_PREFIX_MAX_TOKENS", "65536")),
        }
        env.update(overrides)
        return cls(**env)
