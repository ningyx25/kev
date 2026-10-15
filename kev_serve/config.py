"""Settings: one place for the env-tunable knobs (KEV_*)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    api_key: str | None = None            # unset = open server; set = require Authorization: Bearer <key>
    date_facts: bool = False              # opt-in date preprocessing on every request's state (api.with_date_facts)
    default_model: str = "jev-latest"     # the model name unknown `model` fields fall back to ("" = 404 instead)
    alias_models: tuple = ("kev-latest", "jev-latest")   # names that all resolve to default_model's backend
    # name -> checkpoint path, filled by the CLI (--run name=path); a remote-upstream backend would slot in here too
    models: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env = {
            "api_key": os.environ.get("KEV_API_KEY") or None,
            "date_facts": os.environ.get("KEV_DATE_FACTS", "0") == "1",
            "default_model": os.environ.get("KEV_DEFAULT_MODEL", "jev-latest"),
        }
        env.update(overrides)
        return cls(**env)
