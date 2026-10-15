"""ModelRouter: model names -> backend instances.

resolve() maps the request's `model` field to a backend: an explicit name, an alias
(kev-latest / jev-latest -> the default backend), the default fallback, or 404. The names in
`backends` are the served cards; a future remote backend (an HttpProxy implementing the same
Backend interface) joins the same table and needs no router change.
"""
from __future__ import annotations

from fastapi import HTTPException

from .backends.base import Backend


class ModelRouter:
    def __init__(self, backends: dict[str, Backend], default: str):
        self.backends = backends
        self.default = default

    def resolve(self, model: str) -> Backend:
        if model in self.backends:
            return self.backends[model]
        if model in ("kev-latest", "jev-latest") and self.default in self.backends:
            return self.backends[self.default]
        if self.default and self.default in self.backends:
            return self.backends[self.default]
        raise HTTPException(404, f"model {model!r} not found; served: {', '.join(self.backends) or 'none'}")

    def cards(self, settings) -> list[dict]:
        """One TypeSafe model card per explicit name, plus the alias names on the default backend."""
        cards = []
        for name, b in self.backends.items():
            cards.append({"name": name, "release_date": b.release_date, "kind": b.kind,
                          "run": getattr(b, "requested", None), "device": getattr(b, "device", None), **b.model_card()})
        default_aliases = [a for a in settings.alias_models
                           if a not in self.backends and self.default in self.backends]
        if default_aliases:
            b = self.backends[self.default]
            cards += [{**next(c for c in cards if c["name"] == self.default), "name": a} for a in default_aliases]
        return cards
