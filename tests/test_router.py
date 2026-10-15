"""ModelRouter tests: explicit names, aliases, default fallback, 404, aggregated cards."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kev_serve.config import Settings
from kev_serve.http_app import create_app
from kev_serve.router import ModelRouter
from tests.test_server import FakeBackend


@pytest.fixture()
def two():
    a, b = FakeBackend(), FakeBackend()
    a.requested, b.requested = "ckpt-a", "ckpt-b"
    return a, b


def test_resolve_explicit_alias_default(two):
    a, b = two
    r = ModelRouter({"a": a, "b": b}, default="a")
    assert r.resolve("b") is b
    assert r.resolve("a") is a
    assert r.resolve("jev-latest") is a          # alias -> default
    assert r.resolve("unknown-model") is a       # fallback -> default


def test_resolve_404_without_default(two):
    a, b = two
    r = ModelRouter({"a": a}, default="")
    with pytest.raises(HTTPException) as e:
        r.resolve("nope")
    assert e.value.status_code == 404


def test_route_by_model_field(two):
    a, b = two
    app = create_app(Settings(), ModelRouter({"a": a, "b": b}, "a"))
    c = TestClient(app)
    body = {"questions": {"n": {"type": "noul", "instructions": "?"}}}
    assert c.post("/v1/systemone", json={**body, "model": "b", "state": "via-b"}).status_code == 200
    assert c.post("/v1/systemone", json={**body, "model": "a", "state": "via-a"}).status_code == 200
    assert [d["state"] for d in FakeBackend.seen[-2:]] == ["via-b", "via-a"]   # each hit its backend


def test_cards_aggregate(two):
    a, b = two
    app = create_app(Settings(), ModelRouter({"a": a, "b": b}, "b"))
    cards = TestClient(app).get("/v1/models").json()["models"]
    names = [c["name"] for c in cards]
    assert {"a", "b", "kev-latest", "jev-latest"} <= set(names)
    alias = next(c for c in cards if c["name"] == "jev-latest")
    explicit = next(c for c in cards if c["name"] == "b")
    assert alias["run"] == explicit["run"]        # aliases carry the default backend's card
