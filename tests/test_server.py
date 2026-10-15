"""HTTP tests with a fake backend (no GPU, no checkpoint): /v1/systemone routes, box -> 501, auth,
images gate, multi-model routing, cards. Assembled through create_app + ModelRouter like the real CLI."""
import base64, io
import pytest
from fastapi.testclient import TestClient

from kev_serve.config import Settings
from kev_serve.engine import SyncLockEngine
from kev_serve.http_app import create_app
from kev_serve.router import ModelRouter


class FakeBackend:
    kind = "clef"
    supported_primitives = frozenset({"noul", "choice", "score"})
    release_date = "2026-01-01"
    requested = "fake"
    device = "cpu"
    seen = []

    def __init__(self):
        self.engine = SyncLockEngine(self)

    def answer(self, d):
        FakeBackend.seen.append(d)
        return {"model": d.get("model", "jev-latest"),
                "answers": {"n": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 10, "output_tokens": 1}}

    def model_card(self):
        return {"description": "fake", "supported_primitives": sorted(self.supported_primitives)}

    def close(self):
        pass


@pytest.fixture()
def client():
    FakeBackend.seen = []
    b = FakeBackend()
    settings = Settings()
    return TestClient(create_app(settings, ModelRouter({"main": b}, "main")))


def test_systemone_ok(client):
    r = client.post("/v1/systemone", json={
        "state": "hello", "questions": {"n": {"type": "noul", "instructions": "yes?"}}})
    assert r.status_code == 200
    assert r.json()["answers"]["n"]["noul"] == 0.9
    assert "x-typesafe-request-id" in r.headers


def test_box_returns_501(client):
    # without the images header a box question is gated first (422, like TypeSafe); with it, no
    # backend is registered for box yet -> 501
    r = client.post("/v1/systemone", json={
        "state": "s", "questions": {"b": {"type": "box", "instructions": "the mug in `photo`"}}})
    assert r.status_code == 422
    r = client.post("/v1/systemone", json={
        "state": "s", "questions": {"b": {"type": "box", "instructions": "the mug in `photo`"}}},
        headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert r.status_code == 501
    assert "box" in r.json()["detail"]


def test_auth(client, monkeypatch):
    # auth is on only when Settings.api_key is set; rebuild the app with one
    settings = Settings(api_key="k1")
    app = create_app(settings, ModelRouter({"main": FakeBackend()}, "main"))
    c = TestClient(app)
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"n": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 401
    assert c.get("/v1/models").status_code == 401


def test_models_card(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert names[0] == "main"
    assert set(names[1:]) == {"kev-latest", "jev-latest"}   # aliases onto the default backend
    assert r.json()["models"][0]["supported_primitives"] == ["choice", "noul", "score"]


def _png_b64() -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_assets_require_images_header(client):
    body = {"state": {"photo": "ref1"}, "assets": {"ref1": {"type": "image", "format": "png", "data_base64": _png_b64()}},
            "questions": {"n": {"type": "noul", "instructions": "is the photo red?"}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422 and "typesafe-danger-zone-images" in r.json()["detail"]
    r = client.post("/v1/systemone", json=body, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert r.status_code == 200
    sent = FakeBackend.seen[-1]
    assert sent["images"] and sent["images"][0].size == (4, 4)   # asset decoded to a PIL image


class FakeGuiActorBackend(FakeBackend):
    """ground-only backend, like the real gui_actor one."""
    kind = "gui_actor"
    supported_primitives = frozenset({"ground"})

    def answer(self, d):
        FakeBackend.seen.append(d)
        return {"model": d.get("model", "jev-latest"),
                "answers": {"g": {"type": "ground", "point": [512, 300], "confidence": 0.7,
                                  "topk_points": [[512, 300], [100, 80]]}},
                "usage": {"input_tokens": 99, "output_tokens": 0}}


def test_gui_actor_ground_roundtrip():
    b = FakeGuiActorBackend()
    app = create_app(Settings(), ModelRouter({"grounder": b}, "grounder"))
    c = TestClient(app)
    body = {"state": "s", "questions": {"g": {"type": "ground", "instructions": "the settings icon"}},
            "images": [_png_b64()]}
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 422                       # images gate: no danger-zone header
    r = c.post("/v1/systemone", json=body, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert r.status_code == 200
    a = r.json()["answers"]["g"]
    assert a["point"] == [512, 300] and a["confidence"] == 0.7


def test_gui_actor_rejects_text_primitives():
    b = FakeGuiActorBackend()
    app = create_app(Settings(), ModelRouter({"grounder": b}, "grounder"))
    c = TestClient(app)
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"n": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 501 and "noul" in r.json()["detail"]


def test_clef_rejects_ground(client):
    # the fake clef backend (noul/choice/score) 501s a ground question
    r = client.post("/v1/systemone",
                    json={"state": "s", "questions": {"g": {"type": "ground", "instructions": "x"}}},
                    headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert r.status_code == 501 and "ground" in r.json()["detail"]
