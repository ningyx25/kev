"""HTTP tests with a fake backend (no GPU, no checkpoint): /v1/systemone routes, box -> 501, auth, images header."""
import base64, io, threading
import pytest
from fastapi.testclient import TestClient

from kev_serve import serve as srv


class FakeBackend:
    kind = "clef"
    supported_primitives = frozenset({"noul", "choice", "score"})
    release_date = "2026-01-01"
    requested = "fake"
    device = "cpu"
    seen = []

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
    srv.app.state.server = FakeBackend()
    return TestClient(srv.app)


def test_systemone_ok(client):
    r = client.post("/v1/systemone", json={
        "state": "hello", "questions": {"n": {"type": "noul", "instructions": "yes?"}}})
    assert r.status_code == 200
    assert r.json()["answers"]["n"]["noul"] == 0.9
    assert "x-typesafe-request-id" in r.headers


def test_box_returns_501(client):
    r = client.post("/v1/systemone", json={
        "state": "s", "questions": {"b": {"type": "box", "instructions": "the mug in `photo`"}}})
    assert r.status_code == 501
    assert "box" in r.json()["detail"]


def test_auth(client, monkeypatch):
    monkeypatch.setattr(srv, "API_KEY", "k1")
    r = client.post("/v1/systemone", json={"state": "s", "questions": {"n": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_models_card(client):
    r = client.get("/v1/models", headers={"Authorization": "Bearer k1"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert names == ["kev-latest", "jev-latest"]
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
    assert r.status_code == 422 and srv.IMAGES_HEADER in r.json()["detail"]
    r = client.post("/v1/systemone", json=body, headers={"TypeSafe-Danger-Zone-Images": "true"})
    assert r.status_code == 200
    sent = FakeBackend.seen[-1]
    assert sent["images"] and sent["images"][0].size == (4, 4)   # asset decoded to a PIL image
