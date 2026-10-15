"""The SDK's multipart body: a `request` JSON part plus one part per image, named by the asset's `part`.

Regression: this shape used to 500 — the assets schema demanded data_base64, and a failed validation
whose `input` held raw image bytes crashed FastAPI's error encoder (bytes are not UTF-8)."""
import base64, io, json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from kev_serve.config import Settings
from kev_serve.http_app import create_app
from kev_serve.router import ModelRouter
from tests.test_server import FakeBackend, _png_b64


def _png(color="red", size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _multipart(body: dict, parts: dict[str, bytes], boundary="----t"):
    """The TypeSafe SDK's encoding: the JSON body as the `request` part, then one part per image."""
    chunks = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
              f"Content-Type: application/json\r\n\r\n".encode() + json.dumps(body).encode() + b"\r\n"]
    for name, data in parts.items():
        chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{name}\"\r\n"
                      f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), {"content-type": f"multipart/form-data; boundary={boundary}",
                              "TypeSafe-Danger-Zone-Images": "true"}


@pytest.fixture()
def client():
    FakeBackend.seen = []
    return TestClient(create_app(Settings(), ModelRouter({"main": FakeBackend()}, "main")))


def _body(assets, questions=None):
    return {"model": "main", "state": {"screen": "asset-0"}, "assets": assets,
            "questions": questions or {"q": {"type": "noul", "instructions": "Is there a button in `screen`?"}}}


def test_multipart_image_reaches_the_backend(client):
    content, headers = _multipart(_body({"asset-0": {"type": "image", "part": "asset-0"}}),
                                  {"asset-0": _png(size=(8, 8))})
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 200, r.text
    sent = FakeBackend.seen[-1]
    assert sent["images"][0].size == (8, 8)          # decoded from the part, not from base64


def test_multipart_without_the_images_header_is_gated(client):
    content, headers = _multipart(_body({"asset-0": {"type": "image", "part": "asset-0"}}), {"asset-0": _png()})
    headers.pop("TypeSafe-Danger-Zone-Images")
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 422 and "typesafe-danger-zone-images" in r.json()["detail"]


def test_missing_part_is_a_422_not_a_crash(client):
    content, headers = _multipart(_body({"asset-0": {"type": "image", "part": "asset-7"}}), {"asset-0": _png()})
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 422 and "asset-7" in r.json()["detail"]


def test_unclaimed_part_is_refused(client):
    content, headers = _multipart(_body({"asset-0": {"type": "image", "part": "asset-0"}}),
                                  {"asset-0": _png(), "stowaway": _png()})
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 422 and "stowaway" in r.json()["detail"]


def test_invalid_multipart_body_is_a_422_with_bytes_in_the_error(client):
    """The original 500: validation failed and the error carried the raw PNG bytes."""
    content, headers = _multipart({"model": "main", "questions": {}},   # no `state`
                                  {"asset-0": _png()})
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 422, r.text
    assert "state" in json.dumps(r.json())


def test_asset_needs_exactly_one_of_data_base64_or_part(client):
    content, headers = _multipart(_body({"asset-0": {"type": "image", "format": "png"}}), {"asset-0": _png()})
    r = client.post("/v1/systemone", content=content, headers=headers)
    assert r.status_code == 422


def test_json_body_with_data_base64_still_works(client):
    r = client.post("/v1/systemone", headers={"TypeSafe-Danger-Zone-Images": "true"},
                    json=_body({"asset-0": {"type": "image", "format": "png", "data_base64": _png_b64()}}))
    assert r.status_code == 200, r.text
    assert FakeBackend.seen[-1]["images"][0].size == (4, 4)


def test_json_asset_with_part_but_no_multipart_body(client):
    r = client.post("/v1/systemone", headers={"TypeSafe-Danger-Zone-Images": "true"},
                    json=_body({"asset-0": {"type": "image", "part": "asset-0"}}))
    assert r.status_code == 422 and "asset-0" in r.json()["detail"]
