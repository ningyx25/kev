"""State admission (P0-1): 422 with the old wording, the truncate mode, question budget, opt-out."""
import pytest
from fastapi.testclient import TestClient

from kev_serve.admission import admit, max_state_tokens
from kev_serve.config import Settings
from kev_serve.http_app import create_app
from kev_serve.router import ModelRouter
from tests.test_server import FakeBackend


class _Enc:
    def __init__(self, ids): self.input_ids = ids


class StubTokenizer:
    """Counts words as tokens: enough to pin the admission math without weights. Mirrors the
    transformers call surface the admission uses: tok(text).input_ids and tok.decode(ids)."""
    vocab = {}
    def __call__(self, text, add_special_tokens=False):
        return _Enc(list(range(len(str(text).split()))))    # one id per word
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


class TokenizingBackend(FakeBackend):
    tokenizer = StubTokenizer()


def make(settings, backend=None):
    b = backend or TokenizingBackend()
    return TestClient(create_app(settings, ModelRouter({"main": b}, "main")))


def _req(state):
    return {"model": "main", "state": state, "questions": {"n": {"type": "noul", "instructions": "yes?"}}}


def test_refuses_an_over_length_state():
    c = make(Settings(max_state_tokens=8))
    r = c.post("/v1/systemone", json=_req("one two three four five six seven eight nine"))
    assert r.status_code == 422
    assert r.json()["detail"].startswith("state is 9 tokens, over the 8-token limit")
    assert "KEV_TRUNCATE_STATES=1" in r.json()["detail"]   # the fix is named


def test_admits_a_state_exactly_at_the_limit():
    c = make(Settings(max_state_tokens=8))
    r = c.post("/v1/systemone", json=_req(" ".join(map(str, range(8)))))
    assert r.status_code == 200


def test_truncates_when_opted_in_and_says_so():
    c = make(Settings(max_state_tokens=8, truncate_states=True))
    r = c.post("/v1/systemone", json=_req(" ".join(f"w{i}" for i in range(11))))
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is True
    assert body["usage"]["state_tokens"] == 11 and body["usage"]["state_tokens_used"] == 8
    sent = FakeBackend.seen[-1]
    assert len(sent["state"].split()) == 8 and sent["state"].split()[0] == "t0"   # the backend got the first 8


def test_no_marks_when_not_truncating():
    c = make(Settings(max_state_tokens=100))
    body = c.post("/v1/systemone", json=_req("short state")).json()
    assert "truncated" not in body and "state_tokens" not in body["usage"]


def test_refuses_a_long_question_either_way():
    for truncate in (False, True):
        c = make(Settings(max_state_tokens=64, truncate_states=truncate))
        long_crit = " ".join(f"c{i}" for i in range(9000))     # over the 8192-token question budget
        r = c.post("/v1/systemone", json={"model": "main", "state": "s",
                                          "questions": {"q": {"type": "noul", "instructions": long_crit}}})
        assert r.status_code == 422 and r.json()["detail"].startswith("branch too long")


def test_backend_without_a_tokenizer_opts_out():
    c = make(Settings(max_state_tokens=1), backend=FakeBackend())   # plain FakeBackend: no tokenizer attr
    r = c.post("/v1/systemone", json=_req("way way way over the limit"))
    assert r.status_code == 200


def test_limit_derives_from_settings_then_default():
    b = TokenizingBackend()
    assert max_state_tokens(b, Settings(max_state_tokens=1234)) == 1234
    assert max_state_tokens(b, Settings()) == 65536                 # no model config on the fake: the fallback
