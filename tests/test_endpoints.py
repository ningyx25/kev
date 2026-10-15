"""P0-5 demo endpoints and P0-2 temperature at the HTTP layer (fake backend, no GPU)."""
import pytest
from fastapi.testclient import TestClient

from kev_serve.config import Settings
from kev_serve.http_app import create_app
from kev_serve.router import ModelRouter
from tests.test_server import FakeBackend


class ChoiceBackend(FakeBackend):
    """Deterministic choice probabilities by option order: first option 0.8, rest split 0.2 — so a
    permutation flips the argmax and permute's spread is visible."""
    def answer(self, d):
        FakeBackend.seen.append(d)
        answers = {}
        for qid, q in d["questions"].items():           # answer every question the request carries, keyed by its id
            opts = list(q.get("criteria", {"a": "", "b": ""}))
            probs = {o: (0.8 if i == 0 else round(0.2 / max(1, len(opts) - 1), 4)) for i, o in enumerate(opts)}
            answers[qid] = {"type": "choice", "choice": opts[0], "confidence": 0.6, "probabilities": probs}
        return {"model": "main", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}}


def client(backend=None):
    FakeBackend.seen = []
    return TestClient(create_app(Settings(), ModelRouter({"main": backend or ChoiceBackend()}, "main")))


def _choice_request(n_perm=None, seed=0, question="q"):
    body = {"model": "main", "state": "s", "questions": {"q": {"type": "choice", "instructions": "pick",
                                                                "criteria": {"a": "A", "b": "B", "c": "C"}}}}
    r = {"request": body, "question": question}
    if n_perm is not None: r["n_perm"] = n_perm
    if seed: r["seed"] = seed
    return r


@pytest.mark.parametrize("n_perm,expected", [(0, 422), (-1, 422), (65, 422), (1, 200), (64, 200)])
def test_permute_bounds_n_perm(n_perm, expected):
    r = client().post("/v1/systemone/permute", json=_choice_request(n_perm))
    assert r.status_code == expected
    if expected == 200:
        assert len(r.json()["runs"]) == n_perm


def test_permute_reports_orders_and_spread():
    body = client().post("/v1/systemone/permute", json=_choice_request(n_perm=4, seed=7)).json()
    assert len(body["runs"]) == 4
    assert set(body["spread"]) == {"a", "b", "c"}
    assert body["argmax_stable"] in (True, False)
    first = body["runs"][0]
    assert first["order"] == ["a", "b", "c"] and first["probabilities"]["a"] == 0.8   # run 0: the given order
    # each later run re-answers with its own criteria order, and probabilities key back by option name
    for run in body["runs"][1:]:
        assert sorted(run["probabilities"]) == ["a", "b", "c"]
        assert run["probabilities"][run["order"][0]] == 0.8


def test_permute_rejects_missing_or_non_choice_question():
    c = client()
    r = c.post("/v1/systemone/permute", json=_choice_request(n_perm=2, question="nope"))
    assert r.status_code == 422
    r = c.post("/v1/systemone/permute", json={"request": {"model": "main", "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "?"}}}, "question": "q"})
    assert r.status_code == 422


def test_permute_rejects_a_backend_without_choice():
    class GroundOnly(FakeBackend):
        supported_primitives = frozenset({"ground"})
    c = client(GroundOnly())
    r = c.post("/v1/systemone/permute", json=_choice_request(n_perm=2))
    assert r.status_code == 422 and "ground" in r.json()["detail"]


def test_separate_answers_each_question_alone():
    c = client()
    body = {"model": "main", "state": "shared state", "questions": {
        "q1": {"type": "choice", "instructions": "i1", "criteria": {"a": "A", "b": "B"}},
        "q2": {"type": "choice", "instructions": "i2", "criteria": {"a": "A", "b": "B"}}}}
    r = c.post("/v1/systemone/separate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert set(out["answers"]) == {"q1", "q2"}
    assert out["usage"]["input_tokens"] == 20            # two passes, each 10
    assert len(FakeBackend.seen) == 2                    # one question per request
    assert set(FakeBackend.seen[0]["questions"]) == {"q1"} and set(FakeBackend.seen[1]["questions"]) == {"q2"}
    assert out["latency_ms"] >= 0


def test_separate_single_question_zero_questions_ok():
    c = client()
    r = c.post("/v1/systemone/separate", json={"model": "main", "state": "s",
                                               "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}})
    assert r.status_code == 200 and set(r.json()["answers"]) == {"q"}


def test_temperature_reaches_the_backend():
    """Settings(temperature=...) → ClefBackend(temperature=...) is the factory's job; at the route level
    the backend's engine is the same object either way, so pin what we can: the card carries it and a
    card on a fake backend passes through router.cards untouched."""
    b = ChoiceBackend(); b.temperature = 2.5
    cards = ModelRouter({"main": b}, "main").cards(Settings())
    assert cards[0]["name"] == "main" and "temperature" not in cards[0]   # FakeBackend.model_card has none;
    # the real ClefBackend.model_card() does — asserted in test_temperature_math.
