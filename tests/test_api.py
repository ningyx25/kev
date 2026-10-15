"""Schema tests: the four primitives render / answer in TypeSafe's shapes. No GPU, no checkpoint."""
import math
import pytest
from pydantic import ValidationError

from kev_serve.api import (Box, BoxAnswer, Choice, Noul, Score, SystemOneRequest,
                           choice_confidence, score_confidence, to_answers, to_record)


def test_question_union_accepts_all_four():
    req = SystemOneRequest(state="s", questions={
        "n": {"type": "noul", "instructions": "is it?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        "b": {"type": "box", "instructions": "the mug in `photo`"},
    })
    assert isinstance(req.questions["b"], Box)
    assert req.questions["n"].type == "noul"


def test_box_has_no_criteria_and_needs_instructions():
    assert Box(type="box", instructions="x")
    with pytest.raises(ValidationError):
        Box(type="box", instructions="  ")


def test_box_answer_edge_constraints():
    ok = BoxAnswer(box={"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}, confidence=0.92)
    assert ok.box["left"] < ok.box["right"]
    with pytest.raises(ValidationError):   # edges out of range
        BoxAnswer(box={"top": -0.1, "bottom": 0.5, "left": 0.1, "right": 0.5}, confidence=0.5)
    with pytest.raises(ValidationError):   # left >= right
        BoxAnswer(box={"top": 0.1, "bottom": 0.5, "left": 0.5, "right": 0.5}, confidence=0.5)
    with pytest.raises(ValidationError):   # missing edge
        BoxAnswer(box={"top": 0.1, "bottom": 0.5, "left": 0.1}, confidence=0.5)


def test_to_record_rejects_box():
    req = SystemOneRequest(state="s", questions={"b": {"type": "box", "instructions": "x"}})
    with pytest.raises(ValueError, match="box"):
        to_record(req)


def test_to_answers_shapes():
    meta = [
        {"id": "n", "type": "noul", "keys": ["false", "true"]},
        {"id": "c", "type": "choice", "keys": ["a", "b"]},
        {"id": "s", "type": "score", "keys": ["0", "1"], "legend": {"0": "low", "1": "high"}},
    ]
    out = to_answers([[0.25, 0.75], [0.9, 0.1], [0.8, 0.2]], meta)
    assert out["n"] == {"type": "noul", "noul": 0.75}
    assert out["c"]["choice"] == "a"
    assert out["c"]["confidence"] == choice_confidence([0.9, 0.1])
    assert math.isclose(out["s"]["score"], 0.2)
    assert out["s"]["legend"] == {"0": "low", "1": "high"}


def test_confidence_extremes():
    assert choice_confidence([1.0, 0.0]) == 1.0
    assert choice_confidence([0.5, 0.5]) == 0.0
    assert score_confidence([1.0, 0.0]) == 1.0
    assert score_confidence([0.5, 0.5]) == 0.0
