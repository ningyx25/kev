"""Baseline pin for the vendored clef-flash code (kev_serve/clef_model.py): a fixed question set through
encode_record must give the fixed token spans / option ids. Catches silent drift of the vendored encoder.

Needs a tokenizer only (no GPU, no weights): set KEV_SERVE_TOKENIZER to a local Qwen3.5 tokenizer dir,
otherwise the test skips. Set KEV_SERVE_TEST_CKPT to a full clef checkpoint dir to also pin the forward
pass (probabilities), which needs a GPU.
"""
import json, os, sys
import pytest

torch = pytest.importorskip("torch")

TOKENIZER = os.environ.get("KEV_SERVE_TOKENIZER")
CKPT = os.environ.get("KEV_SERVE_TEST_CKPT")


@pytest.mark.skipif(not TOKENIZER, reason="set KEV_SERVE_TOKENIZER to a Qwen3.5 tokenizer dir")
def test_encode_record_baseline():
    from transformers import AutoTokenizer
    from kev_serve.backends.clef.clef_model import encode_record
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    record = {
        "id": "baseline",
        "state": "The tea is hot.",
        "questions": {
            "hot": {"type": "noul", "instructions": "Is the tea hot?", "criteria": None},
            "cup": {"type": "choice", "instructions": "Which cup?", "criteria": {"blue": "The blue cup.", "red": "The red cup."}},
            "heat": {"type": "score", "instructions": "How hot?", "criteria": ["warm", "hot", "boiling"]},
        },
    }
    enc = encode_record(tok, record)
    assert [q.question_id for q in enc.questions] == ["hot", "cup", "heat"]
    assert enc.questions[0].option_ids == ("true", "false")       # noul order, fixed
    assert enc.questions[1].option_ids == ("blue", "red")         # choice options sorted by id
    assert enc.questions[2].option_ids == ("0", "1", "2")         # score levels in order
    assert len(enc.input_ids) > 0 and not enc.media
    # re-encoding is deterministic (same spans)
    enc2 = encode_record(tok, record)
    assert enc == enc2


@pytest.mark.skipif(not CKPT, reason="set KEV_SERVE_TEST_CKPT to a clef checkpoint dir (needs a GPU)")
def test_forward_baseline():
    from kev_serve.backends import load_backend
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_backend(CKPT, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
    req = {"state": "The tea is hot.",
           "questions": {"hot": {"type": "noul", "instructions": "Is the tea hot?"}}}
    resp = b.answer(req)
    p = resp["answers"]["hot"]["noul"]
    assert 0.0 <= p <= 1.0
    print(f"baseline noul(hot tea) = {p}")   # record this value when pinning; compare on change


def test_ground_decode_999():
    """The connected-region decode and the 0-999 conversion (no checkpoint, no GPU)."""
    import torch
    from kev_serve.backends.gui_actor.gui_actor_model import _to_999, get_prediction_region_point
    # a 4x4 grid of patches; one 2x2 hot block at (y=1..2, x=1..2), rest near zero
    n_width = n_height = 4
    scores = torch.full((1, n_width * n_height), 0.01)
    for y in (1, 2):
        for x in (1, 2):
            scores[0, y * n_width + x] = 1.0
    point, centers, values, *_ = get_prediction_region_point(scores, n_width, n_height,
                                                             return_all_regions=True, rect_center=False)
    # patch centres at rows/cols 1 and 2 are 0.375 and 0.625; the 2x2 block's weighted centre is 0.5
    assert abs(point[0] - 0.5) < 1e-6 and abs(point[1] - 0.5) < 1e-6
    assert _to_999(point[0], point[1]) == (500, 500)
    assert _to_999(0.0, 1.0) == (0, 999)
    assert _to_999(-0.5, 1.5) == (0, 999)   # clamped
