"""GUI-Actor serving: detection, the per-patch answer shape, and the /v1/systemone contract.

The local tests need no GPU, no checkpoint and no server (the CI list includes this file). The
server-marked tests need `python -m kev.serve --run microsoft/GUI-Actor-<...>` up with KEV_BASE_URL
pointing at it; they skip when the server there is not a GUI-Actor one, so `pytest tests` stays
green with any server (or none) up.

Run:
    uv run python -m pytest tests/test_gui_actor.py -q -m "not server"
    KEV_BASE_URL=http://127.0.0.1:8008 uv run python -m pytest tests/test_gui_actor.py -q
"""
import base64, io, json, os

import httpx
import pytest
import torch
from PIL import Image

from kev.gui_actor_model import (MODEL_TYPES, POINTER_CLASS_NAME_QWEN25VL, POINTER_TENSOR_PREFIX, gui_actor_model_type,
                                 is_gui_actor, patch_answer, register_qwen25vl_legacy_keys, systemone)

BASE = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8008")


def test_model_types_are_the_two_served_backbones():
    assert MODEL_TYPES == {"qwen2_5_vl", "qwen3_5"}


def test_gui_actor_model_type_rejects_other_backbones(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_vl"}), encoding="utf-8")
    with pytest.raises(ValueError, match="qwen2_5_vl"):
        gui_actor_model_type(str(tmp_path))


def test_is_gui_actor_reads_the_pointer_tensors(tmp_path):
    """The head in the weights is the signal -- a sharded index, a single safetensors file, or neither."""
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    (sharded / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.embed_tokens.weight": "a.safetensors",
        f"{POINTER_TENSOR_PREFIX}layer_norm.weight": "a.safetensors"}}), encoding="utf-8")
    assert is_gui_actor(sharded)

    single = tmp_path / "single"
    single.mkdir()
    from safetensors.torch import save_file
    save_file({f"{POINTER_TENSOR_PREFIX}layer_norm.weight": torch.zeros(2)}, str(single / "model.safetensors"))
    assert is_gui_actor(single)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    assert not is_gui_actor(plain)
    assert not is_gui_actor(tmp_path / "missing")


def test_patch_answer_is_a_choice_over_the_patch_grid():
    """12 patches, the mass on patch 6 (y=1, x=2 of a 4-wide grid): choice is its index and the point its centre."""
    scores = torch.full((1, 12), 1e-3)
    scores[0, 6] = 0.9
    a = patch_answer(scores, n_width=4, n_height=3, patch_pixels=32, topk=2)
    assert a["type"] == "choice" and a["choice"] == "6"
    assert list(a["probabilities"]) == [str(i) for i in range(12)]
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    assert a["n_width"] == 4 and a["n_height"] == 3 and a["patch_pixels"] == 32
    assert 0 <= a["confidence"] <= 1
    assert a["point"] == [0.625, 0.5]                     # ((2+0.5)/4, (1+0.5)/3)
    assert len(a["topk_points"]) <= 2 and len(a["topk_values"]) <= 2


def test_patch_probabilities_sum_within_tolerance_at_a_2550_patch_grid():
    """4-decimal rounding (kev.api.round_prob's default) drifts a 2,550-key sum by up to 0.13; 6 keeps it < 0.01."""
    probs = torch.softmax(torch.randn(1, 2550), -1)
    a = patch_answer(probs, n_width=34, n_height=75, patch_pixels=32, topk=0)
    assert len(a["probabilities"]) == 2550
    assert abs(sum(a["probabilities"].values()) - 1) < 0.01
    assert "point" not in a and "topk_points" not in a    # topk=0 drops the region decode


def test_register_qwen25vl_legacy_keys_is_idempotent_and_registers_the_renames():
    """Without the renames the 3B/7B checkpoints load randomly initialized; registering twice must not raise."""
    register_qwen25vl_legacy_keys()
    register_qwen25vl_legacy_keys()
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    assert get_checkpoint_conversion_mapping(POINTER_CLASS_NAME_QWEN25VL) is not None


def test_systemone_refuses_requests_it_cannot_answer():
    """The refusals happen before the model is touched, so a None model is enough."""
    choice = {"type": "choice", "instructions": "Click on view results", "criteria": {"1": None}}
    with pytest.raises(ValueError, match="images"):
        systemone(None, None, {"state": "", "questions": {"tap": choice}}, "qwen2_5_vl")
    with pytest.raises(ValueError, match="choice"):
        systemone(None, None, {"state": "", "images": [object()], "questions": {"done": {"type": "noul", "instructions": "x"}}}, "qwen2_5_vl")


def card():
    return httpx.get(f"{BASE}/v1/models", timeout=30).json()["models"][0]


@pytest.fixture(scope="module")
def gui_actor_server():
    """The server-marked tests need a GUI-Actor checkpoint up; anything else (or nothing) skips."""
    try:
        c = card()
    except Exception as e:
        pytest.skip(f"no server at {BASE} ({e})")
    if c.get("kind") != "gui_actor":
        pytest.skip(f"the server at {BASE} serves {c.get('kind')!r}, not a GUI-Actor checkpoint")


def png_b64(size=(1080, 2400)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(body):
    r = httpx.post(f"{BASE}/v1/systemone", json=body, timeout=300)
    assert r.headers["x-typesafe-request-id"]   # every TypeSafe client exposes it as response.request_id
    return r.status_code, r.json()


@pytest.mark.server
def test_choice_answer_is_a_per_patch_choice(gui_actor_server):
    code, r = post({"state": "Open the settings app", "model": "jev-latest",
                    "questions": {"tap": {"type": "choice", "instructions": "Click on the search icon", "criteria": {"1": None}}},
                    "images": [png_b64()]})
    assert code == 200
    a = r["answers"]["tap"]
    assert a["type"] == "choice"
    assert len(a["probabilities"]) == a["n_width"] * a["n_height"] > 0
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    assert abs(sum(a["probabilities"].values()) - 1) < 0.02
    assert a["patch_pixels"] in {28, 32} and 0 <= a["confidence"] <= 1
    assert all(0 <= v <= 1 for v in a["topk_values"]) and all(0 <= c <= 1 for p in a["topk_points"] for c in p)
    assert r["usage"]["output_tokens"] == 0 and r["usage"]["input_tokens"] > 0
    assert "latency_ms" in r


@pytest.mark.server
def test_an_image_is_required(gui_actor_server):
    code, r = post({"state": "x", "questions": {"tap": {"type": "choice", "instructions": "x", "criteria": {"1": None}}}})
    assert code == 422 and "image" in r["detail"]


@pytest.mark.server
def test_only_choice_questions_are_answered(gui_actor_server):
    code, r = post({"state": "x", "images": [png_b64((64, 64))],
                    "questions": {"done": {"type": "noul", "instructions": "Is the goal done?"}}})
    assert code == 422 and "choice" in r["detail"]


@pytest.mark.server
def test_models_card_names_the_family(gui_actor_server):
    c = card()
    assert c["kind"] == "gui_actor" and c["model_type"] in MODEL_TYPES
    assert c["patch_pixels"] in {28, 32} and c["release_date"] and c["run"]
