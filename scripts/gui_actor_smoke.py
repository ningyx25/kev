"""Smoke + fidelity check for GUI-Actor serving (kev.gui_actor_model), on a GPU.

For each checkpoint: loads it through kev.gui_actor_model, answers one /v1/systemone-shaped
request, and prints the patch grid, the top-1 region point, the top-k points, the token count
and the distribution's sum. When the environment still has the GUI-Actor repo's package
editable-installed, it also cross-checks the raw attn_scores against
gui_actor.inference.inference() on the same conversation (that package is not a kev dependency;
the check is skipped when it is absent).

The deployment report's measured 4B case is the pinned baseline:

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/gui_actor_smoke.py \
        --runs microsoft/GUI-Actor-3B-Qwen2.5-VL microsoft/GUI-Actor-7B-Qwen2.5-VL microsoft/GUI-Actor-4B-Qwen3.5 \
        --image ../GUI-Actor/android_control_parsered/parsered/13628/step_006_screenshot.png \
        --instruction "Click on view results" \
        --expect-point 0.7485 0.9610 --expect-bbox 0.5 0.93875 1.0 0.97375

Expected: the 4B's top-1 point lands within --tolerance of --expect-point and inside
--expect-bbox (grid 34x75); the 3B/7B produce a sane point with the distribution summing to 1
(their forward had never been run under transformers 5.17 before this script).
"""
import argparse, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from kev.device import empty_cache
from kev.gui_actor_model import GROUNDING_SYSTEM_MESSAGE, _grounding_pass, load_gui_actor_checkpoint, systemone


def request_for(image, instruction, state=""):
    """The /v1/systemone shape a client sends: state -> system prompt, instructions -> the
    user instruction, images[-1] the screenshot. criteria is required by the API and ignored."""
    return {"model": "gui-actor-smoke", "state": state, "images": [image],
            "questions": {"tap": {"type": "choice", "instructions": instruction, "criteria": {"1": None}}}}


def parity_check(model, processor, image, instruction, system_text):
    """This module's raw attn_scores against gui_actor.inference.inference() on the same pass."""
    try:
        from gui_actor.inference import inference as gui_actor_inference
    except ImportError:
        print("parity: the gui_actor package is not installed here; skipped")
        return True
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]
    theirs = gui_actor_inference(conversation, model, processor.tokenizer, processor, use_placeholder=True, topk=3)
    ours, n_width, n_height, _ = _grounding_pass(model, processor, image, system_text, instruction)
    same_scores = torch.allclose(ours.float().cpu(), torch.tensor(theirs["attn_scores"]).float(), atol=0)
    same_grid = (n_width, n_height) == (theirs["n_width"], theirs["n_height"])
    print(f"parity: attn_scores identical = {same_scores}; grid {n_width}x{n_height} vs "
          f"{theirs['n_width']}x{theirs['n_height']} identical = {same_grid}")
    return bool(same_scores and same_grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="GUI-Actor checkpoint directories")
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", default="Click on view results")
    ap.add_argument("--state", default="", help="system prompt text; empty uses the checkpoints' own grounding message")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--expect-point", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--expect-bbox", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--tolerance", type=float, default=0.02, help="how far the top-1 point may sit from --expect-point")
    ap.add_argument("--no-parity", action="store_true", help="skip the gui_actor.inference() cross-check")
    a = ap.parse_args()

    image = Image.open(a.image).convert("RGB")
    system_text = a.state or GROUNDING_SYSTEM_MESSAGE
    print(f"image {a.image} {image.size}, instruction {a.instruction!r}")
    failures = []
    for run in a.runs:
        print(f"\n=== {run} ===")
        t = time.time()
        model, processor, model_type = load_gui_actor_checkpoint(run, a.device)
        print(f"loaded in {time.time() - t:.1f}s (model_type={model_type})")
        t = time.time()
        body = systemone(model, processor, request_for(image, a.instruction, a.state), model_type, topk=a.topk)
        dt = time.time() - t
        ans = body["answers"]["tap"]
        probs = ans["probabilities"]
        print(f"grid {ans['n_width']}x{ans['n_height']} ({len(probs)} patches), patch_pixels {ans['patch_pixels']}")
        print(f"choice {ans['choice']} (confidence {ans['confidence']}); sum(probabilities) = {sum(probs.values()):.6f}")
        print(f"point {ans['point']}  topk_points {ans['topk_points']}  topk_values {[round(v, 4) for v in ans['topk_values']]}")
        print(f"input_tokens {body['usage']['input_tokens']}; one pass {dt:.2f}s")

        if len(probs) != ans["n_width"] * ans["n_height"]:
            failures.append(f"{run}: {len(probs)} probabilities for a {ans['n_width']}x{ans['n_height']} grid")
        if abs(sum(probs.values()) - 1) > 0.02:
            failures.append(f"{run}: probabilities sum to {sum(probs.values()):.4f}")
        if a.expect_point:
            (x, y), point = a.expect_point, ans["point"]
            if abs(point[0] - x) > a.tolerance or abs(point[1] - y) > a.tolerance:
                failures.append(f"{run}: point {point} is not within {a.tolerance} of {a.expect_point}")
            else:
                print(f"point matches the expected {a.expect_point} within {a.tolerance}")
        if a.expect_bbox:
            x1, y1, x2, y2 = a.expect_bbox
            x, y = ans["point"]
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                failures.append(f"{run}: point {(x, y)} is outside the expected bbox {a.expect_bbox}")
            else:
                print(f"point lies inside the expected bbox {a.expect_bbox}")
        if not a.no_parity and not parity_check(model, processor, image, a.instruction, system_text):
            failures.append(f"{run}: attn_scores differ from gui_actor.inference.inference()")

        del model
        empty_cache(a.device)

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
