"""GUI-Actor serving: a pointer head over visual patches on a Qwen2.5-VL / Qwen3.5 backbone.

GUI-Actor neither regresses coordinates nor answers by generating text. The prompt carries the
full click action -- bare arguments JSON, no tool envelope -- whose coordinate list holds the
``<|pointer_start|><|pointer_pad|><|pointer_end|>`` placeholder triple, and the click point is
the attention of the ``<|pointer_pad|>`` hidden state over the image patches. Placeholder mode
is a single prefill (``max_new_tokens=1``): the answer does not depend on the generated token.

The pointer head, the prompt constants and the region decode are vendored from
microsoft/GUI-Actor (MIT License, Copyright (c) 2025 Microsoft) -- the same arrangement
kev.clef_model has with Cloudflare/clef-flash -- so these checkpoints serve with no dependency
on that repository. Two asymmetries in the vendored head are deliberate and must not be
"fixed":

* the encoder side reads the *embed* layer's hidden state (the vision features already
  scattered into the placeholders, exactly what the language model consumed), while the
  decoder side reads the *last* layer's hidden state at ``<|pointer_pad|>``. They are not
  interchangeable: the head compares "what the LM decided to click" against "what the pixels
  look like".
* the head's output is a softmax over patches and training supervised it against a
  distribution (uniform over the in-ground-truth-box patches), not a class index. That is
  what makes the inference-time connected-component decode (``get_prediction_region_point``)
  work at all.

Serving shape (``kev.serve.GUIActorServer``, ``POST /v1/systemone``): the request's ``state``
is the system prompt and each question's ``instructions`` the user instruction; ``images``
must hold the screenshot to ground on and its *last* entry is used. Only ``choice`` questions
are answered: ``probabilities`` is the per-patch distribution keyed by patch index (see
``patch_answer`` for the index -> pixel convention) and ``choice`` the argmax patch index.

Loading: ``load_gui_actor_checkpoint``. The 3B/7B-Qwen2.5-VL checkpoints predate transformers
5.x and load randomly initialized through their own class without
``register_qwen25vl_legacy_keys`` -- read that docstring before touching the loader.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...api import render, round_prob

# The transformers model_types this serves (the GUI-Actor repo calls them qwen25vl / qwen35).
MODEL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5"}
# Pixels spanned by one merged visual token: patch_size * merge_size.
MERGED_TOKEN_PIXELS = {"qwen2_5_vl": 28, "qwen3_vl": 32, "qwen3_vl_moe": 32, "qwen3_5": 32}
PATCH_PROB_DECIMALS = 6   # a 2,550-patch grid at kev.api.round_prob's 4 decimals can drift the sum by 0.13; 6 bounds it at 0.0013
TOPK = 5                  # region points per answer (kev.serve overrides from KEV_GUI_ACTOR_TOPK)
POINTER_TENSOR_PREFIX = "multi_patch_pointer_head."


# ── vendored constants (gui_actor/constants.py, renamed to kev's UPPER style) ─────────────
POINTER_START_TOKEN = "<|pointer_start|>"
POINTER_END_TOKEN = "<|pointer_end|>"
POINTER_PAD_TOKEN = "<|pointer_pad|>"
# One click point, as the three special tokens that stand in for it.
POINTER_PLACEHOLDER = POINTER_START_TOKEN + POINTER_PAD_TOKEN + POINTER_END_TOKEN
# The assistant reply *is* the action, so it is just the arguments JSON: no tool-name
# envelope and no <tool_call> wrapper (the tags only made sense while the reply carried a
# name/arguments pair).  ``%s`` is whatever goes inside the JSON coordinate list: the
# pointer placeholder here, or literal numbers when the model spells the point out itself.
CLICK_ACTION_JSON = '{"action": "click", "coordinate": [%s]}'
# The grounding system prompt the checkpoints were trained against (qwen2vl alone used a
# shorter one; no such checkpoint is served here). The server sends it only when a request's
# state renders to nothing -- otherwise the state IS the system prompt.
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. You "
    "should output a click action on the correct position. To indicate the click location, "
    "we will use some special tokens, which is used to refer to a visual patch later. "
    "For each function call, return its arguments as a json object:\n"
    f"{CLICK_ACTION_JSON % '<your_special_token_here>'}"
)
# The Qwen2-VL-style inline chat template the model was trained with (its own checkpoints'
# saved templates are the base Qwen templates and carry no pointer content).
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
ADDITIONAL_SPECIAL_TOKENS = ["<|recipient|>", "<|diff_marker|>", POINTER_START_TOKEN, POINTER_END_TOKEN, POINTER_PAD_TOKEN]


def format_click_action(coordinate: str = POINTER_PLACEHOLDER) -> str:
    """Render the click action, defaulting to the pointer placeholder."""
    return CLICK_ACTION_JSON % coordinate


# ── vendored: the pointer head (gui_actor/modeling_base.py: VisionHead_MultiPatch) ────────
class VisionHead_MultiPatch(nn.Module):
    """Scores every visual patch against every ``<|pointer_pad|>`` query.

    Shape-agnostic in ``n_enc`` (the number of image patches), which is why a single
    checkpoint transfers across screenshot resolutions.
    """

    def __init__(self, d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1, temperature=1.0):
        super().__init__()
        self.d_model = d_model

        # Post-hoc calibration scalar, read from ``config.pointer_temperature`` and applied
        # by the *caller* to the confidence it derives -- never inside ``forward()``.  The
        # returned distribution feeds a threshold + BFS region post-process, so rescaling it
        # here would move the predicted point; dividing after the region is fixed is what
        # keeps every existing output bit-identical.
        #
        # A plain attribute on purpose: a buffer would enter ``state_dict()`` and break
        # strict loading of every checkpoint saved before this feature existed, and a
        # parameter would join ``named_parameters()``.
        self.temperature = float(temperature)

        # No extra normalisation on the inputs: the backbones already emit RMSNorm'd
        # hidden states.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )

        # The vision tower's output carries no text context, so the patches interact
        # with each other once before being matched against the query.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_state_enc, hidden_state_dec, labels=None, return_logits=False):
        """Returns ``(attn_weights, loss)``, plus ``patch_logits`` when ``return_logits``.

        ``attn_weights`` is *always* the raw softmax (temperature 1).  The fitted
        temperature is applied by the caller to the confidence scalar only, after the
        predicted region has been extracted from these weights; see ``self.temperature``.
        """
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(query=enc_input, key=enc_input, value=enc_input, need_weights=False)
        enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output)).squeeze(0)

        proj_enc = self.projection_enc(enc_ctx)                  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)         # [n_dec, d_model]

        # Scaling by sqrt(d_model) keeps the logits well-conditioned for any n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if labels is not None:                                   # training only; kev serves the head
            epsilon = 1e-8
            labels_float = labels.float()
            # Uniform over the in-box patches: every patch the box touches is correct.
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)
            # fp32: bf16 log_softmax is too coarse for a KL target and the model runs bf16
            # under --bf16, so the logits would otherwise stay in bf16.
            pred_log_probs = F.log_softmax(patch_logits.float(), dim=-1)
            loss = F.kl_div(pred_log_probs, target_dist, reduction="batchmean")

        if return_logits:
            return attn_weights, loss, patch_logits
        return attn_weights, loss


def build_pointer_head(config, hidden_size: int | None = None) -> VisionHead_MultiPatch:
    """The pointer head is square: it projects into ``d_model`` and back.

    For every released Qwen VL backbone the vision tower's ``out_hidden_size`` already
    equals the language model's ``hidden_size``, so the encoder (vision) and decoder
    (language) sides share one width and a single ``d_model`` is enough.

    ``config.pointer_temperature`` is the fitted post-hoc calibration scalar.  Like the
    ``pointer_*_token_id`` fields it is a plain extra key in ``config.json``, so it
    survives ``save_pretrained`` / ``from_pretrained`` without a custom config class.
    """
    if hidden_size is None:
        hidden_size = config.text_config.hidden_size
    temperature = getattr(config, "pointer_temperature", 1.0)
    if temperature is None:
        temperature = 1.0
    temperature = float(temperature)  # a hand-edited config.json can carry a string
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"pointer_temperature must be finite and positive, got {temperature}")
    return VisionHead_MultiPatch(hidden_size, hidden_size, temperature=temperature)


def pointer_temperature(model, override: float | None = None) -> float:
    """The temperature the confidence is produced at: ``override``, else the head's fitted value."""
    if override is not None:
        value = float(override)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"temperature must be finite and positive, got {value}")
        return value
    head = getattr(model, "multi_patch_pointer_head", None)
    return float(getattr(head, "temperature", 1.0))


# ── the backbone classes (gui_actor/modeling_qwen25vl.py, modeling_qwen3_5.py) ────────────
class _PointerHeadMixin:
    """Attach the vendored head so a checkpoint's ``multi_patch_pointer_head.*`` tensors load.

    The GUI-Actor repo's classes also mix in its ``PointerForwardMixin``, whose forward exists
    only to train the head (loss, embed-layer hook). Serving never calls it -- the pass is
    ``generate`` plus a direct call to the head -- so it is not vendored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_patch_pointer_head = build_pointer_head(self.config)
        self.post_init()


_POINTER_CLASSES: dict[str, type] = {}


def _pointer_model_class(hf_model_type: str) -> type:
    """The ``...WithPointer`` class for a transformers model_type, built on first use.

    Building it lazily keeps the transformers model modules out of kev.serve's import path:
    only a GUI-Actor checkpoint pays for them. The class *name* matters -- it is what the
    checkpoint's ``config.architectures`` names and what ``register_qwen25vl_legacy_keys``
    keys the legacy key renaming on.
    """
    if hf_model_type in _POINTER_CLASSES:
        return _POINTER_CLASSES[hf_model_type]
    if hf_model_type == "qwen2_5_vl":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as base
        name = "Qwen2_5_VLForConditionalGenerationWithPointer"
    elif hf_model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as base
        name = "Qwen3_5ForConditionalGenerationWithPointer"
    elif hf_model_type in ("qwen3_vl", "qwen3_vl_moe"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration as base
        name = "Qwen3_VLForConditionalGenerationWithPointer"
    else:
        raise ValueError(f"unsupported GUI-Actor backbone {hf_model_type!r}; expected one of {sorted(MODEL_TYPES)}")
    _POINTER_CLASSES[hf_model_type] = type(name, (_PointerHeadMixin, base), {"__module__": __name__})
    return _POINTER_CLASSES[hf_model_type]


# ── vendored: the region decode (gui_actor/inference.py) ─────────────────────────────────
def region_confidence(patch_logits, region_indices, temperature: float = 1.0) -> float:
    """Probability mass the temperature-scaled patch distribution puts on a region.

    ``patch_logits`` is one row of *raw* (pre-softmax) scores -- a torch tensor or a
    sequence of floats.  Taking the raw scores (rather than ``attn_scores``, which the
    bf16 forward pass has already rounded through a softmax) is what lets the scalar be
    recomputed at any temperature without re-running the model.  Subtracting the max is
    exactly softmax-shift invariant and keeps large logits from overflowing.
    """
    z = torch.as_tensor(patch_logits, dtype=torch.float64).detach()
    z = z - z.max()
    probs = torch.softmax(z / float(temperature), dim=-1)
    # float64 accumulation: in float32 a whole-grid region (hundreds of patches) can sum
    # to slightly more than 1, and a probability mass must never exceed one.
    return min(1.0, float(probs[torch.as_tensor(region_indices, dtype=torch.long)].sum()))


def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30, activation_threshold=0.3,
                                return_all_regions=True, rect_center=False, return_region_indices=False):
    """Threshold the patch scores, split the survivors into 4-neighbour connected regions, and
    return the regions' weighted centres, best first.

    1. Select activated patches (> activation_threshold * max score)
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point

    Patch ``idx`` sits at ``y = idx // n_width, x = idx % n_width``; a region's centre is the
    activation-weighted average of its patches' normalised centres, and regions are ranked by
    their mean activation. Coordinates are normalised to the model's (smart-resized) input.

    With ``return_region_indices`` the flat patch indices of every region are returned too
    (sorted alongside the centres and scores), which is what the confidence sums over.
    """
    if return_region_indices and not return_all_regions:
        raise ValueError("return_region_indices requires return_all_regions=True")
    # Get the highest activation value and threshold
    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    # Select all patches above the threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices

    # Convert indices to 2D coordinates
    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    # Divide into connected regions.  ``index_of`` makes the neighbour lookup O(1); scanning
    # topk_coords inside the BFS made it O(k^2), which is seconds per screenshot once a few
    # thousand patches pass the threshold.
    regions = []
    visited = set()
    index_of = {idx: i for i, (_, _, idx) in enumerate(topk_coords)}
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue

        # Start a new region
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = deque([(y, x, idx, topk_values[i].item())])

        # BFS to find connected points
        while queue:
            cy, cx, c_idx, c_val = queue.popleft()

            # Check 4 adjacent directions
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < n_height and 0 <= nx < n_width):
                    continue
                n_idx = ny * n_width + nx

                # Check if this adjacent point is in the topk list
                j = index_of.get(n_idx)
                if j is not None and n_idx not in visited:
                    visited.add(n_idx)
                    region.append((ny, nx, n_idx, topk_values[j].item()))
                    queue.append((ny, nx, n_idx, topk_values[j].item()))

        regions.append(region)

    # Calculate the average activation value for each region
    region_scores = []
    region_centers = []
    region_points = []
    region_indices = []

    for region in regions:
        # Calculate average score for the region
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)
        region_indices.append([item[2] for item in region])

        # Calculate normalized center coordinates for each patch, then take the average
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()

        for y, x, _, score in region:
            # Normalized coordinates of the center point for each patch
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)

            y_coords.add(center_y)
            x_coords.add(center_x)

        region_points.append(normalized_centers)

        # Calculate the average of normalized coordinates as the region center
        if not rect_center:
            # Weighted average
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))

    # Select the region with the highest average activation value
    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    sorted_region_indices = [region_indices[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        # 1. best_point: the center point of the region with the highest average activation value
        # 2. sorted_centers: every region's center point, by average activation, descending
        # 3. sorted_scores: every region's average activation, descending
        # 4. sorted_points: every patch's normalized center, by its region's average activation
        # 5. sorted_region_indices (opt-in): the flat patch indices of every region, same order
        if return_region_indices:
            return best_point, sorted_centers, sorted_scores, sorted_points, sorted_region_indices
        return best_point, sorted_centers, sorted_scores, sorted_points
    return best_point


# ── checkpoint detection and loading ──────────────────────────────────────────────────────
def _pointer_tensor_names(path) -> list[str]:
    """A checkpoint directory's tensor names, from the safetensors index or the single shard's
    header (no tensor data is read)."""
    p = Path(path)
    index = p / "model.safetensors.index.json"
    if index.exists():
        return list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    shards = sorted(p.glob("*.safetensors"))
    if not shards:
        return []
    from safetensors import safe_open
    with safe_open(shards[0], framework="pt") as f:
        return list(f.keys())


def is_gui_actor_checkpoint(path) -> bool:
    """True for a GUI-Actor checkpoint directory: its weights carry the pointer head.

    The head, not config.json, is the signal -- a config can claim pointer token ids without
    the trained head, and such a model would serve random grounding scores."""
    try:
        return any(name.startswith(POINTER_TENSOR_PREFIX) for name in _pointer_tensor_names(path))
    except Exception:
        return False


# Alias used by kev.serve (which imports `is_gui_actor`, as it does `is_clef`).
is_gui_actor = is_gui_actor_checkpoint

POINTER_CLASS_NAME_QWEN25VL = "Qwen2_5_VLForConditionalGenerationWithPointer"


def register_qwen25vl_legacy_keys() -> None:
    """Teach transformers that the Qwen2.5-VL pointer class reads 4.x-layout checkpoints.

    transformers 5.17 made every Qwen VL model composite (``model.visual``,
    ``model.language_model``) and renames a legacy checkpoint's keys on load -- but only for
    the *plain* class name: the conversion table is keyed by class name with no ``qwen2_5_vl``
    fallback, so the ``...WithPointer`` subclass gets none of it and every backbone tensor goes
    unmatched. The model then loads randomly initialized and still reports success, which is
    exactly the failure this call prevents. Verified against microsoft/GUI-Actor-3B/7B-Qwen2.5-VL
    (zero missing, zero unexpected keys). The 4B-Qwen3.5 checkpoint was saved by transformers
    5.17 itself and needs none of this. Idempotent: the registry raises on a second
    registration, so a lookup guards the call.
    """
    from transformers.conversion_mapping import (
        WeightRenaming, get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping,
    )
    if get_checkpoint_conversion_mapping(POINTER_CLASS_NAME_QWEN25VL) is not None:
        return
    register_checkpoint_conversion_mapping(POINTER_CLASS_NAME_QWEN25VL, [
        WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
        WeightRenaming(source_patterns=r"^model(?!\.(language_model|visual))", target_patterns="model.language_model"),
    ])


def gui_actor_model_type(path: str) -> str:
    """A checkpoint's transformers model_type, validated against MODEL_TYPES."""
    from transformers import AutoConfig
    hf_model_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
    if hf_model_type not in MODEL_TYPES:
        raise ValueError(f"unsupported GUI-Actor backbone {hf_model_type!r}; expected one of {sorted(MODEL_TYPES)}")
    return hf_model_type


def get_merged_token_pixels(model_type: str) -> int:
    """Pixels covered by one merged visual token -- the model's effective resolution."""
    return MERGED_TOKEN_PIXELS[model_type]


def _require_pointer_tokens(tokenizer, path) -> None:
    """A GUI-Actor checkpoint ships the five special tokens (the head reads two of them); a
    checkpoint without them cannot ground at all, so fail loudly instead of resizing."""
    missing = [t for t in ADDITIONAL_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"{path} is missing GUI-Actor's special tokens {missing}; it does not look like a GUI-Actor checkpoint")


def load_gui_actor_checkpoint(path: str, device: str, dtype: torch.dtype = torch.bfloat16, attn: str | None = None):
    """GUI-Actor checkpoint directory -> (model eval, processor, transformers model_type).

    `device` is kev.device.default_device()'s string; a device_map is built from it (flash-attn
    is not required -- `attn=None` leaves the implementation to transformers, sdpa here).
    """
    from transformers import AutoProcessor
    hf_model_type = gui_actor_model_type(path)
    if hf_model_type == "qwen2_5_vl":
        register_qwen25vl_legacy_keys()
    if hf_model_type == "qwen3_5" and not str(device).startswith("cuda"):
        raise RuntimeError("the Qwen3.5 backbone runs on the causal-conv1d / flash-linear-attention CUDA kernels only; "
                           "serve this checkpoint on a GPU (CUDA_VISIBLE_DEVICES=<n>)")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    _require_pointer_tokens(processor.tokenizer, path)
    # the GPU kev_serve.device.select() made current (`--run name=ckpt@N`), not a fixed cuda:0, so
    # several models in one process can sit on different GPUs
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype,
                                   "device_map": torch.cuda.current_device() if device == "cuda" else device}
    if attn is not None:
        load_kwargs["attn_implementation"] = attn
    model = _pointer_model_class(hf_model_type).from_pretrained(path, **load_kwargs).eval()
    return model, processor, hf_model_type


# ── the grounding pass ────────────────────────────────────────────────────────────────────
def build_prompt(processor, image, system_text: str, instruction: str) -> str:
    """The click-action prompt: system turn, user turn (image + instruction), assistant head. The
    instruction is what varies between requests over one screenshot, so the text up to and
    including the image is stable -- that stable prefix is what the cache holds (image_prefix)."""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, chat_template=CHAT_TEMPLATE)
    return text + "<|im_start|>assistant<|recipient|>os\n" + format_click_action()


def split_at_image(processor, input_ids):
    """The index just past the image's last token: the prefix (system + user header + image) is
    everything up to here, and the instruction + assistant head is the continuation. The pointer
    pad lives in the continuation."""
    ve = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    hit = (input_ids[0] == ve).nonzero(as_tuple=True)[0]
    if hit.numel() == 0:
        raise ValueError("no <|vision_end|> in the prompt; the chat template did not render the image")
    return int(hit[-1].item()) + 1


@torch.inference_mode()
def image_prefix(model, processor, image, system_text: str, instruction: str):
    """The cacheable first stage: run the prompt's image prefix (system + user header + image) once.

    -> (cache, image_embeds, prefix_len, n_width, n_height, grid_thw). The pointer head's image side
    is the *embed layer* hidden state at the image tokens; it is stable for a given (image, system
    text), so it is cached with the KV. Continuations supply the instruction and read the pointer
    pad's last-layer hidden state (see _grounding_pass with `prefix=`).

    ``instruction`` only selects the split point (a real prompt's tokens up to <|vision_end|>); it
    is not consumed here — the prefix prompt cannot be rebuilt with an empty instruction, because
    that moves <|im_end|> and changes the token count. The prefix tokens are exactly the same for
    every instruction that shares this (image, system text).

    Qwen2.5-VL / Qwen3-VL / Qwen3.5 all use M-RoPE: the continuation's 3D position ids must be the
    tail of the *joined* sequence's ids, so get_rope_index runs once over the full prompt and the
    two stages slice it. Letting the model infer positions from a stale rope_deltas (the prefix
    length, not the joined one) does not reproduce a one-pass forward.
    """
    text = build_prompt(processor, image, system_text, instruction)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    ids = inputs["input_ids"]
    p = split_at_image(processor, ids)
    inner = model.model                                  # Qwen2_5_VLModel / Qwen3_VLModel / Qwen3_5Model: owns get_rope_index
    pos3 = inner.get_rope_index(input_ids=ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs.get("attention_mask"))[0]
    out = model(input_ids=ids[:, :p], attention_mask=inputs["attention_mask"][:, :p], position_ids=pos3[:, :, :p],
                pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=inputs["mm_token_type_ids"][:, :p], use_cache=True,
                output_hidden_states=True, return_dict=True)
    image_embeds = out.hidden_states[0][0][ids[0, :p] == model.config.image_token_id]
    _, n_height, n_width = (inputs["image_grid_thw"][0] // model.model.visual.spatial_merge_size).tolist()
    return out.past_key_values, image_embeds, p, n_width, n_height, inputs["image_grid_thw"]


def _replicate_cache(cache):
    """A copy of a prefix cache whose tensor state the continuation may consume: DynamicLayer.update
    writes in place, so continuing on the stored cache would append the instruction to it and the next
    hit would attend over a longer prefix. Shallow layer copies + cloned tensors (the layers here hold
    keys/values tensors, unlike the hybrid backbone's dict-indexed recurrent state)."""
    import copy as _copy
    def cloned(v):
        if isinstance(v, torch.Tensor): return v.clone()
        if isinstance(v, list): return [cloned(x) for x in v]
        if isinstance(v, tuple): return tuple(cloned(x) for x in v)
        if isinstance(v, dict): return {k: cloned(x) for k, x in v.items()}
        return v
    dup = _copy.copy(cache)
    dup.layers = []
    for layer in cache.layers:
        d = _copy.copy(layer)
        for name, v in list(vars(d).items()):
            c = cloned(v)
            if c is not v: setattr(d, name, c)
        dup.layers.append(d)
    return dup


@torch.inference_mode()
def _grounding_pass(model, processor, image, system_text: str, instruction: str, prefix=None):
    """One placeholder-mode pass -> (attn_scores [1, n_patches], patch_logits [n_patches],
    n_width, n_height, input tokens).

    The steps mirror gui_actor.inference.inference() -- the path the deployment report measured
    end to end -- from the conversation through the two hidden states the head compares.
    ``patch_logits`` is the raw pre-softmax row 0 (float32): the confidence is recomputed from
    it at any temperature without re-running the model.
    Placeholder mode needs no logits processor: the prompt already ends inside the pointer
    triple, and nothing generated is read. The processor's inputs are passed on unfiltered:
    ``mm_token_type_ids`` is required for Qwen3.x's multimodal RoPE (raises without it) and is
    what Qwen2.5-VL's 3D position ids are computed from.

    ``prefix`` (from image_prefix) reuses the image's KV + embed-layer hidden states and runs only
    the instruction + assistant head; the result matches the one-pass forward to bf16 noise
    (probed on GUI-Actor-3B-Qwen2.5-VL and the Qwen3.5-2B checkpoint).
    """
    text = build_prompt(processor, image, system_text, instruction)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    pointer_pad_mask = inputs["input_ids"][0] == model.config.pointer_pad_token_id
    if pointer_pad_mask.sum().item() == 0:
        raise ValueError("no <|pointer_pad|> token in the prompt; the chat template did not render the click action")
    image_mask = inputs["input_ids"][0] == model.config.image_token_id

    if prefix is None:
        results = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)
        decoder_hidden_states = results.hidden_states[0][-1][0][pointer_pad_mask]
        image_embeds = results.hidden_states[0][0][0][image_mask]
        n_tokens = int(inputs["input_ids"].shape[1])
        _, n_height, n_width = (inputs["image_grid_thw"][0] // model.model.visual.spatial_merge_size).tolist()
    else:
        cache, image_embeds, p, n_width, n_height, grid_thw = prefix
        ids = inputs["input_ids"]
        if p > ids.shape[1]:
            raise ValueError("the cached prefix is longer than this prompt; it does not match the image")
        inner = model.model
        pos3 = inner.get_rope_index(input_ids=ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                    image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs.get("attention_mask"))[0]
        out = model(input_ids=ids[:, p:], attention_mask=inputs["attention_mask"], position_ids=pos3[:, :, p:],
                    past_key_values=_replicate_cache(cache), use_cache=False, output_hidden_states=True, return_dict=True)
        decoder_hidden_states = out.hidden_states[-1][0][ids[0, p:] == model.config.pointer_pad_token_id]
        n_tokens = p + int(ids[:, p:].shape[1])
        _, n_height, n_width = (grid_thw[0] // model.model.visual.spatial_merge_size).tolist()

    attn_scores, _, patch_logits = model.multi_patch_pointer_head(
        image_embeds, decoder_hidden_states, return_logits=True
    )
    return attn_scores, patch_logits[0].float(), n_width, n_height, n_tokens


def _to_999(x: float, y: float) -> tuple[int, int]:
    """Normalised [0,1] region-centre coordinates -> 0-999 integers (the Qwen-VL grounding
    convention: x across the width, y down the height, top-left origin)."""
    return min(999, max(0, round(x * 999))), min(999, max(0, round(y * 999)))


def ground_answer(attn_scores, patch_logits, n_width: int, n_height: int,
                  topk: int = TOPK, temperature: float = 1.0) -> dict:
    """The Ground answer for one grounding pass: the connected region's activation-weighted
    centre (get_prediction_region_point, rect_center=False -- the official inference path) as
    [x, y] in 0-999 integers.

    ``confidence`` is the probability mass the (temperature-scaled) patch distribution puts on
    the predicted region -- the fitted temperature only rescales it, never the point.
    ``topk_points`` are the other regions' centres, same 0-999 scale; ``topk=0`` drops them.
    Region extraction reads the *raw* distribution on CPU first (the decode calls .item() per
    activated patch), so the temperature cannot move the point.
    """
    point, points, values, _, region_indices = get_prediction_region_point(
        attn_scores.float().cpu(), n_width, n_height,
        return_all_regions=True, rect_center=False, return_region_indices=True)
    top_region = [int(i) for i in region_indices[0]]
    answer = {"type": "ground",
              "point": list(_to_999(point[0], point[1])),
              "confidence": round_prob(region_confidence(patch_logits, top_region, temperature), PATCH_PROB_DECIMALS)}
    if topk:
        answer["topk_points"] = [list(_to_999(x, y)) for x, y in points[:topk]]
    return answer


@torch.inference_mode()
def systemone(model, processor, request: dict, model_type: str, topk: int = TOPK,
              temperature: float | None = None, prefix=None) -> dict:
    """Answer a /v1/systemone request dict with ground answers (kev_serve adds latency_ms).

    The request's ``state`` is the system prompt (the checkpoints' own grounding prompt only
    when it renders to nothing) and each ground question's ``instructions`` the user instruction.
    ``images`` must hold PIL images -- the HTTP layer decodes the base64 -- and the *last* one is
    the screenshot to ground on (the model was trained single-image; older frames can be sent
    before it). One grounding pass per distinct instruction: questions sharing an instruction
    share the pass.
    """
    images = request.get("images") or []
    if not images:
        raise ValueError("a GUI-Actor request needs at least one image in `images` (base64 PNG/JPEG, raw or data-URI)")
    unsupported = sorted({q["type"] for q in (request.get("questions") or {}).values()} - {"ground"})
    if unsupported:
        raise ValueError(f"a GUI-Actor backend answers ground questions only; this request has {', '.join(unsupported)}")

    image = images[-1]
    system_text = render(request.get("state")).strip() or GROUNDING_SYSTEM_MESSAGE
    temperature = pointer_temperature(model, temperature)   # KEV_TEMPERATURE override, else the fitted scalar
    answers, input_tokens, passes = {}, 0, {}
    for qid, question in (request.get("questions") or {}).items():
        instruction = render(question.get("instructions"))
        if instruction not in passes:   # questions sharing an instruction share the pass
            passes[instruction] = _grounding_pass(model, processor, image, system_text, instruction, prefix=prefix)
        attn_scores, patch_logits, n_width, n_height, n_tokens = passes[instruction]
        input_tokens += n_tokens
        answers[qid] = ground_answer(attn_scores, patch_logits, n_width, n_height, topk,
                                     temperature=temperature)
    return {"model": request.get("model", "kev-latest"), "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
