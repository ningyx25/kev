"""Vision-language decision model: Qwen3VL backbone + PointerHead.

Extends the kev scoring interface (kev.model.SCORING_INTERFACE) to support a
single image per record alongside the state text.  The image is prepended to the
state as <|vision_start|> + N×<|image_pad|> + <|vision_end|> tokens; the rest of
the sequence (question branches) is unchanged.

Only the row form is used — the packed block-causal mask is not supported because
mrope (3D rotary) position-id computation is per-sequence, not across a packed batch.
This is the same trade-off as hybrid backbones (kev.model.is_hybrid).

Usage (training)
----------------
    from kev.vision_model import VisionDecisionModel, load_vision_processor
    proc = load_vision_processor("/path/to/qwen3vl_checkpoint")
    model = VisionDecisionModel("/path/to/qwen3vl_checkpoint", tok, device, lora=16)
    enc = model.encode_vision(tok, rec, image_path="/abs/path/img.png", processor=proc)
    probs = model.probs(enc)

Records with no image fall back to the plain text encode() path and are fully
compatible with VisionDecisionModel.
"""

import copy, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from .model import (
    MAX_STATE, MAX_BRANCH, MAX_PACKED,
    SERVE_MAX_STATE, SERVE_MAX_BRANCH,
    OPT_NONE, OPT_DECIDE,
    ContextOverflow,
    PointerHead,
    SCORING_INTERFACE,
    encode, rows_of, rows_per_pass,
    load_tokenizer, pad_id, user_tokens,
    branch_mask_batch,
)

# ── image-token ids (Qwen3VL; same ids as Qwen2VL) ─────────────────────────────
_VISION_START_TOKEN = "<|vision_start|>"
_VISION_END_TOKEN   = "<|vision_end|>"
_IMAGE_PAD_TOKEN    = "<|image_pad|>"

# mm_token_type_ids values (Qwen3VL convention)
_MM_TEXT  = 0
_MM_IMAGE = 1

# merge_size used by the Qwen2VL image processor (spatial pooling factor)
_MERGE_SIZE = 2


def load_vision_processor(base: str):
    """Load Qwen2VLImageProcessorFast from a local checkpoint directory.
    Requires torchvision (listed in pyproject.toml under dependencies).
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
        Qwen2VLImageProcessorFast,
    )
    return Qwen2VLImageProcessorFast.from_pretrained(base, local_files_only=True)


def _image_token_count(image_grid_thw: torch.Tensor) -> int:
    """Number of <|image_pad|> tokens for one image, after spatial merging.

    image_grid_thw: [1, 3] tensor of (T, H, W) in patch units.  The vision model
    spatially merges H and W by _MERGE_SIZE, so the final token count is
    T * (H // merge_size) * (W // merge_size).
    """
    t, h, w = image_grid_thw[0].tolist()
    return int(t) * (int(h) // _MERGE_SIZE) * (int(w) // _MERGE_SIZE)


def encode_vision(tok, rec, image_path: str, processor,
                  max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False,
                  option_isolation=False):
    """Like kev.model.encode() but prepends a vision prefix to the state.

    The prefix is:  <|vision_start|>  <|image_pad|>×N  <|vision_end|>
    where N = _image_token_count(image_grid_thw).

    Additional keys in the returned dict:
      pixel_values   – [total_patches, C*t*h*w] float tensor (from the processor)
      image_grid_thw – [1, 3] long tensor (T, H, W in patch units)
      n_img_tokens   – int, the N above (length of the image_pad span)
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    proc_out = processor(images=[img], return_tensors="pt")
    pixel_values   = proc_out["pixel_values"]        # [total_patches, C*t*h*w]
    image_grid_thw = proc_out["image_grid_thw"]      # [1, 3]

    n_img = _image_token_count(image_grid_thw)

    # Build the image-prefix token ids once
    vs_id  = tok.convert_tokens_to_ids(_VISION_START_TOKEN)
    ip_id  = tok.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
    ve_id  = tok.convert_tokens_to_ids(_VISION_END_TOKEN)
    img_prefix_ids = [vs_id] + [ip_id] * n_img + [ve_id]   # length N+2
    n_prefix = len(img_prefix_ids)

    # Encode the text record as usual (max_state reduced by the prefix length so
    # the combined row still fits; at least 1 state token must remain)
    adjusted_max_state = max(1, max_state - n_prefix)
    enc = encode(tok, rec, max_state=adjusted_max_state, max_branch=max_branch,
                 strict=strict, option_isolation=option_isolation)

    # Prepend image tokens (all segment 0, positions 0..n_prefix-1)
    # and shift existing positions by n_prefix
    enc["ids"]  = img_prefix_ids + enc["ids"]
    enc["seg"]  = [0] * n_prefix + enc["seg"]
    enc["pos"]  = list(range(n_prefix)) + [p + n_prefix for p in enc["pos"]]
    enc["opt"]  = [OPT_NONE] * n_prefix + enc["opt"]

    # Shift readout indices
    enc["decide_idx"] = [d + n_prefix for d in enc["decide_idx"]]
    enc["opt_idx"]    = [[o + n_prefix for o in oi] for oi in enc["opt_idx"]]

    # Attach vision tensors
    enc["pixel_values"]   = pixel_values
    enc["image_grid_thw"] = image_grid_thw
    enc["n_img_tokens"]   = n_img

    return enc


def _mm_token_type_ids(ids_list: list[list[int]], image_pad_id: int,
                       device: torch.device) -> torch.Tensor:
    """Build [B, L_max] mm_token_type_ids: 1 where token == image_pad_id, else 0.

    Qwen3VL's compute_3d_position_ids uses this to assign 3D mrope positions to
    image tokens and 1D positions to text tokens.  vision_start / vision_end tokens
    are treated as image tokens here so the spatial grid spans them cleanly.
    """
    L = max(len(ids) for ids in ids_list)
    out = torch.zeros(len(ids_list), L, dtype=torch.int, device=device)
    for b, ids in enumerate(ids_list):
        for i, tok_id in enumerate(ids):
            if tok_id == image_pad_id:
                out[b, i] = _MM_IMAGE
    return out


class VisionDecisionModel(nn.Module):
    """Decision model with a Qwen3VL backbone.

    Identical scoring interface to kev.model.DecisionModel; always runs in the
    row form (hybrid=True) because mrope is per-sequence.
    """

    backend = "torch"
    hybrid  = True        # always row form; blocks packed mask path
    graphs  = None
    option_isolation = False
    # class-level placeholders so hasattr() checks pass before instantiation
    head   = None
    device = None

    def __init__(self, name: str, tok, device, lora=None, revision=None,
                 head_dim=256, lora_targets="all", dtype=torch.float32,
                 weights=None, processor=None):
        """
        name     – local path or Hub id of the Qwen3VL checkpoint
        tok      – tokenizer (from load_tokenizer(name))
        lora     – LoRA rank, or None for no adapter
        weights  – full-weight checkpoint directory (overrides base backbone)
        """
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        attn  = "eager"   # sdpa on CUDA is fine too but eager is always safe
        load_kw = {"torch_dtype": dtype, "attn_implementation": attn,
                   "local_files_only": True}
        src = weights if weights else name
        vl_model = Qwen3VLForConditionalGeneration.from_pretrained(src, **load_kw)
        # .model is Qwen3VLModel — exposes the full VL forward (vision + text)
        self.lm = vl_model.model

        self.pad_id = pad_id(tok)
        self._image_pad_id     = tok.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
        self._vision_start_id  = tok.convert_tokens_to_ids(_VISION_START_TOKEN)
        self._vision_end_id    = tok.convert_tokens_to_ids(_VISION_END_TOKEN)

        if lora:
            from peft import LoraConfig, get_peft_model
            targets = {
                "all":  ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "qv":   ["q_proj", "v_proj"],
            }[lora_targets]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION",
                             r=lora, lora_alpha=2 * lora, lora_dropout=0.05,
                             target_modules=targets)
            self.lm = get_peft_model(self.lm, cfg)

        self.head   = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    @property
    def prefix_min_tokens(self):
        return 0   # always cache (row form, like hybrid backbone)

    @property
    def dtype(self):
        return str(next(self.lm.parameters()).dtype).removeprefix("torch.")

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(self, tok, rec, **kw):
        """Text-only encode (no image).  For vision records use encode_vision()."""
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    # ── forward / hidden states ──────────────────────────────────────────────

    SHAPE_BUCKET = int(os.environ.get("KEV_SHAPE_BUCKET", "64"))

    def _pad_rows(self, rows):
        """Pad (ids, pos) rows into [N, L] tensors + attention mask."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training:
            L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id,  device=self.device)
        pos  = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att  = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, :len(rid)]  = torch.tensor(rid,  device=self.device)
            pos[i, :len(rpos)] = torch.tensor(rpos, device=self.device)
            att[i, :len(rid)]  = 1
        return ids, pos, att

    def _lm_forward(self, ids, att, pixel_values=None, image_grid_thw=None):
        """One forward through Qwen3VLModel.  Passes pixel_values/image_grid_thw
        and lets the model compute 3D mrope position_ids via compute_3d_position_ids
        (position_ids=None).  mm_token_type_ids is built from the ids tensor."""
        kw = {}
        if pixel_values is not None and image_grid_thw is not None:
            kw["pixel_values"]   = pixel_values.to(self.device,
                                                    next(self.lm.parameters()).dtype)
            kw["image_grid_thw"] = image_grid_thw.to(self.device)
            # mm_token_type_ids is required by compute_3d_position_ids when
            # image_grid_thw is present
            ids_list = [ids[b, :att[b].sum().item()].tolist()
                        for b in range(ids.shape[0])]
            kw["mm_token_type_ids"] = _mm_token_type_ids(
                ids_list, self._image_pad_id, self.device)
        return self.lm(input_ids=ids, attention_mask=att,
                       position_ids=None, **kw).last_hidden_state.float()

    def _rows_hidden(self, rows, cache=None, prefix_len=0,
                     pixel_values=None, image_grid_thw=None):
        """Hidden states for causal rows, one [L_i, d] tensor per row.

        pixel_values / image_grid_thw are passed on the first chunk only (they
        belong to the state; subsequent branch-only chunks inherit from the KV
        cache).
        """
        chunk = (len(rows) if self.training
                 else rows_per_pass([ids for ids, _ in rows], prefix_len))
        out = []
        first_chunk = True
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            past = {}
            if cache is not None:
                replica = copy.copy(cache)
                replica.layers = [copy.copy(layer) for layer in cache.layers]
                for source, target in zip(cache.layers, replica.layers):
                    if isinstance(source, LinearAttentionCacheLayerMixin):
                        target.conv_states = source.conv_states.copy()
                        target.recurrent_states = source.recurrent_states.copy()
                        target.is_conv_states_initialized = source.is_conv_states_initialized.copy()
                        target.is_recurrent_states_initialized = source.is_recurrent_states_initialized.copy()
                        target.has_previous_state = source.has_previous_state.copy()
                        target.conv_kernel_size = source.conv_kernel_size.copy()
                replica.reorder_cache(
                    torch.zeros(len(part), dtype=torch.long, device=self.device))
                att = torch.cat([
                    torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device),
                    att], 1)
                past = {"past_key_values": replica, "use_cache": True}
            # only the first chunk carries pixel_values (state row)
            pv  = pixel_values   if (first_chunk and cache is None) else None
            thw = image_grid_thw if (first_chunk and cache is None) else None
            lm_out = self.lm(
                input_ids=ids, attention_mask=att, position_ids=None,
                pixel_values=(pv.to(self.device, next(self.lm.parameters()).dtype)
                              if pv is not None else None),
                image_grid_thw=(thw.to(self.device) if thw is not None else None),
                mm_token_type_ids=(_mm_token_type_ids(
                    [ids[b, :int(att[b, prefix_len:].sum())].tolist()
                     for b in range(ids.shape[0])],
                    self._image_pad_id, self.device)
                    if pv is not None else None),
                **past,
            )
            h = lm_out.last_hidden_state.float()
            out += [h[i, :len(row_ids)] for i, (row_ids, _) in enumerate(part)]
            first_chunk = False
        return out

    def rows_form(self, encs):
        return True   # always row form for vision

    def forward_rows_batch(self, encs):
        """Row form: each question = state+branch as one causal sequence."""
        rows, readouts = [], []
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            pv  = e.get("pixel_values")
            thw = e.get("image_grid_thw")
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"]))
                readouts.append((b, len(S) + r["decide"],
                                 [len(S) + o for o in r["opts"]],
                                 pv, thw))
        out = [[] for _ in encs]
        chunk = (len(rows) if self.training
                 else rows_per_pass([ids for ids, _ in rows]))
        for start in range(0, len(rows), chunk):
            part_rows = rows[start:start + chunk]
            part_ro   = readouts[start:start + chunk]
            # only pass pixel_values for rows that carry them (state rows)
            pv_list = [ro[3] for ro in part_ro]
            thw_list = [ro[4] for ro in part_ro]
            # if any row in this chunk has pixel_values, pass the first one
            # (training batches a single record; evaluation uses rows_per_pass=1
            # for very long states so each chunk is one record)
            pv  = pv_list[0]  if pv_list[0]  is not None else None
            thw = thw_list[0] if thw_list[0] is not None else None
            ids, _, att = self._pad_rows(part_rows)
            lm_out = self.lm(
                input_ids=ids, attention_mask=att, position_ids=None,
                pixel_values=(pv.to(self.device, next(self.lm.parameters()).dtype)
                              if pv is not None else None),
                image_grid_thw=(thw.to(self.device) if thw is not None else None),
                mm_token_type_ids=(_mm_token_type_ids(
                    [ids[b, :int(att[b].sum())].tolist()
                     for b in range(ids.shape[0])],
                    self._image_pad_id, self.device)
                    if pv is not None else None),
            )
            hs = lm_out.last_hidden_state.float()
            for i, (ro, (row_ids, _)) in enumerate(zip(part_ro, part_rows)):
                b, d, oi = ro[0], ro[1], ro[2]
                h = hs[i, :len(row_ids)]
                out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs, shared_prefix=False):
        return self.forward_rows_batch(encs)

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)])
                for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    # ── inference (scoring interface) ────────────────────────────────────────

    @torch.no_grad()
    def probs(self, enc):
        return self.probs_and_prefix(enc)[0]

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens (image + text) once; return (n_state, cache, None)."""
        Ls = enc["seg"].count(0)
        state_ids = enc["ids"][:Ls]
        ids = torch.tensor([state_ids], device=self.device)
        att = torch.ones((1, Ls), dtype=torch.long, device=self.device)
        pv  = enc.get("pixel_values")
        thw = enc.get("image_grid_thw")
        out = self.lm(
            input_ids=ids, attention_mask=att, position_ids=None,
            pixel_values=(pv.to(self.device, next(self.lm.parameters()).dtype)
                          if pv is not None else None),
            image_grid_thw=(thw.to(self.device) if thw is not None else None),
            mm_token_type_ids=(_mm_token_type_ids(
                [state_ids], self._image_pad_id, self.device)
                if pv is not None else None),
            past_key_values=DynamicCache(config=self.lm.config),
            use_cache=True,
        )
        return Ls, out.past_key_values, None   # hidden states not cached (hybrid path)

    def _branch_rows_from_prefix(self, enc, cache):
        """Branch-only rows continuing from the cached state (no pixel_values)."""
        S, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows],
                               cache=cache, prefix_len=len(S))
        ps = [F.softmax(self.head(h[r["decide"]],
                                  h[torch.tensor(r["opts"], device=self.device)]), -1)
              for h, r in zip(hs, rows)]
        return list(torch.cat(ps).cpu().split([len(p) for p in ps]))

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        Ls, cache, _ = self.prefix(enc)
        ps = self._branch_rows_from_prefix(enc, cache)
        return ps, (Ls, cache, None)

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        Ls, cache, _ = prefix
        if enc["seg"].count(0) != Ls:
            raise ValueError("prefix does not match this record's state")
        return self._branch_rows_from_prefix(enc, cache)

    @torch.no_grad()
    def probs_batch(self, encs, prefixes, keep):
        """Serial scoring (no CUDA graphs for the vision model)."""
        from .model import probs_one
        results = [probs_one(self, enc, pre, k)
                   for enc, pre, k in zip(encs, prefixes, keep)]
        return [r[0] for r in results], [r[1] for r in results]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# Verify the scoring interface is fully satisfied
assert all(hasattr(VisionDecisionModel, m) for m in SCORING_INTERFACE), \
    f"missing SCORING_INTERFACE methods: {[m for m in SCORING_INTERFACE if not hasattr(VisionDecisionModel, m)]}"
