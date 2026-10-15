"""ClefDecisionModel: JointSchemaHead on a Qwen3.5 VL backbone, compatible with kev.train.

The head architecture is vendored from Cloudflare/clef-flash (Apache-2.0):
  EvidenceRoutingLayer + JointSchemaHead (two-path scoring: logit-lens prior + gated joint residual).

Differences from PointerHead (kev.model):
  - Encoding uses a chat template (system / STATE / SCHEMA FIELDS / think header) rather than kev's
    packed segment format.  encode() returns an EncodedRecord (not a kev dict).
  - forward_batch() accepts list[EncodedRecord] and returns list[list[Tensor]] — same outer shape as
    DecisionModel.forward_batch(), so kev.train.batch_loss is unchanged.
  - Images are passed as PIL Images inside the record dict (processor handles pixel_values / grid_thw
    natively); there is no separate encode_vision() step.
  - option ordering: choice options are sorted alphabetically by option id (matching clef-flash's
    inference behaviour); label keys in _meta must use the same sort order to line up with logits.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_min import pad_id as _pad_id, load_tokenizer, SCORING_INTERFACE  # noqa: F401 (interface reference)


# ── prompt constants (clef-flash verbatim) ────────────────────────────────────
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER  = "<|vision_start|><|video_pad|><|vision_end|>"
MEDIA_BATCH_KEYS   = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
MEDIA_TOKEN_KEYS   = ("mm_token_type_ids",)
QUESTION_TYPES     = {"noul": 0, "choice": 1, "score": 2}

DEFAULT_HEAD_CONFIG = dict(width=1024, routing_layers=2, layers=4, heads=16, feedforward=4096)

# ── span dataclasses ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EncodedQuestion:
    question_id:   str
    question_type: int
    question_span: tuple[int, int]
    option_spans:  tuple[tuple[int, int], ...]
    option_ids:    tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids:  tuple[int, ...]
    questions:  tuple[EncodedQuestion, ...]
    record_id:  str
    media:      dict[str, Any] | None = dc_field(default=None, compare=False, repr=False)
    prefix_len: int = 0   # tokens before the schema: system + user header + media + state (see encode_record); the
                          # prefix-cache unit — everything up to here depends only on the state, not the questions


# ── encoding helpers ──────────────────────────────────────────────────────────

def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _toks(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _question_options(question: dict[str, Any],
                      max_option_chars: int | None = None) -> list[tuple[str, Any]]:
    """(option_id, description) pairs in the encoder's order. max_option_chars truncates each
    description: tap_target-style records put raw screen text in an option label, and one such
    30k-char dump makes the schema alone exceed max_length (a 8-option record needed 38k tokens).
    Only 0.05% of options are over 256 chars, so the cap is invisible on normal data."""
    def cut(v: Any) -> Any:
        if max_option_chars and isinstance(v, str) and len(v) > max_option_chars:
            return v[:max_option_chars]
        return v
    qtype = str(question["type"])
    if qtype == "noul":
        defaults = {"true": "The proposition is true or the answer is yes.",
                    "false": "The proposition is false or the answer is no."}
        defaults.update(question.get("criteria") or {})
        return [(k, cut(defaults[k])) for k in ("true", "false")]
    if qtype == "choice":
        return sorted((str(k), cut(v)) for k, v in question["criteria"].items())
    return [(str(i), cut(v)) for i, v in enumerate(question["criteria"])]


def _encode_media(processor: Any, record: dict[str, Any],
                  tokenizer: Any = None) -> tuple[list[int], dict | None]:
    images = list(record.get("images") or [])
    videos = list(record.get("videos") or [])
    if not images and not videos:
        return [], None
    if processor is None:
        raise ValueError("records with images or videos require a processor")

    tok = tokenizer or getattr(processor, "tokenizer", None)
    img_proc = getattr(processor, "image_processor", processor)

    # Try unified AutoProcessor call first
    if tok is not None:
        text = IMAGE_PLACEHOLDER * len(images) + VIDEO_PLACEHOLDER * len(videos) + "\n"
        try:
            encoded = processor(text=[text], images=images or None, videos=videos or None,
                                return_tensors="pt", **(record.get("media_kwargs") or {}))
            media = {k: encoded[k] for k in MEDIA_BATCH_KEYS if k in encoded}
            for k in MEDIA_TOKEN_KEYS:
                if k in encoded:
                    media[k] = encoded[k][0].tolist()
            return encoded["input_ids"][0].tolist(), media
        except (TypeError, ValueError):
            pass  # fall through to split approach

    # Split approach: process images with image processor, build token IDs manually.
    # spatial_merge_size = 2 (same constant as vision_model._MERGE_SIZE)
    proc_out = img_proc(images=images or None, return_tensors="pt",
                        **(record.get("media_kwargs") or {}))
    media = {k: proc_out[k] for k in MEDIA_BATCH_KEYS if k in proc_out}

    if tok is None:
        raise ValueError("_encode_media: no tokenizer available to build media token IDs")

    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ip_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")

    media_ids:   list[int] = []
    mm_type_ids: list[int] = []
    grid_thw = proc_out.get("image_grid_thw")   # [N, 3] tensor
    for i in range(len(images)):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        n = t * (h // 2) * (w // 2)             # spatial_merge_size = 2
        media_ids   += [vs_id] + [ip_id] * n + [ve_id]
        mm_type_ids += [0]     + [1]     * n + [0]
    # trailing newline
    nl = tok("\n", add_special_tokens=False).input_ids
    media_ids   += nl
    mm_type_ids += [0] * len(nl)

    if any(mm_type_ids):
        media["mm_token_type_ids"] = mm_type_ids
    return media_ids, media


def encode_record(tokenizer: Any, record: dict[str, Any],
                  max_length: int = 16384, max_state_tokens: int | None = None,
                  processor: Any | None = None,
                  max_option_chars: int | None = 256) -> EncodedRecord:
    """Encode one labelled request as a clef prompt with span offsets for the head.

    max_option_chars truncates each option description (see _question_options): the schema is kept
    whole and the state right-truncated, so an untruncated pathological description would raise
    (the schema alone would not fit)."""
    schema_ids: list[int] = _toks(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for q_idx, (qid, question) in enumerate(record["questions"].items()):
        schema_ids.extend(_toks(tokenizer,
            f"\nFIELD {q_idx + 1}\nID: {qid}\nTYPE: {question['type']}\nINSTRUCTION: "))
        q_start = len(schema_ids)
        instr = question.get("instructions") or str(qid)
        schema_ids.extend(_toks(tokenizer, _render(instr)))
        q_end = len(schema_ids)
        schema_ids.extend(_toks(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans: list[tuple[int, int]] = []
        option_ids:   list[str]             = []
        for o_idx, (oid, desc) in enumerate(_question_options(question, max_option_chars)):
            schema_ids.extend(_toks(tokenizer, f"OPTION {o_idx + 1}: "))
            o_start = len(schema_ids)
            semantics: dict[str, Any] = {"option_id": oid}
            if desc is not None:
                semantics["description"] = desc
            schema_ids.extend(_toks(tokenizer, _render(semantics)))
            option_spans.append((o_start, len(schema_ids)))
            option_ids.append(oid)
            schema_ids.extend(_toks(tokenizer, "\n"))
        schema_ids.extend(_toks(tokenizer, "END FIELD\n"))
        questions.append(EncodedQuestion(
            question_id=str(qid), question_type=QUESTION_TYPES[str(question["type"])],
            question_span=(q_start, q_end),
            option_spans=tuple(option_spans), option_ids=tuple(option_ids),
        ))
    prefix_ids = _toks(tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix_ids = _toks(tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    media_ids, media = _encode_media(processor, record, tokenizer=tokenizer)
    if media is not None:
        media["token_offset"] = len(prefix_ids)
        prefix_ids = prefix_ids + media_ids
    state_ids = _toks(tokenizer, _render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens (max_length={max_length})")
    state_ids = state_ids[:max_length - fixed]
    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id=q.question_id, question_type=q.question_type,
            question_span=(q.question_span[0] + offset, q.question_span[1] + offset),
            option_spans=tuple((s + offset, e + offset) for s, e in q.option_spans),
            option_ids=q.option_ids,
        )
        for q in questions
    )
    return EncodedRecord(
        input_ids=tuple(prefix_ids + state_ids + schema_ids + suffix_ids),
        questions=shifted,
        record_id=str(record.get("id", "unknown")),
        media=media,
        prefix_len=len(prefix_ids) + len(state_ids),
    )


def collate_records(records: list[EncodedRecord], pad_token_id: int,
                    device: torch.device) -> dict[str, Any]:
    max_len = max(len(r.input_ids) for r in records)
    input_ids      = torch.full((len(records), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
    for i, rec in enumerate(records):
        n = len(rec.input_ids)
        input_ids[i, :n]      = torch.tensor(rec.input_ids, device=device)
        attention_mask[i, :n] = 1
    media: dict[str, torch.Tensor] = {}
    for k in MEDIA_BATCH_KEYS:
        vals = [r.media[k] for r in records if r.media and k in r.media]
        if vals:
            media[k] = torch.cat(vals, dim=0).to(device)
    for k in MEDIA_TOKEN_KEYS:
        if any(r.media and k in r.media for r in records):
            tv = torch.zeros((len(records), max_len), dtype=torch.long, device=device)
            for i, r in enumerate(records):
                if r.media and k in r.media:
                    off = r.media["token_offset"]
                    v   = torch.tensor(r.media[k], dtype=torch.long, device=device)
                    tv[i, off:off + len(v)] = v
            media[k] = tv
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "records": records, "media": media}


# ── head modules (vendored from Cloudflare/clef-flash, Apache-2.0) ────────────

class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm  = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention   = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm  = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        nq = self.query_norm(queries)
        routed, _ = self.attention(nq, self.memory_norm(memory), self.memory_norm(memory),
                                   need_weights=False)
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size: int, width: int, routing_layers: int,
                 layers: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_norm              = nn.LayerNorm(hidden_size)
        self.memory_projection        = nn.Linear(hidden_size, width, bias=False)
        self.question_projection      = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection        = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding           = nn.Embedding(3, width)
        self.evidence_layers          = nn.ModuleList([
            EvidenceRoutingLayer(width=width, heads=heads, feedforward=feedforward, dropout=dropout)
            for _ in range(routing_layers)
        ])
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.field_norm        = nn.LayerNorm(width)
        self.option_norm       = nn.LayerNorm(width)
        self.residual_scorer   = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )
        self.prior_logit_scale  = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale  = nn.Parameter(torch.zeros(()))
        self.residual_gate      = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(v: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        return v[span[0]:span[1]].mean(dim=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, records: list[EncodedRecord],
                output_embedding_weight: torch.Tensor) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        nh = self.hidden_norm(hidden_states)
        for bi, record in enumerate(records):
            seq_len  = int(attention_mask[bi].sum().item())
            seq_h    = nh[bi, :seq_len]
            memory   = self.memory_projection(seq_h).unsqueeze(0)
            global_v = seq_h[-1]
            q_vecs   = torch.stack([self._mean_span(seq_h, q.question_span) for q in record.questions])
            type_ids = torch.tensor([q.question_type for q in record.questions], device=hidden_states.device)
            opt_contexts: list[torch.Tensor] = []
            lexical_opts: list[torch.Tensor] = []
            opt_counts = []
            for q in record.questions:
                ctx = torch.stack([self._mean_span(seq_h, s) for s in q.option_spans])
                lex = torch.stack([
                    output_embedding_weight[input_ids[bi, s:e]].mean(dim=0)
                    for s, e in q.option_spans
                ])
                opt_contexts.append(ctx); lexical_opts.append(lex)
                opt_counts.append(len(q.option_spans))
            opt_queries = []
            for qi, (ctx, lex) in enumerate(zip(opt_contexts, lexical_opts)):
                opt_queries.append(
                    self.option_context_projection(ctx)
                    + self.option_lexical_projection(lex)
                    + self.option_question_projection(q_vecs[qi]).unsqueeze(0)
                )
            routed = torch.cat(opt_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            routed = routed[0]
            split  = list(torch.split(routed, opt_counts, dim=0))
            base_fields = self.question_projection(q_vecs)
            summaries = []
            for f, opts in zip(base_fields, split):
                w = torch.softmax(torch.matmul(opts, f) / math.sqrt(opts.shape[-1]), dim=0)
                summaries.append(torch.sum(w.unsqueeze(-1) * opts, dim=0))
            fields = (base_fields + self.option_summary_norm(torch.stack(summaries))
                      + self.global_projection(global_v).unsqueeze(0)
                      + self.type_embedding(type_ids))
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])
            rec_logits: list[torch.Tensor] = []
            for field, q, lex, rot in zip(fields, record.questions, lexical_opts, split):
                anchor  = F.normalize(q_vecs[len(rec_logits)] + global_v, dim=-1)
                lex_anc = F.normalize(lex, dim=-1)
                ps = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior   = ps * torch.matmul(lex_anc, anchor)
                opts    = self.option_norm(rot)
                rep_f   = field.unsqueeze(0).expand_as(opts)
                cosine  = F.cosine_similarity(rep_f, opts, dim=-1)
                feats   = torch.cat([rep_f, opts, rep_f * opts, torch.abs(rep_f - opts)], dim=-1)
                residual = self.residual_scorer(feats).squeeze(-1)
                js  = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = js * cosine + residual
                rec_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            results.append(rec_logits)
        return results


# ── ClefDecisionModel ─────────────────────────────────────────────────────────

_INSTANCE_ATTRS = {"head", "device"}


class ClefDecisionModel(nn.Module):
    """kev.train-compatible decision model with JointSchemaHead.

    encode() returns EncodedRecord; forward_batch() accepts list[EncodedRecord].
    The outer shape of forward_batch's return is list[list[Tensor]], identical to
    DecisionModel.forward_batch(), so kev.train.batch_loss needs no changes.
    """

    backend          = "torch"
    option_isolation = False
    prefix_min_tokens = 0
    hybrid           = True   # Qwen3.5 is always hybrid (GatedDeltaNet)

    def __init__(self, name: str, tok, device, lora: int | None = None,
                 head_config: dict | None = None, dtype=torch.float32,
                 direct_load: bool = False, weights: str | None = None,
                 processor=None, max_option_chars: int | None = 256):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        cfg = AutoConfig.from_pretrained(weights or name, local_files_only=True)
        if cfg.model_type != "qwen3_5":
            raise ValueError(
                f"ClefDecisionModel requires model_type 'qwen3_5', got {cfg.model_type!r}"
            )
        load_kw: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
        if direct_load:
            load_kw["device_map"] = {"": torch.cuda.current_device() if device == "cuda" else device}
        self.lm = Qwen3_5ForConditionalGeneration.from_pretrained(weights or name, **load_kw)
        self.lm.config.use_cache = False
        hidden_size = self.lm.config.text_config.hidden_size

        # LoRA on the full VL model (backbone + vision encoder)
        self._lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            cfg_lora = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora,
                                  lora_alpha=2 * lora, lora_dropout=0.05,
                                  target_modules=targets)
            self.lm = get_peft_model(self.lm, cfg_lora)

        hcfg = {**DEFAULT_HEAD_CONFIG, **(head_config or {})}
        self.head    = JointSchemaHead(hidden_size=hidden_size, **hcfg)
        self._hcfg   = hcfg
        self.device  = device
        self.hybrid  = True   # Qwen3.5 is always hybrid (GatedDeltaNet)
        self._pad_id = _pad_id(tok)
        self.max_option_chars = max_option_chars   # cap on option-description length (see encode_record)
        # always use AutoProcessor (handles text + image); the caller may pass an image-only
        # Qwen2VLImageProcessor which lacks the tokenizer half needed by _encode_media
        if processor is not None and hasattr(processor, "tokenizer"):
            self._processor = processor   # already a full processor
        else:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(weights or name, local_files_only=True)
        self.to(device)

    # ── interface ─────────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, tok, rec: dict, max_state: int = 8192, max_branch: int = 0,
               strict: bool = False, option_isolation: bool = False,
               processor=None, image_paths: list[str] | None = None,
               max_pixels_caps=None, max_option_chars: int | None = None) -> EncodedRecord:
        """Encode a kev internal record as an EncodedRecord for forward_batch.

        image_paths: list of absolute paths (history-then-current order, may be empty).
        max_pixels_caps: int or list[int] per image — pixels cap before PIL resize.
        max_option_chars: option-description cap (None = the model's self.max_option_chars)."""
        if max_option_chars is None:
            max_option_chars = self.max_option_chars
        from PIL import Image as PILImage

        proc = processor or self._processor
        imgs: list = []
        if image_paths:
            if isinstance(max_pixels_caps, (int, float)) or max_pixels_caps is None:
                caps = [max_pixels_caps] * len(image_paths)
            else:
                caps = list(max_pixels_caps)
                if len(caps) == 1:
                    caps = caps * len(image_paths)
            for path, cap in zip(image_paths, caps):
                img = PILImage.open(path).convert("RGB")
                if cap and img.width * img.height > cap:
                    scale = (cap / (img.width * img.height)) ** 0.5
                    img = img.resize((max(1, int(img.width * scale)),
                                      max(1, int(img.height * scale))), PILImage.LANCZOS)
                imgs.append(img)

        # Build a minimal record dict for encode_record.
        # Accept either a labelled request (questions as a dict {qid: {type, instructions, criteria, ...}})
        # or a kev internal record (questions as a list [{qtype, instr, options, qid, ...}]).
        raw_qs = rec.get("questions", {})
        if isinstance(raw_qs, dict):
            # labelled request format — pass through directly (criteria is already a dict)
            questions_dict = {qid: {"type": q["type"],
                                    "instructions": q.get("instructions"),
                                    "criteria": q.get("criteria", {})}
                              for qid, q in raw_qs.items()}
        else:
            # kev internal format — convert from list with qtype/instr/options
            questions_dict = {q["qid"]: {"type": q["qtype"],
                                          "instructions": q.get("instr"),
                                          "criteria": q.get("criteria", {})}
                              for q in raw_qs}
        rec_dict = {
            "id":        rec.get("_meta", {}).get("id", "unknown"),
            "state":     rec["state"],
            "questions": questions_dict,
        }
        if imgs:
            rec_dict["images"] = imgs

        return encode_record(tok, rec_dict, max_length=max_state + 4096,
                             max_state_tokens=max_state, processor=proc,
                             max_option_chars=max_option_chars)

    def forward_batch(self, encs: list[EncodedRecord],
                      shared_prefix: bool = False) -> list[list[torch.Tensor]]:
        """Batch forward: list[EncodedRecord] → list[list[Tensor]] (per record, per question logits)."""
        dev = next(self.parameters()).device
        batch = collate_records(encs, self._pad_id, dev)
        # resolve base model and embeddings through PEFT wrapper if present
        lm = self.lm
        base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        media = batch.get("media") or {}
        text_model = base.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        outputs = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        return self.head(
            outputs.last_hidden_state,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            base.get_output_embeddings().weight,
        )

    # ── prefix caching (kev_serve's QueueBatchEngine) ───────────────────────────────────────
    # The cached unit is the record's prefix: system + user header + state (prefix_len tokens), everything
    # the questions do not touch. The head cross-attends over the WHOLE prompt's hidden states, so a
    # prefix is (past_key_values, prefix hidden states); a continuation re-attaches them — exact, because
    # causality says a prefix token's hidden states do not depend on the schema that follows. Text-only
    # records cache (a media record's 3D mrope positions are computed over the whole prompt by the VL
    # wrapper and do not slice cleanly); media records run the one-pass forward_batch.

    def _base(self):
        lm = self.lm
        return lm.get_base_model() if hasattr(lm, "get_base_model") else lm

    def _text_model(self):
        model = self._base().model
        return model.language_model if hasattr(model, "language_model") else model

    @torch.inference_mode()
    def state_pass(self, enc: EncodedRecord):
        """Run the record's prefix tokens only. -> (cache kept exactly at the prefix, prefix hidden states)."""
        from transformers import DynamicCache
        ids = torch.tensor([list(enc.input_ids[:enc.prefix_len])], device=next(self.parameters()).device)
        out = self._text_model()(input_ids=ids, use_cache=True, return_dict=True)
        return out.past_key_values, out.last_hidden_state[0]

    @staticmethod
    def _replicate(cache):
        """A copy of the prefix cache whose tensor state the continuation may consume: the stored prefix
        must stay pristine for the next request (kev's old rows_hidden replicated for the same reason).
        Shallow copies down to the tensors — but the hybrid layers keep their state in *dicts*
        (conv_states / recurrent_states indexed by slot), and those containers must be rebuilt or the
        clone's in-place writes (copy_, update_recurrent_state) land on the stored prefix."""
        import copy as _copy
        def cloned(v):
            if isinstance(v, torch.Tensor): return v.clone()
            if isinstance(v, dict): return {k: cloned(x) for k, x in v.items()}
            if isinstance(v, list): return [cloned(x) for x in v]
            if isinstance(v, tuple): return tuple(cloned(x) for x in v)
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
    def forward_continuation(self, enc: EncodedRecord, prefix) -> list[torch.Tensor]:
        """Per-question logits for a record whose prefix tokens equal the cached prefix's; only the
        schema runs, and the prefix hidden states are re-attached so the head sees the same full-prompt
        layout as in a one-pass forward (spans keep their absolute coordinates). The prefix cache is
        consumed through a replica: callers pass the stored prefix itself and it must not change."""
        cache, h_prefix = prefix
        p = enc.prefix_len
        dev = h_prefix.device
        new_ids = torch.tensor([list(enc.input_ids[p:])], device=dev)
        full_len = len(enc.input_ids)
        mask = torch.ones((1, full_len), dtype=torch.long, device=dev)   # the full causal context, no padding
        out = self._text_model()(input_ids=new_ids, attention_mask=mask, past_key_values=self._replicate(cache),
                                 use_cache=True, return_dict=True)
        hidden = torch.cat([h_prefix.unsqueeze(0), out.last_hidden_state], 1)
        return self.head(hidden, torch.tensor([list(enc.input_ids)], device=dev), mask, [enc],
                         self._base().get_output_embeddings().weight)[0]

    def forward(self, enc: EncodedRecord) -> list[torch.Tensor]:
        return self.forward_batch([enc])[0]

    @staticmethod
    def _softmax(logits: torch.Tensor, temperature: float) -> list[float]:
        """Probabilities at a calibration temperature: logits / T before softmax (the TypeSafe
        KEV_TEMPERATURE semantics — a scalar on the logits, not a post-hoc reshape of p)."""
        z = logits.float()
        return (z / temperature if temperature != 1.0 else z).softmax(-1).tolist()

    @torch.inference_mode()
    def probs(self, enc: EncodedRecord, temperature: float = 1.0) -> list[list[float]]:
        return [self._softmax(z, temperature) for z in self.forward(enc)]

    @torch.inference_mode()
    def probs_batch(self, encs: list[EncodedRecord], temperature: float = 1.0) -> list[list[list[float]]]:
        return [[self._softmax(z, temperature) for z in logits] for logits in self.forward_batch(encs)]

    @torch.inference_mode()
    def probs_with_prefix(self, enc: EncodedRecord, prefix, temperature: float = 1.0) -> list[list[float]]:
        """probs() for a record whose prefix tokens equal the cached prefix's (prefix is the
        (cache, hidden) pair state_pass returned); only the schema runs — forward_continuation
        replicates the cache itself, so a stored prefix survives any number of reuses."""
        if enc.prefix_len == 0: raise ValueError("prefix does not match this record (it has no prefix)")
        return [self._softmax(z, temperature) for z in self.forward_continuation(enc, prefix)]

    @torch.inference_mode()
    def probs_and_prefix(self, enc: EncodedRecord, temperature: float = 1.0):
        """One record's probabilities plus its fresh prefix (state_pass + continuation: the miss path
        the engine takes when keep[i] says the prefix will outlive the batch)."""
        if enc.prefix_len == 0: return self.probs(enc, temperature), None
        prefix = self.state_pass(enc)
        logits = self.forward_continuation(enc, prefix)
        return [self._softmax(z, temperature) for z in logits], prefix

    graphs           = None
    dtype            = property(lambda self: next(self.parameters()).dtype)


# make sure the class satisfies the interface kev code checks for
assert all(
    hasattr(ClefDecisionModel, a) for a in SCORING_INTERFACE if a not in _INSTANCE_ATTRS
), "ClefDecisionModel is missing SCORING_INTERFACE attributes"


# ── checkpoint helpers ────────────────────────────────────────────────────────

def is_clef_checkpoint(path) -> bool:
    """True for a clef-flash style checkpoint directory (has joint_head_config.json)."""
    p = Path(path)
    return (p / "joint_head_config.json").exists() and (p / "joint_head.safetensors").exists()


def save_clef_head(model: ClefDecisionModel, out_dir: str):
    """Save JointSchemaHead in clef-flash checkpoint format."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    state = {k: v.contiguous() for k, v in model.head.state_dict().items()}
    save_file(state, str(out / "joint_head.safetensors"))
    (out / "joint_head_config.json").write_text(
        json.dumps({**model._hcfg, "hidden_size": model.head.hidden_norm.normalized_shape[0]},
                   indent=2)
    )


def load_clef_head(model: ClefDecisionModel, path: str):
    """Load JointSchemaHead weights from a clef-flash checkpoint directory."""
    from safetensors.torch import load_file
    state = load_file(str(Path(path) / "joint_head.safetensors"), device="cpu")
    model.head.load_state_dict(state, strict=True)


# ── serve / eval convenience wrappers ────────────────────────────────────────

# Alias used by kev.serve (which imports `is_clef`).
is_clef = is_clef_checkpoint


def load_clef_checkpoint(
    path: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    fused: bool | None = None,
) -> tuple[ClefDecisionModel, Any]:
    """Load a clef-flash checkpoint directory -> (ClefDecisionModel eval, processor).

    The backbone (Qwen3_5ForConditionalGeneration) is loaded from the checkpoint directory
    itself (full weights); the JointSchemaHead is loaded via load_clef_head().

    `fused` (KEV_FUSED): True applies the fused Triton layers to the Qwen3.5 text backbone and fails
    loudly if they cannot run; None ("auto") applies them on CUDA when flash-linear-attention is
    importable at the pinned version and silently keeps eager otherwise; False declines."""
    from transformers import AutoProcessor
    p = Path(path)
    head_cfg = json.loads((p / "joint_head_config.json").read_text())
    head_cfg.pop("hidden_size", None)   # stored in config file but ClefDecisionModel reads it from backbone
    tok = load_tokenizer(str(p))
    processor = AutoProcessor.from_pretrained(str(p), local_files_only=True)
    model = ClefDecisionModel(str(p), tok, device, head_config=head_cfg,
                              dtype=dtype, direct_load=True, weights=str(p),
                              processor=processor)
    load_clef_head(model, str(p))
    model.head.to(device=device, dtype=dtype)   # safetensors loads as fp32; cast to match backbone
    if fused or (fused is None and str(device).startswith("cuda")):
        from .fused_layers import apply_fused, fused_available
        if fused or fused_available():
            try:
                apply_fused(model.lm)
                model._fused = True
                print("fused Qwen3.5 kernels on")
            except NotImplementedError as e:   # MoE MLPs: fused kernels refuse them; eager keeps serving
                if fused: raise
                print(f"fused Qwen3.5 kernels off: {e}")
        elif fused:
            raise RuntimeError("KEV_FUSED=1 but flash-linear-attention at the pinned FLA_VERSION is not importable")
        else:
            print("fused Qwen3.5 kernels off: install flash-linear-attention at the pinned FLA_VERSION to turn them on")
    return model.eval(), processor


def _systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """Convert per-option probabilities for one question into a TypeSafe answer dict."""
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "noul", "noul": round(probabilities.get("true", 0.0), 4)}
    if qtype == "choice":
        options = [str(o) for o in question["criteria"]]
        choice = max(options, key=lambda o: probabilities.get(o, 0.0))
        return {"type": "choice", "choice": choice,
                "confidence": round(probabilities.get(choice, 0.0), 4),
                "probabilities": {o: round(probabilities.get(o, 0.0), 4) for o in options}}
    # score
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score",
            "score": round(sum(i * probabilities.get(lv, 0.0) for i, lv in enumerate(levels)), 4),
            "confidence": round(max(probabilities.get(lv, 0.0) for lv in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {lv: round(probabilities.get(lv, 0.0), 4) for lv in levels}}


@torch.no_grad()
def systemone(model: ClefDecisionModel, processor: Any, request: dict[str, Any],
              max_length: int = 16384, temperature: float = 1.0) -> dict[str, Any]:
    """Answer a TypeSafe /v1/systemone request dict with a TypeSafe response dict.

    Images in request['images'] must be PIL Images (already decoded from base64 by the HTTP layer).
    `temperature` divides the head's logits before the softmax (the calibration scalar; 1.0 = raw).
    """
    questions = request.get("questions")
    r: dict[str, Any] = {"state": request["state"], "questions": questions}
    imgs = request.get("images")
    if imgs:
        r["images"] = imgs
    tok = processor.tokenizer
    enc = encode_record(tok, r, max_length=max_length, processor=processor if imgs else None)
    logits = model.forward_batch([enc])[0]
    answers = {
        q.question_id: _systemone_answer(
            questions[q.question_id],
            dict(zip(q.option_ids, model._softmax(ql, temperature))),
        )
        for q, ql in zip(enc.questions, logits)
    }
    return {"model": request.get("model", "clef"), "answers": answers,
            "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}
