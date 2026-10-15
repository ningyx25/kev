"""The three symbols kev_serve.clef_model needs from kev.model, inlined so the package has no
kev dependency (kev/model.py:60-66, kev/model.py:228-229)."""
from transformers import AutoTokenizer


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """The id used to right-pad token rows (never attended to); Qwen tokenizers define one, others fall back to 0."""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "probs_batch", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "option_isolation", "prefix_min_tokens")
