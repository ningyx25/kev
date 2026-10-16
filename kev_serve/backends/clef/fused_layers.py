"""The fused-kernel loading helpers for the clef backend (vendored fused_qwen35 sits beside this).

fused_qwen35 imports `fla` at module level, so anything that must work *without* flash-linear-attention
(the availability probe, the eager fallback, the text-layer locator) lives here and imports the vendored
module lazily — exactly the shape kev/checkpoint.py's fused_available used."""
import torch


def fused_available() -> bool:
    """Whether the fused kernels can run: flash-linear-attention importable at the pinned FLA_VERSION.
    Non-throwing (the KEV_FUSED auto path); an explicit fused=True skips this and fails loudly."""
    try:
        import fla
        from .fused_qwen35 import FLA_VERSION
        return fla.__version__ == FLA_VERSION
    except Exception:
        return False


def text_model(vl_model):
    """The Qwen3.5 text backbone of a VL composite: model = {visual, language_model}, and fuse()
    walks lm.layers — bare `vl.layers` would hit ForConditionalGeneration, which has none (the same
    walk _text_model() in clef_model.py does)."""
    m = getattr(vl_model, "model", vl_model)
    return m.language_model if hasattr(m, "language_model") else m


def apply_fused(vl_model) -> bool:
    """Rewrite the text backbone's decoder layers in place. Idempotent: fuse() deletes `gate_proj`, so
    its own MoE check (`all layers have gate_proj`) would misread an already-fused dense model as MoE —
    a second call is a no-op. True when fused, raises NotImplementedError for a real MoE stack (fuse()'s
    own refusal: those layers have neither gate_proj nor gate_up)."""
    from .fused_qwen35 import fuse
    tm = text_model(vl_model)
    if all(hasattr(layer.mlp, "gate_up") for layer in tm.layers):
        return True     # already fused: fuse() cannot run again (it would re-concat gate_up)
    fuse(tm)
    return True


def is_fused(vl_model) -> bool:
    """Whether the text backbone's layers already carry the fused projections."""
    tm = text_model(vl_model)
    return bool(tm.layers) and all(hasattr(layer.mlp, "gate_up") for layer in tm.layers)


def fused_torch_ok() -> bool:
    """The kernels are CUDA Triton: fp32 on CPU keeps the eager path even when fla is installed."""
    return torch.cuda.is_available()
