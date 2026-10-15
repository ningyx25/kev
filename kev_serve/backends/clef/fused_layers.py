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


def text_layers(vl_model):
    """The decoder-layer list of a Qwen3.5 *VL* model: fuse() rewrites lm.layers, which for the
    composite lives at model.language_model.layers (or model.layers on a bare-text layout)."""
    m = vl_model.model if hasattr(vl_model, "model") and hasattr(vl_model.model, "layers") else vl_model
    if hasattr(m, "language_model"): m = m.language_model
    return m.layers


def apply_fused(vl_model) -> bool:
    """Rewrite the text backbone's decoder layers in place. True when fused, False when the model
    layout is not a dense Qwen3.5 stack (the MoE refusal) and eager layers were kept."""
    from .fused_qwen35 import fuse
    fuse(text_layers(vl_model))
    return True


def fused_torch_ok() -> bool:
    """The kernels are CUDA Triton: fp32 on CPU keeps the eager path even when fla is installed."""
    return torch.cuda.is_available()
