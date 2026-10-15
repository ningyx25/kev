"""clef backend: JointSchemaHead on Qwen3.5 VL (vendored clef_model.py). noul / choice / score."""
from .backend import ClefBackend
from ..base import register_backend
from .clef_model import is_clef, load_clef_checkpoint


def _factory(path: str, device: str, dtype):
    model, processor = load_clef_checkpoint(path, device, dtype)
    return ClefBackend(model, processor, path, device)


register_backend("clef", is_clef, _factory)
