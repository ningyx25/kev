"""clef backend: JointSchemaHead on Qwen3.5 VL (vendored clef_model.py). noul / choice / score."""
from .backend import ClefBackend
from ..base import register_backend
from .clef_model import is_clef, load_clef_checkpoint


def _factory(path: str, device: str, dtype, settings=None):
    from ...engine import make_engine
    model, processor = load_clef_checkpoint(path, device, dtype, fused=settings.fused if settings else None)
    b = ClefBackend(model, processor, path, device,
                    temperature=settings.temperature if settings else None)
    b.engine = make_engine(b, settings) if settings is not None else b.engine   # clef batches: settings chooses the engine
    return b


register_backend("clef", is_clef, _factory)
