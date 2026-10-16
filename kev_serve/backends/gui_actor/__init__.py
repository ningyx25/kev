"""GUI-Actor backend: the vendored pointer head on Qwen2.5-VL / Qwen3-VL / Qwen3.5 backbones.

Answers `ground` questions only (one click point in 0-999 + region confidence)."""
from .backend import GuiActorBackend
from ..base import register_backend
from .gui_actor_model import is_gui_actor_checkpoint, load_gui_actor_checkpoint


def _factory(path: str, device: str, dtype, settings=None):
    from ...engine import make_engine
    model, processor, model_type = load_gui_actor_checkpoint(path, device, dtype)
    b = GuiActorBackend(model, processor, path, device, model_type,
                        temperature=settings.temperature if settings else None)
    b.engine = make_engine(b, settings) if settings is not None else b.engine   # caching on: settings picks the engine
    return b


register_backend("gui_actor", is_gui_actor_checkpoint, _factory)
