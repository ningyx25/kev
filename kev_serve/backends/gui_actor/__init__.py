"""GUI-Actor backend: the vendored pointer head on Qwen2.5-VL / Qwen3-VL / Qwen3.5 backbones.

Answers `ground` questions only (one click point in 0-999 + region confidence)."""
from .backend import GuiActorBackend
from ..base import register_backend
from .gui_actor_model import is_gui_actor_checkpoint, load_gui_actor_checkpoint


def _factory(path: str, device: str, dtype):
    model, processor, model_type = load_gui_actor_checkpoint(path, device, dtype)
    return GuiActorBackend(model, processor, path, device, model_type)


register_backend("gui_actor", is_gui_actor_checkpoint, _factory)
