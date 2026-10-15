"""Backend registry. Importing a backend package registers it; third parties can also ship the
"kev_serve.backends" entry-point group (base.discover_entry_points)."""
from .base import Backend, discover_entry_points, load_backend, register_backend

# built-in backends: import = register
from . import clef      # noqa: F401  (clef: noul / choice / score)
from . import gui_actor # noqa: F401  (gui_actor: ground)

__all__ = ["Backend", "discover_entry_points", "load_backend", "register_backend"]
