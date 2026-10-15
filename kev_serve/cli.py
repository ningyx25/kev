"""CLI: python -m kev_serve --run name=ckpt[@gpu] [--run name2=ckpt2[@gpu] ...] --host --port --device.

Each --run names one model; several --runs load several checkpoints into one process and the
request's `model` field routes between them (ModelRouter). `--default` picks the fallback name.

`@gpu` after a checkpoint puts that model on one GPU of a multi-GPU box (`--run main=clef@2`
`--run grounder=gui@3`); without it, every model uses `--device`."""
import argparse

import torch

from .backends import load_backend
from .config import Settings
from .device import DEVICE_HELP, _cuda_index, select
from .http_app import create_app
from .router import ModelRouter


def parse_run(spec: str) -> tuple[str, str, str | None]:
    """`name=ckpt`, `name=ckpt@N`, or `name=ckpt@cuda:N` -> (name, ckpt_path, device or None).
    A Hub id keeps its own `@revision`: only a trailing `@N` / `@cuda:N` is a GPU."""
    name, _, rest = spec.partition("=")
    if not name or not rest: raise SystemExit(f"--run expects name=checkpoint[@gpu], got {spec!r}")
    path, device = rest, None
    head, sep, tail = rest.rpartition("@")
    if sep and tail.isdigit():
        path, device = head, f"cuda:{tail}"
    elif sep and tail.startswith("cuda:"):
        path, device = head, tail
    if not path: raise SystemExit(f"--run expects name=checkpoint[@gpu], got {spec!r}")
    return name, path, device


def device_for(gpu: str | None, default: str, resolved: dict, set_device=None) -> str:
    """Resolve one model's device string, probing each distinct one once and caching it in `resolved`.

    A `cuda:N` also makes N current (what the loaders' `torch.cuda.current_device()` reads); the cache
    keeps a later model from inheriting an earlier `@N`, and keeps a repeated `cuda`/`auto` from
    re-probing — an explicit accelerator fails closed, so re-probing a busy GPU 0 would abort startup.
    `set_device` is `torch.cuda.set_device` (injected so this is testable without a GPU)."""
    want = gpu or default
    if want not in resolved:
        dev = select(want, dtype=torch.bfloat16)
        index = _cuda_index(want) if want != "auto" else 0   # auto probes GPU 0, so it resolves there
        resolved[want] = (dev, index if dev == "cuda" else None)
    dev, index = resolved[want]
    if dev == "cuda" and index is not None and set_device is not None:
        set_device(index)
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="name=ckpt[@gpu]",
                    help="a model name, its checkpoint directory, and optionally the GPU (name=ckpt@2); repeat per model")
    ap.add_argument("--default", default=None, help="model name unknown `model` fields fall back to (default: the first --run)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="auto", help=DEVICE_HELP)
    a = ap.parse_args()

    settings = Settings.from_env()
    runs = [parse_run(spec) for spec in a.run]
    resolved: dict = {}    # device string -> (family, GPU index); see device_for
    backends = {}
    for name, path, gpu in runs:
        dev = device_for(gpu, a.device, resolved, set_device=torch.cuda.set_device)
        dtype = torch.bfloat16 if dev != "cpu" else torch.float32
        print(f"loading {name} <- {path} on {dev} ({dtype})")
        try:
            backends[name] = load_backend(path, dev, dtype)
        except (ValueError, RuntimeError) as e:   # a config/checkpoint error, not a bug: exit cleanly
            raise SystemExit(f"could not load {name} from {path}: {e}") from None
    settings.models = {name: b.requested for name, b in backends.items()}
    settings.default_model = a.default or next(iter(backends))

    app = create_app(settings, ModelRouter(backends, settings.default_model))
    print(f"serving {', '.join(backends)} (default {settings.default_model}) on {a.host}:{a.port}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)
