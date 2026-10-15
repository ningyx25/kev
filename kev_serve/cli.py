"""CLI: python -m kev_serve --run name=ckpt [--run name2=ckpt2 ...] --host --port --device.

Each --run names one model; several --runs load several checkpoints into one process and the
request's `model` field routes between them (ModelRouter). `--default` picks the fallback name."""
import argparse

from .backends import load_backend
from .config import Settings
from .device import DEVICES, DEVICE_HELP, select
from .http_app import create_app
from .router import ModelRouter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="name=ckpt",
                    help="a model name and its checkpoint directory; repeat for several models")
    ap.add_argument("--default", default=None, help="model name unknown `model` fields fall back to (default: the first --run)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    a = ap.parse_args()

    settings = Settings.from_env()
    import torch
    dev = select(a.device, dtype=torch.bfloat16)
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32

    backends = {}
    for spec in a.run:
        name, _, path = spec.partition("=")
        if not name or not path: raise SystemExit(f"--run expects name=checkpoint, got {spec!r}")
        print(f"loading {name} <- {path} on {dev} ({dtype})")
        backends[name] = load_backend(path, dev, dtype)
    settings.models = {name: b.requested for name, b in backends.items()}
    settings.default_model = a.default or next(iter(backends))

    app = create_app(settings, ModelRouter(backends, settings.default_model))
    print(f"serving {', '.join(backends)} (default {settings.default_model}) on {dev} {a.host}:{a.port}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)
