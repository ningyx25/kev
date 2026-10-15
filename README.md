# kev-serve

Standalone deployment service with a Jev-like (TypeSafe System One) API, extracted from the kev
monorepo onto an orphan branch: no shared history, no `kev` package dependency.

## Primitives

| Type       | Request                                                             | Answer                                                                                                   |
| ---------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `noul`   | `{"type": "noul", "instructions", "criteria"?: {"true","false"}}` | `{"type": "noul", "noul": p(true)}`                                                                    |
| `choice` | `{"type": "choice", "instructions", "criteria": {name: desc}}`    | `{"choice", "confidence", "probabilities"}`                                                            |
| `score`  | `{"type": "score", "instructions", "criteria": [levels]}`         | `{"score", "legend", "probabilities", "confidence"}`                                                   |
| `box`    | `{"type": "box", "instructions"}` (no criteria; needs an image)   | `{"box": {top, bottom, left, right}, "confidence"}` — **501 until a Box backend is registered** |

Confidence formulas mirror TypeSafe's `system-one-adapter 0.2.1`. Schemas: `kev_serve/api.py`.

## Run

```bash
uv sync --extra serve
uv run --extra serve python -m kev_serve --run <clef_checkpoint> --port 8009
```

The checkpoint is a clef-flash directory (`joint_head_config.json` + `joint_head.safetensors` on a
Qwen3.5 VL backbone). Env: `KEV_API_KEY` (bearer auth), `KEV_DATE_FACTS=1`, `KEV_DEVICE_PROBE_TIMEOUT`.

## API

- `POST /v1/systemone` — `{state, model, questions: {id: Question}, assets?, images?}`. Images either
  the Jev way (top-level `assets` map, referenced by key) or legacy base64 `images`; both need the
  `TypeSafe-Danger-Zone-Images: true` header. Box questions answer 501 for now.
- `GET /v1/models` — TypeSafe model cards (`kind`, `supported_primitives`).
- Every response carries `x-typesafe-request-id` and `server-timing`.

Works with the TypeSafe SDKs pointed at the service (`TypeSafeClient(base_url=..., api_key=...)`)
— `jev-latest` and `kev-latest` both name the loaded checkpoint.

## Backends

```
kev_serve/
├── api/            # schema layer (pure pydantic): primitives, request, confidence
├── backends/       # Backend registry: base.py + one package per backend (clef/ today)
├── engine/         # ExecutionEngine: SyncLockEngine today; queue+batch+prefix-cache slot for the kev pointer head
├── middlewares/    # telemetry / auth / images_gate; build_stack assembles them from Settings
├── router.py       # ModelRouter: model name -> backend (explicit / alias / default / 404)
├── http_app.py     # create_app(settings, router): middleware stack + routes
├── config.py       # Settings (KEV_* env)
└── cli.py          # python -m kev_serve --run name=ckpt [--run name2=ckpt2 ...]
```

Today: clef (noul / choice / score). Box is interface-only; the planned backend is GUI-Actor's
per-patch distribution aggregated into a connected region (normalized edges).

**Add a backend**: a package under `kev_serve/backends/` whose `__init__.py` calls
`register_backend(kind, sniff, factory)` with a `Backend` implementing `kind`,
`supported_primitives`, `answer()`, `model_card()`, `close()` (plus `answer_box()` to open box).
Import it in `backends/__init__.py` — or ship the `kev_serve.backends` entry-point group — and the
CLI, router and routes pick it up unchanged.

**Add a middleware**: a module in `kev_serve/middlewares/` exposing
`async def mw(request, call_next) -> Response`, plus one line in `build_stack` (middlewares/base.py).

**Multi-model**: repeat `--run name=path`; the request's `model` field routes between them
(`kev-latest` / `jev-latest` alias the default; unknown names fall back to `--default`, else 404).
A future remote backend implements the same Backend interface and joins the same table.

Vendored code (clef-flash `JointSchemaHead`, Apache-2.0) is pinned by `tests/test_baseline.py`; see
`kev_serve/backends/clef/ATTRIBUTION.md`.

## Tests

```bash
pytest tests/test_api.py tests/test_server.py          # no GPU
KEV_SERVE_TOKENIZER=<qwen3.5 tokenizer> pytest tests/test_baseline.py
KEV_SERVE_TEST_CKPT=<clef ckpt> pytest tests/test_baseline.py   # forward pass, needs a GPU
```
