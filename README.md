# kev-serve

Standalone deployment service with a Jev-like (TypeSafe System One) API, extracted from the kev
monorepo onto an orphan branch: no shared history, no `kev` package dependency.

## Primitives

| Type | Request | Answer |
|---|---|---|
| `noul` | `{"type": "noul", "instructions", "criteria"?: {"true","false"}}` | `{"type": "noul", "noul": p(true)}` |
| `choice` | `{"type": "choice", "instructions", "criteria": {name: desc}}` | `{"choice", "confidence", "probabilities"}` |
| `score` | `{"type": "score", "instructions", "criteria": [levels]}` | `{"score", "legend", "probabilities", "confidence"}` |
| `box` | `{"type": "box", "instructions"}` (no criteria; needs an image) | `{"box": {top, bottom, left, right}, "confidence"}` — **501 until a Box backend is registered** |

Confidence formulas mirror TypeSafe's `system-one-adapter 0.2.1`. Schemas: `kev_serve/api.py`.

## Run

```bash
pip install -e ".[test]"
python -m kev_serve --run <clef_checkpoint> --host 0.0.0.0 --port 8008   # or scripts/serve.sh <ckpt> 8008
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

`kev_serve/backends.py` defines the `Backend` protocol and a registry (`load_backend`). Today: clef
(noul / choice / score). Box is interface-only; the planned backend is GUI-Actor's per-patch
distribution aggregated into a connected region (normalized edges). Adding a backend does not touch
the routes.

Vendored code (clef-flash `JointSchemaHead`, Apache-2.0) is pinned by `tests/test_baseline.py`; see
`kev_serve/ATTRIBUTION.md`.

## Tests

```bash
pytest tests/test_api.py tests/test_server.py          # no GPU
KEV_SERVE_TOKENIZER=<qwen3.5 tokenizer> pytest tests/test_baseline.py
KEV_SERVE_TEST_CKPT=<clef ckpt> pytest tests/test_baseline.py   # forward pass, needs a GPU
```
