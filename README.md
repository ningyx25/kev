# kev-serve

A standalone deployment service with a **Jev-like** (TypeSafe System One) API: send state and typed
questions, get typed answers your code can consume directly — no generated text.

Extracted from the `kev` monorepo onto an **orphan branch**: it shares no git history with `kev` and
imports nothing from the `kev` package. Third-party model code is vendored in-tree (see
[Attribution](#attribution)).

---

## Primitives

Five question types. A request carries one or many, all judged against the same `state`; a backend
answers the subset it declares and the service `501`s the rest.

| Type | Request | Answer |
| --- | --- | --- |
| `noul` | `{"type": "noul", "instructions", "criteria"?: {"true","false"}}` | `{"type": "noul", "noul": p(true)}` |
| `choice` | `{"type": "choice", "instructions", "criteria": {name: desc}}` | `{"type": "choice", "choice", "confidence", "probabilities"}` |
| `score` | `{"type": "score", "instructions", "criteria": [level, …]}` | `{"type": "score", "score", "legend", "probabilities", "confidence"}` |
| `box` | `{"type": "box", "instructions"}` — no criteria; needs an image | `{"type": "box", "box": {top, bottom, left, right}, "confidence"}` — edges are fractions of width/height. **Answered by: none yet (501)** |
| `ground` | `{"type": "ground", "instructions"}` — no criteria; needs an image | `{"type": "ground", "point": [x, y], "confidence", "topk_points"?}` — x/y are **0-999 integers**. **Answered by: gui_actor** |

- **Confidence** mirrors TypeSafe's `system-one-adapter 0.2.1`: `choice` is `(p_max − 1/K)/(1 − 1/K)`,
  `score` is a spread-relative concentration. `noul` returns the probability of *yes* directly.
- **Box** follows the official shape (`docs.typesafe.ai/primitives/box`): edges are `0..1` with
  `left < right` and `top < bottom`. The schema is served today; a backend that implements
  `answer_box()` opens it.
- **Ground** is the Qwen-VL grounding convention: `point` is `[x, y]`, each `0..999`, `x` across the
  width and `y` down the height from the top-left. It is the connected-region centre of the GUI-Actor
  pointer head; `topk_points` are the other region centres (`KEV_GUI_ACTOR_TOPK=0` drops them).

Schemas live in `kev_serve/api/primitives.py`.

---

## Install & run

Needs Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra serve             # the service + the TypeSafe SDK;  --extra vision-kernels for the Qwen3.5 flash kernels

uv run --extra serve python -m kev_serve --run main=<clef_checkpoint> --port 8008
```

Load several models into one process; the request's `model` field routes between them:

```bash
uv run --extra serve python -m kev_serve --run main=<clef_ckpt> --run grounder=<gui_actor_ckpt> --port 8008
# noul / choice / score -> main     ground -> grounder     the wrong primitive on either -> 501
```

`scripts/serve.sh <ckpt> [port]` wraps the single-model case. Flags: `--host` (default `127.0.0.1`;
use `0.0.0.0` in a container), `--port`, `--device auto|cpu|cuda|mps`, `--default <name>`.

**Checkpoints.** `clef` is a clef-flash directory (`joint_head_config.json` + `joint_head.safetensors`
on a Qwen3.5 VL backbone). `gui_actor` is a directory whose weights carry the vendored pointer head.

**Env.** `KEV_API_KEY` (set = require `Authorization: Bearer <key>`; unset = open server),
`KEV_DATE_FACTS=1` (date preprocessing on every state), `KEV_DEFAULT_MODEL`, `KEV_DEVICE_PROBE_TIMEOUT`.

### Docker

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN uv pip install --system torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
WORKDIR /app
COPY . .
RUN uv pip install --system .
EXPOSE 8008
CMD ["python", "-m", "kev_serve", "--run", "main=/models/clef", "--host", "0.0.0.0", "--port", "8008"]
```

Mount checkpoints, set `KEV_API_KEY`, and run with `--gpus all`.

---

## API

### `POST /v1/systemone`

```jsonc
{
  "model": "main",
  "state": { "ticket": { "text": "…" } },          // string | object | array; field names are kept as labels
  "questions": {
    "urgent": { "type": "noul",   "instructions": "Is this ticket urgent?" },
    "team":   { "type": "choice", "instructions": "Which team?", "criteria": { "billing": "Money", "sre": "Uptime" } },
    "sev":    { "type": "score",  "instructions": "Severity?",   "criteria": ["low", "medium", "high"] }
  }
}
```

```bash
curl localhost:8008/v1/systemone -H 'content-type: application/json' \
  -H "Authorization: Bearer $KEV_API_KEY" \
  -d '{"model":"main","state":"The server is on fire.","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'
```

One prefill pass per request; several questions share it. Response:

```jsonc
{
  "model": "main",
  "answers": {
    "urgent": { "type": "noul", "noul": 0.98 },
    "team":   { "type": "choice", "choice": "sre", "confidence": 0.83, "probabilities": { "billing": 0.09, "sre": 0.91 } },
    "sev":    { "type": "score", "score": 2.1, "legend": { "0": "low", "1": "medium", "2": "high" },
                "probabilities": { "0": 0.05, "1": 0.80, "2": 0.15 }, "confidence": 0.72 }
  },
  "usage": { "input_tokens": 128, "output_tokens": 21 },
  "latency_ms": 12.4
}
```

**Images.** Send them either the Jev way — a top-level `assets` map, referenced by key from `state` /
`instructions` — or the legacy base64 `images` list. Both need the `TypeSafe-Danger-Zone-Images: true`
header (TypeSafe gates images behind it), else `422`.

```jsonc
{
  "model": "grounder",
  "assets": { "shot": { "type": "image", "format": "png", "data_base64": "iVBORw0KGgo…" } },
  "state": { "screenshot": "shot" },
  "questions": { "icon": { "type": "ground", "instructions": "the settings gear in `screenshot`" } }
}
```

### `GET /v1/models`

One TypeSafe model card per served name (`kev-latest` / `jev-latest` alias the default):
`kind`, `supported_primitives`, `run`, `device`, `release_date`, plus backend-specific fields.

### Errors & headers

`422` invalid schema · `401` bad/missing API key (when `KEV_API_KEY` is set) · `404` unknown model
(and no `--default`) · `422` a gated image request without the header, or a backend-level input error ·
`501` a primitive the routed backend does not answer. Every response carries `x-typesafe-request-id`
(echoed or minted) and `server-timing`.

### TypeSafe SDK

Point the official SDKs at the service — both `jev-latest` and `kev-latest` name the default checkpoint:

```python
from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
client = TypeSafeClient(base_url="http://localhost:8008", api_key=os.environ["KEV_API_KEY"])
```

---

## Backends

```
kev_serve/
├── api/            # schema layer (pure pydantic): primitives, request, confidence
├── backends/       # Backend registry: base.py + one package per backend
│   ├── clef/       #   noul / choice / score  (vendored clef-flash JointSchemaHead)
│   └── gui_actor/  #   ground                 (vendored microsoft/GUI-Actor pointer head)
├── engine/         # ExecutionEngine: SyncLockEngine today; a queue+batch+prefix-cache slot waits for the kev pointer head
├── middlewares/    # telemetry / auth / images_gate; build_stack assembles them from Settings
├── router.py       # ModelRouter: model name -> backend (explicit / alias / default / 404)
├── http_app.py     # create_app(settings, router): middleware stack + the two routes
├── config.py       # Settings (KEV_* env)
├── cli.py          # python -m kev_serve --run name=ckpt [--run name2=ckpt2 ...]
└── device.py       # device probe / select
```

| Backend | Primitives | Backbones |
| --- | --- | --- |
| `clef` | noul, choice, score | Qwen3.5 VL |
| `gui_actor` | ground | Qwen2.5-VL (`qwen2_5_vl`, 28 px/token) · Qwen3-VL (`qwen3_vl`, `qwen3_vl_moe`, 32 px) · Qwen3.5 (`qwen3_5`, 32 px) |

### Extending

- **Add a backend** — a package under `kev_serve/backends/` whose `__init__.py` calls
  `register_backend(kind, sniff, factory)` with a `Backend` implementing `kind`,
  `supported_primitives`, `answer()`, `model_card()`, `close()` (plus `answer_box()` to open box).
  Import it in `backends/__init__.py`, or ship the `kev_serve.backends` entry-point group for
  third-party discovery. The CLI, router and routes pick it up unchanged.
- **Add a primitive** — a model in `api/primitives.py` (add to `Question` / `QUESTION_TYPES`), then a
  backend that lists it in `supported_primitives`.
- **Add a middleware** — a module in `kev_serve/middlewares/` exposing
  `async def mw(request, call_next) -> Response`, plus one line in `build_stack` (`middlewares/base.py`).
- **Multi-model** — repeat `--run name=path`; the request's `model` field routes between them
  (`kev-latest` / `jev-latest` alias the default; unknown names fall back to `--default`, else `404`).
  A remote backend implementing the same `Backend` interface joins the same table.

---

## Tests

```bash
uv run --extra test pytest tests/                                       # schema, server, router — no GPU
KEV_SERVE_TOKENIZER=<qwen3.5-tokenizer> uv run --extra test pytest tests/test_baseline.py   # clef encode spans
KEV_SERVE_TEST_CKPT=<clef-ckpt>         uv run --extra test pytest tests/test_baseline.py   # clef forward pass (GPU)
KEV_SERVE_TEST_GUI_CKPT=<gui-actor-ckpt> uv run --extra test pytest tests/test_baseline.py  # gui_actor load (GPU)
```

---

## Attribution

Vendored code is copied in-tree (no editable-package dependency), keeps its upstream licence, and is
pinned by `tests/test_baseline.py`:

- `kev_serve/backends/clef/clef_model.py` — clef-flash `JointSchemaHead` (Apache-2.0),
  via the `kev` monorepo. See `kev_serve/backends/clef/ATTRIBUTION.md`.
- `kev_serve/backends/gui_actor/gui_actor_model.py` — microsoft/GUI-Actor pointer head (MIT),
  via the `kev` monorepo. See `kev_serve/backends/gui_actor/ATTRIBUTION.md`.