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

**Several GPUs.** Put each model on its own GPU with `@N` after the checkpoint (or `@cuda:N`, which
avoids the ambiguity if a checkpoint path itself ends in a number):

```bash
uv run --extra serve python -m kev_serve \
  --run main=<clef_ckpt>@2 --run grounder=<gui_actor_ckpt>@3 --port 8008
```

Without `@N` a model uses `--device` (all of them share it). `--device` takes
`auto` (probe, then fall back to CPU), `cpu`, `cuda` / `cuda:N`, or `mps`; `auto` picks GPU 0, so on a
box where GPU 0 is busy name the GPU explicitly.

`scripts/serve.sh <ckpt> [name] [port]` wraps the single-model case. Flags: `--host` (default
`127.0.0.1`; use `0.0.0.0` in a container), `--port`, `--default <name>`, `--fallback <ckpt>`
(a checkpoint to load when a `--run` path is missing; unset = fail instead).

**Checkpoints.** `clef` is a clef-flash directory (`joint_head_config.json` + `joint_head.safetensors`
on a Qwen3.5 VL backbone). `gui_actor` is a directory whose weights carry the vendored pointer head.
A path that does not exist and looks like `org/name[@revision]` is downloaded from the HuggingFace
Hub first (`snapshot_download`, cached under `HF_HOME`; the card's `release_date` then comes from the
Hub commit). A trailing `@N` is a GPU, not a revision — pin a numeric revision with `@cuda:N` free of
ambiguity by putting the GPU on another model or using `--device`.

**Env.** `KEV_API_KEY` (set = require `Authorization: Bearer <key>`; unset = open server),
`KEV_DATE_FACTS=1` (date preprocessing on every state), `KEV_DEFAULT_MODEL`, `KEV_DEVICE_PROBE_TIMEOUT`,
`KEV_GUI_ACTOR_TOPK`. Serving: `KEV_MAX_STATE_TOKENS` (admission limit; default = the backbone's
window, else 65,536), `KEV_TRUNCATE_STATES=1` (read a state's first tokens instead of a 422; responses
then carry `truncated` + `usage.state_tokens`/`state_tokens_used`), `KEV_TEMPERATURE` (calibration
scalar: clef divides the head's logits by T; gui_actor rescales the confidence only, never the point),
`KEV_MAX_BATCH` (64), `KEV_PREFIX_CACHE` (states kept, 4; 0 disables), `KEV_PREFIX_MIN_TOKENS`,
`KEV_PREFIX_MAX_TOKENS` (65,536-token budget), `KEV_FUSED` (`1` insist / `0` decline / unset = auto on
CUDA when flash-linear-attention==0.5.2 is importable; else the eager layers, silently).

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
`kind`, `supported_primitives`, `run`, `device`, `release_date`, `temperature`, plus serving stats
the engine keeps (`batches`, `prefix_cache` hits/misses/oom_retries) and backend-specific fields.

### `POST /v1/systemone/permute` and `/v1/systemone/separate`

Research endpoints over the same pipeline. `permute` re-runs one `choice` question under
`n_perm` (1..64) shuffled option orders — `{runs: [{order, probabilities, choice, latency_ms}],
argmax_stable, spread}` — for measuring option-order sensitivity. `separate` asks each question in
its own pass against the same state and aggregates the answers, for the packed-vs-separate
comparison. Both require a backend that answers `choice`.

```bash
curl localhost:8008/v1/systemone/permute -H 'content-type: application/json' -d '{
  "request": {"model":"main","state":"…","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Money","sre":"Uptime"}}}},
  "question":"team","n_perm":6,"seed":7}'
```

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
├── engine/         # ExecutionEngine: QueueBatchEngine (continuous batching + PrefixCache + OOM retry) for
│                   #   batch-capable backends (clef); SyncLockEngine (one locked pass) for the rest
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

### Serving engine

`QueueBatchEngine` runs the batch-capable backends (today: clef). One model thread drains the queue
into batches (up to `KEV_MAX_BATCH`) and keeps a **state-prefix cache**: the tokens up to the schema
(system + user header + state) run once and their KV / DeltaNet states (plus the prefix hidden states
the joint head cross-attends over) are reused by every later request over the same state — the second
request over a state pays only for its schema. The cached prefix is deep-copied before each
continuation, so reuse is **bit-identical to the full forward in fp32** (bf16 rounding aside); a batch
that runs out of device memory while states are cached drops the cache and runs once more.
`latency_ms` is the batch's model time, and `/v1/models` carries `batches` / `prefix_cache` counters.

GUI-Actor's grounding passes are batch=1 (`model.generate`), so it keeps the `SyncLockEngine`.
An image-prefix cache here (its prefill is dominated by the image tokens, and the pointer head's
image-side input is the *embed-layer* hidden state at the image positions, cacheable alongside the
KV) is blocked on Qwen2.5-VL's **M-RoPE continuation**: a two-stage forward — image prefix, then the
instruction + assistant head — does not reproduce the one-pass result, because the decoder's 3D
position ids must be recomputed for the continuation (`rope_deltas` stays at the prefix length and
the M-RoPE `[3, B, S]` ids must be rebuilt for the joined sequence). Probed on
`GUI-Actor-3B-Qwen2.5-VL`; a correct implementation has to replicate `get_rope_index`'s incremental
branch, which is why it is deferred rather than shipped half-working. A single direct forward *does*
match `generate()` bit-for-bit, so the pass itself is sound — only the two-stage split is open.
Requests a backend cannot cache are simply passed with no key — the engine runs them in one full pass.

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
- `kev_serve/backends/clef/fused_qwen35.py` — the fused Qwen3.5 Triton layer rewrite (Apache-2.0,
  via `kev`), targeting flash-linear-attention at its pinned version; optional (`KEV_FUSED`).
- `kev_serve/backends/gui_actor/gui_actor_model.py` — microsoft/GUI-Actor pointer head (MIT),
  via the `kev` monorepo. See `kev_serve/backends/gui_actor/ATTRIBUTION.md`.