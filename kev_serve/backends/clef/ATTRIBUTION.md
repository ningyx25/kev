# Vendored code

## clef-flash (`kev_serve/clef_model.py`)

`EvidenceRoutingLayer` + `JointSchemaHead` are vendored from
[Cloudflare/clef-flash](https://github.com/CloudflareAI/clef-flash) (Apache-2.0),
copied via `kev/clef_model.py` of the kev monorepo (commit b6883b0, 2026-10).
The `ClefDecisionModel` wrapper and checkpoint helpers around them are kev's own.

## Attribution policy

Vendored code is copied in (no editable-package dependency), keeps its upstream
licence notice in the file header, and is pinned by `tests/test_baseline.py`.
