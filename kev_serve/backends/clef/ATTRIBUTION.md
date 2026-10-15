# Vendored code

## clef-flash (`kev_serve/clef_model.py`)

`EvidenceRoutingLayer` + `JointSchemaHead` are vendored from
[Cloudflare/clef-flash](https://github.com/CloudflareAI/clef-flash) (Apache-2.0),
copied via `kev/clef_model.py` of the kev monorepo (commit b6883b0, 2026-10).
The `ClefDecisionModel` wrapper and checkpoint helpers around them are kev's own.

## Attribution policy

Vendored code is copied in (no editable-package dependency), keeps its upstream
licence notice in the file header, and is pinned by `tests/test_baseline.py`.

## fused_qwen35.py (`fused_qwen35.py`)

The fused Qwen3.5 Triton layer rewrite (DeltaNet + attention + MLP + norms) is vendored from the
kev monorepo's `kev/fused_qwen35.py` (Apache-2.0), which itself targets the `fla`
(flash-linear-attention) kernels at the pinned `FLA_VERSION`. `text_layers` and the fused-path probe
are added here so `fuse()` reaches the Qwen3.5 **VL** composite's text stack. Needs flash-linear-attention
and CUDA; falls back to eager layers when either is absent (`KEV_FUSED`).
