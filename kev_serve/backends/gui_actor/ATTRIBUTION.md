# Vendored code

## GUI-Actor (`gui_actor_model.py`)

`VisionHead_MultiPatch` (the pointer head), the prompt constants, the chat template and the
region decode (`get_prediction_region_point`) are vendored from
[microsoft/GUI-Actor](https://github.com/microsoft/GUI-Actor) (MIT), copied via
`kev/gui_actor_model.py` of the kev monorepo (commit b6883b0, 2026-10). Adaptations in this copy:
imports retargeted to `kev_serve.api`, Qwen3-VL (`qwen3_vl` / `qwen3_vl_moe`) backbones added,
and the serving surface reduced to the `ground` primitive (ground_answer / systemone).

## Attribution policy

Vendored code is copied in (no editable-package dependency), keeps its upstream licence notice
in the file header, and is pinned by `tests/test_baseline.py`.
