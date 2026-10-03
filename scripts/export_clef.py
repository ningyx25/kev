"""Export a deployable clef checkpoint from a kev training run.

A kev `--clef 1` run saves its state in one of two shapes:

  * a resume point (mid-training): `<run>/resume/step-<N>/` holding `adapter/` (a PEFT LoRA
    adapter), `head.pt` (JointSchemaHead weights, fp32) and `opt.pt` (optimizer state);
  * a finished run: `<run>/` holding `adapter_model.safetensors` (+ `adapter_config.json`)
    and `joint_head.safetensors` (+ `joint_head_config.json`).

Neither is directly deployable: the adapter is not merged into the backbone, and a resume
point's head is named/located for resumption, not for the clef-flash release layout.

This script assembles the release layout (what `Cloudflare/clef-flash` and `load_clef_checkpoint`
expect):

    <out>/config.json                 Qwen3_5ForConditionalGeneration config (merged backbone)
    <out>/model-*.safetensors         merged backbone weights (+ model.safetensors.index.json)
    <out>/joint_head.safetensors      JointSchemaHead, release format
    <out>/joint_head_config.json      its config
    <out>/tokenizer.json, tokenizer_config.json, chat_template.jinja, processor_config.json

Usage:

    # from a resume point (mid-training), 9B backbone, bf16
    uv run python scripts/export_clef.py runs/clef-v2-qwen35-9B-lora -o /tmp/clef-export

    # a specific resume step, pinned base, fp32
    uv run python scripts/export_clef.py runs/clef-v2-qwen35-9B-lora --step 550 \
        --base ./Qwen/Qwen3.5-9B --dtype fp32 -o /tmp/clef-export

    # a finished run (adapter + joint_head already at the top level)
    uv run python scripts/export_clef.py runs/clef-v2-qwen35-9B-lora -o /tmp/clef-export

    # verify the export loads and answers a request
    uv run python scripts/export_clef.py runs/... -o /tmp/clef-export --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def _find_checkpoint(run: Path, step: int | None) -> tuple[Path, Path, str]:
    """-> (dir with the head, dir with the adapter, kind) for a run directory.

    kind is "resume" (mid-training: head.pt + adapter/) or "final" (a finished run: the
    adapter and joint_head at the top level)."""
    resume = run / "resume"
    if resume.is_dir() and (steps := sorted(resume.glob("step-*"))):
        target = resume / f"step-{step:07d}" if step is not None else steps[-1]
        if step is not None and not target.is_dir():
            raise SystemExit(f"{run}: no resume point at step {step}; have {[s.name for s in steps]}")
        if not (target / "head.pt").exists():
            raise SystemExit(f"{target}: not a resume point (no head.pt)")
        return target, target / "adapter", "resume"
    if (run / "joint_head.safetensors").exists():
        return run, run, "final"
    raise SystemExit(
        f"{run}: not a kev clef run (no resume/step-*/head.pt and no joint_head.safetensors)"
    )


def _infer_head_config(state: dict, heads: int) -> dict:
    """The JointSchemaHead config implied by its state dict (hidden_size, width, layer counts,
    feedforward). `heads` cannot be read from a shape, so it is passed in (from the training
    config, or --heads)."""
    def n_of(prefix: str) -> int:
        return len({k.split(".")[1] for k in state if k.startswith(prefix)})
    # memory_projection: [width, hidden_size] (Linear(hidden_size, width, bias=False))
    return {
        "hidden_size": int(state["memory_projection.weight"].shape[1]),
        "width": int(state["memory_projection.weight"].shape[0]),
        "routing_layers": n_of("evidence_layers."),
        "layers": n_of("layers."),
        "heads": heads,
        "feedforward": int(state["evidence_layers.0.feedforward.0.weight"].shape[0]),
    }


def _head_config_from_run(run: Path, state: dict) -> dict:
    """head config: the run's written joint_head_config.json / training_config.json if present,
    else inferred from the state dict."""
    for name in ("joint_head_config.json",):
        cfg_path = run / name
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            return {k: cfg[k] for k in ("hidden_size", "width", "routing_layers", "layers", "heads", "feedforward") if k in cfg}
    cfg_path = run / "training_config.json"
    if cfg_path.exists():
        a = json.loads(cfg_path.read_text()).get("args", {})
        if a.get("clef"):
            return {"width": a["clef_width"], "routing_layers": a["clef_routing_layers"],
                    "layers": a["clef_layers"], "heads": a["clef_heads"],
                    "feedforward": a["clef_feedforward"]}
    return {}


def _base_of(run: Path, adapter_dir: Path, override: str | None) -> str:
    """The base backbone to merge onto: --base, else the run's --vision_base, else the adapter's
    base_model_name_or_path."""
    if override:
        return override
    cfg_path = run / "training_config.json"
    if cfg_path.exists():
        a = json.loads(cfg_path.read_text()).get("args", {})
        if a.get("vision_base"):
            return a["vision_base"]
    acfg = adapter_dir / "adapter_config.json"
    if acfg.exists():
        return json.loads(acfg.read_text())["base_model_name_or_path"]
    raise SystemExit("cannot find the base backbone; pass --base")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a kev clef run directory (runs/<name>)")
    ap.add_argument("-o", "--out", required=True, help="output directory for the deployable checkpoint")
    ap.add_argument("--step", type=int, default=None, help="resume step to export (default: the newest)")
    ap.add_argument("--base", default=None, help="base backbone to merge onto (default: the run's --vision_base)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="backbone dtype to save (default bf16)")
    ap.add_argument("--heads", type=int, default=None, help="clef head attention heads, if it cannot be read from the run")
    ap.add_argument("--max-shard-size", default="5GB", help="save_pretrained shard size (default 5GB)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to merge on")
    ap.add_argument("--verify", action="store_true", help="load the export and answer a sample request")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out directory")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, so `kev` imports
    from kev.clef_model import JointSchemaHead, load_clef_checkpoint, save_clef_head

    run = Path(a.run)
    ckpt_dir, adapter_dir, kind = _find_checkpoint(run, a.step)
    out = Path(a.out)
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    base = _base_of(run, adapter_dir, a.base)
    dev = a.device

    # 1. head: read the weights, work out their config, save in release format
    head_file = ckpt_dir / "head.pt" if kind == "resume" else ckpt_dir / "joint_head.safetensors"
    print(f"head: {head_file}", flush=True)
    if kind == "resume":
        state = torch.load(head_file, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        state = load_file(str(head_file))
    from_run = _head_config_from_run(run, state)
    heads = a.heads or from_run.get("heads") or 16
    cfg = {**_infer_head_config(state, heads), **{k: from_run[k] for k in from_run if k != "hidden_size"}}
    print(f"  head config: {cfg}", flush=True)
    head = JointSchemaHead(**cfg)
    head.load_state_dict(state, strict=True)
    head = head.to(dtype=dtype)

    class _Shell:   # save_clef_head wants a model with .head and ._hcfg
        pass
    shell = _Shell()
    shell.head = head
    shell._hcfg = {k: v for k, v in cfg.items() if k != "hidden_size"}
    save_clef_head(shell, str(out))

    # 2. backbone: load the base, fold the LoRA in, save the merged weights
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    print(f"backbone: {base} + {adapter_dir} -> merged", flush=True)
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32, local_files_only=True)
    from peft import PeftModel
    lm = PeftModel.from_pretrained(lm, str(adapter_dir), torch_device="cpu")
    lm = lm.merge_and_unload()          # W += delta: fp32 math, one rounding to the save dtype
    lm = lm.to(dtype)
    lm.config.use_cache = False
    lm.save_pretrained(out, max_shard_size=a.max_shard_size, safe_serialization=True)

    # 3. tokenizer + processor + chat template (from the base; the adapter dir carries only the adapter)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tok.save_pretrained(out)
    try:
        AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(out)
    except Exception as error:   # a text-only base has no processor_config.json to write
        print(f"  (no processor saved: {error})", flush=True)
    for name in ("chat_template.jinja", "generation_config.json"):
        if (src := Path(base) / name).exists():
            shutil.copy(src, out / name)

    print(f"exported -> {out}", flush=True)
    for p in sorted(out.iterdir()):
        print(f"  {p.name}", flush=True)

    if a.verify:
        print("\nverifying: loading the export...", flush=True)
        model, processor = load_clef_checkpoint(str(out), dev, dtype)
        from kev.clef_model import systemone
        resp = systemone(model, processor, {
            "model": "clef-export", "state": "The service is returning 500s and checkout is blocked.",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which team handles this?",
                         "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
                "urgent": {"type": "noul", "instructions": "Is a service down?"}}})
        print(json.dumps(resp["answers"], indent=2), flush=True)
        print("verify OK", flush=True)


if __name__ == "__main__":
    main()