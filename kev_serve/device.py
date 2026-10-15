"""The accelerator this process uses: cuda, then mps, then cpu."""
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

# This file is the probe child (`python kev/device.py`). A native crash inside torch's import must not dump a core
# into the caller's directory (issue #170). The limit is set before the import; importing this module as kev.device is not the child.
if __name__ == "__main__" and os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import torch

DEVICES = ("cpu", "cuda", "mps")
# A `cuda:N` string (which GPU of a multi-GPU box) is accepted anywhere a device string is; see select().
# The child prints one line with this prefix. ROCm and other wheels write startup text to stdout around it;
# a bare "cuda" line in that text is not a result.
PROBE_PREFIX = "kev-device "
DEVICE_HELP = ("auto probes in a child process and falls back to cpu (CUDA/ROCm first, then MPS; "
               "each probe waits KEV_DEVICE_PROBE_TIMEOUT seconds, default 30); "
               "cuda / cuda:N picks that GPU (default 0); an explicit accelerator fails closed; cpu skips discovery")


def default_device():
    return select()


def probe_timeout():
    """Seconds one probe may run. KEV_DEVICE_PROBE_TIMEOUT overrides the default; auto may run two (CUDA, then MPS)."""
    raw = os.environ.get("KEV_DEVICE_PROBE_TIMEOUT", "30")
    try:
        timeout = float(raw)
    except ValueError:
        raise ValueError(f"KEV_DEVICE_PROBE_TIMEOUT must be a positive number of seconds; got {raw!r}") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"KEV_DEVICE_PROBE_TIMEOUT must be a positive number of seconds; got {raw!r}")
    return timeout


def _cuda_index(device: str) -> int | None:
    """The GPU index of a `cuda` / `cuda:N` string, or None for a non-CUDA device. A bad index raises."""
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        raw = device.split(":", 1)[1]
        if not raw.isdigit():
            raise ValueError(f"unsupported device {device!r}; the GPU index after 'cuda:' must be a number")
        return int(raw)
    return None


def _check_device(device: str) -> None:
    """Reject a bad device string here (no CUDA call in this process: the probe child owns that and fails closed)."""
    if device == "auto" or device in DEVICES:
        return
    index = _cuda_index(device)
    if index is None or index < 0:
        raise ValueError(f"unsupported device {device!r}; choose auto, {', '.join(DEVICES)}, or cuda:N")


def select(device="auto", *, dtype=torch.float32, timeout=None):
    """Resolve a device before loading weights. A fresh interpreter contains native driver crashes.

    Explicit CPU never inspects accelerators. Auto tries CUDA/ROCm and, if that probe fails, MPS, then falls back to CPU.
    `cuda:N` probes and then selects that GPU (`torch.cuda.set_device(N)`), so a later `torch.cuda.current_device()`
    — the clef loader's device_map — lands on the right one. An explicit accelerator fails closed. This tests one
    allocation/matmul/sync, not every model kernel or whether the checkpoint fits.
    """
    _check_device(device)
    if device == "cpu":
        return device
    if timeout is None:
        timeout = probe_timeout()
    attempts = ("auto", "mps") if device == "auto" else (device,)
    failures = []
    for attempt in attempts:
        try:
            resolved = _run_probe(attempt, dtype, timeout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:   # includes CalledProcessError
            failures.append((attempt, error))
            continue
        index = _cuda_index(attempt)   # the GPU the caller asked for ("cuda" -> 0), not the child's family echo
        if index is not None:
            torch.cuda.set_device(index)
        return resolved
    message = "Device probe failed (" + "; ".join(f"{attempt}: {_probe_detail(error)}" for attempt, error in failures) + ")."
    if device != "auto":
        raise RuntimeError(f"{message} Fix the accelerator runtime or pass --device cpu.") from failures[-1][1]
    warnings.warn(f"{message} Falling back to CPU; pass --device cpu to skip probing. "
                  "CPU defaults to fp32; KEV_DTYPE=bf16 reduces serving weight memory.", RuntimeWarning, stacklevel=2)
    return "cpu"


def _probe_detail(error):
    detail = str(error)
    if isinstance(error, subprocess.CalledProcessError) and (stderr := (error.stderr or "").strip()):
        detail += f": {stderr.splitlines()[-1]}"
    return detail


def _run_probe(device, dtype, timeout):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(dtype).removeprefix("torch.")],
                            capture_output=True, text=True, errors="replace", timeout=timeout, check=True)
    reported = [line[len(PROBE_PREFIX):].strip() for line in result.stdout.splitlines() if line.startswith(PROBE_PREFIX)]
    resolved = reported[-1] if reported else ""
    # a `cuda:N` request (or the child's own echo) reduces to the family "cuda"; "auto" accepts cpu/cuda/mps
    family = "cuda" if _cuda_index(resolved) is not None else resolved
    if family not in DEVICES or (device != "auto" and family != "cuda" and family != device):
        tail = next((line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()), "")
        raise ValueError(f"unexpected probe result {(resolved or tail)!r}")
    return family


def _probe(device, dtype):
    """Child-process entry point: no accelerator operations from here may run in the selecting process."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "cpu":
        with torch.no_grad():
            matrix = torch.ones(8, 8, device=device, dtype=dtype)
            matrix @ matrix
            sync(device)
    return device


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    return 0


if __name__ == "__main__":
    print(PROBE_PREFIX + _probe(sys.argv[1], getattr(torch, sys.argv[2])), flush=True)
