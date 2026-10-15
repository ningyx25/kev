"""Device-string and --run spec parsing (no GPU, no checkpoint). The GPU probe itself needs hardware."""
import pytest

from kev_serve.cli import device_for, parse_run
from kev_serve.device import _cuda_index, _check_device


def test_parse_run_plain_and_gpu():
    assert parse_run("main=clef-flash") == ("main", "clef-flash", None)
    assert parse_run("g=/models/gui@2") == ("g", "/models/gui", "cuda:2")
    assert parse_run("g=org/gui@cuda:3") == ("g", "org/gui", "cuda:3")


def test_parse_run_keeps_hub_revision():
    # a Hub id's @revision is not a GPU: only a trailing @N / @cuda:N is
    assert parse_run("m=owner/name@sha256abc") == ("m", "owner/name@sha256abc", None)
    assert parse_run("m=owner/name@1234") == ("m", "owner/name", "cuda:1234")   # a numeric revision looks like a GPU; pass cuda:N explicitly to be sure


def test_parse_run_rejects_bad_specs():
    for spec in ("nopath", "=x", "name="):
        with pytest.raises(SystemExit):
            parse_run(spec)


def test_cuda_index():
    assert _cuda_index("cuda") == 0
    assert _cuda_index("cuda:3") == 3
    assert _cuda_index("cpu") is None and _cuda_index("mps") is None and _cuda_index("auto") is None
    with pytest.raises(ValueError):
        _cuda_index("cuda:x")


def test_check_device():
    for ok in ("auto", "cpu", "cuda", "cuda:0", "cuda:7", "mps"):
        _check_device(ok)              # no raise
    for bad in ("cuda:-1", "cuda:x", "gpu", "tpu"):
        with pytest.raises(ValueError):
            _check_device(bad)

def test_device_for_caches_and_repins(monkeypatch):
    """Each distinct device string is probed once; @N re-points the process, and a later default goes back."""
    calls, pinned = [], []
    monkeypatch.setattr("kev_serve.cli.select", lambda want, **k: (calls.append(want), "cuda" if want.startswith("cuda") or want == "auto" else "cpu")[1])
    resolved = {}
    order = ["cuda:2", "cuda:3", "cuda:2", None]     # last None -> the default ("auto"),
    devs = [device_for(g, "auto", resolved, set_device=pinned.append) for g in order]
    assert devs == ["cuda", "cuda", "cuda", "cuda"]
    assert calls == ["cuda:2", "cuda:3", "auto"]     # probe once per distinct string
    assert pinned == [2, 3, 2, 0]                    # auto resolves to GPU 0 and re-pins after @2


def test_device_for_cpu_default_never_pins(monkeypatch):
    monkeypatch.setattr("kev_serve.cli.select", lambda want, **k: "cpu")
    pinned = []
    assert device_for(None, "cpu", {}, set_device=pinned.append) == "cpu"
    assert pinned == []
