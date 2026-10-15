"""P0-4: Hub id recognition and the @revision vs @N-GPU ambiguity (offline; no network)."""
import pytest

from kev_serve.cli import parse_run
from kev_serve.hub import is_hub_id


@pytest.mark.parametrize("run", ["jaredpalmer/kev-4b", "org.name/sub-name", "org/model@v1.0",
                                 "org/model@0123abc456789", "G/m"])
def test_hub_ids(run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # nothing here is a local directory
    assert is_hub_id(run)


@pytest.mark.parametrize("run", ["/abs/path", "a/b/c/d", "no-dot-no-slash", "org/model@", "with space/x"])
def test_not_hub_ids(run):
    assert not is_hub_id(run)


@pytest.mark.parametrize("run", ["runs/kev", "./local"])
def test_hub_shaped_relative_paths_resolve_by_existence(tmp_path, monkeypatch, run):
    """`runs/kev` matches the org/name shape, so existence decides: a present directory is a path, a
    missing one reads as a Hub id (the old is_hub_id's isdir check, kept verbatim)."""
    monkeypatch.chdir(tmp_path)
    assert is_hub_id(run)                                   # missing here: hub-shaped
    (tmp_path / run.lstrip("./")).mkdir(parents=True)
    assert not is_hub_id(run)                               # present now: a local path


def test_a_local_directory_wins_over_the_hub_shape(tmp_path, monkeypatch):
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert not is_hub_id("org/model")     # a real directory on disk is a path, not a Hub id


def test_parse_run_keeps_revision_and_splits_gpu():
    assert parse_run("main=org/model@v1.0") == ("main", "org/model@v1.0", None)        # a word revision stays on the id
    assert parse_run("main=org/model@abc123") == ("main", "org/model@abc123", None)
    assert parse_run("main=org/model@0") == ("main", "org/model", "cuda:0")             # a NUMERIC revision reads as a GPU — documented ambiguity
    assert parse_run("main=runs/kef@cuda:3") == ("main", "runs/kef", "cuda:3")
    assert parse_run("main=/path/ckpt") == ("main", "/path/ckpt", None)


def test_resolve_checkpoint_passes_local_through(tmp_path):
    from kev_serve.hub import resolve_checkpoint
    d = tmp_path / "ckpt"; d.mkdir()
    assert resolve_checkpoint(str(d)) == str(d)     # no download for an existing directory


def test_resolve_checkpoint_downloads_a_hub_id(tmp_path, monkeypatch):
    import kev_serve.hub as hub
    calls = {}
    def fake_download(repo, revision=None, allow_patterns=None):
        calls.update(repo=repo, revision=revision, patterns=allow_patterns)
        return str(tmp_path / "snapshot")
    monkeypatch.setattr(hub, "is_hub_id", lambda r: True)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    out = hub.resolve_checkpoint("org/model@v2")
    assert out == str(tmp_path / "snapshot")
    assert calls == {"repo": "org/model", "revision": "v2", "patterns": hub.ALLOW_PATTERNS}


def test_resolve_checkpoint_never_downloads_an_existing_dir(tmp_path, monkeypatch):
    import kev_serve.hub as hub
    d = tmp_path / "here"; d.mkdir()
    def boom(*a, **k): raise AssertionError("must not download")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    assert hub.resolve_checkpoint(str(d)) == str(d)


def test_hub_release_date_falls_back_offline(monkeypatch):
    import kev_serve.hub as hub
    import huggingface_hub
    def boom(*a, **k): raise RuntimeError("offline")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: None)   # the code does HfApi() then catches everything
    assert hub.hub_release_date("org/model") is None              # any failure -> None, caller uses mtimes
