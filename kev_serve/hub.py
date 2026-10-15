"""HuggingFace Hub checkpoints: `--run name=org/model@revision` downloads once and serves locally.

The loaders all pass local_files_only=True (a vendored-model property worth keeping), so a Hub id is
resolved to a local snapshot path here, before load_backend. `@revision` parsing must not eat the
`@N` GPU suffix — cli.parse_run already decided that (only a trailing `@N` / `@cuda:N` is a GPU), so
by the time a path reaches this module it is either local or a Hub id with its own revision."""
from __future__ import annotations

import os
import re

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")
# what a checkpoint needs to load; the same set kev/checkpoint.py downloads
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]


def is_hub_id(run: str) -> bool:
    """A Hub id is `org/name` (optionally `@revision`) that is not a local directory."""
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_checkpoint(path: str) -> str:
    """A local path passes through; a Hub id downloads to the HF cache and returns that directory."""
    if os.path.isdir(path) or not is_hub_id(path):
        return path
    from huggingface_hub import snapshot_download
    repo, _, revision = path.partition("@")
    print(f"{path} is not a local directory; downloading {repo}@{revision or 'main'} from the Hub")
    return snapshot_download(repo, revision=revision or None, allow_patterns=ALLOW_PATTERNS)


def hub_release_date(requested: str) -> str | None:
    """The Hub commit date for a Hub checkpoint, or None (offline, private, unknown revision) so the
    caller can fall back to file mtimes."""
    if not is_hub_id(requested): return None
    from huggingface_hub import HfApi
    repo, _, revision = requested.partition("@")
    try:
        return HfApi().model_info(repo, revision=revision or None).last_modified.date().isoformat()
    except Exception:
        return None
