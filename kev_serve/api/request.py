"""The /v1/systemone request body."""
from pydantic import BaseModel, Field

from .primitives import Asset, JSONContent, Question


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "jev-latest"
    questions: dict[str, Question] = Field(min_length=1)
    assets: dict[str, Asset] | None = None   # Jev-style image assets, referenced by key from state / instructions
    images: list[str] | None = None          # legacy kev form: base64 images (raw or data-URI), clef checkpoints only
