"""The four question primitives and the image asset / box answer models (Jev-like, docs.typesafe.ai).

Noul   -> 2 options [false, true];            answer = p(true)
Choice -> options 'name' or 'name: desc';      answer = argmax, probabilities by name, confidence
Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index
Box    -> instructions name a target in an image asset; answer = {top, bottom, left, right} in [0, 1] + confidence
         (schema served; a backend must register for box before it is answered, else the HTTP layer returns 501)
"""
from typing import Literal, Union
from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
IMAGES_HEADER = "typesafe-danger-zone-images"


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS: raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


class Box(BaseModel):
    """A Box finds where something is in an image. No criteria; the instructions name the target and the image
    (docs.typesafe.ai/primitives/box). Requires an image in the request (assets) and the images header."""
    type: Literal["box"]
    instructions: str

    @model_validator(mode="after")
    def _check(self):
        if not self.instructions or not self.instructions.strip(): raise ValueError("box instructions must name the target")
        return self


Question = Union[Noul, Choice, Score, Box]
QUESTION_TYPES = ("noul", "choice", "score", "box")


class Asset(BaseModel):
    """One image in the request's top-level `assets` map, referenced by its key wherever the image belongs
    (docs.typesafe.ai/images)."""
    type: Literal["image"]
    format: str = "png"                      # jpg / png / webp ...
    data_base64: str


class BoxAnswer(BaseModel):
    """The four edges as fractions of the image's width / height from the top-left corner; left < right, top < bottom."""
    type: Literal["box"] = "box"
    box: dict[str, float]
    confidence: float

    @model_validator(mode="after")
    def _check(self):
        e = self.box
        for k in ("top", "bottom", "left", "right"):
            if k not in e: raise ValueError(f"box edges need a {k!r}")
            if not 0.0 <= e[k] <= 1.0: raise ValueError(f"{k} must be in [0, 1]")
        if not e["left"] < e["right"]: raise ValueError("left must be < right")
        if not e["top"] < e["bottom"]: raise ValueError("top must be < bottom")
        if not 0.0 <= self.confidence <= 1.0: raise ValueError("confidence must be in [0, 1]")
        return self
