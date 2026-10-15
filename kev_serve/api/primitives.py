"""The four question primitives and the image asset / box answer models (Jev-like, docs.typesafe.ai).

Noul   -> 2 options [false, true];            answer = p(true)
Choice -> options 'name' or 'name: desc';      answer = argmax, probabilities by name, confidence
Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index
Box    -> instructions name a target in an image asset; answer = {top, bottom, left, right} in [0, 1] + confidence
         (schema served; a backend must register for box before it is answered, else the HTTP layer returns 501)
Ground -> instructions name a target in an image; answer = the click point [x, y] normalised to 0-999 integers
         + confidence (the GUI-Actor pointer head's connected-region centre)
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


class Ground(BaseModel):
    """A Ground locates something in an image as a click point. No criteria; the instructions name the
    target and the image. Answer coordinates are 0-999 integers (the Qwen-VL grounding convention)."""
    type: Literal["ground"]
    instructions: str

    @model_validator(mode="after")
    def _check(self):
        if not self.instructions or not self.instructions.strip(): raise ValueError("ground instructions must name the target")
        return self


Question = Union[Noul, Choice, Score, Box, Ground]
QUESTION_TYPES = ("noul", "choice", "score", "box", "ground")


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


class GroundAnswer(BaseModel):
    """The click point as [x, y], each 0-999 integers (x across the width, y down the height,
    top-left origin), plus the region confidence. topk_points: alternative region centres,
    same scale, optional."""
    type: Literal["ground"] = "ground"
    point: tuple[int, int]
    confidence: float
    topk_points: list[tuple[int, int]] | None = None

    @model_validator(mode="after")
    def _check(self):
        if not (0 <= self.point[0] <= 999 and 0 <= self.point[1] <= 999): raise ValueError("point coordinates must be 0-999")
        if not 0.0 <= self.confidence <= 1.0: raise ValueError("confidence must be in [0, 1]")
        for p in self.topk_points or []:
            if not (0 <= p[0] <= 999 and 0 <= p[1] <= 999): raise ValueError("topk_points coordinates must be 0-999")
        return self
