"""P0-2: the temperature math, without weights.

clef: probabilities = softmax(logits / T) (the _softmax helper is the single place it happens).
gui_actor: the temperature rescales the confidence only, never the point (region_confidence from raw
logits; the region decode reads the raw distribution)."""
import math

import pytest
import torch


def test_clef_softmax_at_temperature():
    from kev_serve.backends.clef.clef_model import ClefDecisionModel
    z = torch.tensor([2.0, 1.0, 0.0])
    p1 = ClefDecisionModel._softmax(z, 1.0)
    assert abs(sum(p1) - 1.0) < 1e-6 and p1[0] > p1[1] > p1[2]
    p2 = ClefDecisionModel._softmax(z, 2.0)               # higher T flattens
    assert p2[0] < p1[0] and abs(sum(p2) - 1.0) < 1e-5
    assert p2 == pytest.approx((z / 2.0).softmax(-1).tolist())
    p05 = ClefDecisionModel._softmax(z, 0.5)             # lower T sharpens
    assert p05[0] > p1[0]


def test_gui_actor_temperature_moves_confidence_not_the_point():
    """region_confidence reads the RAW logits at the given T; get_prediction_region_point reads the
    softmax at T=1. A different T therefore changes the scalar, not the region."""
    from kev_serve.backends.gui_actor.gui_actor_model import get_prediction_region_point, region_confidence
    n = 4
    scores = torch.full((1, n * n), 0.01)
    for y in (1, 2):
        for x in (1, 2):
            scores[0, y * n + x] = 1.0
    point, _, _, _, region = get_prediction_region_point(scores.float(), n, n, return_region_indices=True)
    logits = torch.log(scores[0])
    c1 = region_confidence(logits, region[0], 1.0)
    c3 = region_confidence(logits, region[0], 3.0)
    assert c3 < c1                                   # a hotter distribution spreads mass off the region
    point2, *_ = get_prediction_region_point((scores / 1000).contiguous(), n, n)   # scaling the whole
    assert abs(point2[0] - point[0]) < 1e-6          # distribution (a monotone transform of the logits) moves nothing
    assert 0.0 <= c3 <= 1.0 and 0.0 <= c1 <= 1.0


def test_gui_actor_pointer_temperature_override():
    from types import SimpleNamespace
    from kev_serve.backends.gui_actor.gui_actor_model import pointer_temperature
    model = SimpleNamespace(multi_patch_pointer_head=SimpleNamespace(temperature=2.41))
    assert pointer_temperature(model) == 2.41                       # the fitted value
    assert pointer_temperature(model, 1.0) == 1.0                   # KEV_TEMPERATURE override
    assert pointer_temperature(model, None) == 2.41                 # explicit None = keep fitted
    with pytest.raises(ValueError):
        pointer_temperature(model, 0.0)
    with pytest.raises(ValueError):
        pointer_temperature(model, float("nan"))
