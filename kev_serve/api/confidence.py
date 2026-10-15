"""Rendering, date facts, answer assembly and the confidence formulas.

Both confidence formulas mirror TypeSafe's reference adapter, system-one-adapter 0.2.1
(src/system_one_adapter/_utils/confidence_metrics.py): p is normalised to sum 1 (all zeros -> uniform),
one option -> 1. Do not change their semantics.
"""
import json
import re
from datetime import datetime
from typing import Any

from .primitives import JSONContent
from .request import SystemOneRequest


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def date_facts(text: str) -> str:
    """Deterministic date arithmetic for the model: every pair of absolute dates found in `text`, as one sentence each
    ("August 3, 2026 is 12 days after July 22, 2026."). The model cannot subtract dates reliably; it can use a
    stated day count. Returns "" when fewer than two dates are found. Dates are listed in order of first appearance."""
    found = []
    for m in _DATE.finditer(text):
        raw = m.group(0)
        try: d = datetime.strptime(raw, "%B %d, %Y") if "," in raw else datetime.strptime(raw, "%Y-%m-%d")
        except ValueError: continue
        if raw not in [r for r, _ in found]: found.append((raw, d))
    facts = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            facts.append(f"{found[j][0]} is {abs(n)} day{'s' if abs(n) != 1 else ''} {'after' if n > 0 else 'before'} {found[i][0]}." if n else f"{found[j][0]} is the same day as {found[i][0]}.")
    return " ".join(facts)


def with_date_facts(state):
    """State with a `date_facts` field (object states) or an appended paragraph (string states) when two or more absolute
    dates appear. Opt-in preprocessing (Settings.date_facts)."""
    facts = date_facts(render(state))
    if not facts: return state
    if isinstance(state, dict): return {**state, "date_facts": facts}
    if isinstance(state, list): return state + [{"date_facts": facts}]
    return f"{state}\n\ndate_facts: {facts}"


def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria names (choice),
    ["false", "true"] (noul), the level indices as strings (score). Labels, targets and anchors use the same keys."""
    if qtype == "choice": return list(criteria)
    if qtype == "noul": return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: SystemOneRequest):
    """-> internal record for encode(), plus per-question metadata ({"id", "type", "keys", "legend" for score}) to map
    probabilities back. Box questions are not encodable (no options); the caller filters them out first."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        if q.type == "box": raise ValueError(f"question {qid!r}: box questions cannot be encoded (no options)")
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q.instructions), "options": opts, "label": 0}); meta.append(m)
    return {"state": render(req.state), "questions": qs}, meta


def _normalize(p: list[float]) -> list[float]:
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: list[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)


def score_confidence(p: list[float]) -> float:
    """max(0, 1 - E|level - mode| / D), D = mean absolute deviation of a uniform distribution over the L levels around its
    mean (L-1)/2; mode = first most likely level. 1 when all mass is on one level, 0 at uniform or anything as spread."""
    L = len(p)
    if L == 1: return 1.0
    p = _normalize(p); mode = max(range(L), key=p.__getitem__)
    D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)


def round_prob(x: float, decimals: int = 4) -> float:
    """Serialization precision for probabilities and derived scalars. 4 decimals keeps the sum of a rounded distribution
    within TypeSafe's tolerance (|sum - 1| < 0.02) at the 255-option maximum: 255 * 0.00005 < 0.02."""
    return round(float(x), decimals)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])], "confidence": round_prob(choice_confidence(p)), "probabilities": dist}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": round_prob(score), "legend": m["legend"], "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)}, "confidence": round_prob(score_confidence(p))}
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style figure: tokens of the serialised answers. Not a measure of generation (there is none)."""
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
