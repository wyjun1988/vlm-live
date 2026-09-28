"""Question interpretation for question-time routing: what kind of question is this and which objects does it
name - from any phrasing, in any language.

The routed facts (I-27 room facts, I-28 closest-point distances, I-33 floor-plan directions) are looked up by object
name and attached only to the question types they help. The evaluation harness read those names with regular
expressions over VSI-Bench's fixed templates - exact on the benchmark, useless anywhere else. Here the reading is a
step of its own, with two backends:

  rules  the VSI templates (the regular expressions below; scripts/eval_vsi_local.py imports them, one source);
  llm    the model reads the question first and returns one line of JSON - type, objects, roles and an English
         rendering (a Korean question comes back in English) - with the rules as the fallback when the JSON does
         not parse or is incomplete.

Either result renders back to a canonical VSI-style question (`Interpretation.canonical()`), so the fact router
keeps working on template text: the interpreter decides *what is asked*, the router *what to attach*. A LoRA for
the reading step (the direction agreed on 2026-09-28) would be trained on (question -> JSON) pairs of this same
schema; scripts/interpret_check.py measures how far the zero-shot model already gets, and on which types.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

TYPES = ("room_size", "abs_distance", "rel_distance", "rel_direction", "count", "size", "order", "route", "other")
REQUIRED = {"abs_distance": ("a", "b"), "rel_distance": ("target", "options"),
            "rel_direction": ("standing", "facing", "target"), "count": ("object",), "size": ("object",),
            "order": ("names",), "route": ("start", "facing", "goal"), "room_size": (), "other": ()}

# VSI-Bench templates. The harness's fact router reads names with these.
ROOM_KEYS = ("size", "square meter", "area")
ABS_RE = re.compile(r"distance between the (.+?) and the (.+?) \(in meters\)")
SIZE_RE = re.compile(r"dimension \(length, width, or height\) of the (.+?), measured in centimeters")
ORDER_RE = re.compile(r"first-time appearance order of the following categories in the video: (.+?)\?")
COUNT_RE = re.compile(r"How many (.+?)\(s\) are in this room")
REL_DIST_RE = re.compile(r"which of these objects \((.+?)\) is the closest to the (.+?)\?")
DIRECTION_RE = re.compile(r"standing by the (.+?) and facing the (.+?), is the (.+?) to ")
ROUTE_HEAD_RE = re.compile(r"beginning at the (.+?) (?:and )?facing the (.+?)\. You want to navigate to the (.+?)\.")

# the answer choices of the three direction difficulties, as the router recognises them ("front-left" -> hard,
# "back" -> medium, else easy)
CHOICES = {
    "easy": "to the left or the right of the {b}?",
    "medium": "to my left, right, or back? An object is to my back if I would have to turn at least 135 degrees in "
              "order to face it.",
    "hard": "to my front-left, front-right, back-left, or back-right? The directions refer to the quadrants of a "
            "Cartesian plane (if I am standing at the origin and facing along the positive y-axis, then front-left "
            "is the second quadrant, front-right the first, back-left the third, and back-right the fourth).",
}


@dataclass
class Interpretation:
    type: str
    objects: list[str] = field(default_factory=list)
    roles: dict = field(default_factory=dict)
    question_en: str = ""
    original: str = ""
    source: str = "rules"

    def complete(self) -> bool:
        return self.type in REQUIRED and all(self.roles.get(k) for k in REQUIRED[self.type])

    def canonical(self) -> str:
        """A VSI-style question of this type naming these objects - the text the fact router reads. `route` and
        `other` (and an incomplete reading) return the original: a route's steps are only ever read from the
        template itself."""
        r = self.roles
        if not self.complete():
            return self.original
        if self.type == "room_size":
            return "What is the size of this room (in square meters)?"
        if self.type == "abs_distance":
            return ("Measuring from the closest point of each object, what is the distance between the "
                    f"{r['a']} and the {r['b']} (in meters)?")
        if self.type == "size":
            return (f"What is the length of the longest dimension (length, width, or height) of the {r['object']}, "
                    "measured in centimeters?")
        if self.type == "count":
            return f"How many {r['object']}(s) are in this room?"
        if self.type == "order":
            return ("What will be the first-time appearance order of the following categories in the video: "
                    f"{', '.join(r['names'])}?")
        if self.type == "rel_distance":
            return ("Measuring from the closest point of each object, which of these objects "
                    f"({', '.join(r['options'])}) is the closest to the {r['target']}?")
        if self.type == "rel_direction":
            tail = CHOICES.get(r.get("choices", "medium"), CHOICES["medium"]).format(b=r["facing"])
            return f"If I am standing by the {r['standing']} and facing the {r['facing']}, is the {r['target']} {tail}"
        return self.original


def _clean(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().strip(".,;:").lower()


def interpret_rules(question: str) -> Interpretation:
    """The VSI templates. Anything else is `other`."""
    q = question
    m = DIRECTION_RE.search(q)
    if m:
        a, b, c = (_clean(g) for g in m.groups())
        choices = "hard" if "front-left" in q else ("medium" if "back" in q else "easy")
        return Interpretation("rel_direction", [a, b, c], {"standing": a, "facing": b, "target": c,
                                                           "choices": choices}, q, q)
    m = ROUTE_HEAD_RE.search(q)
    if m:
        s, f, g = (_clean(x) for x in m.groups())
        return Interpretation("route", [s, f, g], {"start": s, "facing": f, "goal": g}, q, q)
    m = REL_DIST_RE.search(q)
    if m:
        target = _clean(m.group(2))
        opts = [_clean(o) for o in m.group(1).split(",") if o.strip()]
        return Interpretation("rel_distance", opts + [target], {"target": target, "options": opts}, q, q)
    m = ABS_RE.search(q)
    if m:
        a, b = (_clean(g) for g in m.groups())
        return Interpretation("abs_distance", [a, b], {"a": a, "b": b}, q, q)
    m = SIZE_RE.search(q)
    if m:
        o = _clean(m.group(1))
        return Interpretation("size", [o], {"object": o}, q, q)
    m = ORDER_RE.search(q)
    if m:
        names = [_clean(n) for n in m.group(1).split(",") if n.strip()]
        return Interpretation("order", names, {"names": names}, q, q)
    m = COUNT_RE.search(q)
    if m:
        o = _clean(m.group(1))
        return Interpretation("count", [o], {"object": o}, q, q)
    ql = q.lower()
    if "room" in ql and any(k in ql for k in ROOM_KEYS):
        return Interpretation("room_size", [], {}, q, q)
    return Interpretation("other", [], {}, q, q)


PROMPT = """You read a question about a video of an indoor scene and extract what it asks. Reply with ONE line of JSON and nothing else, with these keys:
"type": one of room_size, abs_distance, rel_distance, rel_direction, count, size, order, route, other
"objects": the objects the question names, in English, lower-case, singular, in the order they appear
"roles": depends on the type -
  abs_distance: {{"a": object, "b": object}}
  rel_distance: {{"target": the reference object, "options": [the candidate objects]}}
  rel_direction: {{"standing": object, "facing": object, "target": object, "choices": "easy" (left/right), "medium" (left/right/back) or "hard" (front-left/front-right/back-left/back-right)}}
  count: {{"object": object}}   size: {{"object": object}}   order: {{"names": [objects]}}   room_size: {{}}
  route: {{"start": object, "facing": object, "goal": object}}   other: {{}}
"question_en": the question in English (translate it if it is not in English)

Q: How many chairs are in this room?
A: {{"type": "count", "objects": ["chair"], "roles": {{"object": "chair"}}, "question_en": "How many chair(s) are in this room?"}}
Q: 소파에서 TV까지 거리가 몇 미터야?
A: {{"type": "abs_distance", "objects": ["sofa", "tv"], "roles": {{"a": "sofa", "b": "tv"}}, "question_en": "What is the distance between the sofa and the tv in meters?"}}
Q: 냉장고 옆에 서서 문을 바라보면 침대는 왼쪽이야 오른쪽이야?
A: {{"type": "rel_direction", "objects": ["refrigerator", "door", "bed"], "roles": {{"standing": "refrigerator", "facing": "door", "target": "bed", "choices": "easy"}}, "question_en": "If I am standing by the refrigerator and facing the door, is the bed to the left or the right?"}}
Q: What material is the table made of?
A: {{"type": "other", "objects": ["table"], "roles": {{}}, "question_en": "What material is the table made of?"}}
Q: {question}
A:"""


PROMPT_COMPACT = """You read a question about a video of an indoor scene and extract what it asks. Reply with ONE line of JSON and nothing else:
"type": one of room_size, abs_distance, rel_distance, rel_direction, count, size, order, route, other
"roles": objects in English, lower-case, singular -
  abs_distance {{"a", "b"}} · rel_distance {{"target", "options": [...]}} · rel_direction {{"standing", "facing", "target", "choices": "easy" (left/right) | "medium" (left/right/back) | "hard" (front-left/front-right/back-left/back-right)}}
  count {{"object"}} · size {{"object"}} · order {{"names": [...]}} · route {{"start", "facing", "goal"}} · room_size {{}} · other {{}}
"question_en": the question translated to English - ONLY when the question is not in English; omit it otherwise

Q: How many chairs are in this room?
A: {{"type": "count", "roles": {{"object": "chair"}}}}
Q: 소파에서 TV까지 거리가 몇 미터야?
A: {{"type": "abs_distance", "roles": {{"a": "sofa", "b": "tv"}}, "question_en": "What is the distance between the sofa and the tv in meters?"}}
Q: If I am standing by the refrigerator and facing the door, is the bed to my left, right, or back?
A: {{"type": "rel_direction", "roles": {{"standing": "refrigerator", "facing": "door", "target": "bed", "choices": "medium"}}}}
Q: 책상 앞에 서서 창문을 보면 침대는 왼쪽이야 오른쪽이야?
A: {{"type": "rel_direction", "roles": {{"standing": "desk", "facing": "window", "target": "bed", "choices": "easy"}}, "question_en": "Standing by the desk and facing the window, is the bed to the left or the right?"}}
Q: Which of the lamp, the chair and the stool is closest to the sofa?
A: {{"type": "rel_distance", "roles": {{"target": "sofa", "options": ["lamp", "chair", "stool"]}}}}
Q: What material is the table made of?
A: {{"type": "other", "roles": {{}}}}
Q: {question}
A:"""


# One reasoning hint per question type, attached at question time to that type only (I-40). No measurement in it -
# it tests whether the model lacks a scaffold or the metric information (the routed facts say the latter).
HINTS = {
    "rel_direction": "Hint: first fix your pose - you stand at the {standing} and face the {facing}; then decide on "
                     "which side of that line of sight the {target} lies.",
    "abs_distance": "Hint: estimate in meters, using typical object sizes as a scale (a door is about 2 m tall, "
                    "a chair seat about 0.45 m high).",
    "rel_distance": "Hint: compare the closest point of each candidate to the {target}; the nearest one wins.",
    "count": "Hint: count distinct {object}s; the same one seen in several frames counts once.",
    "size": "Hint: give the longest side in centimeters, judged against typical objects (a door is about 200 cm "
            "tall, a chair seat about 45 cm high).",
    "room_size": "Hint: estimate the floor area in square meters from the room's width and length.",
    "order": "Hint: answer in the order in which the objects first become visible in the video.",
    "route": "Hint: keep track of your heading along the path; at each turn decide left, right or back relative to "
             "the direction you were walking.",
}


def type_hint(it: Interpretation) -> str:
    """The hint for this reading, with the named objects filled in; "" for `other` or an incomplete reading."""
    if it.type not in HINTS or not it.complete():
        return ""
    return HINTS[it.type].format(**{k: (", ".join(v) if isinstance(v, list) else v) for k, v in it.roles.items()}) + "\n"


def describe_prompt(it: Interpretation) -> str:
    """The first turn of the focused two-turn variant (I-41): the model describes only the objects the question
    names, before the question is asked; "" when the reading names nothing."""
    if not it.objects:
        return ""
    return ("In one or two sentences, where are the " + ", ".join(it.objects)
            + " in the room relative to each other and to where the camera moved?")


def parse_llm(text: str) -> dict | None:
    """The first {...} block of the reply that parses as JSON (the model may wrap it in prose or a code fence)."""
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[start:i + 1])
                        return d if isinstance(d, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def from_llm_json(d: dict, question: str) -> Interpretation | None:
    """Validate and normalise the model's JSON; None when the type is unknown or the roles do not suffice."""
    t = _clean(d.get("type", ""))
    if t not in TYPES:
        return None
    roles_in = d.get("roles") if isinstance(d.get("roles"), dict) else {}
    roles: dict = {}
    for k, v in roles_in.items():
        if isinstance(v, list):
            roles[k] = [_clean(x) for x in v if _clean(x)]
        elif v is not None:
            roles[k] = _clean(v)
    if t == "rel_direction" and roles.get("choices") not in ("easy", "medium", "hard"):
        roles["choices"] = "medium"
    if isinstance(d.get("objects"), list):
        objects = [_clean(o) for o in d["objects"] if _clean(o)]
    else:                                   # the compact format carries no object list: the roles name them
        objects = []
        for k in REQUIRED.get(t, ()):
            v = roles.get(k)
            objects += [x for x in (v if isinstance(v, list) else [v]) if x and x not in objects]
    it = Interpretation(t, objects, roles, str(d.get("question_en") or question).strip(), question, "llm")
    return it if it.complete() else None


def interpret(question: str, ask: Callable[[str], str] | None = None, compact: bool = False) -> Interpretation:
    """`ask(prompt) -> reply` is the model; None reads with the rules only. A reply that does not yield a complete
    reading falls back to the rules (source "rules-fallback"). compact=True asks for the short reply (no object
    list, the English rendering only for non-English questions): a third of the output tokens, the same routing."""
    if ask is None:
        return interpret_rules(question)
    try:
        reply = ask((PROMPT_COMPACT if compact else PROMPT).format(question=question.strip()))
    except Exception:  # noqa: BLE001 - the reading step must never take the answer down
        reply = ""
    d = parse_llm(reply or "")
    it = from_llm_json(d, question) if d else None
    if it is None:
        it = interpret_rules(question)
        it.source = "rules-fallback"
    return it
