"""Question interpretation (live3r.serve.interpret): the rules read every VSI template, the canonical rendering
round-trips through the same regular expressions the fact router uses, and the LLM path parses, validates and
falls back."""

import json
from pathlib import Path

import pytest

from live3r.serve import interpret as I

TEMPLATES = {
    "count": "How many chair(s) are in this room?",
    "size": "What is the length of the longest dimension (length, width, or height) of the stove, measured in centimeters?",
    "room_size": "What is the size of this room (in square meters)?  If multiple rooms are shown, estimate the size of the combined space.",
    "abs_distance": "Measuring from the closest point of each object, what is the distance between the sofa and the stove (in meters)?",
    "rel_direction_hard": "If I am standing by the stove and facing the sofa, is the tv to my front-left, front-right, back-left, or back-right? The directions refer to the quadrants of a Cartesian plane.",
    "rel_direction_medium": "If I am standing by the stove and facing the sofa, is the tv to my left, right, or back? An object is to my back if I would have to turn at least 135 degrees in order to face it.",
    "rel_direction_easy": "If I am standing by the stove and facing the sofa, is the tv to the left or the right of the sofa?",
    "rel_distance": "Measuring from the closest point of each object, which of these objects (chair, stool, stove, sofa) is the closest to the tv?",
    "order": "What will be the first-time appearance order of the following categories in the video: ceiling light, cup, heater, door?",
    "route": "You are a robot beginning at the bed facing the tv. You want to navigate to the toilet. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. Go forward until the door is on your left. 2. [please fill in] 3. Go forward until the toilet. You have reached the destination.",
}


def test_rules_read_every_vsi_template():
    r = {k: I.interpret_rules(q) for k, q in TEMPLATES.items()}
    assert r["count"].type == "count" and r["count"].roles == {"object": "chair"}
    assert r["size"].roles == {"object": "stove"}
    assert r["room_size"].type == "room_size"
    assert r["abs_distance"].roles == {"a": "sofa", "b": "stove"}
    for d in ("hard", "medium", "easy"):
        it = r[f"rel_direction_{d}"]
        assert it.type == "rel_direction" and it.roles == {"standing": "stove", "facing": "sofa", "target": "tv",
                                                           "choices": d}
    assert r["rel_distance"].roles == {"target": "tv", "options": ["chair", "stool", "stove", "sofa"]}
    assert r["order"].roles == {"names": ["ceiling light", "cup", "heater", "door"]}
    assert r["route"].type == "route" and r["route"].roles == {"start": "bed", "facing": "tv", "goal": "toilet"}
    assert I.interpret_rules("What color is the sofa?").type == "other"
    assert all(it.source == "rules" and it.canonical() for it in r.values())


def test_canonical_rendering_round_trips_through_the_rules():
    """What the LLM path produces is read by the same regular expressions as the templates."""
    for k, q in TEMPLATES.items():
        it = I.interpret_rules(q)
        again = I.interpret_rules(it.canonical())
        assert (again.type, again.roles) == (it.type, it.roles), k
    # a route or an unknown question keeps its original text (the steps are only read from the template)
    assert I.interpret_rules(TEMPLATES["route"]).canonical() == TEMPLATES["route"]
    assert I.interpret_rules("What color is the sofa?").canonical() == "What color is the sofa?"


def test_llm_reply_is_parsed_normalised_and_rendered():
    reply = ('Sure! ```json\n{"type": "Rel_Direction", "objects": ["Refrigerator", "door", "bed"], '
             '"roles": {"standing": "Refrigerator ", "facing": "door", "target": "bed"}, '
             '"question_en": "If I am standing by the refrigerator and facing the door, is the bed left or right?"}\n```')
    it = I.interpret("냉장고 옆에 서서 문을 바라보면 침대는 왼쪽이야 오른쪽이야?", ask=lambda p: reply)
    assert it.source == "llm" and it.type == "rel_direction"
    assert it.roles == {"standing": "refrigerator", "facing": "door", "target": "bed", "choices": "medium"}
    assert it.objects == ["refrigerator", "door", "bed"]
    m = I.DIRECTION_RE.search(it.canonical())
    assert m and m.groups() == ("refrigerator", "door", "bed") and "back" in it.canonical()


def test_llm_path_falls_back_to_the_rules_when_the_reply_is_unusable():
    q = TEMPLATES["abs_distance"]
    for reply in ("I cannot tell.", '{"type": "distance"}', '{"type": "abs_distance", "roles": {"a": "sofa"}}', ""):
        it = I.interpret(q, ask=lambda p, r=reply: r)
        assert it.source == "rules-fallback" and it.roles == {"a": "sofa", "b": "stove"}, reply

    def boom(p):
        raise RuntimeError("no model")

    assert I.interpret(q, ask=boom).source == "rules-fallback"


def test_prompt_carries_the_question_and_the_schema():
    text = I.PROMPT.format(question="How far is the bed from the door?")
    assert text.rstrip().endswith("Q: How far is the bed from the door?\nA:")
    assert all(t in text for t in I.TYPES)
    assert I.parse_llm('x {"a": {"b": 1}} y') == {"a": {"b": 1}} and I.parse_llm("nothing") is None


def test_paraphrase_fixture_is_consistent_with_the_schema():
    rows = [json.loads(ln) for ln in (Path(__file__).parent / "fixtures" / "question_paraphrases.jsonl").read_text()
            .splitlines() if ln.strip()]
    assert len(rows) >= 24 and {r["type"] for r in rows} == set(I.TYPES)
    for r in rows:
        it = I.Interpretation(r["type"], [], r["roles"], r["question"], r["question"], "fixture")
        assert it.complete(), r
        if r["type"] not in ("route", "other"):
            back = I.interpret_rules(it.canonical())
            assert back.type == r["type"] and all(back.roles[k] == v for k, v in r["roles"].items()), r
    with pytest.raises(KeyError):
        _ = I.REQUIRED["bogus"]


def test_compact_prompt_and_reply_route_like_the_full_one():
    text = I.PROMPT_COMPACT.format(question="침대의 가장 긴 변은 몇 센티미터야?")
    assert text.rstrip().endswith("Q: 침대의 가장 긴 변은 몇 센티미터야?\nA:") and '"objects"' not in text
    reply = '{"type": "size", "roles": {"object": "bed"}, "question_en": "What is the longest dimension of the bed in cm?"}'
    seen = {}

    def ask(p):
        seen["prompt"] = p
        return reply

    it = I.interpret("침대의 가장 긴 변은 몇 센티미터야?", ask, compact=True)
    assert seen["prompt"].startswith(I.PROMPT_COMPACT[:40]) and it.source == "llm"
    assert it.type == "size" and it.objects == ["bed"]                    # objects derived from the roles
    assert I.SIZE_RE.search(it.canonical()).group(1) == "bed"
    it2 = I.interpret("x", lambda p: '{"type": "rel_distance", "roles": {"target": "tv", "options": ["sofa", "chair"]}}')
    assert it2.objects == ["tv", "sofa", "chair"]


def test_type_hints_and_describe_prompt_follow_the_reading():
    it = I.interpret_rules(TEMPLATES["rel_direction_medium"])
    hint = I.type_hint(it, "all")
    assert hint.startswith("Hint: first fix your pose - you stand at the stove and face the sofa") and "tv" in hint
    assert hint.endswith("\n")
    assert I.type_hint(I.interpret_rules("What color is the sofa?"), "all") == ""
    assert "distinct chairs" in I.type_hint(I.interpret_rules(TEMPLATES["count"]), "all")
    for k, q in TEMPLATES.items():
        assert isinstance(I.type_hint(I.interpret_rules(q), "all"), str)
    dq = I.describe_prompt(it)
    assert dq.startswith("In one or two sentences, where are the stove, sofa, tv")
    assert I.describe_prompt(I.interpret_rules("Describe the room.")) == ""


def test_type_hints_default_to_the_prior_hints_only():
    direction = I.interpret_rules(TEMPLATES["rel_direction_medium"])
    room = I.interpret_rules(TEMPLATES["room_size"])
    assert I.type_hint(direction) == "" and I.type_hint(room).startswith("Hint: estimate the floor area")
    assert I.type_hint(direction, "all").startswith("Hint: first fix your pose")
    assert I.type_hint(direction, ("rel_direction",)) and I.type_hint(room, ("rel_direction",)) == ""
