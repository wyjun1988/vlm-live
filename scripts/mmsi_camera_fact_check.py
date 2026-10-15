"""Do CUT3R's camera-motion sentences agree with MMSI-Bench's answers? (the convention check, no model involved)

    python scripts/mmsi_camera_fact_check.py outputs/mmsi_local/cam_4b.json

For every Motion (Cam.) / Cam-Cam question that got a fact (scripts/mmsi_probe_local.py, variant `camera`), the
fact's direction words (left / right, forward / backward, up / down, turned left / right) are compared with the
words of the correct option. Agreement well above chance says the pose convention (OpenCV axes, yaw > 0 = right)
is right; agreement below chance on one axis says that axis is flipped. Questions whose correct option names none of
these words are skipped.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

MOVE_AXES = {
    "lateral": (("right",), ("left",)),
    "depth": (("forward", "ahead", "closer", "toward"), ("backward", "back", "away", "farther")),
    "vertical": (("up", "upward", "higher"), ("down", "downward", "lower")),
}
TURN_RE = re.compile(r"(turn(?:ed|ing|s)?\s+(?:to\s+the\s+)?(right|left)|rotat\w*\s+(?:to\s+the\s+)?(right|left)"
                     r"|(counter-?clockwise)|(clockwise))")
AXES = tuple(MOVE_AXES) + ("turn",)


def signs(text: str) -> dict[str, int]:
    """+1 / -1 per axis where the text names a direction: the turn from turn phrases (right / clockwise = +1), the
    moves from the text with the turn phrases removed, the first-named side of each axis."""
    t = text.lower()
    out = {}
    m = TURN_RE.search(t)
    if m:
        out["turn"] = -1 if (m.group(4) or (m.group(2) or m.group(3)) == "left") else 1
        t = TURN_RE.sub(" ", t)
    for axis, (pos, neg) in MOVE_AXES.items():
        p = min((mm.start() for w in pos for mm in re.finditer(r"\b" + w, t)), default=None)
        n = min((mm.start() for w in neg for mm in re.finditer(r"\b" + w, t)), default=None)
        if p is None and n is None:
            continue
        out[axis] = 1 if (n is None or (p is not None and p < n)) else -1
    return out


def fact_signs(fact: str) -> dict[str, int]:
    """The fact's words (live3r.serve.camera_facts): the same reader, on the part after 'the camera'."""
    m = re.search(r"the camera (.+)$", fact.strip(), flags=re.S)
    return signs(m.group(1)) if m else {}


def main() -> int:
    from datasets import load_dataset

    path = sys.argv[1]
    r = json.load(open(path))
    ds = load_dataset("RunsenXu/MMSI-Bench", split="test")
    by_id = {row["id"]: row for row in ds}
    agree, total = Counter(), Counter()
    examples = []
    for rec in r["records"]:
        v = rec["variants"].get("camera")
        if not v or not v.get("fact"):
            continue
        doc = by_id[rec["id"]]
        m = re.search(rf"\b{rec['gt']}:\s*(.+?)(?:,\s*[A-D]:|$)", doc["question"].replace("\n", " "))
        if not m:
            continue
        gt_text = m.group(1)
        g, f = signs(gt_text), fact_signs(v["fact"])
        for axis in g:
            if axis in f:
                total[axis] += 1
                agree[axis] += g[axis] == f[axis]
                if len(examples) < 6:
                    examples.append((rec["id"], axis, gt_text[:60], v["fact"][len("Measured from a 3D reconstruction of the two images: "):][:110]))
    print(f"{len(r['records'])} records · facts compared on {sum(total.values())} axis judgements")
    for axis in AXES:
        if total[axis]:
            print(f"  {axis:9s} agree {agree[axis]}/{total[axis]} = {100 * agree[axis] / total[axis]:.0f} % (chance 50)")
    print("examples:")
    for e in examples:
        print(f"  [{e[0]}] {e[1]}: truth '{e[2]}' | fact '{e[3]}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
