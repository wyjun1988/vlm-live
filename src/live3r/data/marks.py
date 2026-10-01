"""Marks the question refers to, drawn onto the frames - our implementation, from the data format and the reference
team's description of its rule (docs/REFERENCE_70_ANALYSIS.md §9; no code of theirs).

SPAR-style records (configs/data/vgllm.yaml: spar_234k, spar_234k_cot) ask about objects marked in the images: "the
pillar (red point)", "the desk (red bbox) ... the mug (blue bbox)", "Object3 (bbox3)". The record carries the marks
in `spar_info` (a JSON string or object):

    {color}_point   [[x, y]]                 color: red, blue, green, yellow
    {color}_bbox    [[x1, y1, x2, y2]]
    point_list      [[[x, y], ...]]          listed objects (Object{i})
    bbox_list       [[[x1, y1, x2, y2], ...]]
    point_img_idx   [[i, ...]]               the image each mark is on; bbox_img_idx the same for boxes
    type            the question type

Coordinates are per axis in 0..1000 of the image.

Placement (which frame each colored mark is on) follows a fixed order per question type: the k-th entry of the index
list belongs to the k-th color of `TYPE_ORDER[type]` (e.g. the scene-imagination questions: green = Object0, blue =
Object1, red = Object2, yellow = Object3). A few types follow other rules (`ALL_ON_FIRST_INDEX`, `PRESENT_IN_ORDER`,
`FIRST_IMAGE`), the listed-object types put item i on index i, and `NOT_DRAWN` types are kept with nothing drawn (the
reference never draws them; e.g. the "hard" scene-imagination variants). A single image takes every mark. A
multi-image record of an unknown type falls back to the order in which its question names the colors; when that
does not settle it, `plan` returns None and the dataset leaves the record out rather than guess a frame.

What is drawn: the kind of mark the question refers to - "(red point)" -> the points only, "(green bbox)" -> the
boxes only (both kinds when it names both or neither); compared with the reference's own renderings of the same
records (a depth question: two dots, no boxes; a scene-imagination question: labelled boxes, no dots).

Style (as the reference draws, on the original-resolution image before any resizing, so the VLM and CUT3R see the
same marked frames): points as filled discs of radius 20 px, boxes 20 px wide just outside the object box; the video
and map types label their marks at the box's lower right ("object2": white on the mark's color; listed objects in
blue with green labels). One deliberate difference: a mark named "yellow" is drawn yellow (the reference draws it
cyan in the video types).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
LABEL_GREEN = (0, 200, 0)
# a color named as a mark: "(red point)", "(green bbox)", "shown with a red bbox", "the red point of bed"
_MENTION = re.compile(r"\b(red|blue|green|yellow)\s+(?:point|bbox|box|dot|mark)", re.I)
_KIND = {"point": re.compile(r"\b(?:red|blue|green|yellow)\s+(?:point|dot)", re.I),
         "bbox": re.compile(r"\b(?:red|blue|green|yellow)\s+(?:bbox|box)", re.I)}

# the k-th index belongs to the k-th color
TYPE_ORDER = {
    "depth_prediction_oc_mv": ("red", "blue"),
    "distance_prediction_oo_mv": ("red", "blue"),
    "distance_prediction_oo_video": ("red", "blue"),
    "depth_prediction_oo_mv": ("red", "blue", "green"),
    "distance_prediction_oc_mv": ("red",),
    "distance_infer_center_oo_mv": ("red", "green", "blue"),
    "spatial_imagination_oc_mv": ("green", "blue", "red"),
    "spatial_imagination_oo_mv": ("green", "blue", "red", "yellow"),
    "spatial_imagination_oc_video": ("green", "blue", "red", "yellow"),
    "spatial_imagination_oo_video": ("green", "blue", "red", "yellow"),
    "obj_frame_locate": ("red",),
}
ALL_ON_FIRST_INDEX = {"obj_spatial_relation_oc_mv"}                     # every mark on idx[0]
PRESENT_IN_ORDER = {"obj_spatial_relation_oo_mv": ("red", "blue", "green")}   # the present colors -> idx[0], idx[1]
FIRST_IMAGE = {"position_matching"}                                    # the first image, whatever the index says
NOT_DRAWN = {"distance_infer_center_oc", "distance_infer_center_oc_mv", "view_change_infer", "camera_motion_infer",
             "appearance_order", "room_size", "obj_count", "nav"}
POINT_RADIUS = 20          # px on the original image
LINE_WIDTH = 20
EXPAND = {"plain": 17.5, "labelled": 20.0}   # boxes drawn this far outside the object box


@dataclass(frozen=True)
class Mark:
    kind: str                       # "point" | "bbox"
    xy: tuple[float, ...]           # (x, y) or (x1, y1, x2, y2), per axis in 0..1000
    image: int
    color: tuple[int, int, int]
    label: str | None = None
    label_style: str = "on_color"   # "on_color" (white text on the mark's color) | "green" (green text)


def parse(info) -> dict | None:
    """spar_info (a JSON string or object) -> a dict with at least one mark, or None (no marks: nothing to draw)."""
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return None
    if not isinstance(info, dict):
        return None
    has = any(k.endswith(("_point", "_bbox")) and info[k] for k in info) or bool(info.get("point_list")) \
        or bool(info.get("bbox_list"))
    return info if has else None


def _coords(v) -> list[list[float]]:
    """Any nesting of coordinate lists -> [[x, y] or [x1, y1, x2, y2], ...] (the data wraps each list once more)."""
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return [list(v)]
    out: list[list[float]] = []
    for x in v if isinstance(v, list) else []:
        out += _coords(x)
    return out


def _ints(v) -> list[int]:
    """[[16, 21, 16, 22]] -> [16, 21, 16, 22]."""
    if isinstance(v, (int, float)):
        return [int(v)]
    out: list[int] = []
    for x in v if isinstance(v, list) else []:
        out += _ints(x)
    return out


def mention_order(question: str) -> list[str]:
    """The mark colors in the order the question names them (each once)."""
    out: list[str] = []
    for m in _MENTION.finditer(question):
        c = m.group(1).lower()
        if c not in out:
            out.append(c)
    return out


def named_kinds(question: str) -> set[str]:
    """The mark kinds the question refers to ("red point" -> point, "green bbox" -> bbox); empty when it names none."""
    return {k for k, rx in _KIND.items() if rx.search(question)}


def _where(qtype: str, colors: list[str], idx: list[int], question: str, n_images: int) -> dict[str, int] | None:
    """The image of each present color of one mark kind, or None when it cannot be known."""
    if n_images == 1 or qtype in FIRST_IMAGE or not idx:
        return {c: 0 for c in colors}
    if qtype in ALL_ON_FIRST_INDEX:
        return {c: idx[0] for c in colors}
    if qtype in PRESENT_IN_ORDER:
        ordered = [c for c in PRESENT_IN_ORDER[qtype] if c in colors]
        return dict(zip(ordered, idx)) if len(idx) >= len(ordered) and set(ordered) == set(colors) else None
    if qtype in TYPE_ORDER:
        order = TYPE_ORDER[qtype]
        if not set(colors) <= set(order) or len(idx) < max(order.index(c) for c in colors) + 1:
            return None
        return {c: idx[order.index(c)] for c in colors}
    named = mention_order(question)                          # an unknown type: the order the question names them
    if not set(colors) <= set(named):
        return None
    ordered = [c for c in named if c in colors]
    return dict(zip(ordered, idx)) if len(idx) == len(ordered) else None


def plan(info: dict, question: str, n_images: int) -> list[Mark] | None:
    """Where each mark goes: a list (empty = keep the record, draw nothing), or None (placement not certain)."""
    if n_images <= 0:
        return None
    qtype = str(info.get("type") or "")
    if qtype in NOT_DRAWN or qtype.endswith("_video_hard"):
        return []
    labelled = qtype.endswith("_video")
    kinds = named_kinds(question) or {"point", "bbox"}
    marks: list[Mark] = []
    for kind in ("point", "bbox"):
        if kind not in kinds:
            continue
        colors = [c for c in COLORS if _coords(info.get(f"{c}_{kind}"))]
        if not colors:
            continue
        where = _where(qtype, colors, _ints(info.get(f"{kind}_img_idx")), question, n_images)
        if where is None:
            return None
        order = TYPE_ORDER.get(qtype, ())
        for c in colors:
            label = f"object{order.index(c)}" if labelled and c in order else None
            for xy in _coords(info[f"{c}_{kind}"]):
                marks.append(Mark(kind, tuple(float(v) for v in xy), where[c], COLORS[c], label=label))
    for kind, key in (("bbox", "bbox_list"), ("point", "point_list")):
        items = _coords(info.get(key))
        if not items:
            continue
        idx = _ints(info.get(f"{kind}_img_idx")) or [0] * len(items)
        if len(idx) != len(items):
            return None
        # the listed-object types: blue marks with green labels - "object{i}"; the anchor-first point list of
        # distance_infer_center_oo_video reads objectA, object0, object1, ...
        anchor = qtype == "distance_infer_center_oo_video" and kind == "point"
        for i, (xy, im) in enumerate(zip(items, idx)):
            label = ("objectA" if i == 0 else f"object{i - 1}") if anchor else f"object{i}"
            marks.append(Mark(kind, tuple(float(v) for v in xy), im, COLORS["blue"], label=label, label_style="green"))
        break                                                # boxes when there are boxes, else points
    for m in marks:
        if not 0 <= m.image < n_images or len(m.xy) != (2 if m.kind == "point" else 4):
            return None
    return marks


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                        # Pillow < 10.1: a fixed bitmap font
        return ImageFont.load_default()


def _label(d, corner: tuple[float, float], text: str, color, style: str, size: int) -> None:
    """The label with its lower-right corner at `corner`."""
    font = _font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    xy = (corner[0] - (r - l) - 6, corner[1] - (b - t) - 6 - t)
    if style == "green":
        d.text(xy, text, fill=LABEL_GREEN, font=font)
        return
    tb = d.textbbox(xy, text, font=font)
    d.rectangle((tb[0] - 6, tb[1] - 6, tb[2] + 6, tb[3] + 6), fill=color)
    d.text(xy, text, fill=(255, 255, 255), font=font)


def draw(images: list, marks: list[Mark]) -> list:
    """Copies of the PIL images (original resolution) with the marks drawn in fixed pixel sizes."""
    from PIL import ImageDraw

    out = [im.copy() for im in images]
    for m in marks:
        im = out[m.image]
        w, h = im.size
        d = ImageDraw.Draw(im)
        sx, sy = w / 1000.0, h / 1000.0
        if m.kind == "point":
            x, y = m.xy[0] * sx, m.xy[1] * sy
            r = POINT_RADIUS
            d.ellipse((x - r, y - r, x + r, y + r), fill=m.color)
            if m.label:
                _label(d, (x + r + 200, y + r), m.label, m.color, m.label_style, 48)
        else:
            x1, x2 = sorted((m.xy[0] * sx, m.xy[2] * sx))
            y1, y2 = sorted((m.xy[1] * sy, m.xy[3] * sy))
            e = EXPAND["labelled" if m.label else "plain"]
            d.rectangle((x1 - e, y1 - e, x2 + e, y2 + e), outline=m.color, width=LINE_WIDTH)
            if m.label:                                       # at the box's lower right, as the reference
                _label(d, (min(w, x2 + e), min(h, y2 + e)), m.label, m.color, m.label_style, 48)
    return out
