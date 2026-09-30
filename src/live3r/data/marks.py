"""Marks the question refers to, drawn onto the frames - our implementation, from the data format.

SPAR-style records (configs/data/vgllm.yaml: spar_234k, spar_234k_cot) ask about objects marked in the images: "the
pillar (red point)", "the desk (red bbox) ... the mug (blue bbox)", "Object3 (bbox3)". The record carries the marks
in `spar_info` (a JSON string or object). The format, read from the data (2026-09-30 data pass):

    {color}_point   [[x, y]]                 color: red, blue, green, yellow
    {color}_bbox    [[x1, y1, x2, y2]]
    point_list      [[[x, y], ...]]          objects the question lists as Object{i} (bbox{i})
    bbox_list       [[[x1, y1, x2, y2], ...]]
    point_img_idx   [[i, ...]]               the image each point is on; bbox_img_idx the same for boxes
    type            the question type (not a mark)

Coordinates are per axis in 0..1000 of the image. Which image a colored mark is on: the index lists follow the order in
which the question names the marks ("tv (in Frame-16, Object0) (green bbox) ... books (in Frame-21, Object1) (blue
bbox)" with [[16, 21, ...]]) - all 13 sampled records with marks agree. A single image needs no index; a multi-image
record without an index has its marks on the first image ("shown with a red bbox in the first image"). A multi-image
record whose question does not name every color it has an index for is ambiguous: `plan` returns None and the dataset
leaves the record out rather than draw a mark on a guessed frame.

Style (ours): points as filled discs with a white rim, boxes as outlines, both sized to the image; list boxes carry
their label ("bbox3") so the question's Object3 can be found.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
# a color named as a mark: "(red point)", "(green bbox)", "shown with a red bbox", "the red point of bed"
_MENTION = re.compile(r"\b(red|blue|green|yellow)\s+(?:point|bbox|box|dot|mark)", re.I)


@dataclass(frozen=True)
class Mark:
    kind: str                       # "point" | "bbox"
    xy: tuple[float, ...]           # (x, y) or (x1, y1, x2, y2), per axis in 0..1000
    image: int
    color: tuple[int, int, int]
    label: str | None = None


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


def plan(info: dict, question: str, n_images: int) -> list[Mark] | None:
    """Where each mark goes; None when that cannot be known for sure (the record is then left out)."""
    if n_images <= 0:
        return None
    marks: list[Mark] = []
    named = mention_order(question)
    for kind in ("point", "bbox"):
        colors = [c for c in COLORS if _coords(info.get(f"{c}_{kind}"))]
        if not colors:
            continue
        idx = _ints(info.get(f"{kind}_img_idx"))
        if n_images == 1 or not idx:
            where = {c: 0 for c in colors}
        else:
            if not set(colors) <= set(named):
                return None                                  # the order of the index list is not known
            ordered = [c for c in named if c in colors]
            if len(idx) != len(ordered):
                return None
            where = dict(zip(ordered, idx))
        for c in colors:
            for xy in _coords(info[f"{c}_{kind}"]):
                marks.append(Mark(kind, tuple(float(v) for v in xy), where[c], COLORS[c]))
    for kind, key in (("bbox", "bbox_list"), ("point", "point_list")):
        items = _coords(info.get(key))
        if not items:
            continue
        idx = _ints(info.get(f"{kind}_img_idx")) or [0] * len(items)
        if len(idx) != len(items):
            return None
        for i, (xy, im) in enumerate(zip(items, idx)):
            marks.append(Mark(kind, tuple(float(v) for v in xy), im, COLORS["red"], label=f"{kind}{i}"))
        break                                                # boxes when there are boxes, else points
    for m in marks:
        if not 0 <= m.image < n_images or len(m.xy) != (2 if m.kind == "point" else 4):
            return None
    return marks or None


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                        # Pillow < 10.1: a fixed bitmap font
        return ImageFont.load_default()


def draw(images: list, marks: list[Mark]) -> list:
    """Copies of the PIL images with the marks drawn (sizes relative to each image)."""
    from PIL import ImageDraw

    out = [im.copy() for im in images]
    for m in marks:
        im = out[m.image]
        w, h = im.size
        d = ImageDraw.Draw(im)
        sx, sy = w / 1000.0, h / 1000.0
        if m.kind == "point":
            x, y = m.xy[0] * sx, m.xy[1] * sy
            r = max(3, round(0.012 * min(w, h)))
            d.ellipse((x - r, y - r, x + r, y + r), fill=m.color, outline=(255, 255, 255), width=max(1, r // 3))
            if m.label:
                d.text((x + r + 2, y - r), m.label, fill=m.color, font=_font(max(10, 2 * r)))
        else:
            x1, y1, x2, y2 = m.xy[0] * sx, m.xy[1] * sy, m.xy[2] * sx, m.xy[3] * sy
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            lw = max(2, round(0.005 * max(w, h)))
            d.rectangle((x1, y1, x2, y2), outline=m.color, width=lw)
            if m.label:
                font = _font(max(10, round(0.03 * min(w, h))))
                tb = d.textbbox((x1 + lw, y1 + lw), m.label, font=font)
                d.rectangle(tb, fill=m.color)
                d.text((x1 + lw, y1 + lw), m.label, fill=(255, 255, 255), font=font)
    return out
