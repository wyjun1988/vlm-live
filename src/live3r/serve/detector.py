"""Open-vocabulary object detector for the object map (I-09 in docs/BRAINSTORM.md).

OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0, 622 MB) with a fixed indoor vocabulary — question-agnostic,
so results are valid for the live system (rule I-24). The VLM's own grounding also works but is slow on M2
(7-18 s per frame) and needs the object names up front.

Boxes come back normalised to the original image ([0,1], x1 y1 x2 y2), the same format as `parse_detections`,
so `ObjectMap.add` takes either.
"""

from __future__ import annotations

import numpy as np
import torch

# Common indoor object categories (hand-written, not taken from any benchmark's questions).
INDOOR_VOCAB = [
    "chair", "armchair", "office chair", "stool", "bench", "sofa", "couch", "table", "coffee table", "dining table",
    "desk", "bed", "nightstand", "dresser", "wardrobe", "cabinet", "kitchen cabinet", "shelf", "bookshelf",
    "bookcase", "tv", "monitor", "computer", "laptop", "keyboard", "printer", "telephone", "door", "window",
    "curtain", "blinds", "mirror", "picture", "poster", "whiteboard", "clock", "lamp", "ceiling light", "fan",
    "heater", "radiator", "air conditioner", "fireplace", "plant", "potted plant", "vase", "pillow", "blanket",
    "rug", "carpet", "towel", "trash bin", "trash can", "recycling bin", "basket", "box", "bag", "backpack",
    "suitcase", "shoes", "bottle", "cup", "mug", "bowl", "plate", "kettle", "toaster", "microwave", "oven", "stove",
    "refrigerator", "sink", "faucet", "dishwasher", "washing machine", "dryer", "toilet", "bathtub", "shower",
    "bathroom vanity", "soap dispenser", "toilet paper", "counter", "kitchen counter", "piano", "guitar",
    "speaker", "book", "light switch", "outlet", "coat rack", "clothes", "jacket", "hat", "umbrella", "ladder",
    "stairs", "column", "board", "cart", "printer paper", "tissue box", "remote control", "headphones",
]


class OWLv2Detector:
    """Callable: HxWx3 uint8 frame -> [{"bbox": [x1, y1, x2, y2] in 0..1, "label": str, "score": float}]."""

    def __init__(self, path: str = "checkpoints/owlv2-base-patch16-ensemble", device: str | torch.device = "cpu",
                 vocab: list[str] | None = None, threshold: float = 0.25, nms_iou: float = 0.5,
                 max_per_label: int = 8) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = torch.device(device)
        self.proc = Owlv2Processor.from_pretrained(path)
        self.model = Owlv2ForObjectDetection.from_pretrained(path).to(self.device).eval()
        self.vocab = list(vocab or INDOOR_VOCAB)
        self.queries = [f"a photo of a {v}" for v in self.vocab]
        self.threshold, self.nms_iou, self.max_per_label = threshold, nms_iou, max_per_label

    @torch.no_grad()
    def __call__(self, frame, vocab: list[str] | None = None) -> list[dict]:
        """vocab: override the object names for this call (e.g. an oracle vocabulary for an upper-bound probe)."""
        from PIL import Image
        from torchvision.ops import batched_nms

        names = list(vocab) if vocab else self.vocab
        queries = [f"a photo of a {v}" for v in names] if vocab else self.queries
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
        w, h = img.size
        inputs = self.proc(text=[queries], images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        side = max(w, h)  # OWLv2 pads to a square (bottom/right), boxes come back in that square
        res = self.proc.post_process_grounded_object_detection(out, threshold=self.threshold,
                                                               target_sizes=[(side, side)])[0]
        boxes, scores, labels = res["boxes"].cpu(), res["scores"].cpu(), res["labels"].cpu()
        if len(boxes) == 0:
            return []
        keep = batched_nms(boxes, scores, labels, self.nms_iou)
        dets, per = [], {}
        for k in keep.tolist():
            lab = int(labels[k])
            if per.get(lab, 0) >= self.max_per_label:
                continue
            x1, y1, x2, y2 = boxes[k].tolist()
            box = [max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            per[lab] = per.get(lab, 0) + 1
            dets.append({"bbox": box, "label": names[lab], "score": float(scores[k])})
        return dets
