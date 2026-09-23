"""CUT3R 어댑터 — 스트리밍 기하 인코더 기본값.

CUT3R 은 고정 크기 implicit memory 를 재귀 갱신하는 RNN 형태라 프레임당 O(1) 이다.
VLM-3R 이 검증한 경로이므로 재현 기준선으로 쓴다.

**이 어댑터는 CUT3R 저장소가 PYTHONPATH 에 있어야 동작한다.**
    git clone https://github.com/CUT3R/CUT3R && export PYTHONPATH=$PWD/CUT3R/src:$PYTHONPATH
    # 체크포인트: cut3r_512_dpt_4_64.pth (512x384+, 4-64 views, DPT head)

내부 구조(클래스명·블록 속성명)는 저장소 리비전에 따라 달라질 수 있어 **설정으로 뺐다**.
GPU 머신에서 `python scripts/verify_geometry_adapter.py --name cut3r` 를 먼저 돌려
탭 레이어 차원과 프레임당 지연을 실측한 뒤 configs/*.yaml 의 hidden_size 를 맞춰라.
"""

from __future__ import annotations

import importlib
import logging

import torch

from ..config import GeometryConfig
from .base import GeomOutput, GeometryStream
from .registry import register

logger = logging.getLogger(__name__)

# 저장소 리비전에 따라 달라지는 지점들. 실패 시 아래 후보를 순서대로 시도한다.
_MODEL_PATHS = (
    "dust3r.model.ARCroco3DStereo",
    "src.dust3r.model.ARCroco3DStereo",
    "cut3r.model.ARCroco3DStereo",
)
_BLOCK_ATTRS = ("dec_blocks", "decoder.blocks", "dec_blocks2", "enc_blocks")


class CUT3RStream(GeometryStream):
    """CUT3R 을 GeometryStream 인터페이스에 맞춘 래퍼.

    탭 토큰은 디코더 블록에 forward hook 을 걸어 가져온다 — 원본 코드를 고치지 않는다.
    """

    is_streaming = True

    def __init__(
        self,
        checkpoint: str,
        tap_layers: tuple[int, ...] = (11, 17, 23),
        image_size: int = 512,
        hidden_size: int | None = None,
        model_path: str | None = None,
        block_attr: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        net = _load_cut3r(checkpoint, model_path, device)
        blocks = _find_blocks(net, block_attr)
        bad = [t for t in tap_layers if t >= len(blocks)]
        if bad:
            raise ValueError(f"탭 레이어 {bad} 가 블록 수 {len(blocks)} 를 넘는다")

        inferred = hidden_size or getattr(getattr(net, "dec_embed_dim", None), "__int__", lambda: None)()
        super().__init__(tap_layers, inferred or 768)
        self.net = net
        self.blocks = blocks
        self.image_size = image_size
        self._taps: dict[int, torch.Tensor] = {}
        self._handles = [
            blocks[t].register_forward_hook(self._make_tap(t)) for t in self.tap_layers
        ]
        self._mem = None
        self.reset()

    def _make_tap(self, idx: int):
        def hook(module, args, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(t, torch.Tensor):
                self._taps[idx] = t.detach()
        return hook

    def reset(self) -> None:
        self._frame_index = 0
        self._taps.clear()
        # CUT3R 은 내부에 재귀 상태를 들고 있다. 초기화 API 이름이 리비전마다 다르다.
        for name in ("reset", "init_state", "_init_state"):
            fn = getattr(self.net, name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except TypeError:
                    continue
        self._mem = None

    @torch.no_grad()
    def ingest(self, frames: torch.Tensor) -> GeomOutput:
        raise NotImplementedError(
            "CUT3R 의 프레임 단위 추론 진입점을 저장소 리비전에 맞춰 연결해야 한다.\n"
            "할 일: CUT3R/demo.py 에서 프레임 루프가 호출하는 함수를 찾아 여기에 연결하고,\n"
            "       hook 이 채운 self._taps 로 GeomOutput 을 만들어라.\n"
            "확인:  python scripts/verify_geometry_adapter.py --name cut3r --checkpoint <경로>\n"
            "지금은 파이프라인 검증에 geometry.name=dummy 를 써라."
        )

    def state_bytes(self) -> int:
        if self._mem is None:
            return 0
        if isinstance(self._mem, torch.Tensor):
            return self._mem.numel() * self._mem.element_size()
        return sum(t.numel() * t.element_size() for t in self._mem if isinstance(t, torch.Tensor))


def _load_cut3r(checkpoint: str, model_path: str | None, device):
    paths = (model_path,) if model_path else _MODEL_PATHS
    last = None
    for path in paths:
        try:
            mod_name, cls_name = path.rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except Exception as exc:
            last = exc
            continue
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = ckpt.get("args", None)
        net = cls.from_pretrained(checkpoint) if hasattr(cls, "from_pretrained") else cls(args)
        if not hasattr(cls, "from_pretrained"):
            net.load_state_dict(ckpt.get("model", ckpt), strict=False)
        return net.to(device).eval()
    raise ImportError(
        f"CUT3R 모델 클래스를 못 찾았다 (마지막 오류: {last}).\n"
        "  git clone https://github.com/CUT3R/CUT3R\n"
        "  export PYTHONPATH=$PWD/CUT3R/src:$PYTHONPATH\n"
        "클래스 경로가 다르면 GeometryConfig 에 model_path 를 넘겨라."
    )


def _find_blocks(net, block_attr: str | None):
    for attr in ((block_attr,) if block_attr else _BLOCK_ATTRS):
        obj = net
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise AttributeError(
        f"CUT3R 디코더 블록 리스트를 못 찾았다. 시도한 경로: {_BLOCK_ATTRS}. "
        "GeometryConfig 에 block_attr 을 명시해라."
    )


@register("cut3r")
def _build(cfg: GeometryConfig) -> CUT3RStream:
    if not cfg.checkpoint:
        raise ValueError("geometry.checkpoint 가 필요하다 (cut3r_512_dpt_4_64.pth)")
    return CUT3RStream(
        checkpoint=cfg.checkpoint,
        tap_layers=cfg.tap_layers,
        image_size=cfg.image_size,
        hidden_size=cfg.hidden_size,
    )
