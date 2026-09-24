"""장면 지도 프롬프트 — CUT3R 스트림 출력을 VLM 이 **이미 읽을 줄 아는 형식**(이미지·텍스트)으로.

왜 임베딩이 아니라 지도인가 (2026-09-24 사용자 방향):
    3D 인코더 임베딩을 LLM 에 넣으면 LLM 에게는 새 모달리티라 대량 학습이 필요하다. 문헌에서도 같은 데이터로
    기하 없이 학습한 쪽 대비 기하 인코더의 몫은 작다 — VLM-3R +3.2 (57.7 → 60.9), VG-LLM +0.9 (49.8 → 50.7).
    M2 파일럿에서는 프로젝터가 기하 내용 대신 답 형식만 배웠다. 반면 사전학습된 VLM 은 지도·도면 이미지와
    숫자 텍스트를 **이미 읽는다** — 기하를 그 형식으로 바꿔 주면 학습 없이(zero-shot) 쓸 수 있다.
    참고: See&Trek (BEV 궤적 그림, 학습 없이 VSI +1.4~3.5), GPT4Scene (BEV + 객체 마커),
    Thinking in Space (정답 인지 지도를 주면 상대 거리 +20~32%).

라이브 설계 (사용자 결정 2026-09-24 — 질문 후 1초 안에 답):
    지도는 스트림 중에 계속 갱신되고 프롬프트(프리픽스 캐시)에 **미리** 들어가 있다. 질문이 오면 질문
    토큰만 처리한다. 지도는 키프레임 뒤·질문 앞에 두어, 지도만 바뀌면 꼬리만 다시 계산하면 된다.

좌표: CUT3R 점맵은 첫 프레임 카메라 좌표계(= 스트림 월드), 미터 단위로 학습됐다. 위쪽은 카메라들의
평균 "위" 방향(OpenCV 카메라의 −y)으로 잡는다 — 손에 들거나 로봇에 단 영상은 카메라가 대체로 서 있다.
지도는 벽에 맞춰 돌린다 (점들의 바닥 투영 bbox 가 가장 작아지는 각) → 직사각형 방이 축에 맞게 보인다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 오래된 Pillow
        return ImageFont.load_default()


class SceneMap:
    """스트리밍 3D 점·카메라 → 위에서 본 지도 이미지 + 측정값 텍스트.

    Args:
        stride: 점맵에서 몇 픽셀마다 한 점을 쓸지 (512×384 → stride 4 면 프레임당 1.2만 점)
        conf_quantile: 프레임마다 신뢰도 하위 이 비율을 버린다
        max_points: 넘으면 복셀(voxel m) 다운샘플 — 긴 스트림에서도 메모리 상수
    """

    def __init__(self, stride: int = 4, conf_quantile: float = 0.5, voxel: float = 0.05,
                 max_points: int = 400_000) -> None:
        self.stride = stride
        self.conf_quantile = conf_quantile
        self.voxel = voxel
        self.max_points = max_points
        self.reset()

    def reset(self) -> None:
        self._pts = np.zeros((0, 3), np.float32)
        self._cams: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # frame → (중심, 아래축, 앞축)

    @property
    def n_points(self) -> int:
        return len(self._pts)

    @property
    def n_frames(self) -> int:
        return len(self._cams)

    # ------------------------------------------------------------------- 누적
    def add(self, out, frame_index: int | None = None) -> None:
        """CUT3R GeomOutput (decode_points=True 로 만든 것) 한 프레임을 넣는다."""
        if out.pointmap is None or "c2w" not in out.extra:
            raise ValueError("점맵·포즈가 없다 — CUT3RStream.decode_points = True 로 인제스트해야 한다")
        idx = out.frame_index if frame_index is None else frame_index
        pm = out.pointmap[0, :, :: self.stride, :: self.stride].reshape(3, -1).T.float().cpu().numpy()
        cf = out.conf[0, 0, :: self.stride, :: self.stride].reshape(-1).float().cpu().numpy()
        keep = np.isfinite(pm).all(1) & (cf >= np.quantile(cf, self.conf_quantile))
        self._pts = np.concatenate([self._pts, pm[keep].astype(np.float32)])
        c2w = out.extra["c2w"][0].float().cpu().numpy()
        self._cams[idx] = (c2w[:3, 3].copy(), c2w[:3, 1].copy(), c2w[:3, 2].copy())
        if len(self._pts) > self.max_points:
            q = np.floor(self._pts / self.voxel).astype(np.int64)
            _, first = np.unique(q, axis=0, return_index=True)
            self._pts = self._pts[np.sort(first)]
            if len(self._pts) > self.max_points:  # 복셀로도 모자라면 균등 부분표본 (결정적)
                pick = np.random.default_rng(len(self._cams)).choice(len(self._pts), self.max_points, replace=False)
                self._pts = self._pts[np.sort(pick)]

    # ------------------------------------------------------------ 좌표 정렬
    def _frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(바닥 평면 2D 점 [N,2], 높이 [N], 월드→지도 회전 R[3,3]). 지도 x·y 는 벽에 맞춘 축, z 는 위."""
        downs = np.stack([d for _, d, _ in self._cams.values()])
        up = -downs.mean(0)
        up /= np.linalg.norm(up) + 1e-9
        ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(up, ref)
        ax /= np.linalg.norm(ax)
        ay = np.cross(up, ax)
        R = np.stack([ax, ay, up])                              # 행 = 새 축
        p = self._pts @ R.T
        # 벽 정렬: 바닥 투영의 bbox(2~98 백분위)가 가장 작은 각 (0~89°)
        sub = p[:: max(1, len(p) // 20000), :2]
        best, best_area = 0.0, np.inf
        for deg in range(0, 90, 2):
            t = np.deg2rad(deg)
            rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            q = sub @ rot.T
            lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)
            area = float(np.prod(hi - lo))
            if area < best_area:
                best, best_area = t, area
        rot2 = np.array([[np.cos(best), -np.sin(best), 0], [np.sin(best), np.cos(best), 0], [0, 0, 1]])
        R = rot2 @ R
        p = self._pts @ R.T
        return p[:, :2], p[:, 2], R

    def to_map(self, xyz: np.ndarray) -> np.ndarray:
        """World points [N,3] -> floor-plan coordinates [N,3] (x, y wall-aligned metres; z above the floor)."""
        if self.n_points < 100 or not self._cams:
            return np.asarray(xyz, dtype=np.float64)
        _, z, R = self._frame()
        p = np.asarray(xyz, dtype=np.float64) @ R.T
        p[:, 2] -= np.percentile(z, 2)
        return p

    def facts(self) -> dict:
        """측정값 — 방 크기(바닥 bbox 2~98 백분위), 천장 높이, 카메라 이동 거리."""
        if self.n_points < 100 or not self._cams:
            return {}
        xy, z, R = self._frame()
        lo, hi = np.percentile(xy, 2, 0), np.percentile(xy, 98, 0)
        w, l = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        zl, zh = np.percentile(z, 2), np.percentile(z, 98)
        cams = np.stack([c for c, _, _ in (self._cams[k] for k in sorted(self._cams))])
        path = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum()) if len(cams) > 1 else 0.0
        return {"width_m": w, "length_m": l, "area_m2": w * l, "height_m": float(zh - zl), "path_m": path}

    def text(self, image: bool = True) -> str:
        """Prompt text. image=False: measured facts only (no map image in the prompt)."""
        f = self.facts()
        if not f:
            return ""
        if not image:
            # Zero-shot M2 run (2026-09-24): the map *image* hurt appearance-order / relative-direction questions
            # (-7..-10) while the measured area helped room size (+13..+19) -> try the facts alone.
            return (
                "Measured from a 3D reconstruction of the video: "
                f"the room is about {f['width_m']:.1f} m by {f['length_m']:.1f} m "
                f"(about {f['area_m2']:.0f} square meters), about {f['height_m']:.1f} m high, "
                f"and the camera moved about {f['path_m']:.1f} m. "
                "Answer directly in the requested format without explanation.\n"
            )
        return (
            "The image after the video frames is a top-down map of the scene reconstructed from the video "
            "(grid lines every 1 meter; numbered dots are the camera positions at the video frames, numbered in time order; "
            "the arrow shows where the camera faces at the end). "
            f"Measured from the reconstruction: the room is about {f['width_m']:.1f} m by {f['length_m']:.1f} m "
            f"(about {f['area_m2']:.0f} square meters), about {f['height_m']:.1f} m high, "
            f"and the camera moved about {f['path_m']:.1f} m. "
            # 지도·측정값이 앞에 붙으면 베이스가 설명형으로 답하기 시작한다 (M2 스모크: 선택형 형식 실패 31%).
            # 이미지 모드와 같은 현상 — 형식 지시로 되돌린다 (See&Trek 등 프롬프팅 방법도 설계된 텍스트를 쓴다).
            "Use the map if it helps, and answer directly in the requested format without explanation.\n"
        )

    # -------------------------------------------------------------- 렌더링
    def render(self, size: int = 512, labels: dict[int, int] | None = None):
        """지도 이미지 (PIL). labels: {프레임 인덱스: 표시할 번호} — 보통 키프레임 순번 (1부터)."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (size, size), (0, 0, 0))
        if self.n_points < 100 or not self._cams:
            return img
        xy, z, R = self._frame()
        cams = {k: (R @ c, R @ f) for k, (c, _, f) in self._cams.items()}
        lo = np.minimum(np.percentile(xy, 1, 0), np.min([c[:2] for c, _ in cams.values()], 0)) - 0.5
        hi = np.maximum(np.percentile(xy, 99, 0), np.max([c[:2] for c, _ in cams.values()], 0)) + 0.5
        span = float(max(hi - lo))
        center = (lo + hi) / 2
        scale = (size - 24) / span                                # m → 픽셀
        org = center - span / 2

        def to_px(p2):  # 지도 y 는 위쪽이 + → 이미지 행은 아래로 증가
            u = 12 + (p2[..., 0] - org[0]) * scale
            v = size - 12 - (p2[..., 1] - org[1]) * scale
            return u, v

        # 점: 칸마다 최고 높이로 색, 밀도로 밝기 (벽·가구는 밝게, 바닥은 어둡게)
        u, v = to_px(xy)
        ui, vi = u.astype(int), v.astype(int)
        ok = (ui >= 0) & (ui < size) & (vi >= 0) & (vi < size)
        ui, vi, zz = ui[ok], vi[ok], z[ok]
        zn = np.clip((zz - np.percentile(zz, 2)) / max(1e-3, np.ptp(zz)), 0, 1)
        cnt = np.zeros((size, size), np.float32)
        top = np.zeros((size, size), np.float32)
        np.add.at(cnt, (vi, ui), 1)
        np.maximum.at(top, (vi, ui), zn)
        bright = np.clip(np.log1p(cnt) / np.log1p(max(1.0, np.percentile(cnt[cnt > 0], 95))), 0, 1)
        bright = np.where(cnt > 0, 0.35 + 0.65 * bright, 0.0)   # 점이 드문 바닥도 보이게
        rgb = np.stack([0.25 + 0.75 * top, 0.35 + 0.5 * (1 - np.abs(top - 0.5) * 2), 0.9 - 0.7 * top], -1)
        arr = (rgb * bright[..., None] * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)

        # 1m 격자 + 축척
        for k in range(int(np.floor(org[0])), int(np.ceil(org[0] + span)) + 1):
            x = 12 + (k - org[0]) * scale
            d.line([(x, 0), (x, size)], fill=(55, 55, 55), width=1)
        for k in range(int(np.floor(org[1])), int(np.ceil(org[1] + span)) + 1):
            y = size - 12 - (k - org[1]) * scale
            d.line([(0, y), (size, y)], fill=(55, 55, 55), width=1)
        d.line([(12, size - 8), (12 + scale, size - 8)], fill=(255, 255, 255), width=3)
        d.text((16 + scale, size - 18), "1 m", fill=(255, 255, 255), font=_font(12))

        # 카메라 궤적 + 번호 + 마지막 시선 방향
        order = sorted(cams)
        path = [tuple(float(t) for t in to_px(cams[k][0][:2])) for k in order]
        if len(path) > 1:
            d.line(path, fill=(255, 60, 60), width=2)
        f = _font(12)
        for k in order:
            if labels and k in labels:
                x, y = to_px(cams[k][0][:2])
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 60, 60))
                d.text((x - 5 if labels[k] < 10 else x - 8, y - 7), str(labels[k]), fill=(255, 255, 255), font=f)
        c, fw = cams[order[-1]]
        x, y = to_px(c[:2])
        dirv = fw[:2] / (np.linalg.norm(fw[:2]) + 1e-9)
        tip = (x + 22 * dirv[0], y - 22 * dirv[1])
        d.line([(x, y), tip], fill=(255, 255, 0), width=3)
        d.ellipse([tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4], fill=(255, 255, 0))
        return img


def keyframe_labels(frame_indices: list[int]) -> dict[int, int]:
    """키프레임 프레임 인덱스 → 지도에 쓸 번호 (1부터, 시간순)."""
    return {fi: n + 1 for n, fi in enumerate(sorted(frame_indices))}


def zeros_like_embeds(embeds: list[torch.Tensor], n: int) -> list[torch.Tensor]:
    """지도 이미지 토큰에는 기하 주입이 없다 — 레이어별 0 임베딩 n 개."""
    return [torch.zeros(n, e.shape[-1], dtype=e.dtype, device=e.device) for e in embeds]
