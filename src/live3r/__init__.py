"""Live3R — streaming 3D-aware VLM.

스트리밍 3D 기하 인코더의 잠재 토큰을 Qwen3.5 소형 VLM 디코더에 다깊이 residual 로 주입해
프레임당 상수 비용으로 공간을 이해하는 모델. 설계는 docs/DESIGN.md 참고.
"""

__version__ = "0.0.1"
