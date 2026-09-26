"""프롬프트 빌더 — Qwen3.5 공식 채팅 템플릿 + 프로세서와 **토큰 단위로 같은** 입력을 만든다.

공식 출력 (transformers 5.16 Qwen3VLProcessor 로 직접 뽑은 것, tests 가 대조한다):

    이미지:  <|vision_start|><|image_pad|>×n<|vision_end|>          (n = 그 이미지의 LLM 토큰 수)
    비디오:  <|vision_start|>                                        ← 템플릿이 남긴 바깥 래핑
               <0.2 seconds><|vision_start|><|video_pad|>×n<|vision_end|>
               <1.2 seconds><|vision_start|><|video_pad|>×n<|vision_end|>
             <|vision_end|>
    턴:      <|im_start|>user\\n{content}<|im_end|>\\n
             <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>\\n

규칙 (공식 템플릿에서 확인한 것):
  * thinking off — 사내 기준선(73.3)도 off 였고, 사용자 결정으로 전부 끈다.
    ⚠️ 템플릿 **기본값에 기대면 안 된다.** Qwen3.5 템플릿은 크기마다 기본이 반대다 (2026-09-24 실측):
       0.8B·2B → enable_thinking 미지정이면 꺼짐 / 4B·9B → 미지정이면 **켜짐**
    그래서 빈 think 블록(`<think>\n\n</think>\n\n`)을 여기서 직접 넣는다 — 어느 크기든 off 다.
  * content 는 앞뒤 공백을 trim 한다 (템플릿의 `|trim`).
  * 학습 타깃은 `{answer}<|im_end|>`. <|im_end|> 가 생성 종료 토큰(tokenizer.eos_token)이다.
  * 멀티턴: 공식 템플릿은 **과거** assistant 턴을 think 블록 없이 렌더링한다. 우리는 모든 턴에
    빈 think 블록을 넣는다 — 추론 때 각 답변은 항상 빈 think 블록 뒤에서 생성되므로, **감독받는
    토큰의 조건**이 추론과 같아지는 쪽을 택했다. 싱글턴은 공식과 완전히 같다.

라벨은 문자 오프셋으로 마스킹한다 — 부분 문자열을 따로 토크나이즈해서 개수를 세면
BPE 경계에서 한 토큰씩 어긋날 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# thinking ON (the model writes its reasoning, then `</think>`, then the answer) - probes only (I-21); the live
# system and training use the empty block above
THINK_PREFIX = "<|im_start|>assistant\n<think>\n"

_PLACEHOLDER_RE = re.compile(r"<image>|<video>")


class PromptError(ValueError):
    """레코드가 템플릿 규칙과 안 맞는다 (placeholder 수 불일치 등). 해당 샘플은 건너뛴다."""


@dataclass
class BuiltPrompt:
    input_ids: torch.Tensor          # [1, L]
    labels: torch.Tensor             # [1, L]  (-100 = 손실 제외)
    mm_token_type_ids: torch.Tensor  # [1, L]  text=0, image=1, video=2
    text: str
    turns_used: int


class PromptBuilder:
    """Qwen3.5 입력 문자열·토큰을 만든다.

    Args:
        tokenizer: Qwen3.5 토크나이저 (fast — 오프셋 필요)
        image_token_id, video_token_id, vision_start_id, vision_end_id: 모델 config 의 ID.
            문자열은 토크나이저에서 역으로 얻는다 — 하드코딩하면 모델이 바뀔 때 조용히 틀린다.
        video_outer_wrap: 비디오 전체를 바깥 vision_start/end 로 한 번 더 감쌀지.
            transformers 5.x 프로세서는 템플릿의 `<|video_pad|>` 만 치환해서 바깥 래핑이 남는다
            (실측). 4.57 이하는 span 전체를 치환해서 안 남는다. 평가 파이프라인과 같게 맞춘다.
    """

    def __init__(
        self,
        tokenizer,
        image_token_id: int,
        video_token_id: int,
        vision_start_id: int,
        vision_end_id: int,
        video_outer_wrap: bool = True,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("fast 토크나이저가 필요하다 (라벨을 문자 오프셋으로 마스킹한다)")
        self.tok = tokenizer
        conv = tokenizer.convert_ids_to_tokens
        self.image_pad = conv(image_token_id)
        self.video_pad = conv(video_token_id)
        self.vs = conv(vision_start_id)
        self.ve = conv(vision_end_id)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        for name, s in (("image_pad", self.image_pad), ("video_pad", self.video_pad),
                        ("vision_start", self.vs), ("vision_end", self.ve)):
            if not s or s == tokenizer.unk_token:
                raise ValueError(f"{name} 토큰을 토크나이저에서 못 찾았다 — 모델/토크나이저 불일치")
        self.im_end = "<|im_end|>"
        if tokenizer.convert_tokens_to_ids(self.im_end) in (None, tokenizer.unk_token_id):
            raise ValueError("<|im_end|> 가 없다 — Qwen 채팅 토크나이저가 아니다")
        self.video_outer_wrap = video_outer_wrap

    @classmethod
    def from_model(cls, model, tokenizer=None, **kw) -> "PromptBuilder":
        tok = tokenizer or model.tokenizer
        return cls(tok, model.image_token_id, model.video_token_id,
                   model.vision_start_id, model.vision_end_id, **kw)

    # ------------------------------------------------------------- 비전 세그먼트
    def image_segment(self, n_tokens: int) -> str:
        return self.vs + self.image_pad * n_tokens + self.ve

    def video_segment(self, n_tokens_per_step: int, timestamps: list[float]) -> str:
        body = "".join(
            f"<{t:.1f} seconds>" + self.vs + self.video_pad * n_tokens_per_step + self.ve
            for t in timestamps
        )
        return (self.vs + body + self.ve) if self.video_outer_wrap else body

    # --------------------------------------------------------------- 턴 렌더링
    @staticmethod
    def substitute(turns: list[tuple[str, str]], segments: list[str], media_type: str) -> list[tuple[str, str]]:
        """user 텍스트의 <image>/<video> 를 세그먼트로 순서대로 치환한다.

        placeholder 가 하나도 없으면 첫 user 턴 앞에 모든 세그먼트를 붙인다
        (공식 템플릿에서 content 리스트 앞에 이미지를 두는 것과 같다).
        개수가 안 맞으면 PromptError — 어느 질문이 어느 이미지를 가리키는지 보장할 수 없다.
        """
        total = sum(len(_PLACEHOLDER_RE.findall(u)) for u, _ in turns)
        if total == 0:
            if not segments:
                return turns
            u0, a0 = turns[0]
            return [("".join(segments) + u0, a0)] + list(turns[1:])
        if total != len(segments):
            raise PromptError(
                f"placeholder {total}개 != {media_type} {len(segments)}개 — "
                "어느 질문이 어느 이미지를 가리키는지 보장할 수 없어 샘플을 버린다"
            )
        queue = iter(segments)
        out = []
        for u, a in turns:
            out.append((_PLACEHOLDER_RE.sub(lambda _m: next(queue), u), a))
        return out

    def render(self, turns: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
        """(전체 문자열, 감독 구간 [(start, end), ...] 문자 오프셋)."""
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        pos = 0
        for u, a in turns:
            head = f"<|im_start|>user\n{u.strip()}{self.im_end}\n{ASSISTANT_PREFIX}"
            parts.append(head)
            pos += len(head)
            target = a.strip() + self.im_end
            parts.append(target)
            spans.append((pos, pos + len(target)))
            pos += len(target)
            parts.append("\n")
            pos += 1
        return "".join(parts), spans

    def build(
        self,
        turns: list[tuple[str, str]],
        segments: list[str],
        media_type: str,
        max_length: int | None = None,
    ) -> BuiltPrompt:
        """토큰·라벨·modality 마스크를 만든다. 길면 **뒤쪽 턴부터** 버린다 (비전은 자르지 않는다)."""
        if not turns:
            raise PromptError("대화 턴이 없다")
        subst = self.substitute(turns, segments, media_type)
        n = len(subst)
        while n >= 1:
            text, spans = self.render(subst[:n])
            enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
            if max_length is None or len(enc.input_ids) <= max_length:
                break
            # 비전 세그먼트가 뒤쪽 턴에 있으면 그 턴을 버릴 때 이미지도 같이 빠져야 한다 —
            # placeholder 가 첫 턴에만 있는 일반적인 경우만 안전하게 자른다.
            if any(_PLACEHOLDER_RE.search(u) for u, _ in turns[1:n]) or n == 1:
                raise PromptError(f"시퀀스 {len(enc.input_ids)} 토큰 > max_length {max_length}")
            n -= 1
        ids = torch.tensor(enc.input_ids, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for i, (s, e) in enumerate(enc.offset_mapping):
            if e > s and any(ss <= s and e <= ee for ss, ee in spans):
                labels[i] = ids[i]
        mm = torch.zeros_like(ids)
        mm[ids == self.image_token_id] = 1
        mm[ids == self.video_token_id] = 2
        if not bool((labels != -100).any()):
            raise PromptError("감독할 답변 토큰이 없다")
        return BuiltPrompt(
            input_ids=ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            mm_token_type_ids=mm.unsqueeze(0),
            text=text,
            turns_used=n,
        )

    def build_query(self, question: str, segments: list[str],
                    assistant_prefix: str = ASSISTANT_PREFIX) -> torch.Tensor:
        """추론용 — 답변 없이 생성 프롬프트까지만. 스트리밍 평가가 쓴다. assistant_prefix=THINK_PREFIX 는 thinking 켬."""
        (u, _), = self.substitute([(question, "")], segments, "visual")
        text = f"<|im_start|>user\n{u.strip()}{self.im_end}\n{assistant_prefix}"
        return torch.tensor(self.tok(text, add_special_tokens=False).input_ids).unsqueeze(0)

    def summarize(self, built: BuiltPrompt, width: int = 400) -> str:
        """디버그 출력 — 패드 런을 ×n 으로 줄이고 감독 구간을 ⟦ ⟧ 로 표시한다."""
        text = built.text
        text = re.sub(
            f"({re.escape(self.image_pad)})+",
            lambda m: f"{self.image_pad}×{m.group(0).count(self.image_pad)}", text,
        )
        text = re.sub(
            f"({re.escape(self.video_pad)})+",
            lambda m: f"{self.video_pad}×{m.group(0).count(self.video_pad)}", text,
        )
        sup = self.tok.decode(built.labels[0][built.labels[0] != -100].tolist())
        return f"{text[:width]}{'…' if len(text) > width else ''}\n  ⟦감독⟧ {sup!r}"
