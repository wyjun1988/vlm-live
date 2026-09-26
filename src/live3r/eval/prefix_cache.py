"""시각 프리픽스 캐시 — 같은 시각 입력에 여러 질문이 올 때, 시각 부분은 **한 번만** 프리필한다.

라이브 설계의 핵심과 같은 구조다: 영상은 한 번 먹고, 질문이 오면 질문 토큰만 처리한다.
(VSI-Bench 는 영상 하나에 평균 18문항 — 매번 키프레임 32장을 다시 프리필하면 18배 낭비다.)

    프롬프트 = <|im_start|>user\\n [시각 세그먼트들] [질문] <|im_end|>\\n<|im_start|>assistant\\n<think>…
               └──────────── 프리픽스 (공유) ─────────┘└──── 질문마다 다름 ────┘

Qwen3.5 는 32층 중 24층이 선형 어텐션(재귀 상태), 8층이 일반 어텐션(KV). 프리픽스를 한 번
돌린 뒤의 캐시(상태 + KV)를 질문마다 복제해서 이어 쓴다. 기하 주입은 프리픽스(시각 토큰이 있는
구간)에서만 일어나므로 캐시에 이미 반영돼 있다.

M-RoPE 주의: 프리픽스 forward 가 `model.rope_deltas` 를 설정하고, 이어지는 생성은 그 값으로
텍스트 위치를 계산한다. 다른 입력의 forward 가 중간에 끼면 덮어써지므로 답하기 직전에 복원한다.

정합성: 캐시를 쓴 답 == 매번 전체를 프리필한 답 (greedy) — tests/test_prefix_cache.py.
"""

from __future__ import annotations

import copy

import torch


class PrefixCache:
    def __init__(self, model, prompt, segments: list[str], pixel_kwargs: dict, geometry=None,
                 device: torch.device | str = "cpu") -> None:
        self.model = model
        self.prompt = prompt
        self.segments = segments
        self.device = torch.device(device)
        tok = prompt.tok
        # 프리픽스는 비전 세그먼트의 끝(<|vision_end|>, 특수 토큰)에서 끊는다 → BPE 경계가 깔끔하다
        prefix_text = "<|im_start|>user\n" + "".join(segments)
        self.prefix_ids = torch.tensor(tok(prefix_text, add_special_tokens=False).input_ids).unsqueeze(0)
        ids = self.prefix_ids.to(self.device)
        mm = torch.zeros_like(ids)
        mm[ids == model.image_token_id] = 1
        mm[ids == model.video_token_id] = 2
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                  use_cache=True, **{k: v.to(self.device) for k, v in pixel_kwargs.items()})
        with torch.no_grad():
            if geometry is not None and model.injector is not None:
                with model.injector.primed(geometry.embeds, model.visual_pos_mask(ids)):
                    out = model.base(**kw)
            else:
                out = model.base(**kw)
        self.cache = out.past_key_values
        self.rope_deltas = _inner(model.base).rope_deltas

    @torch.no_grad()
    def answer(self, question: str, thinking: bool = False, **gen_kwargs) -> tuple[str, torch.Tensor]:
        """질문 토큰만 처리한다. (답, 전체 입력 id) 를 돌려준다.

        thinking=True: the model reasons first (THINK_PREFIX); the returned text is everything it wrote, with the
        `</think>` marker kept so the caller can split it (see `split_thinking`)."""
        from ..data.prompt import ASSISTANT_PREFIX, THINK_PREFIX

        full = self.prompt.build_query(question, self.segments, THINK_PREFIX if thinking else ASSISTANT_PREFIX)
        p = self.prefix_ids.shape[1]
        if full.shape[1] <= p or not torch.equal(full[0, :p], self.prefix_ids[0]):
            raise RuntimeError("질문 프롬프트가 캐시한 프리픽스로 시작하지 않는다 — 세그먼트/템플릿 불일치")
        ids = full.to(self.device)
        _inner(self.model.base).rope_deltas = self.rope_deltas  # 다른 입력이 덮어썼을 수 있다
        out = self.model.base.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            past_key_values=copy.deepcopy(self.cache), **gen_kwargs,
        )
        if not thinking:
            return self.prompt.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), full
        text = self.prompt.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        for t in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(t, "")
        return text, full


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(reasoning, answer, closed). Unclosed = the budget ran out before `</think>`: the answer is then empty,
    which scores as wrong - that is the honest cost of thinking under a token budget."""
    if "</think>" in text:
        think, ans = text.split("</think>", 1)
        return think.strip(), ans.strip(), True
    return text.strip(), "", False


def _inner(base):
    """Qwen3_5ForConditionalGeneration → Qwen3_5Model (rope_deltas 를 들고 있는 쪽)."""
    m = getattr(base, "model", base)
    return getattr(m, "model", m) if not hasattr(m, "rope_deltas") else m
