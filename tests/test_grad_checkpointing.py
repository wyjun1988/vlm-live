"""gradient checkpointing + 주입 훅 — 역전파 재계산 때도 같은 주입이 일어나야 한다.

checkpointing 은 역전파 중 레이어 forward 를 다시 돌리고, 그때 forward hook 도 다시 불린다.
주입 상태가 이미 비워져 있으면(컨텍스트 매니저 방식) 재계산 그래프가 원래와 달라진다.
그래서 학습은 keep_primed=True → backward → injector.clear() 순서를 쓴다.
"""

import pytest
import torch
from test_train_step import build, image_batch, trainable_only_projectors


def _grads(m, b, keep_primed: bool):
    m.zero_grad(set_to_none=True)
    out = m(**b, keep_primed=keep_primed)
    out.loss.backward()
    m.injector.clear()
    return torch.cat([p.grad.flatten() for p in m.projectors.parameters()])


def test_checkpointing_gives_same_gradients_with_keep_primed():
    torch.manual_seed(0)
    m = build(zero_init=False)
    trainable_only_projectors(m)
    m.train()
    b = image_batch(m)
    plain = _grads(m, b, keep_primed=True)
    m.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    ckpt = _grads(m, b, keep_primed=True)
    assert torch.allclose(plain, ckpt, rtol=1e-4, atol=1e-6), (plain - ckpt).abs().max()


def test_checkpointing_breaks_without_keep_primed():
    """왜 keep_primed 가 필요한지의 기록 — 컨텍스트 방식이면 재계산이 주입 없이 돈다."""
    torch.manual_seed(0)
    m = build(zero_init=False)
    trainable_only_projectors(m)
    m.train()
    b = image_batch(m)
    plain = _grads(m, b, keep_primed=True)
    m.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    try:
        broken = _grads(m, b, keep_primed=False)
    except Exception:
        return  # CheckpointError — 이것도 '깨짐' 이다
    assert not torch.allclose(plain, broken, rtol=1e-4, atol=1e-6), \
        "keep_primed 없이도 같다면 이 테스트의 전제가 바뀐 것이다 — 훅 구조를 다시 확인해라"
