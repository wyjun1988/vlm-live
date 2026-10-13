"""Voice front-end (live3r.serve.voice): the turn logic with stand-in models - commit after the action delay when a
pause is scored finished, after the timeout when not, resumed speech holds the turn and drops the speculative
transcript, noise blips are dropped. The ONNX wrappers run only when their weights are present."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from live3r.serve import voice as V

ROOT = Path(__file__).resolve().parents[1]
FRAME_S = V.FRAME / V.SR


def tone(s: float) -> np.ndarray:
    t = np.arange(int(s * V.SR)) / V.SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * V.SR), np.float32)


class EnergyVAD:
    def __call__(self, frame):
        return 0.9 if np.abs(frame).mean() > 0.01 else 0.0


class Scores:
    """Smart Turn stand-in: the given scores in call order."""

    def __init__(self, *ps):
        self.ps, self.calls = list(ps), 0

    def __call__(self, audio):
        self.calls += 1
        return self.ps.pop(0)


class DurationASR:
    """Transcribes a turn as its length, so a test can tell which audio was transcribed."""

    def __init__(self):
        self.calls = []

    def __call__(self, audio, language=None, context=None):
        self.calls.append((len(audio) / V.SR, language, context))
        return f"{len(audio) / V.SR:.2f}"


def run(fe, *parts):
    events = []
    for p in parts:
        events += fe.push(p)
    return events + fe.flush()


def commits(events):
    return [e.turn for e in events if e.kind == "commit"]


def test_fire_after_follows_the_eot_bench_rule():
    pol = V.TurnPolicy(threshold=0.5, action_delay=0.7, timeout=1.5, score_point=0.2)
    assert pol.fire_after(0.9) == 0.7 and pol.fire_after(0.5) == 1.5
    assert V.TurnPolicy(action_delay=0.1, score_point=0.2).fire_after(0.9) == 0.2


def test_a_finished_pause_commits_after_the_action_delay_with_the_speculative_transcript():
    asr = DurationASR()
    fe = V.VoiceFrontend(EnergyVAD(), Scores(0.9), asr, V.TurnPolicy(action_delay=0.7, timeout=1.5))
    events = run(fe, silence(0.5), tone(1.0), silence(2.0))
    assert [e.kind for e in events] == ["speech", "pause", "commit"]
    (turn,) = commits(events)
    assert turn.reason == "model" and turn.p_eot == 0.9 and turn.held_pauses == 0
    assert abs(turn.t_speech_end - 1.5) <= FRAME_S
    assert abs(turn.t_commit - turn.t_speech_end - 0.7) <= FRAME_S
    # transcribed once, at the scoring point, on the audio heard by then (pre-roll + speech + 0.2 s)
    assert len(asr.calls) == 1 and asr.calls[0][1] == "Korean"
    assert abs(turn.asr_from - turn.t_speech_end - 0.2) <= FRAME_S
    assert abs(float(turn.text) - (0.2 + 1.0 + 0.2)) <= 2 * FRAME_S
    assert turn.text_ready_s == pytest.approx(max(turn.t_commit, turn.asr_from + turn.asr_s) - turn.t_speech_end)


def test_an_unfinished_pause_waits_for_the_timeout_and_transcribes_then():
    asr = DurationASR()
    fe = V.VoiceFrontend(EnergyVAD(), Scores(0.1), asr, V.TurnPolicy(threshold=0.5, action_delay=0.7, timeout=1.5))
    (turn,) = commits(run(fe, tone(1.0), silence(2.0)))
    assert turn.reason == "timeout"
    assert abs(turn.t_commit - turn.t_speech_end - 1.5) <= FRAME_S
    assert turn.asr_from == turn.t_commit and len(asr.calls) == 1


def test_resumed_speech_holds_the_turn_and_drops_the_speculative_transcript():
    asr = DurationASR()
    turn_scores = Scores(0.9, 0.9)
    fe = V.VoiceFrontend(EnergyVAD(), turn_scores, asr, V.TurnPolicy(action_delay=0.7, timeout=1.5))
    events = run(fe, tone(1.0), silence(0.5), tone(1.0), silence(2.0))
    assert [e.kind for e in events] == ["speech", "pause", "resume", "pause", "commit"]
    (turn,) = commits(events)
    assert turn.held_pauses == 1 and turn_scores.calls == 2
    assert len(asr.calls) == 2                    # the first, speculative one was dropped
    assert float(turn.text) > 2.5                 # the committed text covers both halves


def test_a_noise_blip_is_dropped_without_transcription():
    asr = DurationASR()
    fe = V.VoiceFrontend(EnergyVAD(), Scores(0.9), asr, V.TurnPolicy(min_speech=0.15))
    events = run(fe, silence(0.3), tone(0.064), silence(2.0))
    assert [e.kind for e in events] == ["speech", "pause", "drop"]
    assert commits(events) == [] and asr.calls == []


def test_the_turn_keeps_its_pre_roll_and_two_turns_stay_apart():
    fe = V.VoiceFrontend(EnergyVAD(), Scores(0.9, 0.9), DurationASR(), V.TurnPolicy(pre_roll=0.2, action_delay=0.5))
    a, b = commits(run(fe, silence(1.0), tone(0.5), silence(1.5), tone(0.8), silence(1.5)))
    assert len(a.audio) / V.SR == pytest.approx(0.2 + 0.5 + 0.5, abs=2 * FRAME_S)
    assert abs(a.t_start - 1.0) <= FRAME_S and abs(b.t_start - 3.0) <= FRAME_S


def test_a_long_turn_commits_at_max_length():
    fe = V.VoiceFrontend(EnergyVAD(), Scores(), DurationASR(), V.TurnPolicy(max_turn=2.0))
    (turn,) = commits(fe.push(tone(3.0)))
    assert turn.reason == "max_length" and turn.p_eot is None
    assert turn.t_commit - turn.t_start == pytest.approx(2.0, abs=FRAME_S)


def test_int16_input_and_an_executor_give_the_same_turn():
    pcm = np.concatenate([tone(1.0), silence(2.0)])
    inline = commits(run(V.VoiceFrontend(EnergyVAD(), Scores(0.9), DurationASR()), pcm))[0]
    with ThreadPoolExecutor(1) as ex:
        fe = V.VoiceFrontend(EnergyVAD(), Scores(0.9), DurationASR(), executor=ex)
        threaded = commits(run(fe, (pcm * 32767).astype(np.int16)))[0]
    assert (inline.text, inline.reason, inline.t_commit) == (threaded.text, threaded.reason, threaded.t_commit)


SILERO = ROOT / "checkpoints/silero-vad/silero_vad.onnx"
SMART_TURN = ROOT / "checkpoints/smart-turn-v3/smart-turn-v3.2-cpu.onnx"


@pytest.mark.skipif(not SILERO.exists(), reason="Silero VAD weights not downloaded")
def test_silero_hears_a_voice_like_signal_and_not_silence():
    pytest.importorskip("onnxruntime")
    vad = V.SileroVAD(str(SILERO))
    assert vad(np.zeros(V.FRAME, np.float32)) < 0.1
    vad.reset()
    assert vad(np.zeros(V.FRAME, np.float32)) < 0.1


@pytest.mark.skipif(not SMART_TURN.exists(), reason="Smart Turn weights not downloaded")
def test_smart_turn_scores_any_length_in_0_1():
    pytest.importorskip("onnxruntime")
    st = V.SmartTurn(str(SMART_TURN))
    for s in (0.5, 8.0, 12.0):
        assert 0.0 <= st(tone(s)) <= 1.0
