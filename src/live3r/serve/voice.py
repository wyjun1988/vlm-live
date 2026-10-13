"""Voice front-end: a spoken question in, its text out - the basic chain of docs/VOICE_FRONTEND.md (section 0b).

    16 kHz audio -> Silero VAD (32 ms frames) -> at each pause Smart Turn v3.2 scores "has the speaker finished?" ->
    the turn commits after `action_delay` of silence when the score says yes, after `timeout` when it says no ->
    Qwen3-ASR transcribes the turn -> the text goes to the reader (live3r.serve.interpret) and the VLM, which read
    Korean as it is (I-38).

The decision rule is eot-bench's (github.com/livekit/eot-bench), so its Korean numbers apply to this code: one Smart
Turn score per pause, taken `score_point` (0.2 s) into the silence on everything heard so far (its last 8 s); a score
above the threshold fires at max(action_delay, score_point), anything else at the timeout.
scripts/voice_eot_eval.py replays the benchmark's Korean turns through the same pieces to choose the operating point.

The transcription starts speculatively at the scoring point when the score says "finished": what follows is only
silence, so the transcript is final unless speech resumes, and then it is dropped. The 0.7 s action delay hides
most of the ASR pass on the M2 (0.4-0.9 s per question: text 0.75-1.1 s after speech in the demo) and should hide
all of it on an H100 (not measured yet).

Weights (gitignored; docs/VOICE_FRONTEND.md): checkpoints/silero-vad/silero_vad.onnx (v6.2, MIT),
checkpoints/smart-turn-v3/smart-turn-v3.2-cpu.onnx (BSD-2), checkpoints/qwen3-asr-1.7b (Qwen/Qwen3-ASR-1.7B-hf,
Apache-2.0, native in transformers >= 5.13). The tests use stand-ins and need none of them.
"""

from __future__ import annotations

import collections
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field

import numpy as np

SR = 16000
FRAME = 512  # Silero's frame at 16 kHz: 32 ms


class SileroVAD:
    """Speech probability of one 32 ms frame (Silero VAD v6, ONNX). Stateful: reset() between streams."""

    CONTEXT = 64  # samples of the previous frame the v5+ model expects in front of each frame

    def __init__(self, path: str = "checkpoints/silero-vad/silero_vad.onnx") -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros((1, self.CONTEXT), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, np.asarray(frame, np.float32).reshape(1, FRAME)], axis=1)
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": np.array(SR, np.int64)})
        self._context = x[:, -self.CONTEXT:]
        return float(out[0, 0])


class SmartTurn:
    """P(the speaker has finished) from the turn's last 8 s (Smart Turn v3.2: Whisper-Tiny encoder, ONNX). The
    preprocessing is the reference implementation's: front-pad to 8 s, Whisper log-mel features, normalised."""

    def __init__(self, path: str = "checkpoints/smart-turn-v3/smart-turn-v3.2-cpu.onnx", seconds: int = 8) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.n = seconds * SR
        self.features = WhisperFeatureExtractor(chunk_length=seconds)

    def __call__(self, audio: np.ndarray) -> float:
        a = np.asarray(audio, np.float32)[-self.n:]
        a = np.pad(a, (self.n - len(a), 0))
        f = self.features(a, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=self.n,
                          truncation=True, do_normalize=True).input_features.astype(np.float32)
        return float(self.sess.run(None, {"input_features": f})[0].reshape(-1)[0])


class QwenASR:
    """Qwen3-ASR through transformers' native support: 16 kHz float audio -> text. `language` forces the output
    language ("Korean"; None detects it); `context` is the system prompt the model was trained to use for
    vocabulary biasing (e.g. "Vocabulary: 소파, 냉장고, TV")."""

    def __init__(self, path: str = "checkpoints/qwen3-asr-1.7b", device: str = "cpu", dtype=None,
                 max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float32 if self.device.type == "cpu" else torch.bfloat16)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(path, dtype=self.dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, audio: np.ndarray, language: str | None = "Korean", context: str | None = None) -> str:
        inputs = self.proc.apply_transcription_request(audio=np.asarray(audio, np.float32), language=language,
                                                       prompt=context)
        inputs = inputs.to(self.device, dtype=self.dtype)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], return_format="transcription_only").strip()


@dataclass
class TurnPolicy:
    """When a pause ends the turn. Defaults: the Korean operating point of scripts/voice_eot_eval.py (eot-bench ko,
    377 real turns, 2026-10-13): 4.5 % of turns truncated - 0.9 % of those under 8 s of speech - at a mean wait of
    0.80 s from the end of speech. The action delay is the knob that matters (0.6 s: 8.0 %, 0.71 s; 0.8 s: 2.9 %,
    0.88 s); the threshold is flat between 0.4 and 0.8."""

    vad_on: float = 0.5          # speech starts at this Silero probability ...
    vad_off: float = 0.35        # ... and continues while above this one (Silero's own hysteresis: on - 0.15)
    score_point: float = 0.2     # seconds into a pause at which Smart Turn scores the turn (eot-bench's point)
    threshold: float = 0.5       # P(finished) above which the turn ends after `action_delay`
    action_delay: float = 0.7    # silence before committing a turn the detector called finished
    timeout: float = 1.5         # silence before committing regardless
    pre_roll: float = 0.2        # audio kept from before the first speech frame
    min_speech: float = 0.15     # a turn with less speech than this is dropped as noise
    max_turn: float = 30.0       # a turn this long commits regardless

    def fire_after(self, p: float) -> float:
        """Seconds of silence after which a pause scored `p` commits the turn."""
        return max(self.action_delay, self.score_point) if p > self.threshold else self.timeout


@dataclass
class Turn:
    """One committed question. Times are stream seconds (samples fed / 16 kHz)."""

    audio: np.ndarray = field(repr=False)  # the turn's audio, pre-roll included, up to the commit
    text: str
    reason: str                  # "model" (scored finished), "timeout", "max_length"
    p_eot: float | None          # the deciding pause's score
    t_start: float               # the first speech frame
    t_speech_end: float          # where the deciding pause began
    t_commit: float
    asr_s: float                 # transcription wall time
    asr_from: float              # stream time the transcription started (the scoring point when speculative)
    held_pauses: int             # scored pauses inside the turn that did not end it

    @property
    def text_ready_s(self) -> float:
        """Seconds from the end of speech until the text is ready: the commit wait, or the transcription where it
        outlasts the wait (a single worker - what a live run sees when frames are processed in real time)."""
        return max(self.t_commit, self.asr_from + self.asr_s) - self.t_speech_end


@dataclass
class VoiceEvent:
    kind: str                    # "speech" (a turn starts), "pause" (scored), "resume", "commit", "drop" (noise)
    t: float
    p: float | None = None
    turn: Turn | None = None
    since: float | None = None   # "pause" / "resume": where the pause began


class VoiceFrontend:
    """Feed audio as it arrives with push(); read the turns from the "commit" events.

        fe = VoiceFrontend(SileroVAD(), SmartTurn(), QwenASR(device="mps"))
        for chunk in mic:                      # 16 kHz mono, float32 in [-1, 1] or int16, any length
            for ev in fe.push(chunk):
                if ev.kind == "commit":
                    ask(ev.turn.text)

    `vad(frame) -> P(speech)`, `turn(audio) -> P(finished)` and `asr(audio, language, context) -> str` are plain
    callables, so stand-ins replace them in tests. With `executor` the transcription runs off the audio thread.
    """

    def __init__(self, vad, turn, asr=None, policy: TurnPolicy | None = None, *, language: str | None = "Korean",
                 context: str | None = None, executor: Executor | None = None) -> None:
        self.vad, self.turn, self.asr = vad, turn, asr
        self.policy = policy or TurnPolicy()
        self.language, self.context, self.executor = language, context, executor
        self.reset()

    def reset(self) -> None:
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        self._buf = np.zeros(0, np.float32)
        self._n = 0  # samples consumed: the stream clock
        self._pre: collections.deque = collections.deque(maxlen=max(1, round(self.policy.pre_roll * SR / FRAME)))
        self._new_turn()

    def _new_turn(self) -> None:
        self._frames: list[np.ndarray] = []
        self._t0: float | None = None       # stream time of the turn's first speech frame; None = no turn
        self._speaking = False
        self._speech = 0                    # speech samples in the turn
        self._silence_from: int | None = None
        self._p: float | None = None        # the current pause's score; None = not scored yet
        self._fire_at: int | None = None    # stream sample at which the current pause commits
        self._spec: tuple[Future, float] | None = None
        self._held = 0

    @property
    def in_turn(self) -> bool:
        return self._t0 is not None

    def push(self, pcm) -> list[VoiceEvent]:
        x = np.asarray(pcm)
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
        self._buf = np.concatenate([self._buf, x.reshape(-1)])
        events: list[VoiceEvent] = []
        while len(self._buf) >= FRAME:
            frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            events += self._step(frame)
        return events

    def flush(self) -> list[VoiceEvent]:
        """End of stream: silence until a turn in progress commits."""
        p = self.policy
        return self.push(np.zeros(int((max(p.timeout, p.action_delay) + p.score_point) * SR) + 2 * FRAME, np.float32))

    def _step(self, frame: np.ndarray) -> list[VoiceEvent]:
        pol = self.policy
        start = self._n
        self._n += FRAME
        prob = self.vad(frame)
        speech = prob >= pol.vad_on or (self._speaking and prob >= pol.vad_off)
        if self._t0 is None:
            self._pre.append(frame)
            if not speech:
                return []
            self._frames = list(self._pre)
            self._t0 = start / SR
            self._speaking, self._speech = True, FRAME
            return [VoiceEvent("speech", self._t0)]

        self._frames.append(frame)
        events: list[VoiceEvent] = []
        if speech:
            if self._silence_from is not None:
                if self._p is not None:
                    events.append(VoiceEvent("resume", start / SR, p=self._p, since=self._silence_from / SR))
                    self._held += 1
                self._silence_from, self._p, self._fire_at, self._spec = None, None, None, None
            self._speaking = True
            self._speech += FRAME
        else:
            if self._silence_from is None:
                self._silence_from = start
            self._speaking = False
            if self._p is None and self._n - self._silence_from >= round(pol.score_point * SR):
                audio = np.concatenate(self._frames)
                self._p = float(self.turn(audio))
                self._fire_at = self._silence_from + round(pol.fire_after(self._p) * SR)
                events.append(VoiceEvent("pause", self._n / SR, p=self._p, since=self._silence_from / SR))
                if self._p > pol.threshold and self.asr is not None and self._speech >= pol.min_speech * SR:
                    self._spec = (self._transcribe(audio), self._n / SR)
            if self._fire_at is not None and self._n >= self._fire_at:
                return events + self._commit("model" if self._p > pol.threshold else "timeout")
        if self._n / SR - self._t0 >= pol.max_turn:
            events += self._commit("max_length")
        return events

    def _transcribe(self, audio: np.ndarray) -> Future:
        def run() -> tuple[str, float]:
            t = time.perf_counter()
            text = self.asr(audio, language=self.language, context=self.context)
            return text, time.perf_counter() - t

        if self.executor is not None:
            return self.executor.submit(run)
        fut: Future = Future()
        fut.set_result(run())
        return fut

    def _commit(self, reason: str) -> list[VoiceEvent]:
        audio = np.concatenate(self._frames)
        t_end = (self._silence_from if self._silence_from is not None else self._n) / SR
        t_commit = self._n / SR
        spec, p, t0, held, speech = self._spec, self._p, self._t0, self._held, self._speech / SR
        self._new_turn()
        if speech < self.policy.min_speech:
            return [VoiceEvent("drop", t_commit)]
        if spec is None:
            spec = (self._transcribe(audio), t_commit) if self.asr is not None else None
        text, asr_s, asr_from = ("", 0.0, t_commit) if spec is None else (*spec[0].result(), spec[1])
        turn = Turn(audio=audio, text=text, reason=reason, p_eot=p, t_start=t0, t_speech_end=t_end,
                    t_commit=t_commit, asr_s=asr_s, asr_from=asr_from, held_pauses=held)
        return [VoiceEvent("commit", t_commit, p=p, turn=turn)]
