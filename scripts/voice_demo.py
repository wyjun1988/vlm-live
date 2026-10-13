"""Speak a question, get its text: the voice front-end (live3r.serve.voice) on audio files, synthetic Korean
questions or the microphone, optionally handing each transcript to the model's question reader (I-38).

    python scripts/voice_demo.py                      # built-in Korean questions, spoken by macOS `say` (Yuna)
    python scripts/voice_demo.py --reader             # ... and the 4B reads each transcript (type, roles)
    python scripts/voice_demo.py --wav a.wav b.wav    # your recordings (any rate; mono is taken)
    python scripts/voice_demo.py --mic --reader       # live: speak, Ctrl+C to stop

Files are streamed in 32 ms frames as fast as they decode, on the stream's own clock, so the waits printed are the
ones a live run would see; "text" adds the transcription when it outlasts the wait (it starts at the scoring point).
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from live3r.serve.voice import SR, QwenASR, SileroVAD, SmartTurn, TurnPolicy, VoiceFrontend  # noqa: E402

# (spoken text with `say` pause marks, expected reading). The pauses sit where people pause: after a filler, after
# the first object, after the viewpoint clause ("-면,") - the trap for an audio-only end-of-turn model.
EXAMPLES = [
    ("이 방에 의자가 몇 개 있어?", ("count", {"object": "chair"})),
    ("음 [[slnc 400]] 소파에서 티비까지 거리가 [[slnc 500]] 몇 미터야?", ("abs_distance", {"a": "sofa", "b": "tv"})),
    ("냉장고 옆에 서서 문을 바라보면 [[slnc 600]] 침대는 왼쪽이야 오른쪽이야?",
     ("rel_direction", {"standing": "refrigerator", "facing": "door", "target": "bed"})),
    ("침대의 가장 긴 변은 몇 센티미터야?", ("size", {"object": "bed"})),
    ("책상하고 [[slnc 300]] 의자랑 램프 중에 [[slnc 400]] 뭐가 소파에 제일 가까워?",
     ("rel_distance", {"target": "sofa", "options": ["desk", "chair", "lamp"]})),
    ("이 방 넓이가 몇 제곱미터야?", ("room_size", {})),
    ("어 [[slnc 300]] 책상 앞에 서서 창문을 보면 [[slnc 900]] 침대는 왼쪽이야 오른쪽이야?",
     ("rel_direction", {"standing": "desk", "facing": "window", "target": "bed"})),
]


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    return resample_poly(a, SR, sr).astype(np.float32) if sr != SR else a


def say(text: str, out: Path, voice: str) -> np.ndarray:
    subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text], check=True)
    return read_wav(str(out))


def describe(ev) -> str:
    t = ev.turn
    return (f"[{t.t_commit:6.2f}s] {t.text!r}\n"
            f"           {t.reason}, p={t.p_eot if t.p_eot is None else round(t.p_eot, 2)}, wait {t.t_commit - t.t_speech_end:.2f} s, "
            f"text {t.text_ready_s:.2f} s after speech (asr {t.asr_s:.2f} s), pauses held {t.held_pauses}")


def run_stream(fe: VoiceFrontend, pcm: np.ndarray, verbose: bool) -> list:
    turns = []
    for i in range(0, len(pcm), 512):
        for ev in fe.push(pcm[i:i + 512]):
            if ev.kind == "commit":
                print(describe(ev), flush=True)
                turns.append(ev.turn)
            elif verbose and ev.kind in ("pause", "resume", "drop"):
                print(f"[{ev.t:6.2f}s]   {ev.kind}" + (f" p={ev.p:.2f}" if ev.p is not None else ""), flush=True)
    for ev in fe.flush():
        if ev.kind == "commit":
            print(describe(ev), flush=True)
            turns.append(ev.turn)
    return turns


def run_mic(fe: VoiceFrontend, on_turn, verbose: bool) -> None:
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=512,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        print("listening - ask a question (Ctrl+C to stop)", flush=True)
        try:
            while True:
                for ev in fe.push(q.get()):
                    if ev.kind == "speech" and verbose:
                        print("  ...", flush=True)
                    elif ev.kind == "commit":
                        print(describe(ev), flush=True)
                        on_turn(ev.turn)
        except KeyboardInterrupt:
            print()


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--wav", nargs="+", help="audio files, streamed one after another with 2 s of silence between")
    src.add_argument("--say", nargs="+", help="Korean questions to synthesise (macOS `say`; [[slnc 500]] = pause)")
    src.add_argument("--mic", action="store_true")
    ap.add_argument("--voice", default="Yuna")
    ap.add_argument("--asr", default="checkpoints/qwen3-asr-1.7b")
    ap.add_argument("--vad", default="checkpoints/silero-vad/silero_vad.onnx")
    ap.add_argument("--smart-turn", default="checkpoints/smart-turn-v3/smart-turn-v3.2-cpu.onnx")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--language", default="Korean", help='ASR output language; "auto" detects it')
    ap.add_argument("--context", help='vocabulary hint for the ASR, e.g. "Vocabulary: 소파, 냉장고, TV"')
    ap.add_argument("--threshold", type=float, default=TurnPolicy.threshold)
    ap.add_argument("--action-delay", type=float, default=TurnPolicy.action_delay)
    ap.add_argument("--timeout", type=float, default=TurnPolicy.timeout)
    ap.add_argument("--reader", action="store_true", help="the 4B reads each transcript into type + roles")
    ap.add_argument("--model", default="checkpoints/qwen3.5-4b", help="the reader")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every scored pause")
    args = ap.parse_args()

    policy = TurnPolicy(threshold=args.threshold, action_delay=args.action_delay, timeout=args.timeout)
    t = time.time()
    asr = QwenASR(args.asr, device=args.device)
    asr(np.zeros(SR, np.float32))  # warm-up: the first MPS pass compiles kernels
    fe = VoiceFrontend(SileroVAD(args.vad), SmartTurn(args.smart_turn), asr, policy,
                       language=None if args.language == "auto" else args.language, context=args.context)
    print(f"loaded in {time.time() - t:.0f} s - policy: score at {policy.score_point} s, p > {policy.threshold} -> "
          f"commit at {policy.action_delay} s, else at {policy.timeout} s")

    expected = []
    if args.mic:
        ask = None
        if args.reader:
            from interpret_check import make_ask
            ask, _ = make_ask(args.model, args.device, 64)

        def on_turn(turn):
            if ask is not None:
                from live3r.serve.interpret import interpret
                it = interpret(turn.text, ask)
                print(f"           reader: {it.type} {it.roles} {it.question_en or ''}", flush=True)

        run_mic(fe, on_turn, args.verbose)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        if args.wav:
            clips = [read_wav(p) for p in args.wav]
        else:
            items = [(s, None) for s in args.say] if args.say else EXAMPLES
            clips = [say(s, Path(tmp) / f"q{i}.wav", args.voice) for i, (s, _) in enumerate(items)]
            expected = [e for _, e in items]
            for (s, _), c in zip(items, clips):
                print(f"  spoken ({len(c) / SR:.1f} s): {s}")
    gap = np.zeros(2 * SR, np.float32)
    pcm = np.concatenate([x for c in clips for x in (gap, c)] + [gap])
    turns = run_stream(fe, pcm, args.verbose)
    print(f"\n{len(turns)} turns from {len(clips)} clips")

    if args.reader and turns:
        del fe, asr
        from interpret_check import make_ask, roles_match
        from live3r.serve.interpret import interpret

        ask, last = make_ask(args.model, args.device, 64)
        ok = 0
        for i, turn in enumerate(turns):
            t = time.perf_counter()
            it = interpret(turn.text, ask)
            dt = time.perf_counter() - t
            want = expected[i] if len(turns) == len(clips) and i < len(expected) else None
            hit = want is not None and it.type == want[0] and roles_match(it.roles, want[1])
            ok += hit
            mark = "" if want is None else (" ok" if hit else f" MISS (want {want[0]} {want[1]})")
            print(f"  {turn.text}\n    -> {it.type} {it.roles}  [{dt:.1f} s, {last.get('tokens')} tok]{mark}")
        if expected and len(turns) == len(clips):
            print(f"reader agrees on {ok}/{len(turns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
