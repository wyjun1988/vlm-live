# Voice front-end — open-model survey and recommendation (2026-10-13)

What sits in front of the live VLM when the question is spoken: the user talks (Korean or English, Korean often with
English object names), and the front-end hands the reader (`live3r.serve.interpret`) one clean English question.
Requirements (user, 2026-10-12): (1) recognise when the question has ended, (2) drop fillers and condense a rambling
question, (3) translate Korean to English. Priority: **accuracy first, latency second**. Open weights only.

Survey date 2026-10-12/13. ✔ = re-checked on the primary source in this session; everything else is as read by the
survey from model cards / papers and not re-checked. Vendor numbers are marked "self".

## 0. Answer

**No single open model does this for Korean.** Every open model that takes turns by itself (MiniCPM-o 4.5,
PersonaPlex, Moshi, NemotronLabs VoiceChat, Raon-SpeechChat) is English or English/Chinese only. None of the
Korean-capable speech LLMs is shown to beat a cascade on Korean. The one open model that could replace ASR + translation (Qwen3-Omni-30B-A3B)
still needs an external end-of-turn detector and most of an 80 GB GPU. So, a cascade:

| Stage | Pick | Why | License |
|---|---|---|---|
| VAD | **Silero VAD v6.2** | < 1 ms per 32 ms chunk on one CPU thread, trained on 6000+ languages | MIT |
| End of turn | **Smart Turn v3.2** (audio) **+ a text check** on the transcript (Namo-v1-Korean, and the reader's slot completeness, §2) | best permissively licensed detector on real Korean speech; 8M params, ~2 ms GPU; training code + data open → fine-tunable on our pauses | BSD-2 ✔ / Apache-2.0 ✔ |
| ASR | **Qwen3-ASR-1.7B** — streaming partials while speaking, one offline pass over the clip at the end, the scene's object names as context | FLEURS-ko CER 2.57 (self ✔), best open English model on the Open ASR Leaderboard (5.40); trained for context biasing ✔ | Apache-2.0 ✔ |
| Clean + translate | **the reader's existing Qwen3.5-4B call, extended** (non-thinking, greedy, JSON) + a deterministic slot check | the reader already renders Korean in English (I-38); in-domain few-shot clean-up is what decides accuracy (Disfl-QA); +0.05–0.1 s instead of a second model | Apache-2.0 |
| Fallback (only if the slot check fails often) | **Hy-MT2-7B** literal translation | best measured open ko→en in its size (FLORES+ XCOMET 95.09, third-party) | Apache-2.0 ✔ (2026-05-21) |

**Do not use Hunyuan-MT / HY-MT1.5**: the Tencent HY Community License "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED
KINGDOM AND SOUTH KOREA" ✔. (Hy-MT2 is plain Apache-2.0 ✔.) Also excluded on license: LiveKit's turn detectors outside
LiveKit Agents, SeamlessM4T/Streaming and Raon-Speech (non-commercial), EXAONE (non-commercial).

### 0b. The basic function only — Korean speech → end of question → Korean text → model (asked 2026-10-13)

**Built the same day — measurements and how to run it: §8.**

**Silero VAD v6.2 → Smart Turn v3.2 → Qwen3-ASR-1.7B**, the transcript straight to the reader (the 4B reads Korean:
I-38, type 100 % / roles 100 % on the Korean paraphrases).
- ASR is the clear part: Qwen3-ASR-1.7B has the best Korean numbers of any commercially usable open model (FLEURS-ko CER
  2.57 vs Nemotron-3.5-ASR-Streaming 7.1–7.6, both self ✔). Run it once over the whole clip when the turn ends (offline
  beats its own streaming mode, 2.69 vs 3.33 ✔), started speculatively at the first pause, object names as context.
  Only Raon-Speech-9B reports better Korean (FLEURS 1.81, KsponSpeech 6.56/6.96, self) — CC BY-NC, 9B, not streaming.
- End of question is a near-tie decided by the license (eot-bench Korean ✔): LiveKit v1-mini cuts off less at short
  waits (12.3 % vs 30.1 % at 0.3 s, 6.5 % vs 9.7 % at 0.6 s), but at the strict end that accuracy-first runs at, the two
  wait the same (712 vs 734 ms at 5 % cut-off). v1-mini runs locally on CPU and covers Korean ✔, but only inside LiveKit
  Agents ✔. Smart Turn has no strings and can be fine-tuned. Pick v1-mini only if the demo is built on LiveKit Agents.
- No open model does both steps: the Korean-capable ASRs emit no end-of-turn signal (Nemotron-3.5 ✔, Qwen3-ASR,
  Voxtral Realtime), and the models that take turns are English/Chinese.
- Korean text ready ≈ 0.7 s after the user stops (the end-of-question wait; the ASR pass fits inside it).

## 1. Pipeline and latency

```
mic 16 kHz ─► Silero VAD ─► speech ─► Qwen3-ASR streaming partials (2 s chunks; display only)
                    │
                    └─► pause ≥ 0.25 s ─► Smart Turn v3.2 on the whole turn (≤ 8 s)
                                          + Qwen3-ASR offline pass on the clip so far (the text checks need the last words)
                                          + Namo-v1-Korean / reader slot completeness on that transcript
                          likely end  ─► start speculatively: read + clean + translate (4B) → router → VLM prefill/decode
                          confirmed   ─► release the answer (all signals agree, or ≥ 0.6–0.7 s of silence)
                          speech again ─► discard, keep listening; hard timeout 1.5–2.0 s
                          speech within ~1 s after release ─► barge-in: cancel, re-run on the merged turn
```

Budget from the end of speech (estimates until measured on the H100; only the VLM's 0.23 s is ours):

| Step | Time |
|---|---|
| Silero sees the pause | ~0.25 s |
| Smart Turn | ~2 ms (GPU) / ~10 ms (CPU) |
| Qwen3-ASR pass over a ≤ 10 s clip | ~0.1–0.2 s (est.) |
| reader + clean + translate, ~40–60 tokens on the 4B | ~0.2–0.4 s (est.) |
| VLM question-only TTFT | 0.23 s (measured) |
| **Sequential, after a 0.7 s commit** | **~1.2–1.5 s** |
| **Speculative (chain starts at the first 0.25 s pause)** | **≈ max(commit wait, chain) ≈ 0.8–1.1 s** |

The detector's own choice matters less than the speculation: on real Korean speech, the open detectors at a 5 % false
cut-off rate still wait ~0.7 s on average (Smart Turn 734 ms vs 900 ms for a plain VAD timeout, §2). The chain must
overlap that wait. Everything above fits on the VLM's H100 (Qwen3-ASR ~5 GB + capped KV; Smart Turn 32 MB; Silero and
Namo on CPU). On the M2 everything but Qwen3-ASR streaming (vLLM only) runs for development.

## 2. End of turn

**Real Korean speech, LiveKit eot-bench** (real human-to-agent turns, every ≥ 100 ms pause annotated; harness
Apache-2.0, data open) ✔ — cut-off = the detector fired in a pause inside the turn:

| Detector | open? | cut-off @0.3 s | @0.6 s | mean wait @5 % cut-off | @10 % |
|---|---|---|---|---|---|
| LiveKit v1 | no (LiveKit Cloud) | 5.5 % | 2.1 % | 324 ms | 228 ms |
| LiveKit v1-mini | weights, Agents-only license | 12.3 % | 6.5 % | 712 ms | 376 ms |
| **Smart Turn v3.2** | BSD-2 | 30.1 % | 9.7 % | 734 ms | 586 ms |
| UltraVAD | license unstated, ~8B | 33.8 % | 12.0 % | 731 ms | 636 ms |
| VAP | MIT | 43.5 % | 11.8 % | 800 ms | 642 ms |
| VAD only | — | 58.6 % | 15.2 % | 900 ms | 700 ms |

English ranks the same way (Smart Turn v3.2 35.2 % / 14.8 % / 1051 ms; v1-mini 27.8 % / 12.1 % / 1070 ms).

- **Smart Turn v3.2** (pipecat-ai/smart-turn) ✔: 8M (Whisper-Tiny encoder), audio only, 23 languages incl. Korean,
  int8 CPU 8 MB / fp32 GPU 32 MB; self-reported Korean 97.6 % on its own (partly synthetic) test set — eot-bench above
  is the honest number. Pipecat's default; runs standalone from ONNX. Training script and datasets open, so it can be
  fine-tuned on Korean spatial questions with mid-question pauses.
- **Namo-Turn-Detector-v1-Korean** (VideoSDK) ✔: DistilBERT-multilingual, ~135 MB ONNX, text only; self 97.3 %
  accuracy / 98.2 % recall on 800+ Korean utterances (not public), < 14 ms. English: the Namo multilingual model.
- **Why both kinds.** Korean endings ("-야? -까? -요? -니?") and connectives ("-면, -고 -는데") are lexical — a text
  model sees them once the ASR has the last syllable; but "-는데? / -고? / -면?" also end casual questions and only the
  final pitch separates them — that is the audio model. The VSI direction question is the canonical trap:
  "냉장고 옆에 서서 문을 바라보면, [pause] 침대는 왼쪽이야 오른쪽이야?".
- **Our own third signal (proposed, unmeasured):** the reader already runs at the pause in the speculative chain; an
  incomplete reading of a VSI type (`Interpretation.complete()` false — e.g. a direction question without its target)
  means "keep listening". Free, domain-specific, and only for the eight VSI types.
- **LiveKit v1-mini** is the best open-weight detector at short delays and runs locally on CPU ✔, but the LiveKit Model
  License allows use only together with LiveKit Agents ("not ... on a standalone basis or with any other frameworks")
  and forbids using its outputs to develop other models ✔. Worth it only if the webcam demo moves to LiveKit for
  the WebRTC transport anyway.
- Not for us: TEN Turn Detection (EN/ZH, extra Agora clause), Easy Turn (Mandarin), FireRedChat (ZH/EN), Vogent,
  Turnsense, parakeet EOU, Kyutai semantic VAD (EN/FR); the LiveKit text model is deprecated.
- No Korean-only turn-taking set exists; eot-bench's Korean subset (377 turns) and smart-turn-data-v3.2-test are the
  only public Korean ones.

## 3. Speech recognition

Korean is CER unless marked WER (Korean WER runs ~2.5–3× CER). KsponSpeech numbers from different labs are not
comparable (normalisation differs; models trained on its conventions get an edge).

| Model | Korean | English | Streaming | License |
|---|---|---|---|---|
| **Qwen3-ASR-1.7B** (2026-01) ✔ | FLEURS 2.57, CV 5.88, MLC-SLM 8.61 (self, paper A.2 ✔); KsponSpeech 9.76/10.22 vs Whisper-v2 13.70/12.64 (one third party) | OAL 5.40 | vLLM only; 2 s chunks; avg WER 3.33 streaming vs 2.69 offline ✔; TTFT ~0.1 s | Apache-2.0 |
| Qwen3-ASR-0.6B | FLEURS 3.72, CV 8.48 (self) | 6.24 | same | Apache-2.0 |
| Whisper large-v3 | FLEURS 4.89 (3rd); KsponSpeech ~12/11 | 7.16 | wrappers (SimulStreaming) | MIT |
| Voxtral Mini 4B Realtime (2026-02) | FLEURS **WER** 15.7 at 480 ms delay, 14.3 at 2.4 s (self; Whisper 14.3) | 8.44 | native, 80 ms tokens, 80–2400 ms delay | Apache-2.0 |
| Nemotron-3.5-ASR-Streaming 0.6B (2026-06) | FLEURS 7.1–7.6 (self) | 9.75 | native, 80–1120 ms chunks | OpenMDW |
| Gemma 4 E4B / 12B (audio ≤ 30 s) | FLEURS 5.3 / 5.7 (self) | — | no | Apache-2.0 |
| Qwen3-Omni-30B-A3B | KsponSpeech 8.46/7.91 (SKT) | — | input streaming | Apache-2.0 |
| Raon-Speech-9B (KRAFTON) | FLEURS 1.81, KsponSpeech 6.56/6.96 (self) — best found | — | no | **CC BY-NC** |

- **Qwen3-ASR-1.7B** is the pick: best Korean and English numbers among commercially usable open models, ~2B fits
  beside the VLM, and **context biasing** — pass the ObjectMap's object names (Korean and English) so that "TV",
  "소파", "협탁" come out right; the object names are the slots that matter. Unknown: filler behaviour, code-switching
  (not evaluated), finalisation latency on our H100.
- **Whisper large-v3** is the A/B baseline and fallback: close on read Korean, weaker on spontaneous Korean and
  English, drops most fillers, hallucinates on silence (~1 % of transcripts — gate with the VAD), weak at Korean→English
  switch points (HiKE PIER 34.4). large-v3-turbo cannot translate.
- **Voxtral Realtime** only if sub-0.5 s live captions become a requirement; on Korean it trails Whisper until 2.4 s.
- **Direct speech translation is not worth it**: Whisper large-v2 `translate` ~21 BLEU on FLEURS ko→en and large-v3's
  looks broken in one evaluation; Seamless is non-commercial and collapses on code-switching (HiKE MER 89). The cascade
  also keeps the Korean transcript for logging and for the slot check.

## 4. Clean-up and translation

**Extend the reader's call; do not add a model on the critical path.**
- ko→en is the easy direction for strong LLMs; among good open MT models the spread is 1–2 quality-estimation points.
  The failure that matters — a self-correction resolved the wrong way, a constraint dropped — is what no MT model is
  trained or benchmarked on.
- **Disfl-QA** (SQuAD questions with spoken repairs): QA F1 89.6 → 61.6 on disfluent questions; a clean-up model trained
  on Switchboard recovers to 65.7, one trained in-domain to 87.2. In-domain examples matter far more than the model —
  hence few-shot examples in our question style (and later the reading LoRA, which can learn this too).
- **DRES**: each model family has a stable over/under-deletion bias; **reasoning models delete fluent words** → keep
  thinking off (it is already off everywhere).
- **Uh-Mazing** (Aug 2026): filled pauses do not hurt translation; false starts cost 3–4 COMET-Kiwi points, mostly by
  omitting a span.
- One pass (clean + translate together) beats clean-then-translate and avoids a second place to drop a constraint.

Prompt pattern (cached system prompt + 8–12 Korean/English few-shots in our question style): *rewrite one spoken
question about the room; delete only fillers (음, 어, 그, 저기, 막, 약간, 뭐지 / um, uh, like), stutters and repeats, and
the words a self-correction replaces (아니, 말고, 그게 아니라 / I mean, no wait) — keep the corrected version; never add,
change or drop an object, number, unit, direction, viewpoint ("standing by X facing Y") or answer option; translate to
English.* Slots before the sentence, so the sentence is written from them: `{type, roles, numbers, options, dropped,
question_en}`. For the eight VSI types `canonical()` already is the condensed question; `question_en` matters for
`other`.

**Slot check (deterministic):** extract numbers + units (incl. Korean numerals), object words (scene list + Korean
aliases), direction words, the viewpoint pattern and answer options from the transcript; every source slot must appear
in the output unless listed in `dropped` as replaced by a correction; every output slot must trace back to the source.
On failure: retry once with MUST/MUST NOT lists (+0.1–0.15 s) → else hand the VLM the raw transcript (the 4B reads
Korean) or, if the check fails often, a Hy-MT2 literal translation → never answer silently after a failed check.

Candidates considered (ko→en FLORES+ XCOMET/CometKiwi, third-party, Xiaomi arXiv 2608.10812; Gemini 3 Pro 95.67/89.24):

| Model | ko→en | Free-form instructions | License |
|---|---|---|---|
| Hy-MT2 1.8B / 7B / 30B-A3B (2026-05) ✔ | 93.27 / **95.09** / 95.12 | templates incl. style, terminology, personalization | Apache-2.0 |
| MiLMMT-46 4B / 12B (Xiaomi, 2026-08) | 95.29 / 95.53 (self) | translation prompt only | Gemma terms |
| Seed-X-PPO-7B | 94.10 | translation only | OpenMDW |
| TranslateGemma 4B / 12B / 27B (2026-01) | 90.33 / 93.14 / 94.47 | fixed template only | Gemma terms |
| Qwen3.5-4B / 9B | none published | yes (IFEval 89.8 / 91.5) | Apache-2.0 |
| HY-MT1.5, Hunyuan-MT | 94.15 (HY-MT1.5-7B) | fixed templates | **excludes South Korea** ✔ |

Upgrade path if the 4B fails the self-correction subset: Qwen3.5-9B (FP8), same tokenizer and prompt.

## 5. One-model alternatives — A/B later, not now

- **Qwen3-Omni-30B-A3B-Instruct** (Apache-2.0, 2025-09): Korean in its 19 speech-input languages; best measured open
  ko→en speech translation (FLEURS ~28–30 BLEU / 87.6 COMET, two third parties); best speech instruction following in
  SKT's comparison (Speech-IFEval 67.6). ~70 GB BF16 (talker off saves ~10 GB) → its own GPU. Test it in the cascade's
  ASR + clean-up role: audio in → {verbatim transcript, clean English question}, after our end-of-turn detector.
- **Gemma 4 12B / E4B** (Apache-2.0): Korean FLEURS CER 5.3–5.7, official transcribe-then-translate prompt, 16–24 GB;
  no ko→en or latency numbers published.
- A Feb 2026 study (arXiv 2602.17598): speech LLMs behave like expensive cascades and degrade more in noise (−3.9 to
  −12.7 % at 0 dB vs −0.5 to −4.2 % for cascades) — relevant for live room audio.
- **Watch:** Qwen3.5-Omni weights (API-only since the 2026-03-30 launch; would be first choice), SKT A.X K2 ALM
  (Korean, built-in context-aware VAD + barge-in; weights "coming soon"), Fun-Audio-Chat-Duplex weights.

## 6. Evaluation — what decides it (proposed)

No public set has spoken spatial questions; build one.
- **Set:** ~300 VSI-Bench questions stratified by type (+ the 28 paraphrases in `tests/fixtures/question_paraphrases.jsonl`),
  spoken in Korean and English by 3–5 people, each in four styles: clean, fillers, self-correction (object, number,
  direction, viewpoint, option), mid-question pause after "-면," / "and". TTS (a Korean-capable one; not surveyed
  here) only to scale pauses and noise, not as the gold. Gold = the original VSI question and its rules reading.
- **KsponSpeech** (ETRI, AI Hub): 969 h of spontaneous Korean with filler (`/`), repetition / self-correction (`+`) and
  noise tags → free verbatim → clean pairs for the clean-up step, and training data for the reading LoRA later.
- **Metrics:** truncation rate (committed before the last word); end-of-speech → question-ready and → first answer token
  (p50/p95); reading agreement with the gold (type, roles — `interpret_check.py` already scores this); slot errors
  (dropped / invented), overall and on the self-correction subset; VSI accuracy spoken vs typed on the same questions.
- **Gates (proposal):** VSI spoken − typed ≥ −1.0 (paired, CI); truncation ≤ 2 %; end of speech → first token p95 ≤ 1.2 s.
- **A/B order:** ASR (Qwen3-ASR-1.7B vs Whisper large-v3, slot errors on our set) → detector thresholds on our
  recordings (Smart Turn alone vs + text check vs + slot completeness) → clean-up (4B vs 9B; vs Qwen3-Omni in one call).

## 7. Not verified

- Smart Turn's Korean test data share of TTS vs human; whether eot-bench counts cut-offs per pause or per turn.
- LiveKit v1-mini's size (not published).
- Qwen3-ASR filler and code-switching behaviour; every H100 latency above except the VLM's.
- Qwen3-Omni Korean numbers (third-party only; Instruct vs Thinking FLEURS WER disagree, 19.85 vs 7.29).
- No published ko→en score for Qwen3.5 or any Korean LLM; no study of modern LLMs on Disfl-QA or of spoken Korean.
- Gemma 4: no ko→en, latency or VRAM numbers published.

## 8. Built — the basic chain (2026-10-13, I-50)

`live3r.serve.voice`: `SileroVAD` → `SmartTurn` → `QwenASR` behind `VoiceFrontend.push(pcm)`, which returns events
("speech", "pause", "resume", "commit" with a `Turn`: text, reason, waits). Weights in `checkpoints/` (`silero-vad`,
`smart-turn-v3`, `qwen3-asr-1.7b`; 4.1 GB). Qwen3-ASR is native in our transformers 5.16.1, so no new environment;
added `onnxruntime`, `soundfile`, `sounddevice` (requirements.txt). `tests/test_voice.py` runs on stand-ins.

```bash
python scripts/voice_demo.py --reader          # built-in Korean questions (macOS say) -> turns -> the 4B's reading
python scripts/voice_demo.py --mic --reader    # speak into the microphone
python scripts/voice_eot_eval.py               # end of turn on eot-bench ko (data/eval/eot_bench, 377 real turns)
python scripts/voice_asr_eval.py               # transcription on the same turns
```

Measured on the M2, eot-bench `ko` (377 real Korean human-to-agent turns, reservation calls):

| | |
|---|---|
| our Smart Turn under eot-bench's protocol vs the published numbers | 9.7 % cut-offs at 0.6 s (published 9.7 %), 710 ms at 5 % (734), the VAD baseline identical |
| **end to end, per turn, default policy** (p > 0.5 → commit at 0.7 s, else 1.5 s) | **4.5 % truncated; wait mean 0.80 s, median 0.68 s, p95 1.48 s**; 85 % of turns end at 0.7 s |
| … turns with < 4 s / 4–8 s / > 8 s of speech | 0 % / 1.9 % / 10.3 % truncated |
| action delay 0.6 s / 0.8 s instead | 8.0 % at 0.71 s / 2.9 % at 0.88 s |
| a plain silence timeout, 1.0 s / 1.5 s | 3.4 % at 1.03 s / 0.8 % at 1.52 s |
| + a text detector (Namo-v1-Korean on the reference words up to each pause) | no measurable gain — not attached |
| Qwen3-ASR-1.7B, CER vs the benchmark's reference transcript | 10.3 % (7.5 % without numbers, Latin letters, truncated references); the machine-made reference is often the wrong side |
| ASR time per question, M2 (MPS) | 0.44 s for turns under 6 s (RTF 0.12) |
| demo: 7 synthetic Korean questions, pauses up to 0.9 s | 7 turns, none truncated, one character wrong (변 → 편); reader 7/7 |

Reading: questions are short, and short turns are where the detector is safe (one mid-question pause in ten turns
under 4 s); the truncations come from long multi-sentence turns. The policy was picked on these 377 turns. What
remains: our own recorded spatial questions (the gate in §6), and the ASR pass on the H100 (expected well inside the
0.7 s wait, which would make the text ready when the turn commits).

## 9. Speaking the answer — Korean TTS survey (2026-10-13)

Full survey — every candidate, the Korean-company and omni tables, normalisation evidence and tools, the bake-off
recipe and the 30 test sentences: `docs/KOREAN_TTS_SURVEY.md` (sentences also in
`tests/fixtures/korean_answer_readings.jsonl`). The summary:

The VLM's answer, in Korean, spoken back. Answers are short and full of numbers, units and English object names
("소파에서 TV까지 약 2.4m예요", "의자는 4개", "넓이는 약 18㎡"). Same priorities: quality first, then time to first
audio; open weights, commercial license preferred. ✔ = re-checked on the primary source in this session.

**Answer: keep the cascade - VLM text → Korean normalizer → Qwen3-TTS-12Hz-1.7B-CustomVoice with the native Korean
preset voice "Sohee".** No omni model beats it: an omni model would replace the trained VLM or serve as an 8–30B TTS
with weaker or unproven Korean and worse licenses (Qwen3-Omni: Korean CER 26.4 in SKT's run vs 1.67 self-reported;
HyperCLOVA X SEED 8B Omni: best Korean MOS 4.22 self-reported on 10 utterances, needs ~48 GB on 3 GPUs, no streaming
documented; Qwen3.5-Omni: API only).

| Model | Korean evidence | Latency | License |
|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B** (2026-01-22) ✔ | MiniMax 24-language set, Korean WER: Base 1.755 (MiniMax 1.747, ElevenLabs 1.865), CustomVoice 1.741 — best in its table (self ✔); Supertone's run: CER 4.07 (variant unknown). **0.6B-CustomVoice is much worse in Korean (3.053) ✔** | first packet 97–101 ms (self, GPU unnamed); vLLM-Omni serves it online (`/v1/audio/speech`, async-chunk streaming), ~131 ms on H100/H200 per the agent's reading of the vLLM-Omni docs; the official `qwen-tts` package has no streaming example ✔ | Apache-2.0 ✔ |
| Fish Audio S2 Pro (2026-03, 5B) ✔ | best Korean numbers found: MiniMax set WER 1.18, CV3-Eval ko 2.76 (self) | ~100 ms first audio, RTF 0.195 on H200 ✔ | **research / non-commercial** ✔ — commercial use by separate license |
| VoxCPM2 (2026-04, 2B) | MiniMax set WER 1.96, speaker similarity 0.833 (self); CV3-ko 5.69; Supertone's run CER 4.70 | streaming output, RTF ~0.13–0.30; first audio unpublished; MPS path | Apache-2.0 |
| NVIDIA Magpie TTS Multilingual v2607 (2026-07, 364M) | Korean added 07-2026; CER 2.69 on an internal set | 47 ms first audio on H100 (NIM) | NVIDIA Open Model License (commercial) |
| Supertone Supertonic 3 (2026-04, 99M) | Korean CER 3.26 on the MiniMax set (self; competitor's table) | no streaming but RTF ~0.01 on a laptop CPU (v2) — a short answer in tens of ms; ONNX | OpenRAIL-M (commercial with use restrictions); **repo archived 2026-09-09** |
| Avoid | Chatterbox Multilingual (vendor-measured Korean CER 70.9 %), VibeVoice-Realtime (trained on English; one experimental Korean voice), CosyVoice3-0.5B (Korean CER 12.8 without DiffRO), XTTS-v2 (non-commercial, defunct), MeloTTS / GPT-SoVITS (no Korean evaluation; their front-ends misread decimals and units) | | |

**Normalize before the TTS - none of these reads Korean digits and units reliably.** The only published TN benchmark
(Qwen-Audio-3.0-TTS, Chinese/English) has Qwen3-TTS-1.7B reading formulas, units and symbols right 25–44 % of the
time; CosyVoice's front-end sends Hangul through English normalization ("4개" → "four개"); Chatterbox's NFKD turns
"㎡" into "m2". So: keep digits in the VLM's answer (display, VSI scoring) and add a deterministic `speech_text()`
between the VLM and the TTS (~200 lines + a lexicon, < 1 ms): decimals and units ("2.4m" → 이 점 사 미터, "85cm" →
팔십오 센티미터, ㎡ / m² → 제곱미터, ° → 도, % → 퍼센트), native numerals before counters ("4개" → 네 개, "20개" → 스무 개,
"2명", "3대", "21권", "3번째", "2시 방향"), Sino-Korean for floors and labels ("2층"), ranges ("3~4개" → 서너 개), a
closed English lexicon (TV → 티비, PC → 피시, LED → 엘이디, USB → 유에스비, Wi-Fi → 와이파이) with letter-by-letter
fallback. NVIDIA's NeMo-text-processing has a Korean grammar to start from (decimals, kg/km/m/cm/mm/L/%/°, native
numerals before 16 counters; no ㎡, no 대/가지/곳) ✔ (directory exists). Do not ask the VLM to spell numbers: prompting
alone falls well short of a trained or rule-based normalizer ("Ready to Speak", NAVER LABS Europe, 2026).

**Latency.** Answers are one sentence, so the TTS can take the whole answer text; first audio ≈ VLM answer (TTFT
0.23 s + ~10–20 tokens) + TTS first packet (~0.1–0.25 s) ≈ 0.5–0.7 s after the question text, ~1.2–1.4 s after the
user stops speaking. To measure on the H100.

**Bake-off to decide (proposed).** Qwen3-TTS-1.7B (Sohee), VoxCPM2, Magpie v2607, Supertonic 3 (+ Fish S2 Pro as a
non-commercial reference), each on raw vs normalized text: 30 answer-style sentences stressing numbers, units, counters
and English names (expected readings kept alongside, doubling as the normalizer's unit tests) + 100 CV3-Eval
`zero_shot_ko` items. Score CER with our Qwen3-ASR (normalize its output the same way; check numeric values), but
numeral-system errors (네 개 vs 사 개) need 3–5 native listeners: a yes/no on "numbers/units/English read right" + a
5-point MOS. Then time to first audio on the H100 (P50/P95). Decision order: ≥ 29/30 fully right with normalized
text → MOS → latency.

Sources: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [Qwen3-TTS report](https://arxiv.org/abs/2601.15621) ·
[vLLM-Omni TTS engineering](https://vllm.ai/blog/2026-06-23-vllm-omni-tts) ·
[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) · [VoxCPM](https://github.com/OpenBMB/VoxCPM) ·
[Magpie TTS Multilingual](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) ·
[Supertonic 3](https://huggingface.co/Supertone/supertonic-3) ·
[NeMo Korean TN](https://github.com/NVIDIA/NeMo-text-processing/tree/main/nemo_text_processing/text_normalization/ko) ·
[CV3-Eval](https://huggingface.co/datasets/yuekai/CV3-Eval) · [Qwen-Audio-3.0-TTS (TN benchmark)](https://arxiv.org/html/2607.23938) ·
[Ready to Speak](https://arxiv.org/abs/2609.01246) · [HyperCLOVA X SEED Omni](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B) ·
[Raon-Speech report](https://arxiv.org/html/2605.23912).

## Sources

End of turn: [eot-bench](https://github.com/livekit/eot-bench) ·
[Smart Turn](https://github.com/pipecat-ai/smart-turn) ·
[Namo Korean](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Korean) ·
[LiveKit turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/) ·
[Silero VAD](https://github.com/snakers4/silero-vad).
ASR: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR paper](https://arxiv.org/abs/2601.21337) ·
[Voxtral Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) ·
[Nemotron-3.5-ASR-Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
[SimulStreaming](https://github.com/ufal/SimulStreaming) · [HiKE](https://arxiv.org/abs/2509.24613) ·
[Raon-Speech](https://huggingface.co/KRAFTON/Raon-Speech-9B).
Clean-up / MT: [Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B) ·
[HY-MT1.5 license](https://huggingface.co/tencent/HY-MT1.5-7B/blob/main/License.txt) ·
[TranslateGemma](https://huggingface.co/google/translategemma-4b-it) · [MiLMMT](https://arxiv.org/abs/2608.10812) ·
[Disfl-QA](https://arxiv.org/abs/2106.04016) · [DRES](https://arxiv.org/abs/2509.20321) ·
[Uh-Mazing](https://arxiv.org/abs/2608.02138).
One model: [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) ·
[Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) · [SKT A.X K2 ALM](https://huggingface.co/skt/A.X-K2-ALM) ·
[speech LLMs vs cascades](https://arxiv.org/abs/2602.17598) ·
[Qwen3.5-Omni weights not found (HN)](https://brianlovin.com/hn/46219538) ·
[DataLearner Qwen3.5-Omni](https://www.datalearner.com/en/ai-models/pretrained-models/qwen3-5-omni-plus).
