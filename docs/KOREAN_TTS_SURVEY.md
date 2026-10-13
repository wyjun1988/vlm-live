# Speaking the answer in Korean — open TTS survey (2026-10-13)

The live VLM answers a spoken Korean question; the answer, in Korean, has to be spoken back. This note surveys the
open-weight text-to-speech models that can do it and recommends a pipeline. It is the long form of §9 of
`docs/VOICE_FRONTEND.md`; the voice input side (end of question, ASR) is §0–§8 there.

What the answers look like: short, and full of numbers, units and English object names — "소파에서 TV까지 약 2.4m예요",
"의자는 4개예요", "침대는 왼쪽에 있어요", "넓이는 약 18㎡입니다". Priorities, as for the input side: **quality first**
(natural, intelligible Korean; numbers, units and English names read correctly; nothing skipped), **then latency**
(time to first audio). Open weights only; a commercial license is preferred. One consistent assistant voice is
enough. Inference on an H100 shared with the VLM; development on an M2 Max.

Survey date 2026-10-13. ✔ = re-checked on the primary source in this session; everything else as read by the survey
from model cards, papers and repositories. Numbers are the authors' own unless marked 3P (third party). Korean WER and
CER from different papers are not comparable (different test sets, ASR and normalisation).

Test sets referred to below:
- **MM-24**: the MiniMax-Speech multilingual test set — 24 languages × 100 sentences, 2 Common Voice speakers per
  language, scored with Whisper-large-v3 in the MiniMax paper (other papers rarely name their ASR).
- **CV3-ko**: the Korean part of CosyVoice 3's CV3-Eval (`zero_shot_ko`, 500 utterances with voice prompts).

## 1. Recommendation

```
VLM answer (digits kept, for display and scoring)
   -> speech_text(): deterministic Korean normaliser (numbers, units, counters, English -> Hangul)
   -> Qwen3-TTS-12Hz-1.7B-CustomVoice, preset voice "Sohee" (native Korean, female)
   -> audio (streamed through vLLM-Omni on the server)
```

- **TTS: Qwen3-TTS-12Hz-1.7B.** On the only Korean benchmark several systems share (MM-24) it is level with MiniMax
  and ahead of ElevenLabs; it has a native Korean preset voice, an Apache-2.0 license, ~0.1 s to first audio, and
  it is the same family as our VLM and ASR. Use the **1.7B**: the 0.6B CustomVoice is much worse in Korean
  (WER 3.05 vs 1.74) ✔.
- **A normaliser before it.** No TTS model has evidence of reading Korean digits, units and Latin letters
  correctly from raw text, and several are shown to fail (§5).
- **No omni model.** One that speaks would either replace our trained VLM or act as an 8–30B TTS with weaker or
  unproven Korean and a worse license (§4). Revisit only if Qwen3.5-Omni weights appear.
- **Decide by a bake-off** on our own answer sentences, with listeners (§6): the published Korean numbers come from
  different test sets and disagree by 2–3x for the same model depending on who measured.

## 2. Shortlist

| Model | Korean evidence | Latency | License | Use |
|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B** (CustomVoice / Base / VoiceDesign), 2026-01-22 ✔ | MM-24 Korean WER: Base 1.755 (0.6B-Base 1.741; MiniMax 1.747, ElevenLabs 1.865) ✔. Target-speaker set: 1.7B-CustomVoice 1.741 (best in its table), **0.6B-CustomVoice 3.053** ✔. Cross-lingual en→ko 5.14 vs CosyVoice3-1.5B 5.87. 3P (Supertone): CER 4.07, variant not stated | first packet 97 ms (0.6B) / 101 ms (1.7B) at concurrency 1, 333 ms at 6 (report; GPU not named); vLLM-Omni online serving (`/v1/audio/speech`, async-chunk streaming; first packet = prefill + 25 decode steps) — ~131 ms, RTF 0.34 on H100/H200 per its docs; faster-qwen3-tts ~230–240 ms on H100 | Apache-2.0 ✔ | **pick** |
| Fish Audio S2 Pro (4B + 0.4B), 2026-03-09 ✔ | best Korean numbers found: MM-24 WER 1.180, SIM 0.817; CV3-ko 2.76 | ~100 ms first audio, RTF 0.195 on one H200 ✔ | **Fish Audio Research License: research / non-commercial**; commercial use by separate license ✔ | reference only |
| VoxCPM2 (2B), 2026-04 | MM-24 WER 1.962, SIM 0.833 (highest); CV3-ko 5.69; 3P CER 4.70 | streaming output; RTF ~0.30 on RTX 4090, ~0.13 with Nano-vLLM; first audio not published | Apache-2.0 | second candidate |
| NVIDIA Magpie TTS Multilingual v2607 (364M), 2026-07-21 | Korean added in v2607; CER 2.69 on an internal set (ASR not named) | 47 ms first audio on H100 at 1 stream, 275 ms at 64 (via NIM, not the open checkpoint) | NVIDIA Open Model License (commercial) | if lowest latency + commercial matter most |
| Supertone Supertonic 3 (99M), 2026-04-29 | Korean CER 3.26 on the MiniMax set in Supertone's own table (OmniVoice 3.22, Supertonic 2 3.65, Qwen3-TTS 4.07, VoxCPM2 4.70; ASR not named) | not streaming, but v2 runs at RTF 0.012–0.015 on an M4 Pro CPU (a short answer in tens of ms) | OpenRAIL-M (commercial with use restrictions, e.g. disclosing machine-generated speech) | CPU / Mac stand-in; **repo archived 2026-09-09** |

## 3. All Korean-capable candidates

| Model · date | Size | Korean evidence | Streaming / speed | Voices | License | Runtime |
|---|---|---|---|---|---|---|
| Qwen3-TTS-12Hz ✔ · 2026-01-22 | 0.6B, 1.7B | see §2; 10 languages, 9 presets incl. "Sohee" ✔ | see §2; the pip `qwen-tts` package shows no streaming ✔ | presets, 3 s cloning, voice design (1.7B) | Apache-2.0 ✔ | CUDA; vLLM-Omni; Mac via mlx-audio / faster-qwen3-tts (GGML Metal) |
| Fish Audio S2 Pro ✔ · 2026-03-09 | ~5B | see §2 | see §2 | cloning, style tags | research / NC ✔ | SGLang-Omni, vLLM-Omni |
| VoxCPM2 · 2026-04 | 2B | see §2 | see §2; llama.cpp Metal RTF 1.76 on an M4 Pro (slower than real time) | voice design, cloning, LoRA | Apache-2.0 | CUDA, vLLM-Omni, MPS, mlx-audio |
| Fun-CosyVoice3-0.5B-2512 · 2025-12 | 0.5B | CV3-ko CER **12.8** (0.5B), 4.02 with DiffRO (paper; which public checkpoint matches is unclear) | "bi-streaming", ~150 ms (hardware not stated); takes streamed text | cloning, instructions | Apache-2.0 | vLLM, TensorRT-LLM; front-end misroutes Hangul (§5) |
| NVIDIA Magpie TTS Multilingual v2607 · 2026-07-21 | 364M | see §2 | see §2 | 5 presets (cloning removed) | NVIDIA Open Model License | NeMo, Riva, NIM, GGUF C++; no MPS |
| Supertonic 3 / 2 · 2026-04-29 / 2026-01-06 | 99M / 66M | CER 3.26 / 3.65 (Supertone's table) | not streaming; RTF ~0.001–0.005 on RTX 4090 (v2, PyTorch, 2 steps) | presets only (Voice Builder closed 2026-08-31) | OpenRAIL-M; code MIT | ONNX on CPU / WebGPU / mobile |
| OmniVoice (k2-fsa / Xiaomi) · 2026-04-01 | 0.6–0.8B | MM-24 WER 2.651, SIM 0.828; 3P CER 3.22 | non-autoregressive, no streaming; RTF 0.037 on H100 | cloning | **weights CC-BY-NC** | CUDA, MPS, vLLM-Omni |
| MOSS-TTS (Realtime 1.7B, Nano 0.1B, 8B) · 2026-03/04 | | no Korean numbers | Realtime: 180 ms to first byte; Nano streams on a 4-core CPU | cloning | Apache-2.0 (Nano: pending) | CUDA, mlx-audio |
| Higgs TTS 3 (Boson AI) · 2026-06-04 | ~4B | no per-language number (MM-24 average 2.74) | SSE streaming; 617 ms end to end on H100 | cloning, tags | research / NC | SGLang-Omni, vLLM-Omni |
| Raon-Speech-9B (KRAFTON) · 2026-04/05 | 9B speech LM | CER (in-house Zipformer ASR): KSponSpeech-c 4.89, MiniMax-ko 1.57, CV3-ko 3.90; UTMOS 2.36–2.88 | 617 ms first audio, RTF 0.27 on RTX 6000 Pro | speaker-conditioned | **CC BY-NC 4.0** | CUDA, SGLang, vLLM-Omni |
| Pocket-TTS-Korean 300M / 100M (community, seastar105) · 2026-08-27 | 336M / 110M | CV3-ko CER 6.14 / 7.76 (Whisper-large-v3, Korean forced); English "very poor" | Pocket TTS streams on CPU | cloning | CC BY 4.0 | PyTorch (CPU) |
| Chatterbox Multilingual V3 · 2026-06-10 | 0.5B | **vendor-measured Korean CER 70.9 %**, "not production quality" | no streaming in the open release | cloning, watermark | MIT | CUDA, MPS |
| VibeVoice-Realtime-0.5B · 2025-12 | 0.5B | one "experimental" Korean voice; trained on English | ~300 ms | presets | MIT, research | CUDA |
| Step-Audio-EditX · 2025-11 | 3–4B | `[Korean]` tag; no numbers | – | cloning, editing | code Apache-2.0; weights unclear | CUDA, vLLM |
| Orpheus 3B Korean (research release) · 2025-04 | 3B | no evaluation | ~200 ms (generic claim) | presets | tagged Apache-2.0 but Llama-3.2-based; gated | vLLM |
| kani-tts-400m-ko · 2025-10 | 400M | no evaluation | RTF ~0.2 on RTX 4080/5090 | – | card conflicts (LFM vs Apache) | vLLM |
| XTTS-v2 (Coqui) · 2023 | – | Korean speaker similarity 0.741 (T5Gemma-TTS paper) | <200 ms streaming | cloning | **CPML, non-commercial incl. outputs**; company defunct | PyTorch |
| GPT-SoVITS v2+ | small | no Korean evaluation; its front-end leaves decimals and spells "km" as 케이엠 | RTF 0.014 (RTX 4090) | few-shot | MIT | CUDA / CPU / MPS |
| MeloTTS-Korean (MyShell) · 2024 | VITS2 | no evaluation; no number conversion in its front-end | CPU real time | 1 voice | MIT | CPU / GPU |
| OpenAudio S1-mini (Fish) · 2025-06 | 0.5B | no Korean numbers | – | cloning | CC-BY-NC-SA-4.0 | CUDA |

**No Korean, or Korean not usable:** Kyutai TTS (en/fr), Pocket TTS official (7 European languages), Kokoro-82M, Zonos
v0.1, Spark-TTS, IndexTTS-2 / 2.5, Voxtral TTS (no Korean, CC BY-NC), Breeze TTS 2 (en/zh; top of the
English-only Artificial Analysis arena), F5-TTS (base en/zh, CC-BY-NC; no Korean fine-tune found), MMS-TTS kor
(romanised input, CC-BY-NC), Bark (13–14 s cap, unstable), Higgs Audio v2 (MM-24 average WER 49.9).

**Korean companies:** almost no standalone open TTS. Supertone (Supertonic) is the only commercially usable one, and
its repository is archived. Naver: only HyperCLOVA X SEED 8B Omni (§4). Kakao: Kanana-o is API-only. SKT: A.X K2 ALM
not released. KT, NC AI, LG: text / vision models only. KRAFTON Raon-OpenTTS and Nari Labs Dia / Dia2 are
English-only. Community Korean fine-tunes (Qwen3-TTS-0.6B-Korean, CosyVoice2-KO-SFT, neutts-air-ko) publish no
evaluation; old VITS models trained on KSS inherit its CC BY-NC-SA.

**Independent comparisons:** none with Korean MOS. The Artificial Analysis speech arena is English-only. Korean
CER tables come from vendors (Supertone, KRAFTON) running competitors.

## 4. Single models that speak Korean (why the cascade stays)

| Model | Weights / license | Korean speech evidence | Serving | Verdict |
|---|---|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct · 2025-09 | open, Apache-2.0 | Korean is one of 10 output languages; self: MM-24 Korean WER 1.670. 3P: Naver MOS 3.40 (vs HCX 4.22); SKT CER 26.37 on KVoiceBench | 234 ms first packet (theoretical); ≥ 79 GB BF16 | no |
| HyperCLOVA X SEED 8B Omni (Naver) · 2025-12 | open, custom license (commercial under 10M MAU; "Powered by HyperCLOVA X") | self: Korean MOS 4.22, best in its table (Gemini-2.5-flash 4.20, ElevenLabs v3 4.05) — 10 utterances, 30 listeners. 3P CER: 16.7 / 2.64 / 4.52 (KRAFTON), 14.67 (Kakao), 4.16 (SKT) | OmniServe: 3 GPUs, ~48 GB; audio returned as a URL; no streaming documented | bake-off only, as a heavy "read aloud" TTS |
| A.X K2 Raon-Speech-21B-A3B (KRAFTON on SKT's backbone) · 2026 | CC BY-NC 4.0 | self: best Korean aggregate among models ≤ 30B; TTS error 3.6 → 3.1 % after DPO | 42 GB BF16 | research only |
| A.X K2 ALM (SKT) · 2026-07 | weights "coming soon" | self: KVoiceBench CER 1.86 | streaming by design | not available |
| Kanana-o 9.8B (Kakao) · 2026-02 | API only | self: KsponTTS CER 2.23 | streaming API | not open |
| Qwen3.5-Omni · 2026-03 | API only | self: Korean custom-voice WER 1.309 | – | watch for weights |
| MiniCPM-o 4.5, Qwen2.5-Omni, Fun-Audio-Chat, Step-Audio 2 mini, Raon-SpeechChat-9B | open | no usable Korean speech (KRAFTON: CER 70–121 for the first four; Step-Audio 2 mini 23–35) | – | no |

The answer must come from our LoRA + CUT3R-tuned Qwen3.5-4B, so an omni model would either replace it (losing the ~70
VSI work) or be used as a large TTS. With short answers, first audio is dominated by the VLM anyway.

## 5. Korean text normalisation — required before any TTS

**Evidence that raw text fails:**
- CosyVoice's front-end sends any text without Chinese characters — all Hangul — through English normalisation
  (`inflect`), so "4개" can come out as "four개" (frontend.py, issue #1949). Use `text_frontend=False`.
- Chatterbox's Korean path only splits syllables into jamo; digits pass through untouched, and its NFKD step turns
  "㎡" into "m2".
- Supertonic's number and unit handling claims are English-only; the Qwen3-TTS report says nothing about
  normalisation.
- The only published TN benchmark (Qwen-Audio-3.0-TTS paper, 2026-07, Chinese/English): Qwen3-TTS-1.7B reads
  numbers and dates right 74 % / 79 % of the time, but formulas, units and symbols only 25 % / 44 %; CosyVoice3 and
  VoxCPM2 are similar or worse.
- Builders normalise first: KRAFTON normalised Korean text with an 11-category rulebook before synthesising
  KVoiceBench with Qwen3-TTS ("3개" → 세 개, "3분" → 삼 분) and re-synthesised whatever Whisper flagged.
- CoreaSpeech (NeurIPS 2025, with a numeric subset): normalising numerals and English terms helps every TTS
  architecture on Korean.

**Tools:**
- **NVIDIA NeMo-text-processing, Korean grammar** (pynini rules, Apache-2.0; directory confirmed ✔): decimals
  ("3.14m" → 삼점일사 미터), units kg, g, km, m, cm, mm, L, %, °; native numerals 1–39 before 16 counters (개 명 병 마리
  송이 포기 사람 자루 채 켤레 그루 벌 잔 장 권 살); times ("오후 3시 반" → 오후 세시 반); ordinals ("3번째" → 세번째). Missing:
  ㎡ / m², the counters 대 / 가지 / 곳, an English lexicon.
- g2pK: native numerals before a fixed counter list, but reads "2.4" as "이.사" and leaves "3m" as "삼m"; its output is
  spelled as pronounced — right for phoneme TTS, wrong input for LLM-based TTS.
- KoG2P (phone symbols only, GPL-3.0), num2words `ko` (Sino-Korean only), WeTextProcessing (no Korean): not useful.

**Recommendation:**
1. Keep digits in the VLM's answer (display, VSI scoring).
2. Add a deterministic `speech_text()` between the VLM and the TTS (~200 lines + a lexicon, < 1 ms), extending
   NeMo's grammar or as regex rules for our closed domain:
   - lengths with decimals, digit by digit after the point ("2.4m" → 이 점 사 미터, "1.95m" → 일 점 구오 미터, "0.8m" →
     영 점 팔 미터); drop a trailing ".0";
   - symbols before any NFKD step: ㎡ / m² / m2 → 제곱미터, ° → 도, % → 퍼센트;
   - native numerals before counters: 개 명 대 권 마리 가지 번째, and 시 in "N시 방향" ("4개" → 네 개, "20개" → 스무 개,
     "21권" → 스물한 권, "1개" → 한 개); Sino-Korean for floors and labels ("2층" → 이 층);
   - ranges: "3~4개" → 서너 개;
   - English: a closed lexicon (TV → 티비, PC → 피시, LED → 엘이디, USB → 유에스비, Wi-Fi → 와이파이, iPad → 아이패드)
     with letter-by-letter spelling as the fallback;
   - known traps: 번 and 권 used as labels rather than counters.
3. Prompt the VLM only to avoid parentheses, slashes and "~" — not to spell numbers out: "Ready to Speak" (NAVER LABS
   Europe, English, Qwen3-4B) found prompting alone far below alignment training (listening score 51.4 vs 68.3,
   reference 94.7), and an LLM rewriter doubled the latency.
4. Switch off each TTS model's own normalisation where it has one; normalise clause by clause if answers get long.

The 30 sentences in §7 are the normaliser's unit tests (`tests/fixtures/korean_answer_readings.jsonl`).

## 6. Bake-off to decide (proposed)

- **Systems:** Qwen3-TTS-1.7B ("Sohee"), VoxCPM2, Magpie v2607, Supertonic 3; Fish S2 Pro and HyperCLOVA X Omni as
  non-commercial / heavy references; CosyVoice 3 with `text_frontend=False` optional.
- **Two conditions each:** raw VLM text vs normalised text — so the normaliser's value is measured too.
- **Data:** the 30 domain sentences (§7) + 100 items of CV3-Eval `zero_shot_ko` for general intelligibility.
- **Automatic:** CER with our Qwen3-ASR-1.7B (cross-check with Whisper-large-v3, Korean forced); strip punctuation;
  run the ASR output through the same normaliser before scoring (ASR writes digits); a numeric-value check (2.4 vs
  24). ASR cannot tell 네 개 from 사 개, so numeral-system errors need listeners.
- **Listeners:** 3–5 native speakers, blind and shuffled: per item a yes/no "numbers, units and English read
  correctly" and a 5-point MOS (or side by side against the best system). UTMOS only as a sanity check (less reliable
  for Korean).
- **Latency on the H100:** text ready → first audio chunk, and RTF; P50 / P95 over 30 items × 5 warm runs (for
  non-streaming models first audio = full synthesis).
- **Decision order:** ≥ 29/30 fully correct with normalised text → MOS → time to first audio.
- **Compare only in-house numbers:** Qwen3-TTS's Korean error is 1.76 (own WER, MM-24), 4.07 (Supertone's CER) or 5.64
  (Qwen-Audio-3.0 paper, CER on CV3-ko) depending on who measured.

**Latency budget.** Answers are one sentence, so the TTS can take the whole answer: first audio ≈ VLM (TTFT 0.23 s +
~10–20 answer tokens) + TTS first packet (~0.1–0.25 s) ≈ 0.5–0.7 s after the question text, ~1.2–1.4 s after the user
stops speaking. To be measured on the H100.

## 7. Test sentences (expected readings)

Machine-readable with tags: `tests/fixtures/korean_answer_readings.jsonl`.

| # | Answer text | Spoken form |
|---|---|---|
| 1 | 소파에서 TV까지 약 2.4m예요. | 소파에서 티비까지 약 이 점 사 미터예요. |
| 2 | 의자와 책상 사이는 85cm예요. | 의자와 책상 사이는 팔십오 센티미터예요. |
| 3 | 문까지 0.8m 정도예요. | 문까지 영 점 팔 미터 정도예요. |
| 4 | 침대 길이는 1.95m예요. | 침대 길이는 일 점 구오 미터예요. |
| 5 | 냉장고 높이는 180cm예요. | 냉장고 높이는 백팔십 센티미터예요. |
| 6 | 벽에서 3.25m 떨어져 있어요. | 벽에서 삼 점 이오 미터 떨어져 있어요. |
| 7 | 두 의자 사이는 1m가 안 돼요. | 두 의자 사이는 일 미터가 안 돼요. |
| 8 | 침대는 왼쪽에 있어요. (control) | 침대는 왼쪽에 있어요. |
| 9 | 의자는 4개예요. | 의자는 네 개예요. |
| 10 | 쿠션은 모두 11개 있어요. | 쿠션은 모두 열한 개 있어요. |
| 11 | 컵이 20개 보여요. | 컵이 스무 개 보여요. |
| 12 | 방에 사람이 2명 있어요. | 방에 사람이 두 명 있어요. |
| 13 | 모니터 3대가 있어요. | 모니터 세 대가 있어요. |
| 14 | 책이 21권 있어요. | 책이 스물한 권 있어요. |
| 15 | 램프는 1개뿐이에요. | 램프는 한 개뿐이에요. |
| 16 | 의자는 3~4개 정도예요. | 의자는 서너 개 정도예요. |
| 17 | 넓이는 약 18㎡입니다. | 넓이는 약 십팔 제곱미터입니다. |
| 18 | 방 면적은 12.5m²예요. | 방 면적은 십이 점 오 제곱미터예요. |
| 19 | 문은 2시 방향에 있어요. | 문은 두 시 방향에 있어요. |
| 20 | TV는 정면에서 30° 오른쪽이에요. | 티비는 정면에서 삼십 도 오른쪽이에요. |
| 21 | 침대가 방의 40%를 차지해요. | 침대가 방의 사십 퍼센트를 차지해요. |
| 22 | 3번째 의자가 가장 커요. | 세 번째 의자가 가장 커요. |
| 23 | 화분은 2층 선반에 있어요. | 화분은 이 층 선반에 있어요. |
| 24 | PC 모니터는 책상 위에 있어요. | 피시 모니터는 책상 위에 있어요. |
| 25 | LED 스탠드는 침대 오른쪽에 있어요. | 엘이디 스탠드는 침대 오른쪽에 있어요. |
| 26 | Wi-Fi 공유기는 TV 아래에 있어요. | 와이파이 공유기는 티비 아래에 있어요. |
| 27 | iPad는 소파 위에 있어요. | 아이패드는 소파 위에 있어요. |
| 28 | USB 허브는 노트북 왼쪽에 있어요. | 유에스비 허브는 노트북 왼쪽에 있어요. |
| 29 | 의자 4개, 테이블 1개, TV 1대가 있어요. | 의자 네 개, 테이블 한 개, 티비 한 대가 있어요. |
| 30 | TV까지 2.4m, 문까지 3.1m예요. | 티비까지 이 점 사 미터, 문까지 삼 점 일 미터예요. |

## 8. Not verified

- Qwen3-TTS: the GPU behind the 97 ms; whether token-level streaming text input ships (vLLM-Omni issue #1766 open); its
  README still calls vLLM-Omni offline-only while the vLLM-Omni docs describe online streaming; Mac paths
  (mlx-audio, faster-qwen3-tts) not run.
- How any model reads raw Korean digits and units — nothing published; nothing run here yet.
- Which ASR each vendor used for Korean (Fish S2, VoxCPM2, Magpie, Supertonic); Supertone's Qwen3-TTS variant.
- CosyVoice3: which public checkpoint matches the paper's Korean numbers; the hardware behind "150 ms".
- Fish S2 latency on an H100 (only H200 reported); Magpie streaming from the open checkpoint; VoxCPM2 first audio.
- HyperCLOVA X Omni: streaming and single-H100 serving undocumented; its MOS rests on 10 utterances.
- Qwen3-Omni: self-reported Korean WER 1.67 vs SKT's CER 26.37 vs Naver's MOS 3.40 — unreconciled.
- Licenses: Orpheus-ko (Llama-3.2 base), Raon-Speech-9B (seen on a mirror), MOSS-TTS-Nano, Step-Audio-EditX weights,
  kani-tts-ko (contradictory card).
- NeMo's Korean grammar not run: "2시 방향" without 오전/오후, "3~4개", and pynini on macOS unchecked.
- Gated or private test sets: the MiniMax multilingual set's contents, Kakao's KsponTTS, SKT's KVoiceBench subset.

## Sources

TTS models: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [Qwen3-TTS report](https://arxiv.org/abs/2601.15621) ·
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) · [vLLM-Omni TTS engineering](https://vllm.ai/blog/2026-06-23-vllm-omni-tts) ·
[vLLM-Omni TTS docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_tts_model/) ·
[vLLM-Omni issue #1795](https://github.com/vllm-project/vllm-omni/issues/1795) ·
[VoiceChat-TTS (Qwen3-TTS re-measured)](https://arxiv.org/pdf/2608.13831) ·
[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) · [S2 report](https://arxiv.org/abs/2603.08823) ·
[VoxCPM](https://github.com/OpenBMB/VoxCPM) · [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) ·
[CosyVoice 3 paper](https://arxiv.org/abs/2505.17589) ·
[Magpie TTS Multilingual](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) ·
[Magpie blog](https://huggingface.co/blog/nvidia/magpie-tts-multilingual-voice-agents) ·
[Supertonic 3](https://huggingface.co/Supertone/supertonic-3) · [Supertonic 2](https://huggingface.co/Supertone/supertonic-2) ·
[Supertonic repo (archived)](https://github.com/supertone-oss-archive/supertonic) ·
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) · [MOSS-TTS-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) ·
[Higgs TTS 3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) · [Raon-Speech-9B](https://huggingface.co/KRAFTON/Raon-Speech-9B) ·
[Raon report](https://arxiv.org/html/2605.23912) · [Pocket-TTS-Korean](https://huggingface.co/seastar105/pocket-tts-korean-300m) ·
[Chatterbox Multilingual V3](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages) ·
[VibeVoice-Realtime](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) · [Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX) ·
[Orpheus Korean](https://huggingface.co/canopylabs/3b-ko-ft-research_release) · [kani-tts-400m-ko](https://huggingface.co/nineninesix/kani-tts-400m-ko) ·
[XTTS-v2](https://huggingface.co/coqui/XTTS-v2) · [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) ·
[MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean) · [OpenAudio S1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini).
Omni: [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) ·
[HyperCLOVA X SEED Omni](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B) · [HCX paper](https://arxiv.org/html/2601.01792v1) ·
[OmniServe](https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe) ·
[A.X K2 Raon-Speech](https://huggingface.co/KRAFTON/A.X-K2-Raon-Speech-21B-A3B) · [A.X K2 ALM](https://huggingface.co/skt/A.X-K2-ALM) ·
[Kanana-o](https://huggingface.co/kakaocorp/Kanana-1.5-o-9.8B-instruct-2602-API_Doc) ·
[Qwen3.5-Omni report](https://arxiv.org/html/2604.15804v2) · [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5).
Normalisation and evaluation: [NeMo Korean TN](https://github.com/NVIDIA/NeMo-text-processing/tree/main/nemo_text_processing/text_normalization/ko) ·
[g2pK](https://github.com/Kyubyong/g2pK) · [CosyVoice frontend.py](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/frontend.py) ·
[Chatterbox tokenizer.py](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/models/tokenizers/tokenizer.py) ·
[Qwen-Audio-3.0-TTS (TN benchmark)](https://arxiv.org/html/2607.23938) · [KVoiceBench](https://arxiv.org/abs/2605.27984) ·
[Ready to Speak](https://arxiv.org/abs/2609.01246) · [CoreaSpeech](https://coreaspeech.github.io/demo/) ·
[CV3-Eval](https://huggingface.co/datasets/yuekai/CV3-Eval) · [MiniMax-Speech](https://arxiv.org/abs/2505.07916).
