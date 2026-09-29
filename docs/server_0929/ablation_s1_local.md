# S1 geometry ablation on the M2 with the server's S1 checkpoints (2026-09-29)

`scripts/eval_geometry_ablation.py`, 4B, `configs/m2_zeroshot_4b.yaml` (same fusion as `server_4b.yaml`; LoRA is
irrelevant for S1 weights), stage align, 94 held-out Sensenova samples (`data/sensenova_1k/sn1k.holdout.jsonl`;
drawn from the 800K set, so any overlap with the server's 13.6k S1 training samples is ~1.6 % in expectation).
Same samples in both runs (the `none` loss is identical on 94/94 rows). Donors: same grid shape on 72.3 % of rows.
Per-sample rows in `ablation_s1_4b.json` and `ablation_s1_4b_control_vs_control_seed1.json`; logs in
`logs/ablation_s1_4b.log`, `logs/ablation_s1_4b_cc.log`.

| quantity (holdout loss, lower is better; mean ± 95 % CI) | value |
|---|---|
| s1_real: loss real / shuffled / none | 0.6286 / 0.6652 / 2.7657 |
| s1_control: loss real / shuffled (its training condition) / none | 0.6088 / 0.6024 / 2.7657 |
| s1_control_seed1: loss shuffled / real | 0.6029 / 0.6128 |
| **real projector reads content**: shuffled − real within s1_real | **+0.0366 ± 0.0314** (real wins 66.0 %) |
| control projector reads content: shuffled − real within s1_control | −0.0063 ± 0.0123 (42.6 %) |
| **value of content vs control** = s1_control(shuffled) − s1_real(real) | **−0.0262 ± 0.0408** (real wins 52.1 %) |
| seed noise at the training condition = control_seed1(shuffled) − control(shuffled) | +0.0005 ± 0.0214 |
| value of the injection path = none − real (s1_real / s1_control) | +2.137 ± 0.545 / +2.157 ± 0.548 |

Reading:
- The injection path is worth ~2.1 nats of loss for BOTH arms: switching injection off returns the base model's
  loss (2.77) — S1 learned the answer format/distribution through the projector (the M2 pilot's finding, now on the
  server's 4B checkpoints).
- The real projector's loss does depend on the geometry's content (+0.037, lower bound > 0) and the control's does
  not (−0.006): real S1 learned *something* about content. But the control projector reaches a lower loss overall
  (0.602 vs 0.629), so against the control the value of content is −0.026 ± 0.041 — not distinguishable from 0,
  with seed noise far smaller (±0.021). At 13.6k samples (0.016 epoch) the content signal is real but smaller than
  what the arms differ by for other reasons; it cannot pay for itself yet.
- Consistent with the full-set VSI result s2_real − s2_sft = −1.6: at ~2 % of an epoch the geometry path costs more
  than its content returns. The comparison that decides the question (VSI s2_real − s2_control, and the same
  ablation at S2) is still missing, and the data budget has to grow by ~50× before a null is informative.
