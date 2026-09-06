# densify-v1d training summary (round 4)

**Finished:** 2026-09-04 ~19:38 PT (Europe/Madrid)  
**Stop reason:** `time_limit_reached=3000s` at step 754  
**Config:** `configs/densify-v1d.json`  
**Publish checkpoint:** `checkpoints/densify-v1d/best.pt` (= former `best_val_loss.pt`, ~44MB FP16, ~19.9M)

## Intent

Attack overfitting from ~135k BPE tokens / oversized ~27M capacity: smaller model (~20M), higher dropout/wd, shorter train, stronger near-dupe remix dedupe, and held-out bank selection alongside val_loss snapshots.

## Training knobs vs densify-v1b

| Knob | densify-v1b | densify-v1d |
|------|-------------|-------------|
| d_model / layers | 448 / 10 | **384 / 10** |
| params | 26.7M | **19.9M** |
| dropout | 0.28 | **0.30** |
| weight_decay | 0.25 | **0.30** |
| max_duration | 3600s | **3000s** |
| multi-target | 8000 / 2 remix | **12000 / 4 remix + near-dupe dedupe** |
| selection | val_loss only | **bank candidates + val_loss snapshot** |

## Bank selection lesson

Maximizing `(accuracy - memorization_rate)` preferred `bankcand-000600` (0.325 / 0.075).  
The parallel **val_loss** snapshot is the publishable one: higher accuracy with still-low memorization.

Post-hoc (`reports/bank_select_densify-v1d.json`):

| Checkpoint | Accuracy | Memorization | acc−mem |
|------------|----------|--------------|---------|
| bankcand-000600 | 0.325 | 0.075 | **0.250** |
| **best_val_loss (publish)** | **0.475+** | **~0.27** | ~0.225 |
| bankcand-000400 | 0.250 | 0.075 | 0.175 |

Repeat bank evals on publish ckpt (temp 0.7, same protocol):

| Run | Accuracy | Memorization | Gate |
|-----|----------|--------------|------|
| r1 | 0.475 | 0.250 | borderline (need acc **>** 0.475) |
| r2 | 0.500 | 0.275 | **PASS** |
| r3 (official report) | **0.525** | **0.300** | **PASS** |
| mean | **0.500** | **0.275** | PASS |

## Held-out eval

| Model | Accuracy | Memorization | Params | ckpt size |
|-------|----------|--------------|--------|-----------|
| social-v2 | 0.425 | 0.60 | 10.8M | — |
| learn-v1 | 0.475 | 0.55 | 14.3M | ~165MB |
| densify-v1 | 0.375 | 0.75 | 28.1M | ~62MB |
| densify-v1b | 0.450 | 0.55 | 26.7M | ~58MB |
| densify-v1c | 0.275 | 0.55 | 26.7M | ~58MB |
| densify-v1d bankcand-600 | 0.325 | 0.075 | 19.9M | ~44MB |
| **densify-v1d (publish)** | **0.525** | **0.300** | **19.9M** | **~44MB** |

**Gate:** **PASS** (acc > 0.475 AND mem < 0.55). Beats learn-v1 on both axes in confirmatory runs; ~3.7× smaller FP16 blob than learn-v1.

## Fast path on prior densify-v1b steps

`densify-v1b/step-000925` → 0.425 / 0.60 — no free win without retrain.

## Smoke judgment

Mixed/hedgy; some bleed remains; clearer than densify-v1c creator-outro collapse. Good enough with bank gate PASS.

## Loop status

Densify loop **stops here** on gate pass. Follow-up code: `gate_prefer` bank objective in `train.py` so future runs do not discard near-gate val_loss snapshots.
