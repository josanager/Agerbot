# densify-v1c training summary (round 3)

**Finished:** 2026-09-04 ~18:33 PT (Europe/Madrid)  
**Stop reason:** `time_limit_reached=4800s` at step 1237  
**Config:** `configs/densify-v1c.json`  
**Checkpoint:** `checkpoints/densify-v1c/best.pt` (~58MB FP16)

## Intent

Longer train (4800s) with dropout 0.26 / wd 0.22 on the v1b shape (448×10, 26.7M), hoping val would keep falling without round1 memorize.

## Training

Best val_loss **0.6704** at step 1200 (train 0.3991) — lower val than v1b, but bank quality worse.

## Held-out eval

| Model | Accuracy | Memorization | Params |
|-------|----------|--------------|--------|
| social-v2 | 0.425 | 0.60 | 10.8M |
| learn-v1 | **0.475** | **0.55** | 14.3M |
| densify-v1 | 0.375 | 0.75 | 28.1M |
| **densify-v1b (best densify)** | **0.450** | **0.55** | 26.7M |
| densify-v1c | 0.275 | 0.55 | 26.7M |

**Gate:** FAIL. Longer train regenerated content-creation topic-bleed; accuracy collapsed.

## Smoke judgment

Mostly memorizing / wrong-domain (creator outros). Not learning useful friendship/small-talk behavior.

## Loop status

Stopping further densify rounds here: required retrain done; densify-v1b remains best densify checkpoint but still does not beat learn-v1 publish gate (acc>0.475 and mem<0.55).
