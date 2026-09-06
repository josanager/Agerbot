# densify-v1b training summary (round 2)

**Finished:** 2026-09-04 ~17:10 PT (Europe/Madrid)  
**Stop reason:** `time_limit_reached=3600s` at step 925  
**Config:** `configs/densify-v1b.json`  
**Checkpoint:** `checkpoints/densify-v1b/best.pt` (~58MB FP16)

## Changes vs densify-v1

| Knob | densify-v1 | densify-v1b |
|------|------------|-------------|
| d_model / layers | 512 / 8 | **448 / 10** |
| dropout | 0.20 | **0.28** |
| weight_decay | 0.18 | **0.25** |
| lr | 3e-4→2.5e-4 | **2e-4** |
| max_duration | 4800s | **3600s** |
| early_stop patience | 25 | **12** |
| multi-target extras | 4000 | **8000** |
| params | 28.1M | **26.7M** |

## Training

| Metric | Value |
|--------|-------|
| Best val_loss | **1.0414** (step 900, train 0.6793) |
| Last | time limit step 925 |
| Overfit gap | smaller than round1 (train/val closer) |

## Held-out eval

| Model | Accuracy | Memorization | Params | best.pt |
|-------|----------|--------------|--------|---------|
| social-v2 | 0.425 | 0.60 | 10.8M | — |
| learn-v1 | 0.475 | 0.55 | 14.3M | ~165MB |
| densify-v1 | 0.375 | 0.75 | 28.1M | ~62MB |
| **densify-v1b** | **0.450** | **0.55** | **26.7M** | **~58MB** |

**Gate:** still FAIL (need acc > 0.475 and mem < 0.55).  
Improved vs densify-v1 (+0.075 acc, −0.20 mem) but not yet past learn-v1.

## Smoke judgment

Less topic-bleed than round1; more hedging / vague refusals. Partial learning signal, not publishable.

## Next

densify-v1c: keep BPE + ~27M budget, dropout~0.26, wd~0.22, longer 4800s with tighter early stop so val can keep falling without round1-style memorize.
