# densify-v1 training summary (round 1)

**Finished:** 2026-09-04 ~16:05 PT (Europe/Madrid)  
**Stop reason:** `time_limit_reached=4800s` at step 1477  
**Config:** `configs/densify-v1.json`  
**Checkpoint:** `checkpoints/densify-v1/best.pt` (~62MB FP16 weights; not in git — gh release asset)

## Architecture

| Item | Value |
|------|-------|
| Tokenizer | BPE byte-level (HF tokenizers), vocab **5421** |
| Parameters | **28,126,720** (~28.1M) |
| d_model / layers / heads | 512 / 8 / 8 |
| dropout / weight_decay | 0.20 / 0.18 |
| Multi-target | on (remix existing reply variants; chars 529k→552k) |
| Tokens (BPE) | 135,595 |
| Tied lm_head | yes |
| Checkpoint dtype | float16 model weights |

## Training

| Metric | Value |
|--------|-------|
| Device | CPU |
| Best val_loss | **0.6140** (step 1200, train_loss 0.2450) |
| Last logged | step 1400 train_loss 0.1925 val_loss 0.6541 |
| Max duration | 4800s |
| Note | train≪val near end → overfit |

## Held-out eval (`learn_bank_v1`)

Generation: `max_new_tokens=80`, `temperature=0.7`, `top_k=40`.  
Memorization vs `data/processed/agerbot_learn_v1.txt`.

| Model | Accuracy | Memorization rate | Passed / N | Params | best.pt |
|-------|----------|-------------------|------------|--------|---------|
| social-v2 | 0.425 | 0.60 | 17 / 40 | 10.8M | — |
| learn-v1 | 0.475 | 0.55 | 19 / 40 | 14.3M | ~165MB |
| **densify-v1** | **0.375** | **0.75** | **15 / 40** | **28.1M** | **~62MB** |

**Gate:** FAIL — need acc > 0.475 and mem < 0.55 (target better: ≥0.50 / ≤0.40).  
Delta vs learn-v1: **−0.100 accuracy**, **+0.20 memorization_rate**.

## Smoke judgment

`runs/densify-v1/smoke_chat.md`: mostly **memorizing / topic-bleed** (content-creation outros, jokes) rather than answering novel friendship/small-talk prompts. Occasional on-topic lines, but high verbatim overlap with corpus.

## Next

Retrain **densify-v1b**: higher dropout/wd, stronger multi-target, slightly different depth/width within ~30M, shorter duration to cut overfit. No new dialogue text.
