# learn-v1 training summary

**Finished:** 2026-09-04 (Europe/Madrid)  
**Stop reason:** `time_limit_reached=4800s` at step 2008  
**Config:** `configs/learn-v1.json`  
**Checkpoint:** `checkpoints/learn-v1/best.pt` (~165MB; not in git — see release asset if uploaded)

## Training

| Metric | Value |
|--------|-------|
| Device | CPU |
| Parameters | 14,337,792 |
| Tokens | 529,035 |
| Best val_loss | **0.3895** (step 1800, train_loss 0.3637) |
| Last logged | step 2000 train_loss 0.2346 val_loss 0.4493 |
| Max duration | 4800s |
| Early stopping | patience 25 (not triggered) |

## Held-out eval (`data/evaluation/learn_bank_v1.jsonl`)

Same generation flags as social-v2 baseline: `max_new_tokens=80`, `temperature=0.7`, `top_k=40`.  
Memorization measured vs `data/processed/agerbot_learn_v1.txt` (train corpus). Baseline used `agerbot_social_v2.txt`.

| Model | Accuracy | Memorization rate | Passed / N | Params |
|-------|----------|-------------------|------------|--------|
| social-v2 best (baseline) | 0.425 | 0.60 | 17 / 40 | 10,788,864 |
| **learn-v1 best** | **0.475** | **0.55** | **19 / 40** | 14,337,792 |

Delta vs baseline: **+0.050 accuracy**, **-0.05 memorization_rate**.

Reports:
- `reports/learn_bank_social-v2_best.json`
- `reports/learn_bank_learn-v1_best.json`

## Last train.log lines

```
step=1800 train_loss=0.3637 val_loss=0.3895 elapsed=4306.1s
best_checkpoint=checkpoints/learn-v1/best.pt
step=1900 train_loss=0.3484 val_loss=0.5678 elapsed=4537.8s
step=2000 train_loss=0.2346 val_loss=0.4493 elapsed=4782.3s
checkpoint=checkpoints/learn-v1/latest.pt
time_limit_reached=4800s step=2008 checkpoint=checkpoints/learn-v1/latest.pt
```

Eval command:

```bash
.venv/bin/python scripts/eval_learn_bank.py \
  --checkpoint checkpoints/learn-v1/best.pt \
  --bank data/evaluation/learn_bank_v1.jsonl \
  --train-corpus data/processed/agerbot_learn_v1.txt \
  --device cpu --max-new-tokens 80 --temperature 0.7 --top-k 40 \
  --out reports/learn_bank_learn-v1_best.json
```
