# densify-v1e training summary — NO RELEASE

**Finished:** 2026-09-04 ~20:50 PT (Europe/Madrid)  
**Stop reason:** `time_limit_reached=2100s` at step 499  
**Config:** `configs/densify-v1e.json`  
**Init:** weights-only from `checkpoints/densify-v1d/best.pt` (same ~19.9M arch)  
**Studio:** still serving densify-v1d on :4318 (unchanged)

## Intent

Improve novel-prompt chat coherence while keeping bank gate (acc > 0.475, mem < 0.55).  
No new dialogue phrases. Levers: `gate_prefer` selection, finetune from v1d, stronger existing-pair remix + marker-variant rewrite, slightly higher dropout/wd, shorter LR.

## What changed vs densify-v1d

| Knob | densify-v1d | densify-v1e |
|------|-------------|-------------|
| init | from scratch | **v1d best.pt weights-only** |
| d_model / params | 384 / 19.9M | same |
| dropout / wd | 0.30 / 0.30 | **0.32 / 0.32** |
| lr / duration | 1.8e-4 / 3000s | **1e-4 / 2100s** |
| remix | 12k / 4 / thr 0.88 | **16k / 6 / thr 0.85 + 2500 marker variants** |
| bank score | acc−mem | **gate_prefer** |

Code: `init_checkpoint` in `train.py`; optional `marker_variants` in `data.augment_multitarget_text` (existing user/assistant text under alternate markers only).

## Bank results (temp 0.7, full bank)

During train: bankcand-300 = 0.475 / 0.325 (best bank objective; **not** strict gate); bankcand-450 = 0.325 / 0.425 (collapse).

Post-hoc repeats:

| Checkpoint | Run | Accuracy | Memorization | Gate |
|------------|-----|----------|--------------|------|
| best_val_loss | r1 | 0.450 | 0.175 | FAIL |
| best_val_loss | r2 | 0.450 | 0.375 | FAIL |
| best_val_loss | r3 | 0.350 | 0.325 | FAIL |
| best_val_loss | **mean** | **0.417** | **0.292** | **FAIL** |
| best (bank) | r1 | 0.450 | 0.400 | FAIL |
| best (bank) | r2 | 0.500 | 0.350 | PASS (noise) |
| best (bank) | **mean** | **0.475** | **0.375** | **FAIL** (acc not > 0.475) |
| densify-v1d publish | mean~ | **0.500** | **0.275** | **PASS** |

## Smoke judgment

Compared v1d / v1e_best / v1e_bvl on 10 novel prompts (`reports/smoke_chat_densify-v1e.md`).  
**Not clearly less nonsense than v1d** — still hedges, topic bleed (jokes/sofá/gastronomía/creator-ish), off-topic fragments. Decode-only probes on v1d (temp 0.5–0.9) also fail to fix coherence.

## Decision

**Do not release densify-v1e.** Do not switch Studio off densify-v1d.

## Blunt blocker

Cannot improve novel-prompt coherence while holding the bank gate **without new diverse dialogue data** (or a smarter remix of *existing* pairs that does not invent phrases and does not pollute short-intent groups).

What we already know does **not** work here:
- more params / wider model (v1–v1c overfit)
- finetune + heavier reg from the gate-passing v1d snapshot (this run)
- marker-variant rewrite of existing pairs (helped format variety, hurt bank)
- exact-intent multi-target is nearly saturated (~200–300 productive remixes)
- decode/temp/stop tweaks alone (hide nothing useful; nonsense remains)

**Need:** more diverse existing-data remix strategy (or more held-out-friendly paraphrase diversity in the corpus builder), and/or decode fixes grounded in real turn structure — **not** more parameters.
