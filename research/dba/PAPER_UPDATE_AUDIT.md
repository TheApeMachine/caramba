# Paper Update Audit: 10k → 100k

## Summary
The paper was written around 10k-step results and references an older "primary" config (sem16/geo32) that is no longer the focus. The 100k runs used only **baseline** vs **aggressive DBA (sem8/geo32/v40)**. Several legacy numbers from even earlier experiments (168x, WikiText, 6-layer models) also need removal.

---

## CRITICAL: Outdated Numbers to Replace

### Loss/Perplexity (throughout paper)
| Location | Old (10k) | New (100k) |
|----------|-----------|------------|
| Line 47 (abstract) | 3.36 vs 3.38 | 2.67 vs 2.68 |
| Line 98 (contributions) | 3.36 vs 3.38 | 2.67 vs 2.68 |
| Line 322 | 3.36 (PPL 28.9) vs 3.38 (PPL 29.5) | 2.67 (PPL 14.46) vs 2.68 (PPL 14.57) |
| Line 377 | 3.36 vs 3.38 | 2.67 vs 2.68 |
| Line 760 | 3.36 vs 3.38 | 2.67 vs 2.68 |
| Line 783 | 3.36 vs 3.38 | 2.67 vs 2.68 |

### Step Count References
| Line | Change |
|------|--------|
| 47 | "At 10k steps" → "At 100k steps" |
| 96 | "At 10k steps" → "At 100k steps" |
| 97 | "at 10k steps" → "at 100k steps" |
| 98 | "at 10k steps" → "at 100k steps" |
| 247-249 | Table: 10k → 100k |
| 276 | "10k steps" → "100k steps" |
| 300 | Caption: "10k-step" → "100k-step" |
| 321-322 | "10k-step pilot" → "100k-step training" |
| Many more... |

### Training Efficiency Claims (NEED VERIFICATION from 100k data)
| Claim | Line | Current | Need to verify |
|-------|------|---------|----------------|
| Memory reduction | 47, 98, 335, 363 | 27% (55GB→40GB) | Check 100k logs |
| Throughput advantage | 47, 98, 346, 364 | 32% | Check 100k logs |
| Warmup throughput | 346 | 18k tok/s | Check 100k logs |
| Post-warmup baseline | 346 | 13.5k tok/s | Check 100k logs |
| Post-warmup DBA | 346 | 17.5k tok/s | Check 100k logs |

---

## REMOVE: Legacy/Outdated Content

### 1. The "168x" claim (Line 216)
```
This is the origin of the often-quoted ``168$\times$'': \((4096/96)\times 4 \approx 171\)
```
**Action:** REMOVE this footnote entirely. It's from 6-layer WikiText experiments, not validated.

### 2. "Primary" config references (sem16/geo32/v48 = 512/1024/1536)
The 100k runs only used sem8/geo32/v40. References to "primary" config are misleading:
- Line 161-162: "in primary config"
- Line 174: "in our primary configuration"
- Line 202: "$2048\!\rightarrow\!1536$"
- Line 248: "Decoupled (primary)" row
- Line 277: "primary comparison"
- Line 335: "primary decoupled"
- Line 377: "$d_{\text{sem}}{=}512$, $d_{\text{geo}}{=}1024$"
- Line 398: "primary DBA"
- Lines 407, 417, 429, 571: "DBA (16+32)" column

**Action:** Either remove "primary" config entirely OR clearly label it as "not run at 100k"

### 3. Three-way comparison tables
Tables comparing Baseline vs DBA(16+32) vs DBA(8+32) need to become two-way:
- Table in line 403-414 (behavioral probes)
- Table in line 425-437 (qualitative)
- Table in line 567-579 (downstream accuracy)

**Action:** Remove middle column OR note it wasn't run at 100k

---

## UPDATE: Figure References

### Current (10k) → New (100k)
| Current filename | New filename |
|------------------|--------------|
| A100-1b-10k-loss.png | A100-1b-100k-loss.png |
| A100-1b-10k-ppl.png | A100-1b-100k-ppl.png |
| A100-1b-10k-gpu_memory.png | A100-1b-100k-gpu_memory.png |
| A100-1b-10k-tok_s.png | A100-1b-100k-tok_s.png |
| A100-1b-10k-grad_norms.png | A100-1b-100k-grad_norms.png |
| A100-1b-10k-warmup_spikes.png | A100-1b-100k-warmup_spikes.png (if needed) |
| A100-1b-10k-training_summary.tex | A100-1b-100k-training_summary.tex |

Lines to update: 283-286, 293-296, 305-306, 314-315, 330-333, 341-344, 352-355

---

## UPDATE: Table Data

### Table~\ref{tab:configs} (Line 237-258)
Current shows:
- Baseline: 10k Complete
- Decoupled (primary): 10k Complete
- Decoupled (aggressive): 10k Complete

Should be:
- Baseline: 100k Complete
- Decoupled (sem8/geo32/v40): 100k Complete
- Remove or mark "primary" as not run at 100k

### Table~\ref{tab:a100-training-summary-10k}
Referenced at lines 322, 377, 760, 796
**Action:** Rename to `tab:a100-training-summary-100k` and update values

### Downstream accuracy table (Line 564-579)
Currently shows 10k results.
**Action:** Either run 100k downstream eval OR clearly label as "10k preliminary"

---

## STRUCTURAL DECISIONS NEEDED

1. **Primary vs Aggressive config:** Do we keep both in the paper or focus only on sem8/geo32?
   - 100k runs only have baseline + aggressive
   - Behavioral probes were done at 10k for all three

2. **Behavioral probes:** Re-run on 100k checkpoints?
   - Line 558 says "We will re-run this analysis on 100k-step checkpoints"

3. **Downstream accuracy:** Re-run on 100k checkpoints?
   - Currently shows 10k data only

4. **Local suite (5k, 12L, 550M):** Still planned or cut?
   - Lines 252-255, 613-632

---

## NEW DATA TO ADD

From the 100k analysis we already computed:

### Efficiency metrics (need to add)
- Parameter reduction: 10.4% total, 37.5% attention
- KV cache reduction: 37.5% (4096 → 2560 elements/token)

### Training dynamics (need to add)
- DBA has more stable gradient norms (std: 0.094 vs 0.179)
- DBA spends fewer steps at clipping (111 vs 439)

---

## Files to Generate

1. `A100-1b-100k-loss.png` - Loss curves
2. `A100-1b-100k-ppl.png` - Perplexity curves
3. `A100-1b-100k-grad_norms.png` - Gradient norm comparison
4. `A100-1b-100k-training_summary.tex` - LaTeX summary table
5. `A100-1b-100k-loss_gap.png` - Loss gap over training (new!)
6. Update `100k_results_table.tex` - Already created
