Let's explore the space of model insight tooling. I'll think about this from the perspective of "what questions do researchers actually ask" and work backwards to the tools that answer them.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUESTIONS RESEARCHERS ASK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DURING TRAINING                                                           │
│   ───────────────────────────────────────────────────────────────────────   │
│   • "Is this run healthy?"                                                  │
│   • "Why did the loss spike?"                                               │
│   • "Is this layer even learning?"                                          │
│   • "When should I stop?"                                                   │
│                                                                             │
│   AFTER TRAINING                                                            │
│   ───────────────────────────────────────────────────────────────────────   │
│   • "What did the model learn?"                                             │
│   • "Why does it fail on this example?"                                     │
│   • "What's the difference between checkpoint A and B?"                     │
│   • "Which component matters most?"                                         │
│                                                                             │
│   COMPARING ARCHITECTURES                                                   │
│   ───────────────────────────────────────────────────────────────────────   │
│   • "Why is DBA faster but equally good?"                                   │
│   • "What's baseline doing that variant isn't?"                             │
│   • "Where do they diverge in behavior?"                                    │
│                                                                             │
│   MECHANISTIC                                                               │
│   ───────────────────────────────────────────────────────────────────────   │
│   • "What circuit implements this capability?"                              │
│   • "Which heads are doing what?"                                           │
│   • "Is there redundancy I can prune?"                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. The "Autopsy" Tool — Post-Mortem for Failed Runs

When a run fails or produces bad results, you need forensics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RUN AUTOPSY: experiment_dba_s1337                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Status: DIVERGED at step 12,847                                           │
│                                                                             │
│  TIMELINE OF ANOMALIES                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Step 8,200   ⚠ Gradient norm exceeded 10.0 for first time                 │
│               └─ layers.7.attention.W_o: 14.2                               │
│                                                                             │
│  Step 9,100   ⚠ Loss variance increased 3x                                 │
│               └─ Rolling std: 0.04 → 0.12                                   │
│                                                                             │
│  Step 11,500  ⚠ Activation collapse detected                               │
│               └─ layers.9.ffn: 94% of activations < 0.01                    │
│                                                                             │
│  Step 12,400  🔴 First NaN in gradients                                     │
│               └─ layers.11.attention.W_q                                    │
│                                                                             │
│  Step 12,847  💀 Loss = NaN, run terminated                                 │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  PROBABLE CAUSE ANALYSIS                                                    │
│                                                                             │
│  1. Attention output projection (W_o) in layer 7 showed early instability   │
│  2. This propagated forward, causing activation collapse in later FFN       │
│  3. Gradient explosion followed, leading to NaN                             │
│                                                                             │
│  RECOMMENDATIONS                                                            │
│                                                                             │
│  • Consider gradient clipping < 10.0                                        │
│  • Check layer 7 initialization                                             │
│  • Review learning rate for attention layers                                │
│                                                                             │
│  [View checkpoint before divergence] [Compare to healthy run] [Replay]      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What this requires:**
- Anomaly detection running during training (thresholds for grad norm, loss variance, activation statistics)
- Event log with timestamps
- "Last known good" checkpoint tracking
- Pattern matching against known failure modes

---

## 2. The "Diff" Tool — Checkpoint Comparison

Like `git diff` but for model weights:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT DIFF                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Comparing: step_5000 → step_10000                                          │
│                                                                             │
│  SUMMARY                                                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Total parameter change: 2.3% (L2 distance / initial norm)                │
│  • Largest changes in: layers.0-3 (early layers)                            │
│  • Smallest changes in: layers.10-11 (late layers)                          │
│  • Loss improved: 2.8 → 2.4 (14% reduction)                                 │
│                                                                             │
│  PER-MODULE CHANGE (sorted by magnitude)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Module                      │ Δ Norm  │ Δ Mean  │ Δ Std   │ % Changed     │
│  ────────────────────────────┼─────────┼─────────┼─────────┼───────────    │
│  layers.0.attention.W_q      │ 0.142   │ +0.003  │ +0.008  │ ██████ 8.2%   │
│  layers.0.attention.W_k      │ 0.138   │ +0.002  │ +0.007  │ █████▓ 7.9%   │
│  layers.1.ffn.W_1            │ 0.127   │ -0.001  │ +0.012  │ █████░ 7.1%   │
│  layers.0.ffn.W_1            │ 0.121   │ +0.001  │ +0.009  │ ████▓░ 6.8%   │
│  ...                         │         │         │         │               │
│  layers.11.attention.W_o     │ 0.012   │ +0.000  │ +0.001  │ ▓░░░░░ 0.7%   │
│  output.weight               │ 0.008   │ +0.000  │ +0.000  │ ░░░░░░ 0.4%   │
│                                                                             │
│  INTERPRETATION                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Early layers are adapting most, suggesting the model is still learning     │
│  basic features. Late layers are relatively stable. This is normal for      │
│  this stage of training.                                                    │
│                                                                             │
│  [View side-by-side] [Show as heatmap] [Animate interpolation]              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Bonus: Animated Interpolation**

Smoothly morph between two checkpoints and watch how predictions change on a fixed eval set. This reveals the "loss landscape" between two points.

---

## 3. The "Probe" Tool — Interactive Input Analysis

Given a specific input, trace exactly what happens:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT PROBE                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: "The capital of France is"                                          │
│  Prediction: " Paris" (p=0.94)                                              │
│  Checkpoint: step_10000                                                     │
│                                                                             │
│  TOKEN FLOW                                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│     "The"   "capital"   "of"   "France"   "is"                             │
│       │         │        │        │        │                                │
│       ▼         ▼        ▼        ▼        ▼                                │
│     ┌───────────────────────────────────────────┐                           │
│     │              Embedding                    │                           │
│     └───────────────────────────────────────────┘                           │
│       │         │        │        │        │                                │
│       ▼         ▼        ▼        ▼        ▼                                │
│     ┌───────────────────────────────────────────┐                           │
│     │     Layer 0: "France" attends to "capital"│ ← click to expand        │
│     └───────────────────────────────────────────┘                           │
│       │         │        │        │        │                                │
│       ▼         ▼        ▼        ▼        ▼                                │
│                        ...                                                  │
│       │         │        │        │        │                                │
│       ▼         ▼        ▼        ▼        ▼                                │
│     ┌───────────────────────────────────────────┐                           │
│     │     Layer 11: "is" gathers from "France"  │                           │
│     └───────────────────────────────────────────┘                           │
│       │         │        │        │        │                                │
│       ▼         ▼        ▼        ▼        ▼                                │
│     ┌───────────────────────────────────────────┐                           │
│     │              Output: "Paris"              │                           │
│     └───────────────────────────────────────────┘                           │
│                                                                             │
│  KEY INFORMATION FLOW                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • "France" is the most attended token at output position                   │
│  • Layer 6-8 appear to do the "fact lookup"                                 │
│  • Layer 0-2 establish syntactic relationships                              │
│                                                                             │
│  [Perturb input] [Compare to wrong prediction] [Find similar examples]      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**"Perturb Input" Sub-tool:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT PERTURBATION ANALYSIS                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Original: "The capital of France is" → "Paris" (p=0.94)                   │
│                                                                             │
│  SUBSTITUTIONS                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  "The capital of Germany is" → "Berlin" (p=0.91)     ✓ correct             │
│  "The capital of Frnace is"  → "Paris" (p=0.72)      ✓ robust to typo      │
│  "The capital of Narnia is"  → "London" (p=0.31)     ? confused            │
│  "The largest city of France is" → "Paris" (p=0.88) ✓ correct              │
│                                                                             │
│  TOKEN IMPORTANCE (leave-one-out)                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  "The"     → p(Paris) = 0.89  │ ░░░░░░░░░░ -5%                             │
│  "capital" → p(Paris) = 0.42  │ █████████░ -55%  ← most important          │
│  "of"      → p(Paris) = 0.91  │ ░░░░░░░░░░ -3%                             │
│  "France"  → p(Paris) = 0.12  │ ██████████ -87%  ← critical                │
│  "is"      → p(Paris) = 0.93  │ ░░░░░░░░░░ -1%                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The "Head Surgeon" — Attention Head Analysis

For transformers, attention heads are the atomic units of computation. Understanding them is crucial:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ATTENTION HEAD CENSUS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model: dba_s1337_step10000                                                │
│  Total heads: 96 (12 layers × 8 heads)                                      │
│                                                                             │
│  HEAD TYPES (auto-detected)                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Type                │ Count │ Layers      │ Description                   │
│  ────────────────────┼───────┼─────────────┼──────────────────────────     │
│  Previous Token      │ 12    │ 0,1,2       │ Attends to t-1                │
│  Induction           │ 8     │ 3,4,5       │ Pattern matching [A][B]...[A] │
│  Positional          │ 6     │ 0,1         │ Fixed position patterns       │
│  Semantic            │ 18    │ 6,7,8,9     │ Content-based attention       │
│  Rare/Specialized    │ 4     │ 10,11       │ Low-frequency patterns        │
│  Seemingly Dead      │ 3     │ 5,9,11      │ Near-uniform attention        │
│  Unclassified        │ 45    │ various     │ Mixed patterns                │
│                                                                             │
│  DEAD HEAD ALERT                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  L5H3, L9H7, L11H2 show near-uniform attention across eval set.             │
│  These may be candidates for pruning (saves 3% compute).                    │
│                                                                             │
│  [View head details] [Run pruning simulation] [Compare to baseline]         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Individual Head Inspector:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HEAD DETAIL: Layer 4, Head 2 (Induction Head)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SIGNATURE PATTERN                                                          │
│  When input contains [A][B]...[A], this head attends [A] → [B]             │
│                                                                             │
│  EXAMPLE                                                                    │
│  Input: "The cat sat on the mat. The cat"                                  │
│                                                                             │
│         The  cat  sat  on  the  mat   .  The  cat                          │
│  The     █    ░    ░   ░    ░    ░   ░    ░    ░                           │
│  cat     ░    █    ░   ░    ░    ░   ░    ░    ░                           │
│  sat     ░    ░    █   ░    ░    ░   ░    ░    ░                           │
│  on      ░    ░    ░   █    ░    ░   ░    ░    ░                           │
│  the     █    ░    ░   ░    █    ░   ░    ░    ░                           │
│  mat     ░    ░    ░   ░    ░    █   ░    ░    ░                           │
│  .       ░    ░    ░   ░    ░    ░   █    ░    ░                           │
│  The     █    ░    ░   ░    █    ░   ░    █    ░                           │
│  cat     ░    █    ░   ░    ░    ░   ░    ░    █  ← attends to first "cat"│
│                                                   ▲                        │
│                                        enables prediction "sat"            │
│                                                                             │
│  ABLATION IMPACT                                                            │
│  Zeroing this head: perplexity +12% on repetition tasks, +2% overall       │
│                                                                             │
│  EMERGENCE                                                                  │
│  This head's induction behavior emerged at step ~3,000                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The "Dataset Lens" — Understanding Data-Model Interaction

Which examples are easy? Which are hard? Which are the model memorizing?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATASET LENS                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Eval Set: wikitext_val (500 examples)                                      │
│  Checkpoint: step_10000                                                     │
│                                                                             │
│  LOSS DISTRIBUTION                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│              ▄▄▄▄                                                           │
│            ▄██████▄                                                         │
│          ▄██████████▄                                                       │
│        ▄██████████████▄                              ▄                      │
│    ▄▄▄████████████████████▄▄▄               ▄▄▄▄▄▄▄███                      │
│   ─────────────────────────────────────────────────────────                 │
│   0.5     1.0     1.5     2.0     2.5     3.0     4.0+                     │
│   easy ◀─────────────────────────────────────────────▶ hard                │
│                                                                             │
│  EXAMPLE CATEGORIES                                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Very Easy (loss < 1.0)         │ 47 examples (9%)                         │
│  └─ Mostly: common phrases, punctuation prediction                         │
│                                                                             │
│  Easy (loss 1.0-2.0)            │ 312 examples (62%)                       │
│  └─ Standard text, expected predictions                                    │
│                                                                             │
│  Hard (loss 2.0-3.0)            │ 98 examples (20%)                        │
│  └─ Rare words, technical content, long-range dependencies                 │
│                                                                             │
│  Very Hard (loss > 3.0)         │ 43 examples (9%)                         │
│  └─ Names, numbers, novel combinations                                     │
│                                                                             │
│  [View hardest examples] [View easiest examples] [Track over training]      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**"Hardest Examples" Deep Dive:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HARDEST EXAMPLES                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #1 (loss = 8.42)                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Context: "The Treaty of Westphalia was signed in"                         │
│  Target: "1648"                                                            │
│  Prediction: "the" (p=0.23)                                                │
│  Why hard: Rare factual recall, year prediction                            │
│                                                                             │
│  #2 (loss = 7.91)                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Context: "Her phone number is 555-"                                       │
│  Target: "0123"                                                            │
│  Prediction: "1234" (p=0.18)                                               │
│  Why hard: Arbitrary number, no way to "know" this                         │
│                                                                             │
│  #3 (loss = 6.73)                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Context: "The enzyme ribonuclease catalyzes the hydrolysis of"            │
│  Target: "RNA"                                                             │
│  Prediction: "proteins" (p=0.31)                                           │
│  Why hard: Technical domain, requires biochemistry knowledge               │
│                                                                             │
│  PATTERN ANALYSIS                                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 34% of hardest examples involve numbers/dates                           │
│  • 28% involve rare proper nouns                                           │
│  • 21% involve technical/scientific terms                                  │
│  • 17% other                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The "Learning Curve Microscope" — What's Being Learned When?

Track when specific capabilities emerge:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CAPABILITY EMERGENCE TRACKER                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tracking capabilities across training:                                     │
│                                                                             │
│  CAPABILITY              │ STEP    │ THRESHOLD │ TREND                     │
│  ────────────────────────┼─────────┼───────────┼─────────────────────────  │
│  Basic grammar           │ 500     │ >80% acc  │ ████████████████ stable   │
│  Subject-verb agreement  │ 1,200   │ >90% acc  │ ████████████████ stable   │
│  Induction (copying)     │ 2,800   │ >70% acc  │ █████████████░░░ growing  │
│  Factual recall          │ 6,500   │ >40% acc  │ ████████░░░░░░░░ slow     │
│  Multi-hop reasoning     │ —       │ >30% acc  │ ░░░░░░░░░░░░░░░░ not yet  │
│  Code completion         │ 8,200   │ >50% acc  │ ██████░░░░░░░░░░ emerging │
│                                                                             │
│  EMERGENCE PLOT                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  100%│                                              ┌─────────────────      │
│      │                                    ┌─────────┘ grammar               │
│   80%│                          ┌─────────┘                                 │
│      │                ┌─────────┘ subject-verb                              │
│   60%│          ┌─────┘                                                     │
│      │    ┌─────┘ induction                                                 │
│   40%│    │              ┌────── factual                                    │
│      │────┘        ┌─────┘                                                  │
│   20%│       ┌─────┘                                                        │
│      │───────┘ code                                                         │
│    0%└─────────────────────────────────────────────────────────────────     │
│       0      2k      4k      6k      8k      10k                            │
│                                                                             │
│  [Add capability probe] [Compare to other runs] [Export timeline]           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. The "Similarity Search" — Find Related Examples

"Show me other inputs where the model behaves similarly":

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIMILARITY SEARCH                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "The capital of France is" → "Paris"                               │
│  Mode: [●] Similar prediction  [ ] Similar attention  [ ] Similar error    │
│                                                                             │
│  RESULTS                                                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  1. "The capital of Germany is" → "Berlin"                                 │
│     Similarity: 0.94 (same pattern: capital-of-country)                    │
│                                                                             │
│  2. "The largest city in France is" → "Paris"                              │
│     Similarity: 0.91 (same answer, different framing)                      │
│                                                                             │
│  3. "Paris is the capital of" → "France"                                   │
│     Similarity: 0.87 (inverse relationship)                                │
│                                                                             │
│  4. "The president of France is" → "Macron"                                │
│     Similarity: 0.82 (same structure: the X of France is Y)                │
│                                                                             │
│  5. "France's capital city is" → "Paris"                                   │
│     Similarity: 0.79 (rephrased)                                           │
│                                                                             │
│  CLUSTER VISUALIZATION                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│          ○ Germany/Berlin                                                   │
│        ○ Italy/Rome    ○ Spain/Madrid                                       │
│              ● France/Paris (query)                                         │
│         ○ UK/London                                                         │
│                   ○ Japan/Tokyo                                             │
│                                                                             │
│  These examples form a "capital city" cluster in activation space.          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. The "Counterfactual Playground"

"What if I changed this one thing?"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COUNTERFACTUAL ANALYSIS                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Base: "The doctor told her patient that she was pregnant"                 │
│  Target token: "she"                                                       │
│                                                                             │
│  COUNTERFACTUAL INTERVENTIONS                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  What if we change...                                                       │
│                                                                             │
│  "doctor" → "nurse"                                                        │
│  └─ p(she) = 0.72 → 0.78 (+8%)    Model slightly more biased               │
│                                                                             │
│  "doctor" → "surgeon"                                                      │
│  └─ p(she) = 0.72 → 0.45 (-38%)   "Surgeon" activates male bias            │
│                                                                             │
│  "her" → "his"                                                             │
│  └─ p(she) = 0.72 → 0.31 (-57%)   Possessive strongly influences           │
│                                                                             │
│  "patient" → "patients"                                                    │
│  └─ p(she) = 0.72 → 0.69 (-4%)    Minimal effect                           │
│                                                                             │
│  CAUSAL GRAPH                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│      "doctor" ──(0.38)──▶ p("she")                                         │
│                              ▲                                              │
│         "her" ──(0.57)───────┘                                              │
│                              ▲                                              │
│     "patient" ──(0.04)───────┘                                              │
│                                                                             │
│  The possessive "her" has the largest causal effect on pronoun choice.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. The "Training Replay" — Time-Travel Debugging

Scrub through training like a video:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING REPLAY                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ◀◀   ◀   ▶   ▶▶   ⏸                          Step: 5,420 / 10,000        │
│  ════════════════●══════════════════════════════════════════════════════   │
│                                                                             │
│  Probe input: "The quick brown fox"                                        │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐                        │
│  │   ATTENTION (L3H2)   │  │    PREDICTION        │                        │
│  │                      │  │                      │                        │
│  │  ░░▒▒░░░░            │  │  "jumps" (p=0.34)    │                        │
│  │  ░░░░▓▓░░            │  │  "ran"   (p=0.21)    │                        │
│  │  ░░░░░░██            │  │  "sat"   (p=0.12)    │                        │
│  │  ░░░░░░░░            │  │  ...                 │                        │
│  │                      │  │                      │                        │
│  └──────────────────────┘  └──────────────────────┘                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       METRICS AT THIS STEP                           │  │
│  │                                                                      │  │
│  │  Loss: 2.41   |   Grad Norm: 1.2   |   LR: 0.0003                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  [Bookmark this step] [Compare to another step] [Export frame]              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: The Microscope Toolbox

| Tool | Question Answered | Key Insight |
|------|-------------------|-------------|
| **Autopsy** | "Why did this fail?" | Timeline of anomalies + probable cause |
| **Diff** | "What changed?" | Per-module delta between checkpoints |
| **Probe** | "What happens for this input?" | Token flow, attention patterns |
| **Head Surgeon** | "What do attention heads do?" | Head census + individual analysis |
| **Dataset Lens** | "Which examples are hard?" | Loss distribution + pattern analysis |
| **Emergence Tracker** | "When did capability X appear?" | Capability vs training step |
| **Similarity Search** | "What behaves like this?" | Activation space clustering |
| **Counterfactual** | "What if I changed X?" | Causal effect of tokens |
| **Replay** | "What was happening at step N?" | Time-travel through training |

Each of these is a "stain" that reveals different structures. Together, they let you truly see inside the model.
