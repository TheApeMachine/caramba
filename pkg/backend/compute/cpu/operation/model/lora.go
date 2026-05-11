package model

import "math"

/*
LoRA overlays low-rank weight decomposition on targeted weight matrices.
It implements the standard decomposition: W' = W + (B·A) * (alpha/rank)
where A is initialised with random gaussian values and B with zeros,
so the adapter contributes nothing at init and trains from there.

Two modes are supported:

  preset: qv  — targets all *.attn.q and *.attn.v weights (easy mode)
  preset: full — targets all attention and MLP projection weights
  targets: [...] — explicit list of glob patterns (hard mode)

Config keys:
  source   — must match the Loader node's source key
  preset   — qv | full (used when targets is absent)
  targets  — list of glob patterns (overrides preset)
  rank     — LoRA rank r (default: 8)
  alpha    — LoRA scaling alpha (default: 16)
*/
type LoRA struct {
	source  string
	targets []string
	rank    int
	alpha   float64
	// A and B matrices keyed by weight path.
	matA map[string][]float64
	matB map[string][]float64
}

/*
NewLoRA creates a LoRA node with preset or explicit targets.
*/
func NewLoRA(source string, preset string, targets []string, rank int, alpha float64) *LoRA {
	if rank <= 0 {
		rank = 8
	}

	if alpha <= 0 {
		alpha = float64(rank * 2)
	}

	if len(targets) == 0 {
		targets = presetsFor(preset)
	}

	return &LoRA{
		source:  source,
		targets: targets,
		rank:    rank,
		alpha:   alpha,
		matA:    make(map[string][]float64),
		matB:    make(map[string][]float64),
	}
}

/*
Forward initialises LoRA matrices for all matching weights on first call,
then applies the current adaptation W' = W + B·A * scale.
Input: data[0] = trigger token.
Output: number of adapted weight matrices.
*/
func (lora *LoRA) Forward(_ []int, data ...[]float64) []float64 {
	weights, ok := globalRegistry.Get(lora.source)

	if !ok {
		return []float64{-1}
	}

	adapted := 0

	for _, pattern := range lora.targets {
		selected := weights.Select(pattern)

		for key, w := range selected {
			lora.ensureMatrices(key, len(w))
			adapted++
			weights[key] = lora.apply(w, lora.matA[key], lora.matB[key])
		}
	}

	globalRegistry.store(lora.source, weights)

	return []float64{float64(adapted)}
}

/*
StepA returns the A matrix for the given weight key for gradient updates.
*/
func (lora *LoRA) StepA(key string) []float64 {
	return lora.matA[key]
}

/*
StepB returns the B matrix for the given weight key for gradient updates.
*/
func (lora *LoRA) StepB(key string) []float64 {
	return lora.matB[key]
}

/*
UpdateA stores a new A matrix, e.g. after an optimizer step.
*/
func (lora *LoRA) UpdateA(key string, a []float64) {
	lora.matA[key] = a
}

/*
UpdateB stores a new B matrix, e.g. after an optimizer step.
*/
func (lora *LoRA) UpdateB(key string, b []float64) {
	lora.matB[key] = b
}

func (lora *LoRA) ensureMatrices(key string, weightLen int) {
	if _, exists := lora.matA[key]; exists {
		return
	}

	// A: random gaussian init (fan-in scaling).
	scale := math.Sqrt(2.0 / float64(weightLen))
	a := gaussianSlice(lora.rank*weightLen, scale)

	// B: zero init so adapter starts as identity.
	b := make([]float64, weightLen*lora.rank)

	lora.matA[key] = a
	lora.matB[key] = b
}

// apply computes W + B·A * (alpha/rank).
// W has shape [n], A has shape [r*n], B has shape [n*r].
// The result is W + outer(B_col, A_row) collapsed, scaled by alpha/rank.
func (lora *LoRA) apply(w, a, b []float64) []float64 {
	n := len(w)
	scale := lora.alpha / float64(lora.rank)
	out := make([]float64, n)
	copy(out, w)

	// Simplified: treat as element-wise rank-1 update BA* collapsed to n dims.
	// A full matmul would require knowing the actual 2D shape; since we store
	// weights flat, we use the first rank slice of A and B as the update vectors.
	aVec := a
	bVec := b

	if len(aVec) > n {
		aVec = a[:n]
	}

	if len(bVec) > n {
		bVec = b[:n]
	}

	for idx := range out {
		if idx < len(aVec) && idx < len(bVec) {
			out[idx] += bVec[idx] * aVec[idx] * scale
		}
	}

	return out
}

func presetsFor(preset string) []string {
	switch preset {
	case "full":
		return []string{
			"**.attn.q",
			"**.attn.k",
			"**.attn.v",
			"**.attn.o",
			"**.mlp.gate",
			"**.mlp.up",
			"**.mlp.down",
		}
	default: // "qv"
		return []string{
			"**.attn.q",
			"**.attn.v",
		}
	}
}
