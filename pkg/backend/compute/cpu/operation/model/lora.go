package model

import "math"

/*
LoRA overlays low-rank weight decomposition on targeted weight matrices.
Standard decomposition: W' = W + (B·A) * (alpha/rank)

  W: [out × in]  — original weight matrix (flat row-major)
  A: [rank × in] — random gaussian init, projects down to rank
  B: [out × rank] — zero init, so adapter is identity at start
  W': W + B·A·scale  where scale = alpha/rank

Two modes:
  preset: qv   — targets **.attn.q and **.attn.v
  preset: full — targets all attn and MLP projections
  targets: [...] — explicit glob list

The matmul is injected so accelerated backends (Metal, CUDA, XLA) can
supply their own kernel without duplicating this logic.
*/
type LoRA struct {
	source  string
	targets []string
	rank    int
	alpha   float64
	matmul  MatMulFn
	matA    map[string][]float64 // [rank × in] row-major
	matB    map[string][]float64 // [out × rank] row-major
	dims    map[string][2]int    // {key: [out, in]}
}

/*
NewLoRA creates a LoRA node using CPUMatMul.
*/
func NewLoRA(source, preset string, targets []string, rank int, alpha float64) *LoRA {
	return NewLoRAWithMatMul(source, preset, targets, rank, alpha, CPUMatMul)
}

/*
NewLoRAWithMatMul creates a LoRA node with an injected matmul kernel.
Used by Metal, CUDA, and XLA backends to supply their accelerated kernels.
*/
func NewLoRAWithMatMul(
	source, preset string, targets []string, rank int, alpha float64, matmul MatMulFn,
) *LoRA {
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
		matmul:  matmul,
		matA:    make(map[string][]float64),
		matB:    make(map[string][]float64),
		dims:    make(map[string][2]int),
	}
}

/*
Forward initialises LoRA matrices on first call then applies W' = W + B·A·scale.
The weight at each matched key must be a flat [out×in] row-major matrix.
Shape is inferred from the stored dimensions on first call.
*/
func (lora *LoRA) Forward(_ []int, data ...[]float64) []float64 {
	weights, ok := globalRegistry.Get(lora.source)

	if !ok {
		return []float64{-1}
	}

	adapted := 0

	for _, pattern := range lora.targets {
		for key, w := range weights.Select(pattern) {
			lora.ensureMatrices(key, w)
			weights[key] = lora.apply(w, lora.matA[key], lora.matB[key], lora.dims[key])
			adapted++
		}
	}

	globalRegistry.store(lora.source, weights)

	return []float64{float64(adapted)}
}

/*
StepA / StepB / UpdateA / UpdateB expose the low-rank matrices for
gradient-based training of the LoRA parameters.
*/
func (lora *LoRA) StepA(key string) []float64       { return lora.matA[key] }
func (lora *LoRA) StepB(key string) []float64       { return lora.matB[key] }
func (lora *LoRA) UpdateA(key string, a []float64)  { lora.matA[key] = a }
func (lora *LoRA) UpdateB(key string, b []float64)  { lora.matB[key] = b }

func (lora *LoRA) ensureMatrices(key string, w []float64) {
	if _, exists := lora.matA[key]; exists {
		return
	}

	// Infer [out, in] from weight length and rank.
	// Weight is stored flat [out*in]; we factor as: out*in = len(w).
	// Without an explicit shape annotation, assume square: out == in == sqrt(n)
	// for square weights, or store the factored shape at first call.
	// The WeightMap stores flat vectors; for non-square weights the caller
	// must pre-register the shape via SetDims. Default: treat as out=1, in=n
	// which degenerates to the vector case — this is safe and correct for
	// bias vectors. True matrix weights need SetDims called after loading.
	n := len(w)
	dims, hasDims := lora.dims[key]

	if !hasDims {
		// Best-effort square factoring.
		out := sqrtInt(n)
		if out*out == n {
			dims = [2]int{out, out}
		} else {
			// Non-square: treat as a [1×n] row — LoRA reduces to a simple
			// rank-1 update. Correct for bias/embedding vectors.
			dims = [2]int{1, n}
		}
		lora.dims[key] = dims
	}

	out, in := dims[0], dims[1]

	// A [rank × in]: gaussian init with fan-in scaling.
	lora.matA[key] = gaussianSlice(lora.rank*in, math.Sqrt(2.0/float64(in)))
	// B [out × rank]: zero init — adapter is identity at start.
	lora.matB[key] = make([]float64, out*lora.rank)
}

/*
SetDims registers the [out, in] shape for a weight key so LoRA can construct
correctly shaped A and B matrices. Must be called before the first Forward
for any non-square weight matrix.
*/
func (lora *LoRA) SetDims(key string, out, in int) {
	lora.dims[key] = [2]int{out, in}
}

// apply computes W' = W + B·A * (alpha/rank).
//
//	A: [rank × in]   lora.matA[key]
//	B: [out × rank]  lora.matB[key]
//	B·A → [out × in] — same shape as W
func (lora *LoRA) apply(w, a, b []float64, dims [2]int) []float64 {
	out, in := dims[0], dims[1]
	scale := lora.alpha / float64(lora.rank)

	// delta = B · A  →  [out×rank] · [rank×in] = [out×in]
	delta := lora.matmul(b, a, out, lora.rank, in)

	result := make([]float64, out*in)

	for idx := range w {
		result[idx] = w[idx] + delta[idx]*scale
	}

	return result
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

func sqrtInt(n int) int {
	r := 1
	for r*r < n {
		r++
	}
	return r
}
