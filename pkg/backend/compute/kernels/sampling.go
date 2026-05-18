package kernels

import (
	"math"
	"math/rand/v2"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Sampling kernels for LLM decoding: greedy, multinomial, top-k,
top-p (nucleus). Each takes a [vocab] logits row and produces a
single int32 token index in output[0].
*/

type SamplingConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        uint64
}

func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{Temperature: 1.0, TopK: 0, TopP: 1.0, Seed: 0xfeedface}
}

func init() {
	Default.Register(Kernel{
		Name: "greedy_sample",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGreedySample,
	})

	Default.Register(Kernel{
		Name: "topk_sample",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runTopKSampleDefault,
	})

	Default.Register(Kernel{
		Name: "topp_sample",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runTopPSampleDefault,
	})
}

func runGreedySample(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	logits, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	if len(logits) == 0 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	maxIndex := 0
	maxLogit := logits[0]

	for index, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
			maxIndex = index + 1
		}
	}

	out[0] = int32(maxIndex)
	return nil
}

func runTopKSampleDefault(args ...tensor.Tensor) error {
	return TopKSample(DefaultSamplingConfig(), args[0], args[1])
}

func runTopPSampleDefault(args ...tensor.Tensor) error {
	return TopPSample(DefaultSamplingConfig(), args[0], args[1])
}

/*
TopKSample restricts sampling to the K highest-scoring logits and
draws from the resulting truncated distribution.
*/
func TopKSample(config SamplingConfig, logits, output tensor.Tensor) error {
	logitView, err := logits.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Int32Native()

	if err != nil {
		return err
	}

	if len(outView) < 1 {
		return tensor.ErrShapeMismatch
	}

	k := config.TopK

	if k <= 0 || k > len(logitView) {
		k = len(logitView)
	}

	probabilities, indices := softmaxAndSort(logitView, config.Temperature)

	cumulative := float32(0)

	for index := 0; index < k; index++ {
		cumulative += probabilities[index]
	}

	for index := 0; index < k; index++ {
		probabilities[index] /= cumulative
	}

	for index := k; index < len(probabilities); index++ {
		probabilities[index] = 0
	}

	rng := newSamplingRNG(config.Seed)
	choice := drawFrom(probabilities, rng)
	outView[0] = int32(indices[choice])

	return nil
}

/*
TopPSample restricts to the smallest prefix whose cumulative
probability mass meets or exceeds config.TopP, then draws from the
truncated distribution.
*/
func TopPSample(config SamplingConfig, logits, output tensor.Tensor) error {
	logitView, err := logits.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Int32Native()

	if err != nil {
		return err
	}

	if len(outView) < 1 {
		return tensor.ErrShapeMismatch
	}

	probabilities, indices := softmaxAndSort(logitView, config.Temperature)

	cumulative := float32(0)
	cutoff := len(probabilities)

	for index := 0; index < len(probabilities); index++ {
		cumulative += probabilities[index]

		if cumulative >= config.TopP {
			cutoff = index + 1
			break
		}
	}

	cumulative = 0

	for index := 0; index < cutoff; index++ {
		cumulative += probabilities[index]
	}

	for index := 0; index < cutoff; index++ {
		probabilities[index] /= cumulative
	}

	for index := cutoff; index < len(probabilities); index++ {
		probabilities[index] = 0
	}

	rng := newSamplingRNG(config.Seed)
	choice := drawFrom(probabilities, rng)
	outView[0] = int32(indices[choice])

	return nil
}

func softmaxAndSort(logits []float32, temperature float32) ([]float32, []int) {
	if temperature == 0 {
		temperature = 1
	}

	probabilities := make([]float32, len(logits))
	indices := make([]int, len(logits))

	maximum := logits[0]

	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var denominator float64

	for index, value := range logits {
		shifted := math.Exp(float64((value - maximum) / temperature))
		probabilities[index] = float32(shifted)
		indices[index] = index
		denominator += shifted
	}

	scale := float32(1.0 / denominator)

	for index := range probabilities {
		probabilities[index] *= scale
	}

	sort.SliceStable(indices, func(left, right int) bool {
		return probabilities[indices[left]] > probabilities[indices[right]]
	})

	sorted := make([]float32, len(probabilities))

	for resultIndex, originalIndex := range indices {
		sorted[resultIndex] = probabilities[originalIndex]
	}

	return sorted, indices
}

func newSamplingRNG(seed uint64) *rand.Rand {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source)
}

func drawFrom(probabilities []float32, rng *rand.Rand) int {
	target := rng.Float32()
	cumulative := float32(0)

	for index, prob := range probabilities {
		cumulative += prob

		if cumulative >= target {
			return index
		}
	}

	return len(probabilities) - 1
}
