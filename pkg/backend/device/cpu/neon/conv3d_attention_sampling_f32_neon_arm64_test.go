//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"
)

func TestConv3DFloat32NEONParitySizes(t *testing.T) {
	config := DefaultConv3DConfig()
	cases := []struct {
		inD, inH, inW, kD, kH, kW int
	}{
		{1, 7, 7, 1, 3, 3},
		{4, 8, 8, 3, 3, 3},
		{7, 16, 16, 3, 5, 5},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("d=%d_h=%d_k=%dx%dx%d", testCase.inD, testCase.inH, testCase.kD, testCase.kH, testCase.kW)
		t.Run(label, func(t *testing.T) {
			batch, inC := 1, 2
			outC := 2
			outD := testCase.inD - testCase.kD + 1
			outH := testCase.inH - testCase.kH + 1
			outW := testCase.inW - testCase.kW + 1
			input := randFloat32Slice(batch*inC*testCase.inD*testCase.inH*testCase.inW, 0x3E0)
			weight := randFloat32Slice(outC*inC*testCase.kD*testCase.kH*testCase.kW, 0x3E1)
			bias := randFloat32Slice(outC, 0x3E2)
			got := make([]float32, batch*outC*outD*outH*outW)
			want := make([]float32, len(got))

			Conv3DFloat32Native(
				config, input, weight, bias, got,
				batch, inC, testCase.inD, testCase.inH, testCase.inW,
				outC, testCase.kD, testCase.kH, testCase.kW, outD, outH, outW,
			)
			conv3DFloat32Scalar(
				config, input, weight, bias, want,
				batch, inC, testCase.inD, testCase.inH, testCase.inW,
				outC, testCase.kD, testCase.kH, testCase.kW, outD, outH, outW,
			)

			assertFloat32SlicesNear(t, got, want, 1e-4)
		})
	}
}

func TestMultiHeadAttentionNativeParity(t *testing.T) {
	for _, seqLen := range []int{1, 7, 64} {
		t.Run(fmt.Sprintf("seq=%d", seqLen), func(t *testing.T) {
			config := DefaultMultiHeadAttentionConfig()
			config.NumHeads = 4
			config.HeadDim = 16
			depth := config.NumHeads * config.HeadDim
			query := randFloat32Slice(seqLen*depth, 0xA10)
			key := randFloat32Slice(seqLen*depth, 0xA11)
			value := randFloat32Slice(seqLen*depth, 0xA12)
			got := make([]float32, seqLen*depth)
			want := make([]float32, seqLen*depth)

			multiHeadAttentionSlices(config, query, key, value, got, seqLen, seqLen, config.NumHeads)
			multiHeadAttentionSlicesScalar(config, query, key, value, want, seqLen, seqLen, config.NumHeads)

			assertFloat32SlicesNear(t, got, want, 1e-4)
		})
	}
}

func multiHeadAttentionSlicesScalar(
	config MultiHeadAttentionConfig,
	queryView, keyView, valueView, outView []float32,
	seqQ, seqK, kvHeads int,
) {
	scale := float32(1.0 / math.Sqrt(float64(config.HeadDim)))
	headsPerKVHead := config.NumHeads / kvHeads

	for headIndex := 0; headIndex < config.NumHeads; headIndex++ {
		kvHeadIndex := headIndex / headsPerKVHead
		queryHeadOffset := headIndex * config.HeadDim
		kvHeadOffset := kvHeadIndex * config.HeadDim
		queryStride := config.NumHeads * config.HeadDim
		kvStride := kvHeads * config.HeadDim
		scores := make([]float32, seqK)

		for qIndex := range seqQ {
			scalarComputeHeadScores(
				queryView, keyView,
				qIndex, seqK, config.HeadDim,
				queryHeadOffset, kvHeadOffset,
				queryStride, kvStride,
				scale, scores,
				config,
			)
			scalarStableSoftmaxRow(scores)
			scalarWriteHeadOutput(
				scores, valueView, outView,
				qIndex, seqK, config.HeadDim,
				queryHeadOffset, kvHeadOffset,
				queryStride, kvStride,
			)
		}
	}
}

func scalarComputeHeadScores(
	queryView, keyView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
	scale float32,
	scores []float32,
	config MultiHeadAttentionConfig,
) {
	for kIndex := range seqK {
		var dot float32

		for dimIndex := range headDim {
			dot += queryView[qIndex*queryStride+queryHeadOffset+dimIndex] *
				keyView[kIndex*kvStride+kvHeadOffset+dimIndex]
		}

		score := dot * scale

		if config.Causal && kIndex > qIndex {
			score = float32(math.Inf(-1))
		}

		if config.WindowSize > 0 && qIndex-kIndex >= config.WindowSize {
			score = float32(math.Inf(-1))
		}

		if config.ALiBiSlope != 0 {
			score += config.ALiBiSlope * float32(kIndex-qIndex)
		}

		scores[kIndex] = score
	}
}

func scalarStableSoftmaxRow(scores []float32) {
	maximum := scores[0]

	for _, value := range scores[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var sum float32

	for index, value := range scores {
		shifted := float32(math.Exp(float64(value - maximum)))
		scores[index] = shifted
		sum += shifted
	}

	if sum == 0 {
		return
	}

	for index := range scores {
		scores[index] /= sum
	}
}

func scalarWriteHeadOutput(
	scores, valueView, outView []float32,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
) {
	for dimIndex := range headDim {
		var sum float32

		for kIndex := range seqK {
			sum += scores[kIndex] *
				valueView[kIndex*kvStride+kvHeadOffset+dimIndex]
		}

		outView[qIndex*queryStride+queryHeadOffset+dimIndex] = sum
	}
}

func TestGreedySampleNativeParity(t *testing.T) {
	for _, size := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			logits := randFloat32Slice(size, 0x501)
			got := GreedySampleFloat32Native(logits)
			want := scalarGreedySample(logits)
			assertInt32Equal(t, got, want)
		})
	}
}

func TestSamplingSoftmaxRowNativeParity(t *testing.T) {
	for _, size := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			logits := randFloat32Slice(size, 0x502)
			got := make([]float32, size)
			want := make([]float32, size)

			SamplingSoftmaxRowNative(logits, got, 1.25)
			scalarSamplingSoftmaxRow(logits, want, 1.25)

			assertFloat32SlicesNear(t, got, want, 1e-5)
		})
	}
}

func TestTopKSampleNativeParity(t *testing.T) {
	const seed = uint64(0x503)

	for _, size := range []int{7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			logits := randFloat32Slice(size, 0x504)
			got := TopKSampleFloat32Native(logits, 1.25, 4, seed)
			want := scalarTopKSample(logits, 1.25, 4, seed)
			assertInt32Equal(t, got, want)
		})
	}
}

func TestTopPSampleNativeParity(t *testing.T) {
	const seed = uint64(0x505)

	for _, size := range []int{7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			logits := randFloat32Slice(size, 0x506)
			got := TopPSampleFloat32Native(logits, 1.25, 0.9, seed)
			want := scalarTopPSample(logits, 1.25, 0.9, seed)
			assertInt32Equal(t, got, want)
		})
	}
}

func scalarTopKSample(
	logits []float32,
	temperature float32,
	topK int,
	seed uint64,
) int32 {
	elementCount := len(logits)

	if elementCount == 0 {
		return 0
	}

	k := topK

	if k <= 0 || k > elementCount {
		k = elementCount
	}

	probabilities := make([]float32, elementCount)
	indices := make([]int, elementCount)

	scalarSamplingSoftmaxRow(logits, probabilities, temperature)

	for index := range indices {
		indices[index] = index
	}

	sortIndicesByProbability(probabilities, indices)

	cumulative := float32(0)

	for index := 0; index < k; index++ {
		cumulative += probabilities[indices[index]]
	}

	if cumulative == 0 {
		return int32(indices[0])
	}

	scale := float32(1.0 / cumulative)

	for index := 0; index < k; index++ {
		probabilities[indices[index]] *= scale
	}

	for index := k; index < elementCount; index++ {
		probabilities[indices[index]] = 0
	}

	rng := newSamplingRNG(seed)

	return int32(indices[drawFrom(probabilities, rng)])
}

func scalarTopPSample(
	logits []float32,
	temperature float32,
	topP float32,
	seed uint64,
) int32 {
	elementCount := len(logits)

	if elementCount == 0 {
		return 0
	}

	probabilities := make([]float32, elementCount)
	indices := make([]int, elementCount)

	scalarSamplingSoftmaxRow(logits, probabilities, temperature)

	for index := range indices {
		indices[index] = index
	}

	sortIndicesByProbability(probabilities, indices)

	cumulative := float32(0)
	cutoff := elementCount

	for index := 0; index < elementCount; index++ {
		cumulative += probabilities[indices[index]]

		if cumulative >= topP {
			cutoff = index + 1
			break
		}
	}

	cumulative = 0

	for index := 0; index < cutoff; index++ {
		cumulative += probabilities[indices[index]]
	}

	if cumulative == 0 {
		return int32(indices[0])
	}

	scale := float32(1.0 / cumulative)

	for index := 0; index < cutoff; index++ {
		probabilities[indices[index]] *= scale
	}

	for index := cutoff; index < elementCount; index++ {
		probabilities[indices[index]] = 0
	}

	rng := newSamplingRNG(seed)

	return int32(indices[drawFrom(probabilities, rng)])
}

func sortIndicesByProbability(probabilities []float32, indices []int) {
	for left := 1; left < len(indices); left++ {
		current := indices[left]
		currentProb := probabilities[current]
		scanIndex := left - 1

		for scanIndex >= 0 && probabilities[indices[scanIndex]] < currentProb {
			indices[scanIndex+1] = indices[scanIndex]
			scanIndex--
		}

		indices[scanIndex+1] = current
	}
}

func scalarGreedySample(logits []float32) int32 {
	maxIndex := 0
	maxLogit := logits[0]

	for index, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
			maxIndex = index + 1
		}
	}

	return int32(maxIndex)
}

func scalarSamplingSoftmaxRow(logits, probabilities []float32, temperature float32) {
	if temperature == 0 {
		temperature = 1
	}

	for index, value := range logits {
		probabilities[index] = value / temperature
	}

	maximum := probabilities[0]

	for _, value := range probabilities[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var denominator float64

	for index, value := range probabilities {
		shifted := math.Exp(float64(value - maximum))
		probabilities[index] = float32(shifted)
		denominator += shifted
	}

	if denominator == 0 {
		return
	}

	scale := float32(1.0 / denominator)

	for index := range probabilities {
		probabilities[index] *= scale
	}
}

func assertInt32Equal(t *testing.T, got, want int32) {
	t.Helper()

	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}

func BenchmarkConv3DFloat32Native(b *testing.B) {
	config := DefaultConv3DConfig()
	batch, inC, inD, inH, inW := 1, 4, 8, 16, 16
	outC, kD, kH, kW := 4, 3, 3, 3
	outD, outH, outW := inD-kD+1, inH-kH+1, inW-kW+1
	input := randFloat32Slice(batch*inC*inD*inH*inW, 0xB3D)
	weight := randFloat32Slice(outC*inC*kD*kH*kW, 0xB3E)
	bias := randFloat32Slice(outC, 0xB3F)
	out := make([]float32, batch*outC*outD*outH*outW)

	for b.Loop() {
		Conv3DFloat32Native(
			config, input, weight, bias, out,
			batch, inC, inD, inH, inW,
			outC, kD, kH, kW, outD, outH, outW,
		)
	}
}

func BenchmarkGreedySampleNative(b *testing.B) {
	logits := randFloat32Slice(8192, 0xB40)

	for b.Loop() {
		_ = GreedySampleFloat32Native(logits)
	}
}

func BenchmarkTopKSampleNative(b *testing.B) {
	logits := randFloat32Slice(8192, 0xB41)

	for b.Loop() {
		_ = TopKSampleFloat32Native(logits, 1.25, 32, 0xB42)
	}
}

func BenchmarkTopPSampleNative(b *testing.B) {
	logits := randFloat32Slice(8192, 0xB43)

	for b.Loop() {
		_ = TopPSampleFloat32Native(logits, 1.25, 0.9, 0xB44)
	}
}
