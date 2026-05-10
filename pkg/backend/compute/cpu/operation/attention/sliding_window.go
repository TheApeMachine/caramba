package attention

import "math"

// SlidingWindow implements sliding-window attention (local attention).
// Each position i can only attend to positions in [i-Window, i].
//
// shape: [batch, num_heads, seq_len, head_dim]
// data[0]=Q, data[1]=K, data[2]=V  each [batch*num_heads*seq_len*head_dim]
// output: [batch*num_heads*seq_len*head_dim]
type SlidingWindow struct {
	Window int
}

func NewSlidingWindow(window int) *SlidingWindow {
	return &SlidingWindow{Window: window}
}

func (sw *SlidingWindow) Forward(shape []int, data ...[]float64) []float64 {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	Q, K, V := data[0], data[1], data[2]
	headStride := seqLen * headDim
	batchStride := numHeads * headStride
	total := batch * numHeads * headStride
	out := make([]float64, total)

	window := sw.Window
	maskFn := func(i, j int) bool {
		// mask out positions outside [i-window, i]
		return j > i || j < i-window
	}

	for b := range batch {
		for h := range numHeads {
			off := b*batchStride + h*headStride
			sdpaHeadWithInf(
				out[off:off+headStride],
				Q[off:off+headStride],
				K[off:off+headStride],
				V[off:off+headStride],
				seqLen, headDim, maskFn,
			)
		}
	}
	return out
}

// sdpaHeadWithInf is like sdpaHead but supports -inf masking.
func sdpaHeadWithInf(out, q, k, v []float64, seqLen, headDim int, maskFn func(i, j int) bool) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, seqLen)

	for i := range seqLen {
		qRow := q[i*headDim : (i+1)*headDim]

		for j := range seqLen {
			if maskFn(i, j) {
				scores[j] = math.Inf(-1)
			} else {
				kRow := k[j*headDim : (j+1)*headDim]
				scores[j] = dotProduct(qRow, kRow) * scale
			}
		}

		// softmax with -inf handling: find finite max
		max := math.Inf(-1)

		for _, s := range scores {
			if s > max {
				max = s
			}
		}

		var sum float64

		for j, s := range scores {
			if math.IsInf(s, -1) {
				scores[j] = 0
			} else {
				e := math.Exp(s - max)
				scores[j] = e
				sum += e
			}
		}

		if sum > 0 {
			for j := range scores {
				scores[j] /= sum
			}
		}

		outRow := out[i*headDim : (i+1)*headDim]

		for d := range outRow {
			outRow[d] = 0
		}

		for j := range seqLen {
			vRow := v[j*headDim : (j+1)*headDim]
			scaledAdd(outRow, vRow, scores[j])
		}
	}
}
