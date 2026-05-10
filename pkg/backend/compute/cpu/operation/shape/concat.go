package shape

// Concat concatenates multiple tensors along a single axis.
//
// Forward(shape, data[0], data[1], ...) -> concatenated flat buffer.
// shape is the shape of each input tensor — all inputs must have the same
// shape except along Dim, which may differ.  For simplicity the current
// implementation treats all inputs as having the provided shape.
type Concat struct {
	Dim int
}

// NewConcat creates a Concat operation along the given axis.
func NewConcat(dim int) *Concat {
	return &Concat{Dim: dim}
}

// Forward concatenates all inputs along Dim.
// shape is the shape of each individual input tensor.
// All inputs must have the same shape except along Dim.
func (c *Concat) Forward(shape []int, data ...[]float64) []float64 {
	if len(data) == 0 {
		return []float64{}
	}
	if len(data) == 1 {
		out := make([]float64, len(data[0]))
		copy(out, data[0])
		return out
	}

	rank := len(shape)
	dim := c.Dim

	// Compute the total number of elements per "outer" slice above dim,
	// the size of the concat dim, and the "inner" stride below dim.
	outer := 1
	for d := 0; d < dim; d++ {
		outer *= shape[d]
	}
	inner := 1
	for d := dim + 1; d < rank; d++ {
		inner *= shape[d]
	}

	dimSize := shape[dim] // dim size of each individual input
	totalDim := dimSize * len(data)

	// Total output size.
	total := outer * totalDim * inner
	dst := make([]float64, total)

	// For each outer index, write each input's slice, then advance.
	dstOff := 0
	for o := 0; o < outer; o++ {
		for _, src := range data {
			// Copy dimSize * inner elements from src for this outer slice.
			srcOff := o * dimSize * inner
			block := src[srcOff : srcOff+dimSize*inner]
			copyBlock(dst[dstOff:dstOff+len(block)], block)
			dstOff += len(block)
		}
	}

	return dst
}
