package shape

// Split divides a tensor into equal-sized chunks of SplitSize along Dim and
// returns all chunks concatenated into one flat buffer.
// The caller can recover individual chunks because each has the same size:
// (product-of-shape / shape[Dim]) * SplitSize elements.
//
// Forward(shape, data[0]) -> flat buffer containing all chunks in order.
type Split struct {
	SplitSize int
	Dim       int
}

// NewSplit creates a Split operation.
func NewSplit(splitSize, dim int) *Split {
	return &Split{SplitSize: splitSize, Dim: dim}
}

// Forward splits data[0] into chunks of SplitSize along Dim and returns them
// concatenated.  shape is the full tensor shape.
func (s *Split) Forward(shape []int, data ...[]float64) []float64 {
	src := data[0]
	rank := len(shape)
	dim := s.Dim

	outer := 1
	for d := 0; d < dim; d++ {
		outer *= shape[d]
	}
	inner := 1
	for d := dim + 1; d < rank; d++ {
		inner *= shape[d]
	}

	dimSize := shape[dim]
	numChunks := dimSize / s.SplitSize

	// Output is the same data rearranged — total size identical.
	total := len(src)
	dst := make([]float64, total)

	// For each chunk k, collect the corresponding SplitSize rows from each
	// outer slice and place them consecutively.
	chunkElems := outer * s.SplitSize * inner
	for k := 0; k < numChunks; k++ {
		dstOff := k * chunkElems
		for o := 0; o < outer; o++ {
			srcOff := (o*dimSize + k*s.SplitSize) * inner
			n := s.SplitSize * inner
			copy(dst[dstOff:dstOff+n], src[srcOff:srcOff+n])
			dstOff += n
		}
	}

	return dst
}
