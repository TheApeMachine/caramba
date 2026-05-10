package shape

// Transpose swaps two dimensions of an N-D tensor stored in row-major flat
// layout.
//
// Forward(shape, data[0]) -> transposed flat buffer.
// shape describes the full tensor; Dim0 and Dim1 are the axes to swap.
type Transpose struct {
	Dim0 int
	Dim1 int
}

// NewTranspose creates a Transpose operation swapping axes dim0 and dim1.
func NewTranspose(dim0, dim1 int) *Transpose {
	return &Transpose{Dim0: dim0, Dim1: dim1}
}

// Forward returns a new flat buffer with axes Dim0 and Dim1 swapped.
// shape must be the full N-D shape of data[0].
func (t *Transpose) Forward(shape []int, data ...[]float64) []float64 {
	src := data[0]
	rank := len(shape)
	// Total number of elements.
	n := 1
	for _, d := range shape {
		n *= d
	}

	dst := make([]float64, n)

	// Pre-compute strides for each dimension (row-major).
	strides := make([]int, rank)
	strides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	// Compute output shape and strides after swap.
	outShape := make([]int, rank)
	copy(outShape, shape)
	outShape[t.Dim0], outShape[t.Dim1] = outShape[t.Dim1], outShape[t.Dim0]

	outStrides := make([]int, rank)
	outStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		outStrides[i] = outStrides[i+1] * outShape[i+1]
	}

	// Iterate every element by its multi-dimensional index.
	coords := make([]int, rank)
	for i := 0; i < n; i++ {
		// Decode flat index i into coords using input strides.
		rem := i
		for d := 0; d < rank; d++ {
			coords[d] = rem / strides[d]
			rem %= strides[d]
		}

		// Swap the two dimensions in coords to find the output flat index.
		coords[t.Dim0], coords[t.Dim1] = coords[t.Dim1], coords[t.Dim0]
		outIdx := 0
		for d := 0; d < rank; d++ {
			outIdx += coords[d] * outStrides[d]
		}
		// Swap back so next iteration starts clean.
		coords[t.Dim0], coords[t.Dim1] = coords[t.Dim1], coords[t.Dim0]

		dst[outIdx] = src[i]
	}

	return dst
}
