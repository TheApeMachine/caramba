package shape

func transposeGenericKernel(dst, src []float64, shape []int, dim0, dim1 int) {
	rank := len(shape)
	elementCount := 1

	for _, dimension := range shape {
		elementCount *= dimension
	}

	strides := make([]int, rank)
	strides[rank-1] = 1

	for index := rank - 2; index >= 0; index-- {
		strides[index] = strides[index+1] * shape[index+1]
	}

	outShape := make([]int, rank)
	copy(outShape, shape)
	outShape[dim0], outShape[dim1] = outShape[dim1], outShape[dim0]

	outStrides := make([]int, rank)
	outStrides[rank-1] = 1

	for index := rank - 2; index >= 0; index-- {
		outStrides[index] = outStrides[index+1] * outShape[index+1]
	}

	coords := make([]int, rank)

	for index := 0; index < elementCount; index++ {
		remainder := index

		for dimension := 0; dimension < rank; dimension++ {
			coords[dimension] = remainder / strides[dimension]
			remainder %= strides[dimension]
		}

		coords[dim0], coords[dim1] = coords[dim1], coords[dim0]
		outputIndex := 0

		for dimension := 0; dimension < rank; dimension++ {
			outputIndex += coords[dimension] * outStrides[dimension]
		}

		coords[dim0], coords[dim1] = coords[dim1], coords[dim0]
		dst[outputIndex] = src[index]
	}
}
