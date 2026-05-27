package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/execution"
)

/*
LookupSlice returns a resident tensor containing one range from a packed
checkpoint tensor.
*/
func (store *Store) LookupSlice(
	name, axis string,
	start, end int64,
) (tensor.Tensor, error) {
	key := weightSliceKey{name: name, axis: axis, start: start, end: end}

	store.mu.Lock()

	if cached, ok := store.sliceCache[key]; ok {
		store.mu.Unlock()
		return cached, nil
	}

	store.mu.Unlock()

	shape, values, err := store.loadSlicedFloat32Values(name, axis, start, end)

	if err != nil {
		return nil, err
	}

	resident, err := store.memory.Upload(shape, dtype.Float32, float32ValuesToBytes(values))

	if err != nil {
		return nil, fmt.Errorf("weights store: upload sliced %q: %w", name, err)
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	if cached, ok := store.sliceCache[key]; ok {
		return cached, nil
	}

	store.sliceCache[key] = resident

	return resident, nil
}

/*
LookupTransposedSlice returns the row-major transpose of one packed tensor
range.
*/
func (store *Store) LookupTransposedSlice(
	name, axis string,
	start, end int64,
) (tensor.Tensor, error) {
	key := weightSliceKey{name: name, axis: axis, start: start, end: end}

	store.mu.Lock()

	if cached, ok := store.transposedSliceCache[key]; ok {
		store.mu.Unlock()
		return cached, nil
	}

	store.mu.Unlock()

	shape, source, err := store.loadSlicedFloat32Values(name, axis, start, end)

	if err != nil {
		return nil, err
	}

	transposedShape, transposedBytes, err := transposeFloat32Tensor(name, shape, source)

	if err != nil {
		return nil, err
	}

	transposed, err := store.memory.Upload(transposedShape, dtype.Float32, transposedBytes)

	if err != nil {
		return nil, fmt.Errorf("weights store: upload transposed slice %q: %w", name, err)
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	if cached, ok := store.transposedSliceCache[key]; ok {
		return cached, nil
	}

	store.transposedSliceCache[key] = transposed

	return transposed, nil
}

func (store *Store) loadSlicedFloat32Values(
	name, axis string,
	start, end int64,
) (tensor.Shape, []float32, error) {
	entry, ok := store.entries[name]

	if !ok {
		return tensor.Shape{}, nil, execution.ErrWeightNotFound
	}

	shape, values, err := store.loadFloat32Values(name, entry)

	if err != nil {
		return tensor.Shape{}, nil, err
	}

	return sliceFloat32Values(name, shape, values, axis, start, end)
}

func sliceFloat32Values(
	name string,
	shape tensor.Shape,
	values []float32,
	axis string,
	start, end int64,
) (tensor.Shape, []float32, error) {
	dimensions := shape.Dims()
	axisIndex, err := weightAxisIndex(axis, len(dimensions))

	if err != nil {
		return tensor.Shape{}, nil, fmt.Errorf("weights store: slice %q: %w", name, err)
	}

	if start < 0 || end <= start || end > int64(dimensions[axisIndex]) {
		return tensor.Shape{}, nil, fmt.Errorf(
			"weights store: slice %q range [%d:%d) out of bounds for axis %d size %d",
			name, start, end, axisIndex, dimensions[axisIndex],
		)
	}

	slicedDimensions := append([]int(nil), dimensions...)
	slicedDimensions[axisIndex] = int(end - start)
	slicedShape, err := tensor.NewShape(slicedDimensions)

	if err != nil {
		return tensor.Shape{}, nil, fmt.Errorf("weights store: sliced shape for %q: %w", name, err)
	}

	return slicedShape, copySlice(values, dimensions, axisIndex, int(start), int(end)), nil
}

func copySlice(values []float32, dimensions []int, axisIndex, start, end int) []float32 {
	inner := productDimensions(dimensions[axisIndex+1:])
	outer := productDimensions(dimensions[:axisIndex])
	axisSize := dimensions[axisIndex]
	axisLength := end - start
	output := make([]float32, outer*axisLength*inner)

	for outerIndex := range outer {
		sourceBase := (outerIndex*axisSize + start) * inner
		outputBase := outerIndex * axisLength * inner
		copy(output[outputBase:outputBase+axisLength*inner], values[sourceBase:sourceBase+axisLength*inner])
	}

	return output
}

func transposeFloat32Tensor(
	name string,
	shape tensor.Shape,
	source []float32,
) (tensor.Shape, []byte, error) {
	dimensions := shape.Dims()

	if len(dimensions) != 2 {
		return tensor.Shape{}, nil, fmt.Errorf(
			"weights store: %q has rank %d, transpose requires rank 2",
			name, len(dimensions),
		)
	}

	rows, cols := dimensions[0], dimensions[1]
	transposedBytes := make([]byte, rows*cols*4)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowBase := rowIndex * cols

		for colIndex := 0; colIndex < cols; colIndex++ {
			outOffset := (colIndex*rows + rowIndex) * 4

			binary.LittleEndian.PutUint32(
				transposedBytes[outOffset:],
				math.Float32bits(source[rowBase+colIndex]),
			)
		}
	}

	transposedShape, err := tensor.NewShape([]int{cols, rows})

	if err != nil {
		return tensor.Shape{}, nil, fmt.Errorf("weights store: shape for transposed %q: %w", name, err)
	}

	return transposedShape, transposedBytes, nil
}

func weightAxisIndex(axis string, rank int) (int, error) {
	switch axis {
	case "output":
		return 0, nil
	case "input":
		if rank < 2 {
			return 0, fmt.Errorf("input axis requires rank >= 2, got %d", rank)
		}

		return 1, nil
	default:
		return 0, fmt.Errorf("unsupported axis %q", axis)
	}
}

func productDimensions(dimensions []int) int {
	product := 1

	for _, dimension := range dimensions {
		product *= dimension
	}

	return product
}
