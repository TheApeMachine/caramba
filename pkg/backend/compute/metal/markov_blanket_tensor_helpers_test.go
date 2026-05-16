//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func markovShape(test testing.TB, dimensions ...int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape(dimensions)
	if err != nil {
		test.Fatalf("markov blanket shape: %v", err)
	}

	return shape
}

func markovVector(elementCount int, scale float64, offset float64) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(float32(offset + scale*float64(index%23-11)))
	}

	return values
}

func markovWeights(rows int, columns int) []float64 {
	values := make([]float64, rows*columns)

	for index := range values {
		values[index] = float64(float32(0.006 * float64(index%29-14)))
	}

	return values
}

func markovPartitionInputs(elementCount int) ([]float64, [][]float64, []float64, []int) {
	state := markovVector(elementCount, 0.015, -0.25)
	masks := [][]float64{
		make([]float64, elementCount),
		make([]float64, elementCount),
		make([]float64, elementCount),
		make([]float64, elementCount),
	}
	counts := make([]int, 4)

	for index := range elementCount {
		partitionIndex := index % 4
		masks[partitionIndex][index] = 1
		counts[partitionIndex]++
	}

	return state, masks, markovPackMasks(masks), counts
}

func markovPackMasks(masks [][]float64) []float64 {
	length := len(masks[0])
	values := make([]float64, 0, 4*length)

	for _, mask := range masks {
		values = append(values, mask...)
	}

	return values
}

func referenceMarkovPartition(state []float64, masks [][]float64, counts []int) []float64 {
	values := make([]float64, counts[0]+counts[1]+counts[2]+counts[3])
	offsets := []int{0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2]}
	limits := []int{counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2], len(values)}

	for stateIndex, value := range state {
		for maskIndex, mask := range masks {
			if mask[stateIndex] == 0 || offsets[maskIndex] >= limits[maskIndex] {
				continue
			}

			values[offsets[maskIndex]] = value
			offsets[maskIndex]++
			break
		}
	}

	return values
}

func referenceMarkovFlow(input []float64, weights []float64, bias []float64, rows int, columns int) []float64 {
	values := make([]float64, rows)

	for rowIndex := range rows {
		sum := float32(bias[rowIndex])
		for columnIndex := range columns {
			sum += float32(weights[rowIndex*columns+columnIndex]) * float32(input[columnIndex])
		}

		values[rowIndex] = float64(sum)
	}

	return values
}

func markovMutualInformationInputs(samples int, xDimensions int, yDimensions int) ([]float64, []float64) {
	xValues := make([]float64, samples*xDimensions)
	yValues := make([]float64, samples*yDimensions)

	for sampleIndex := range samples {
		for dimensionIndex := range xDimensions {
			value := markovNoise(sampleIndex, dimensionIndex, 17)
			xValues[sampleIndex*xDimensions+dimensionIndex] = float64(value)
		}

		for dimensionIndex := range yDimensions {
			value := markovNoise(sampleIndex, dimensionIndex, 53)
			yValues[sampleIndex*yDimensions+dimensionIndex] = float64(value)
		}
	}

	return xValues, yValues
}

func markovNoise(sampleIndex int, dimensionIndex int, salt uint32) float32 {
	value := uint32(sampleIndex+1)*1664525 + uint32(dimensionIndex+3)*1013904223 + salt*374761393
	value ^= value >> 13
	value *= 1274126177
	value ^= value >> 16

	return float32(int(value%2001)-1000) / 997
}

func referenceMarkovMutualInformation(
	xValues []float64,
	yValues []float64,
	samples int,
	xDimensions int,
	yDimensions int,
) float64 {
	xMean := markovColumnMean(xValues, samples, xDimensions)
	yMean := markovColumnMean(yValues, samples, yDimensions)
	covX := markovCovariance(xValues, xValues, xMean, xMean, samples, xDimensions, xDimensions)
	covY := markovCovariance(yValues, yValues, yMean, yMean, samples, yDimensions, yDimensions)
	covXY := markovCovariance(xValues, yValues, xMean, yMean, samples, xDimensions, yDimensions)
	joint := markovJointCovariance(covX, covY, covXY, xDimensions, yDimensions)
	value := 0.5 * (markovLogDet(covX, xDimensions) + markovLogDet(covY, yDimensions) -
		markovLogDet(joint, xDimensions+yDimensions))

	if value < 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return 0
	}

	return float64(value)
}

func markovColumnMean(values []float64, samples int, dimensions int) []float32 {
	mean := make([]float32, dimensions)

	for sampleIndex := range samples {
		for dimensionIndex := range dimensions {
			mean[dimensionIndex] += float32(values[sampleIndex*dimensions+dimensionIndex])
		}
	}

	for dimensionIndex := range dimensions {
		mean[dimensionIndex] /= float32(samples)
	}

	return mean
}

func markovCovariance(
	left []float64,
	right []float64,
	leftMean []float32,
	rightMean []float32,
	samples int,
	leftDimensions int,
	rightDimensions int,
) []float32 {
	values := make([]float32, leftDimensions*rightDimensions)

	for rowIndex := range leftDimensions {
		for columnIndex := range rightDimensions {
			sum := float32(0)
			for sampleIndex := range samples {
				sum += (float32(left[sampleIndex*leftDimensions+rowIndex]) - leftMean[rowIndex]) *
					(float32(right[sampleIndex*rightDimensions+columnIndex]) - rightMean[columnIndex])
			}

			values[rowIndex*rightDimensions+columnIndex] = sum / float32(samples-1)
		}
	}

	return values
}

func markovJointCovariance(covX []float32, covY []float32, covXY []float32, xDimensions int, yDimensions int) []float32 {
	dimensions := xDimensions + yDimensions
	values := make([]float32, dimensions*dimensions)

	for rowIndex := range xDimensions {
		for columnIndex := range xDimensions {
			values[rowIndex*dimensions+columnIndex] = covX[rowIndex*xDimensions+columnIndex]
		}
	}

	for rowIndex := range yDimensions {
		for columnIndex := range yDimensions {
			values[(xDimensions+rowIndex)*dimensions+xDimensions+columnIndex] = covY[rowIndex*yDimensions+columnIndex]
		}
	}

	for rowIndex := range xDimensions {
		for columnIndex := range yDimensions {
			values[rowIndex*dimensions+xDimensions+columnIndex] = covXY[rowIndex*yDimensions+columnIndex]
			values[(xDimensions+columnIndex)*dimensions+rowIndex] = covXY[rowIndex*yDimensions+columnIndex]
		}
	}

	return values
}

func markovLogDet(matrix []float32, dimensions int) float32 {
	work := make([]float32, len(matrix))
	copy(work, matrix)

	for diagonalIndex := range dimensions {
		work[diagonalIndex*dimensions+diagonalIndex] += 1e-6
	}

	for columnIndex := range dimensions {
		sum := work[columnIndex*dimensions+columnIndex]
		for innerIndex := range columnIndex {
			value := work[columnIndex*dimensions+innerIndex]
			sum -= value * value
		}

		if sum <= 0 {
			return float32(math.NaN())
		}

		diagonal := float32(math.Sqrt(float64(sum)))
		work[columnIndex*dimensions+columnIndex] = diagonal
		inverse := 1 / diagonal

		for rowIndex := columnIndex + 1; rowIndex < dimensions; rowIndex++ {
			accumulator := work[rowIndex*dimensions+columnIndex]
			for innerIndex := range columnIndex {
				accumulator -= work[rowIndex*dimensions+innerIndex] * work[columnIndex*dimensions+innerIndex]
			}

			work[rowIndex*dimensions+columnIndex] = accumulator * inverse
		}
	}

	logDet := float32(0)
	for diagonalIndex := range dimensions {
		logDet += float32(math.Log(float64(work[diagonalIndex*dimensions+diagonalIndex])))
	}

	return 2 * logDet
}

func markovGraph(test testing.TB) (*ir.Graph, []*ir.Node, int64, map[string][]float64, map[string]float64) {
	test.Helper()

	state, masks, _, counts := markovPartitionInputs(16)
	sensory := markovVector(counts[0], 0.012, -0.1)
	internal := markovVector(counts[2], 0.018, 0.03)
	internalWeights := markovWeights(counts[2], counts[0])
	internalBias := markovVector(counts[2], 0.004, 0.01)
	activeWeights := markovWeights(counts[1], counts[2])
	activeBias := markovVector(counts[1], 0.005, -0.02)
	xValues, yValues := markovMutualInformationInputs(18, 2, 2)
	inputs := markovGraphInputs(
		test,
		"partition",
		append([][]float64{state}, masks...)...,
	)
	moreInputs := markovGraphInputs(
		test,
		"flow",
		sensory, internalWeights, internalBias, internal, activeWeights, activeBias, xValues, yValues,
	)
	inputs = append(inputs, moreInputs...)
	graph := ir.NewGraph()

	for _, node := range inputs {
		graph.AddNode(node)
	}

	partition := markovNode(test, "mb_partition", "markov_blanket.partition", []int{16, counts[0], counts[1], counts[2], counts[3]}, inputs[:5]...)
	flowInternal := markovNode(test, "mb_flow_internal", "markov_blanket.flow_internal", []int{counts[2], counts[0], counts[2]}, inputs[5], inputs[6], inputs[7])
	flowActive := markovNode(test, "mb_flow_active", "markov_blanket.flow_active", []int{counts[1], counts[2]}, inputs[8], inputs[9], inputs[10])
	mutualInfo := markovNode(test, "mb_mutual_information", "markov_blanket.mutual_information", []int{2, 2}, inputs[11], inputs[12])

	for _, node := range []*ir.Node{partition, flowInternal, flowActive, mutualInfo} {
		graph.AddNode(node)
	}

	expected := map[string][]float64{
		"mb_partition":          referenceMarkovPartition(state, masks, counts),
		"mb_flow_internal":      referenceMarkovFlow(sensory, internalWeights, internalBias, counts[2], counts[0]),
		"mb_flow_active":        referenceMarkovFlow(internal, activeWeights, activeBias, counts[1], counts[2]),
		"mb_mutual_information": []float64{referenceMarkovMutualInformation(xValues, yValues, 18, 2, 2)},
	}
	tolerances := map[string]float64{
		"mb_partition":          1e-6,
		"mb_flow_internal":      1e-5,
		"mb_flow_active":        1e-5,
		"mb_mutual_information": 2e-4,
	}
	expectedBytes := int64(0)

	for _, node := range inputs {
		expectedBytes += int64(node.Shape().Len() * 4)
	}

	return graph, []*ir.Node{partition, flowInternal, flowActive, mutualInfo}, expectedBytes, expected, tolerances
}

func markovGraphInputs(test testing.TB, prefix string, values ...[]float64) []*ir.Node {
	test.Helper()

	nodes := make([]*ir.Node, len(values))
	for index, value := range values {
		node := ir.NewNode(fmt.Sprintf("mb_%s_input_%02d", prefix, index), ir.OpInput, markovShape(test, len(value)))
		node.SetMetadata("values", value)
		nodes[index] = node
	}

	return nodes
}

func markovNode(test testing.TB, name string, operation ir.OpType, shape []int, inputs ...*ir.Node) *ir.Node {
	test.Helper()

	node := ir.NewNode(name, operation, markovShape(test, shape...))

	for _, input := range inputs {
		node.AddInput(input)
	}

	return node
}

func assertMarkovGraphOutputs(
	results map[string]computetensor.Float64Tensor,
	expected map[string][]float64,
	tolerances map[string]float64,
) {
	for name, output := range results {
		So(output.Location(), ShouldEqual, computetensor.Metal)
		defer func(value computetensor.Float64Tensor) {
			So(value.Close(), ShouldBeNil)
		}(output)

		values, err := output.CloneFloat64()
		So(err, ShouldBeNil)
		assertMetalMaxDiff(values, expected[name], tolerances[name])
	}
}

func markovOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalMarkovBlanket {
	test.Helper()

	markovOps, err := tensorBackend.markovBlanket()
	So(err, ShouldBeNil)

	return markovOps
}

func markovBenchmarkOps(benchmark *testing.B) (*TensorBackend, *MetalMarkovBlanket) {
	benchmark.Helper()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	markovOps, err := tensorBackend.markovBlanket()
	if err != nil {
		benchmark.Fatal(err)
	}

	return tensorBackend, markovOps
}
