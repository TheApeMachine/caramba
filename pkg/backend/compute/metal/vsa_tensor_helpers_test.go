//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func vsaVector(elementCount int, phase int) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(float32(float64(((index+phase)%17)-8) / 16))
	}

	return values
}

func vsaUnitVector(elementCount int, value float64) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(float32(value))
	}

	return values
}

func flattenVSAVectors(vectors [][]float64) []float64 {
	length := 0

	for _, vector := range vectors {
		length += len(vector)
	}

	values := make([]float64, 0, length)

	for _, vector := range vectors {
		values = append(values, vector...)
	}

	return values
}

func referenceVSABind(left []float64, right []float64) []float64 {
	values := make([]float64, len(left))

	for index := range values {
		values[index] = float64(float32(left[index]) * float32(right[index]))
	}

	return values
}

func referenceVSABundle(vectors [][]float64) []float64 {
	values := make([]float64, len(vectors[0]))
	sums := make([]float32, len(values))
	sumSquares := float32(0)

	for index := range sums {
		for _, vector := range vectors {
			sums[index] += float32(vector[index])
		}

		sumSquares += sums[index] * sums[index]
	}

	invNorm := float32(1)
	if sumSquares > 1e-24 {
		invNorm = float32(1 / math.Sqrt(float64(sumSquares)))
	}

	for index, sum := range sums {
		values[index] = float64(sum * invNorm)
	}

	return values
}

func referenceVSASimilarity(left []float64, right []float64) float64 {
	sum := float32(0)

	for index := range left {
		sum += float32(left[index]) * float32(right[index])
	}

	return float64(sum)
}

func referenceVSAPermute(input []float64, shift int) []float64 {
	values := make([]float64, len(input))
	length := len(input)
	normalised := shift % length

	if normalised < 0 {
		normalised += length
	}

	for index := range values {
		source := index - normalised
		if source < 0 {
			source += length
		}

		values[index] = float64(float32(input[source]))
	}

	return values
}

func referenceVSAInversePermute(input []float64, shift int) []float64 {
	return referenceVSAPermute(input, -shift)
}

func vsaGraph(
	shape computetensor.Shape,
	left []float64,
	right []float64,
	third []float64,
) (*ir.Graph, []*ir.Node) {
	similarityShape, err := computetensor.NewShape([]int{1})
	if err != nil {
		panic(err)
	}

	leftNode := ir.NewNode("left", ir.OpInput, shape)
	leftNode.SetMetadata("values", left)
	rightNode := ir.NewNode("right", ir.OpInput, shape)
	rightNode.SetMetadata("values", right)
	thirdNode := ir.NewNode("third", ir.OpInput, shape)
	thirdNode.SetMetadata("values", third)

	bindNode := ir.NewNode("bind", "vsa.bind", shape)
	bindNode.AddInput(leftNode)
	bindNode.AddInput(rightNode)
	bundleNode := ir.NewNode("bundle", "vsa.bundle", shape)
	bundleNode.AddInput(leftNode)
	bundleNode.AddInput(rightNode)
	bundleNode.AddInput(thirdNode)
	similarityNode := ir.NewNode("similarity", "vsa.similarity", similarityShape)
	similarityNode.AddInput(leftNode)
	similarityNode.AddInput(rightNode)
	permuteNode := ir.NewNode("permute", "vsa.permute", shape)
	permuteNode.SetMetadata("k", 2)
	permuteNode.AddInput(leftNode)
	inverseNode := ir.NewNode("inverse", "vsa.inverse_permute", shape)
	inverseNode.SetMetadata("k", 2)
	inverseNode.AddInput(permuteNode)

	graph := ir.NewGraph()
	for _, node := range []*ir.Node{
		leftNode, rightNode, thirdNode, bindNode, bundleNode, similarityNode, permuteNode, inverseNode,
	} {
		graph.AddNode(node)
	}

	return graph, []*ir.Node{bindNode, bundleNode, similarityNode, permuteNode, inverseNode}
}

func assertVSAGraphOutputs(
	results map[string]computetensor.Float64Tensor,
	left []float64,
	right []float64,
	third []float64,
) {
	for _, output := range results {
		So(output.Location(), ShouldEqual, computetensor.Metal)
		defer func(value computetensor.Float64Tensor) {
			So(value.Close(), ShouldBeNil)
		}(output)
	}

	assertVSAGraphOutput(results, "bind", referenceVSABind(left, right), 1e-7)
	assertVSAGraphOutput(results, "bundle", referenceVSABundle([][]float64{left, right, third}), 2e-5)
	assertVSAGraphOutput(results, "similarity", []float64{referenceVSASimilarity(left, right)}, 1e-5)
	assertVSAGraphOutput(results, "permute", referenceVSAPermute(left, 2), 1e-7)
	assertVSAGraphOutput(
		results,
		"inverse",
		referenceVSAInversePermute(referenceVSAPermute(left, 2), 2),
		1e-7,
	)
}

func assertVSAGraphOutput(
	results map[string]computetensor.Float64Tensor,
	name string,
	expected []float64,
	tolerance float64,
) {
	output, ok := results[name]
	So(ok, ShouldBeTrue)

	values, err := output.CloneFloat64()
	So(err, ShouldBeNil)
	assertMetalMaxDiff(values, expected, tolerance)
}

func closeBenchmarkTensors(values ...computetensor.Float64Tensor) {
	for _, value := range values {
		_ = value.Close()
	}
}

func closeBenchmarkOutput(
	benchmark *testing.B,
	output computetensor.Float64Tensor,
	err error,
) {
	benchmark.Helper()

	if err != nil {
		benchmark.Fatal(err)
	}

	if err := output.Close(); err != nil {
		benchmark.Fatal(err)
	}
}
