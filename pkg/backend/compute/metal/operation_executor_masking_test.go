//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_applyMaskingGraph(test *testing.T) {
	Convey("Given Metal masking graph execution", test, func() {
		Convey("It should execute apply-mask without intermediate readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape, err := computetensor.NewShape([]int{64})
			So(err, ShouldBeNil)

			scores := maskingScores(shape.Len())
			mask := maskingMask(shape.Len())
			expected := referenceApplyMask(scores, mask)
			scoreInput := ir.NewNode("scores", ir.OpInput, shape)
			scoreInput.SetMetadata("values", scores)
			maskInput := ir.NewNode("mask", ir.OpInput, shape)
			maskInput.SetMetadata("values", mask)
			output := ir.NewNode("masked", "masking.apply", shape)
			output.AddInput(scoreInput)
			output.AddInput(maskInput)

			values := runMaskingGraphForTest(test, tensorBackend, []*ir.Node{scoreInput, maskInput, output}, output)
			assertMaskValues(values, expected, 1e-6)
		})

		Convey("It should execute causal-mask without input readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			outputShape, err := computetensor.NewShape([]int{8, 8})
			So(err, ShouldBeNil)

			output := ir.NewNode("causal", "masking.causal", outputShape)
			output.SetMetadata("seq_len", 8)

			values := runMaskingGraphForTest(test, tensorBackend, []*ir.Node{output}, output)
			assertCausalMask(values, 8)
		})
	})
}

func runMaskingGraphForTest(
	test *testing.T,
	tensorBackend *TensorBackend,
	nodes []*ir.Node,
	output *ir.Node,
) []float64 {
	test.Helper()

	graph := ir.NewGraph()
	for _, node := range nodes {
		graph.AddNode(node)
	}

	before := tensorBackend.runtime.Metrics()
	results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{output})
	after := tensorBackend.runtime.Metrics()
	So(err, ShouldBeNil)
	So(results, ShouldHaveLength, 1)
	So(results[output.ID()].Location(), ShouldEqual, computetensor.Metal)

	expectedUploadBytes := maskGraphUploadBytes(nodes)
	So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedUploadBytes)

	values, err := results[output.ID()tensorFloat64Values(])
	So(err, ShouldBeNil)
	defer func() {
		So(results[output.ID()].Close(), ShouldBeNil)
	}()

	return values
}

func maskGraphUploadBytes(nodes []*ir.Node) int64 {
	var uploadBytes int64

	for _, node := range nodes {
		if node.OpType() != ir.OpInput {
			continue
		}

		uploadBytes += int64(node.Shape().Len() * 4)
	}

	return uploadBytes
}
