//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_applyPoolingGraph(test *testing.T) {
	Convey("Given Metal pooling graph execution", test, func() {
		Convey("It should execute max_pool2d without intermediate readback", func() {
			shape, outputShape, input := pool2dCase(test)
			expected := referencePool2d(input, shape.Dims(), outputShape.Dims(), maxPoolTestParams(), true)

			runPoolingGraphForTest(test, "pooling.max_pool2d", shape, outputShape, input, expected)
		})

		Convey("It should execute avg_pool2d without intermediate readback", func() {
			shape, outputShape, input := pool2dCase(test)
			params := avgPoolTestParams()
			expected := referencePool2d(input, shape.Dims(), outputShape.Dims(), maxParamsFromAvg(params), false)

			runPoolingGraphForTest(test, "pooling.avg_pool2d", shape, outputShape, input, expected)
		})

		Convey("It should execute adaptive_avg_pool2d without intermediate readback", func() {
			shape, outputShape, input := adaptivePool2dCase(test)
			expected := referenceAdaptivePool2d(input, shape.Dims(), outputShape.Dims(), false)

			runPoolingGraphForTest(test, "pooling.adaptive_avg_pool2d", shape, outputShape, input, expected)
		})

		Convey("It should execute adaptive_max_pool2d without intermediate readback", func() {
			shape, outputShape, input := adaptivePool2dCase(test)
			expected := referenceAdaptivePool2d(input, shape.Dims(), outputShape.Dims(), true)

			runPoolingGraphForTest(test, "pooling.adaptive_max_pool2d", shape, outputShape, input, expected)
		})
	})
}

func runPoolingGraphForTest(
	test *testing.T,
	operationID ir.OpType,
	inputShape computetensor.Shape,
	outputShape computetensor.Shape,
	inputValues []float64,
	expected []float64,
) {
	test.Helper()

	tensorBackend := newMetalTensorBackendForTest(test)
	runner := NewRunnerWithBackend(tensorBackend)
	input := ir.NewNode("input", ir.OpInput, inputShape)
	input.SetMetadata("values", inputValues)
	output := ir.NewNode("pool", operationID, outputShape)
	output.AddInput(input)
	setPoolingGraphMetadata(output)

	graph := ir.NewGraph()
	graph.AddNode(input)
	graph.AddNode(output)

	before := tensorBackend.runtime.Metrics()
	results, err := runner.Execute(context.Background(), graph, []*ir.Node{output})
	after := tensorBackend.runtime.Metrics()

	So(err, ShouldBeNil)
	So(results, ShouldHaveLength, 1)
	So(results["pool"].Location(), ShouldEqual, computetensor.Metal)
	So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(inputShape.Len()*4))

	values, err := tensorFloat64Values(results["pool"])
	So(err, ShouldBeNil)
	defer func() {
		So(results["pool"].Close(), ShouldBeNil)
	}()
	assertMetalMaxDiff(values, expected, 1e-6)
}

func setPoolingGraphMetadata(node *ir.Node) {
	node.SetMetadata("kernel_size", 3)
	node.SetMetadata("stride", 1)
	node.SetMetadata("padding", 1)
	node.SetMetadata("dilation", 1)
}
