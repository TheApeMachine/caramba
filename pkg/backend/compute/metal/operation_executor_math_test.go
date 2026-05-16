//go:build darwin && cgo

package metal

import (
	"context"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_applyMathGraph(test *testing.T) {
	Convey("Given a Metal math graph", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		runner := NewRunnerWithBackend(tensorBackend)

		inputShape, err := computetensor.NewShape([]int{1, 7})
		So(err, ShouldBeNil)

		outputShape, err := computetensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		values := []float64{-2, -1, -0.5, 0, 0.5, 1, 2}
		input := ir.NewNode("input", ir.OpInput, inputShape)
		input.SetMetadata("values", values)
		exp := ir.NewNode("exp", "math.exp", inputShape)
		exp.AddInput(input)
		logSumExp := ir.NewNode("logsumexp", "math.logsumexp", outputShape)
		logSumExp.AddInput(exp)

		graph := ir.NewGraph()
		graph.AddNode(input)
		graph.AddNode(exp)
		graph.AddNode(logSumExp)

		Convey("It should execute without intermediate readback", func() {
			before := tensorBackend.runtime.Metrics()
			results, err := runner.Execute(context.Background(), graph, []*ir.Node{logSumExp})
			after := tensorBackend.runtime.Metrics()

			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(results["logsumexp"].Location(), ShouldEqual, computetensor.Metal)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(inputShape.Len()*4))

			output, err := results["logsumexp"].CloneFloat64()
			So(err, ShouldBeNil)
			defer func() {
				So(results["logsumexp"].Close(), ShouldBeNil)
			}()

			assertMetalMaxDiff(output, []float64{referenceExpLogSumExp(values)}, 3e-5)
		})
	})
}

func TestTensorBackend_applyDropoutGraph(test *testing.T) {
	Convey("Given a Metal dropout graph", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		runner := NewRunnerWithBackend(tensorBackend)

		shape, err := computetensor.NewShape([]int{7})
		So(err, ShouldBeNil)

		values := []float64{-3, -2, -1, 0, 1, 2, 3}
		input := ir.NewNode("input", ir.OpInput, shape)
		input.SetMetadata("values", values)
		dropout := ir.NewNode("dropout", "math.dropout", shape)
		dropout.SetMetadata("p", 0.25)
		dropout.SetMetadata("training", true)
		dropout.SetMetadata("seed", 17)
		dropout.AddInput(input)

		graph := ir.NewGraph()
		graph.AddNode(input)
		graph.AddNode(dropout)

		Convey("It should execute seeded dropout without intermediate readback", func() {
			before := tensorBackend.runtime.Metrics()
			results, err := runner.Execute(context.Background(), graph, []*ir.Node{dropout})
			after := tensorBackend.runtime.Metrics()

			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(results["dropout"].Location(), ShouldEqual, computetensor.Metal)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(shape.Len()*4))

			output, err := results["dropout"].CloneFloat64()
			So(err, ShouldBeNil)
			defer func() {
				So(results["dropout"].Close(), ShouldBeNil)
			}()

			assertMetalMaxDiff(output, referenceDropout(values, 0.25, true, 17), 1e-6)
		})
	})
}

func TestTensorBackend_applyOuterGraph(test *testing.T) {
	Convey("Given a Metal outer graph", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		runner := NewRunnerWithBackend(tensorBackend)

		leftShape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		rightShape, err := computetensor.NewShape([]int{3})
		So(err, ShouldBeNil)

		outputShape, err := computetensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		leftValues := []float64{2, -3}
		rightValues := []float64{4, 5, -6}
		left := ir.NewNode("left", ir.OpInput, leftShape)
		left.SetMetadata("values", leftValues)
		right := ir.NewNode("right", ir.OpInput, rightShape)
		right.SetMetadata("values", rightValues)
		outer := ir.NewNode("outer", "math.outer", outputShape)
		outer.AddInput(left)
		outer.AddInput(right)

		graph := ir.NewGraph()
		graph.AddNode(left)
		graph.AddNode(right)
		graph.AddNode(outer)

		Convey("It should execute outer product without intermediate readback", func() {
			before := tensorBackend.runtime.Metrics()
			results, err := runner.Execute(context.Background(), graph, []*ir.Node{outer})
			after := tensorBackend.runtime.Metrics()

			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(results["outer"].Location(), ShouldEqual, computetensor.Metal)
			So(
				after.TransferBytes-before.TransferBytes,
				ShouldEqual,
				int64((leftShape.Len()+rightShape.Len())*4),
			)

			output, err := results["outer"].CloneFloat64()
			So(err, ShouldBeNil)
			defer func() {
				So(results["outer"].Close(), ShouldBeNil)
			}()

			assertMetalMaxDiff(output, referenceOuter(leftValues, rightValues), 1e-6)
		})
	})
}

func referenceExpLogSumExp(input []float64) float64 {
	transformed := make([]float64, len(input))

	for index, value := range input {
		transformed[index] = math.Exp(float64(float32(value)))
	}

	return referenceLogSumExp(transformed)
}
