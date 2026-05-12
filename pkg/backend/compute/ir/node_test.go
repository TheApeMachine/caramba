package ir

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNode(t *testing.T) {
	Convey("Given a new Node", t, func() {
		ctx := context.Background()
		shape, err := tensor.NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		node := NewNode(ctx, "n1", OpMatmul, shape)

		Convey("It should return its ID", func() {
			So(node.ID(), ShouldEqual, "n1")
		})

		Convey("It should return its OpType", func() {
			So(node.OpType(), ShouldEqual, OpMatmul)
		})

		Convey("It should return its Shape", func() {
			So(node.Shape().Equal(shape), ShouldBeTrue)
		})

		Convey("It should allow adding inputs", func() {
			inputShape, _ := tensor.NewShape([]int{2, 2})
			inputNode := NewNode(ctx, "in1", OpInput, inputShape)

			node.AddInput(inputNode)
			So(len(node.Inputs()), ShouldEqual, 1)
			So(node.Inputs()[0].ID(), ShouldEqual, "in1")
		})

		Convey("It should allow setting and getting metadata", func() {
			node.SetMetadata("activation", "relu")
			metadata := node.Metadata()
			So(metadata["activation"], ShouldEqual, "relu")
		})

		Convey("It should allow configuring buffer reuse", func() {
			So(node.InPlace(), ShouldBeFalse)
			node.SetInPlace(true)
			So(node.InPlace(), ShouldBeTrue)
		})
	})
}

func BenchmarkNode(b *testing.B) {
	ctx := context.Background()
	shape, _ := tensor.NewShape([]int{2, 2})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewNode(ctx, "n1", OpMatmul, shape)
	}
}
