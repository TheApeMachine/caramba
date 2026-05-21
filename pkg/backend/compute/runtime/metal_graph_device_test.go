package runtime

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func TestPackedGLUOutputShape(t *testing.T) {
	convey.Convey("Given a packed gate/up activation tensor", t, func() {
		inputShape, err := tensor.NewShape([]int{2, 19456})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should match the temporary gate and up tensor shape", func() {
			outputShape, err := packedGLUOutputShape(inputShape)

			convey.So(err, convey.ShouldBeNil)
			convey.So(outputShape.Dims(), convey.ShouldResemble, []int{2, 9728})
		})
	})

	convey.Convey("Given a batched packed gate/up activation tensor", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 4100, 18432})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should preserve every leading dimension", func() {
			outputShape, err := packedGLUOutputShape(inputShape)

			convey.So(err, convey.ShouldBeNil)
			convey.So(outputShape.Dims(), convey.ShouldResemble, []int{1, 4100, 9216})
		})
	})
}

func TestUnitRMSNormWeight(t *testing.T) {
	convey.Convey("Given an affine-free RMSNorm input", t, func() {
		memory, err := metal.NewBackend(context.Background(), nil)
		convey.So(err, convey.ShouldBeNil)

		runner := &MetalGraphRunner{memory: memory}
		inputShape, err := tensor.NewShape([]int{1, 4096, 3072})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should create a unit scale vector", func() {
			weight, err := runner.unitRMSNormWeight(inputShape, dtype.Float32)

			convey.So(err, convey.ShouldBeNil)
			convey.So(weight.Shape().Dims(), convey.ShouldResemble, []int{3072})
			convey.So(weight.DType(), convey.ShouldEqual, dtype.Float32)
		})

		convey.Reset(func() {
			if memory != nil {
				memory.Close()
			}
		})
	})
}

func TestNodeOutputShape(t *testing.T) {
	convey.Convey("Given a scalar timestep input", t, func() {
		inputShape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("time_guidance_embed.time_proj", "embedding.timestep", outputShape)
		node.SetOperationID("embedding.timestep")
		node.SetAttribute("dim", ir.IntAttribute(256))

		convey.Convey("It should produce a timestep embedding row", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 256})
		})
	})

	convey.Convey("Given a 3D input to a linear projection", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 4096, 128})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("x_embedder", "projection.linear", outputShape)
		node.SetOperationID("projection.linear")

		convey.Convey("It should preserve the input prefix dimensions", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 4096, 3072})
		})
	})

	convey.Convey("Given two tensors concatenated along the sequence axis", t, func() {
		leftShape, err := tensor.NewShape([]int{1, 4096, 3072})
		convey.So(err, convey.ShouldBeNil)

		rightShape, err := tensor.NewShape([]int{1, 10, 3072})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("conditioning_concat", "shape.concat", outputShape)
		node.SetOperationID("shape.concat")
		node.SetAttribute("dim", ir.IntAttribute(1))

		convey.Convey("It should sum the concat axis at runtime", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{leftShape, rightShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 4106, 3072})
		})
	})

	convey.Convey("Given a 3D input to RMSNorm", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 4106, 3072})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("norm", "math.rmsnorm", outputShape)
		node.SetOperationID("math.rmsnorm")

		convey.Convey("It should preserve the input shape", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 4106, 3072})
		})
	})

	convey.Convey("Given a 3D input and 2x conditioning to adaptive RMSNorm", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 4100, 3072})
		convey.So(err, convey.ShouldBeNil)

		modulationShape, err := tensor.NewShape([]int{1, 6144})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("norm_out.adaptive", "math.adaptive_rmsnorm", outputShape)
		node.SetOperationID("math.adaptive_rmsnorm")

		convey.Convey("It should preserve the normalized input shape", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape, modulationShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 4100, 3072})
		})
	})

	convey.Convey("Given a 3D input to view_as_heads", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 4106, 3072})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 24, 128})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("heads", "shape.view_as_heads", outputShape)
		node.SetOperationID("shape.view_as_heads")

		convey.Convey("It should preserve batch and sequence dimensions", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 4106, 24, 128})
		})
	})

	convey.Convey("Given an NCHW input to conv2d", t, func() {
		inputShape, err := tensor.NewShape([]int{1, 32, 128, 128})
		convey.So(err, convey.ShouldBeNil)

		weightShape, err := tensor.NewShape([]int{512, 32, 3, 3})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 1, 1, 1})
		convey.So(err, convey.ShouldBeNil)

		node := ir.NewNode("conv", "convolution.conv2d", outputShape)
		node.SetOperationID("convolution.conv2d")
		node.SetAttribute("stride_h", ir.IntAttribute(1))
		node.SetAttribute("stride_w", ir.IntAttribute(1))
		node.SetAttribute("pad_h", ir.IntAttribute(1))
		node.SetAttribute("pad_w", ir.IntAttribute(1))

		convey.Convey("It should derive the output channels and image dimensions", func() {
			resolved, err := nodeOutputShape(node, []tensor.Shape{inputShape, weightShape})

			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Dims(), convey.ShouldResemble, []int{1, 512, 128, 128})
		})
	})
}
