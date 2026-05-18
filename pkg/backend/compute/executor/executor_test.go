package executor_test

import (
	"context"
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestExecutor(t *testing.T) {
	Convey("Given a shared graph executor", t, func() {
		backend := computecpu.NewTensorBackend()
		graphExecutor := executor.New(backend)
		shape, err := tensor.NewShape([]int{2})
		So(err, ShouldBeNil)
		data, err := executor.EncodeFloat64([]float64{-1, 2})
		So(err, ShouldBeNil)

		nodes := []executor.NodeSpec{
			{
				ID:     "input",
				Op:     ir.OpInput,
				Shape:  shape.Dims(),
				Target: false,
			},
			{
				ID:     "relu",
				Op:     ir.OpReLU,
				Inputs: []string{"input"},
				Shape:  shape.Dims(),
				Target: true,
			},
		}

		tensors := []executor.TensorSpec{
			{
				ID:    "input",
				Shape: []int64{2},
				Data:  data,
				DType: dtype.Float64,
			},
		}

		Convey("When executing a supported graph", func() {
			outputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)

			Convey("It should execute through the backend kernels", func() {
				So(err, ShouldBeNil)
				So(outputs, ShouldHaveLength, 1)

				values, err := tensorFloat64Values(outputs["relu"])
				So(err, ShouldBeNil)
				So(values, ShouldResemble, []float64{0, 2})
			})
		})

		Convey("When executing repeatedly with closed caller outputs", func() {
			Convey("It should release executor-owned inputs and intermediates", func() {
				for range 256 {
					outputs, err := executor.New(backend).Execute(context.Background(), nodes, tensors)
					So(err, ShouldBeNil)
					So(outputs["relu"].Close(), ShouldBeNil)
				}
			})
		})

		Convey("When reusing the same executor after a successful run", func() {
			firstOutputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)
			So(err, ShouldBeNil)

			secondOutputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)
			So(err, ShouldBeNil)
			defer func() { So(secondOutputs["relu"].Close(), ShouldBeNil) }()
			defer func() { So(firstOutputs["relu"].Close(), ShouldBeNil) }()

			Convey("It should leave caller-owned outputs alive", func() {
				values, err := tensorFloat64Values(firstOutputs["relu"])

				So(err, ShouldBeNil)
				So(values, ShouldResemble, []float64{0, 2})
			})
		})
	})

	Convey("Given a graph with fanout dependencies", t, func() {
		backend := newRecordingBackend()
		graphExecutor := executor.New(backend)
		shape, err := tensor.NewShape([]int{2})
		So(err, ShouldBeNil)
		data, err := executor.EncodeFloat64([]float64{1, 2})
		So(err, ShouldBeNil)

		nodes := []executor.NodeSpec{
			{ID: "input", Op: ir.OpInput, Shape: shape.Dims()},
			{ID: "left", Op: ir.OpReLU, Inputs: []string{"input"}, Shape: shape.Dims()},
			{ID: "right", Op: ir.OpGELU, Inputs: []string{"input"}, Shape: shape.Dims()},
			{
				ID:     "output",
				Op:     ir.OpAdd,
				Inputs: []string{"left", "right"},
				Shape:  shape.Dims(),
				Target: true,
			},
		}
		tensors := []executor.TensorSpec{
			{ID: "input", Shape: []int64{2}, Data: data, DType: dtype.Float64},
		}

		outputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)

		Convey("It should release owned tensors only after their final consumer", func() {
			So(err, ShouldBeNil)
			So(outputs, ShouldHaveLength, 1)
			So(backend.closed["upload:0"], ShouldEqual, 1)
			So(backend.closed["left"], ShouldEqual, 1)
			So(backend.closed["right"], ShouldEqual, 1)
			So(backend.closed["output"], ShouldEqual, 0)
			So(outputs["output"].Close(), ShouldBeNil)
			So(backend.closed["output"], ShouldEqual, 1)
		})
	})
}

func TestWithDerivedMetadata(t *testing.T) {
	Convey("Given an affine-free RMSNorm node", t, func() {
		shape, err := tensor.NewShape([]int{1, 3})
		So(err, ShouldBeNil)

		input := newRecordingTensor(
			"input",
			shape,
			[]float64{1, 2, 3},
			make(map[string]int),
		)
		node := executor.NodeSpec{
			ID:       "norm",
			Op:       ir.OpType("math.rmsnorm"),
			Metadata: map[string]any{"affine": false},
		}

		Convey("It should materialize the unit affine vector from input shape", func() {
			derived := executor.WithDerivedMetadata(
				node,
				[]tensor.Tensor{input},
			)

			So(derived.Metadata["weight"], ShouldResemble, []float64{1, 1, 1})
			_, originalBound := node.Metadata["weight"]
			So(originalBound, ShouldBeFalse)
		})
	})
}

func TestRunOperation(test *testing.T) {
	Convey("Given the host-staged operation executor", test, func() {
		backend := newRecordingBackend()
		backend.location = tensor.Metal
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		input := newRecordingTensor(
			"input",
			shape,
			[]float64{1},
			backend.closed,
		)
		node := executor.NodeSpec{
			ID:    "relu",
			Op:    ir.OpReLU,
			Shape: shape.Dims(),
		}

		Convey("It should reject accelerator execution before downloading inputs", func() {
			output, err := executor.RunOperation(
				context.Background(),
				backend,
				node,
				[]tensor.Tensor{input},
				hostStagedOperation{},
			)

			So(output, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "resident kernel required")
			So(backend.downloads, ShouldEqual, 0)
		})
	})
}

type hostStagedOperation struct{}

func (operation hostStagedOperation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	stateDict.SetOperationOutput([]float64{1})

	return stateDict, nil
}

type recordingBackend struct {
	uploads   int
	downloads int
	location  tensor.Location
	closed    map[string]int
}

func newRecordingBackend() *recordingBackend {
	return &recordingBackend{
		closed: make(map[string]int),
	}
}

func (recordingBackend *recordingBackend) Location() tensor.Location {
	if recordingBackend.location == "" {
		return tensor.Host
	}

	return recordingBackend.location
}

func (recordingBackend *recordingBackend) SupportedDTypes() []dtype.DType {
	return tensor.NewHostBackend().SupportedDTypes()
}

func (recordingBackend *recordingBackend) SupportedLayouts() []tensor.Layout {
	return tensor.NewHostBackend().SupportedLayouts()
}

func (recordingBackend *recordingBackend) Capabilities() tensor.Capabilities {
	return tensor.NewHostBackend().Capabilities()
}

func (recordingBackend *recordingBackend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	id := fmt.Sprintf("upload:%d", recordingBackend.uploads)
	recordingBackend.uploads++

	return newRecordingTensorFromBytes(id, shape, sourceDType, bytes, recordingBackend.closed)
}

func (recordingBackend *recordingBackend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	return recordingBackend.Upload(shape, sourceDType, bytes)
}

func (recordingBackend *recordingBackend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (recordingBackend *recordingBackend) Download(
	value tensor.Tensor,
) (dtype.DType, []byte, error) {
	recordingBackend.downloads++

	return value.RawBytes()
}

func (recordingBackend *recordingBackend) Close() error {
	return nil
}

func (recordingBackend *recordingBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	for _, input := range inputs {
		if _, _, err := input.RawBytes(); err != nil {
			return nil, err
		}
	}

	shape, err := recordingOutputShape(node, inputs)

	if err != nil {
		return nil, err
	}

	return newRecordingTensor(
		node.ID,
		shape,
		make([]float64, shape.Len()),
		recordingBackend.closed,
	), nil
}

func recordingOutputShape(
	node executor.NodeSpec, inputs []tensor.Tensor,
) (tensor.Shape, error) {
	if len(node.Shape) > 0 {
		return tensor.NewShape(node.Shape)
	}

	if len(inputs) == 0 {
		return tensor.NewShape([]int{0})
	}

	return inputs[0].Shape(), nil
}

type recordingTensor struct {
	tensor.Tensor
	id     string
	closed map[string]int
	done   bool
}

func newRecordingTensor(
	id string,
	shape tensor.Shape,
	values []float64,
	closed map[string]int,
) *recordingTensor {
	recordingTensor, err := newRecordingTensorFromBytes(
		id,
		shape,
		dtype.Float64,
		dtypeconvert.Float64ToBytes(values),
		closed,
	)

	if err != nil {
		panic(err)
	}

	return recordingTensor
}

func newRecordingTensorFromBytes(
	id string,
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
	closed map[string]int,
) (*recordingTensor, error) {
	backend := tensor.NewHostBackend()
	inner, err := backend.Upload(shape, sourceDType, bytes)

	if err != nil {
		return nil, err
	}

	return &recordingTensor{
		Tensor: inner,
		id:     id,
		closed: closed,
	}, nil
}

func (recordingTensor *recordingTensor) Close() error {
	if recordingTensor.done {
		return nil
	}

	recordingTensor.done = true
	recordingTensor.closed[recordingTensor.id]++

	return recordingTensor.Tensor.Close()
}

func BenchmarkExecutor_Execute(benchmark *testing.B) {
	backend := computecpu.NewTensorBackend()
	data, err := executor.EncodeFloat64([]float64{1, 2})
	if err != nil {
		benchmark.Fatalf("EncodeFloat64 failed: %v", err)
	}

	nodes := []executor.NodeSpec{
		{ID: "input", Op: ir.OpInput},
		{ID: "relu", Op: ir.OpReLU, Inputs: []string{"input"}, Target: true},
	}

	tensors := []executor.TensorSpec{
		{ID: "input", Shape: []int64{2}, Data: data, DType: dtype.Float64},
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		graphExecutor := executor.New(backend)
		outputs, err := graphExecutor.Execute(context.Background(), nodes, tensors)
		if err != nil {
			benchmark.Fatalf("Execute failed: %v", err)
		}

		for _, output := range outputs {
			if err := output.Close(); err != nil {
				benchmark.Fatalf("Close failed: %v", err)
			}
		}
	}
}

func BenchmarkWithDerivedMetadata(benchmark *testing.B) {
	shape, err := tensor.NewShape([]int{1, 3})
	if err != nil {
		benchmark.Fatalf("NewShape failed: %v", err)
	}

	input := newRecordingTensor(
		"input",
		shape,
		[]float64{1, 2, 3},
		make(map[string]int),
	)
	node := executor.NodeSpec{
		ID:       "norm",
		Op:       ir.OpType("math.rmsnorm"),
		Metadata: map[string]any{"affine": false},
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = executor.WithDerivedMetadata(node, []tensor.Tensor{input})
	}
}

func tensorFloat64Values(value tensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}
