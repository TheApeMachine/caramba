package compute

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestNewBackend(test *testing.T) {
	Convey("Given a compute backend", test, func() {
		backend, err := NewBackend(CPU)
		So(err, ShouldBeNil)
		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should expose runner execution through the backend facade", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			input.SetMetadata("values", []float64{1, 2})
			graph.AddNode(input)

			outputs, err := backend.Execute(context.Background(), graph, []*ir.Node{input})
			So(err, ShouldBeNil)

			values, err := tensorFloat64Values(outputs["input"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2})
			So(outputs["input"].Close(), ShouldBeNil)
			So(backend.Location(), ShouldEqual, tensor.Host)
		})

		Convey("It should route facade execution through backend lowering", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			relu := ir.NewNode("relu", ir.OpReLU, shape)
			relu.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(relu)

			testRunner := &backendTestRunner{location: tensor.Metal}
			backend := &Backend{Runner: testRunner}

			outputs, err := backend.Execute(context.Background(), graph, []*ir.Node{relu})

			So(outputs, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "requires F64 precision")
			So(testRunner.called, ShouldBeFalse)
		})

		Convey("It should call the runner after explicit Metal precision opt-in", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			valueType := ir.ValueType{
				Shape:     shape,
				DType:     dtype.Float64,
				Precision: dtype.Float32,
			}
			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			input.SetValueType(valueType)
			relu := ir.NewNode("relu", ir.OpReLU, shape)
			relu.SetValueType(valueType)
			relu.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(relu)

			testRunner := &backendTestRunner{location: tensor.Metal}
			backend := &Backend{Runner: testRunner}

			outputs, err := backend.Execute(context.Background(), graph, []*ir.Node{relu})

			So(err, ShouldBeNil)
			So(testRunner.called, ShouldBeTrue)
			So(outputs["relu"], ShouldNotBeNil)
			So(outputs["relu"].Close(), ShouldBeNil)
		})

		Convey("It should reject unsupported backend types instead of silently using CPU", func() {
			backend, err := NewBackend(BackendType(255))

			So(backend, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported backend type")
		})
	})
}

func BenchmarkNewBackend(benchmark *testing.B) {
	for benchmark.Loop() {
		backend, err := NewBackend(CPU)
		if err != nil {
			benchmark.Fatal(err)
		}

		_ = backend.Close()
	}
}

func BenchmarkBackend_Execute(benchmark *testing.B) {
	backend, err := NewBackend(CPU)
	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = backend.Close()
	}()

	shape, err := tensor.NewShape([]int{2})
	if err != nil {
		benchmark.Fatal(err)
	}

	graph := ir.NewGraph()
	input := ir.NewNode("input", ir.OpInput, shape)
	input.SetMetadata("values", []float64{1, 2})
	graph.AddNode(input)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		outputs, err := backend.Execute(context.Background(), graph, []*ir.Node{input})

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := outputs["input"].Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

type backendTestRunner struct {
	location tensor.Location
	called   bool
}

func (testRunner *backendTestRunner) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (map[string]tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	testRunner.called = true
	outputs := make(map[string]tensor.Tensor, len(targets))
	hostBackend := tensor.NewHostBackend()

	for _, target := range targets {
		value, err := hostBackend.Upload(
			target.Shape(),
			dtype.Float64,
			dtypeconvert.Float64ToBytes(make([]float64, target.Shape().Len())),
		)

		if err != nil {
			return nil, err
		}

		outputs[target.ID()] = value
	}

	return outputs, nil
}

func tensorFloat64Values(value tensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}

func (testRunner *backendTestRunner) Location() tensor.Location {
	return testRunner.location
}

func (testRunner *backendTestRunner) Close() error {
	return nil
}
