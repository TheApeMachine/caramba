package compute

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/runtime"
	"github.com/theapemachine/caramba/pkg/qpool"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestParseDeviceID(test *testing.T) {
	Convey("Given device id strings", test, func() {
		Convey("It should accept host aliases", func() {
			deviceID, err := ParseDeviceID("cpu")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Host, Index: 0})
		})

		Convey("It should parse indexed gpu ids", func() {
			deviceID, err := ParseDeviceID("metal:2")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Metal, Index: 2})
		})
	})
}

func TestNewBackend(test *testing.T) {
	Convey("Given a compute backend", test, func() {
		backend, err := NewBackend(context.Background(), nil)
		So(err, ShouldBeNil)

		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should always discover host", func() {
			hostDevice, err := backend.Device(DeviceID{Location: tensor.Host, Index: 0})
			So(err, ShouldBeNil)
			So(hostDevice.Executor(), ShouldNotBeNil)
		})

		Convey("It should route execution to a selected device", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			input.SetMetadata("values", []float64{1, 2})
			graph.AddNode(input)

			outputs, err := backend.Execute(
				context.Background(),
				graph,
				[]*ir.Node{input},
				DeviceID{Location: tensor.Host, Index: 0},
			)
			So(err, ShouldBeNil)

			values, err := tensorFloat64Values(outputs["input"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2})
			So(outputs["input"].Close(), ShouldBeNil)
		})

		Convey("It should reject precision mismatch before the executor runs", func() {
			shape, err := tensor.NewShape([]int{2})
			So(err, ShouldBeNil)

			graph := ir.NewGraph()
			input := ir.NewNode("input", ir.OpInput, shape)
			relu := ir.NewNode("relu", ir.OpReLU, shape)
			relu.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(relu)

			routed := backendWithDevice(&Device{
				id:       DeviceID{Location: tensor.Metal, Index: 0},
				memory:   tensor.NewHostBackend(),
				executor: &testExecutor{location: tensor.Metal},
			})

			outputs, err := routed.Execute(
				context.Background(),
				graph,
				[]*ir.Node{relu},
				DeviceID{Location: tensor.Metal, Index: 0},
			)

			So(outputs, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "requires F64 precision")
		})

		Convey("It should call the executor after explicit precision opt-in", func() {
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

			testExec := &testExecutor{location: tensor.Metal}
			routed := backendWithDevice(&Device{
				id:       DeviceID{Location: tensor.Metal, Index: 0},
				memory:   tensor.NewHostBackend(),
				executor: testExec,
			})

			outputs, err := routed.Execute(
				context.Background(),
				graph,
				[]*ir.Node{relu},
				DeviceID{Location: tensor.Metal, Index: 0},
			)

			So(err, ShouldBeNil)
			So(testExec.called, ShouldBeTrue)
			So(outputs["relu"], ShouldNotBeNil)
			So(outputs["relu"].Close(), ShouldBeNil)
		})

		Convey("It should reject unknown device ids", func() {
			_, err := backend.Device(DeviceID{Location: tensor.CUDA, Index: 9})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "device not found")
		})
	})
}

func TestNewBackend_Heterogeneous(test *testing.T) {
	Convey("Given discovery on a host with optional accelerators", test, func() {
		backend, err := NewBackend(context.Background(), qpool.NewQ(context.Background(), 1, 1, nil))
		So(err, ShouldBeNil)

		defer func() {
			So(backend.Close(), ShouldBeNil)
		}()

		Convey("It should keep host resident while metal memory is present", func() {
			locations := make([]tensor.Location, 0, len(backend.Devices()))

			for _, device := range backend.Devices() {
				locations = append(locations, device.Location())
			}

			So(locations, ShouldContain, tensor.Host)

			if len(locations) > 1 {
				So(locations, ShouldContain, tensor.Metal)
			}
		})
	})
}

func backendWithDevice(device *Device) *Backend {
	return &Backend{
		ctx:     context.Background(),
		devices: []*Device{device},
		byID:    map[DeviceID]*Device{device.id: device},
		mesh:    buildMesh([]*Device{device}),
	}
}

type testExecutor struct {
	location tensor.Location
	called   bool
}

func (testExecutor *testExecutor) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (map[string]tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	testExecutor.called = true
	outputs := make(map[string]tensor.Tensor, len(targets))
	hostMemory := tensor.NewHostBackend()

	for _, target := range targets {
		value, err := hostMemory.Upload(
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

func (testExecutor *testExecutor) Location() tensor.Location {
	return testExecutor.location
}

func (testExecutor *testExecutor) Close() error {
	return nil
}

var _ runtime.Executor = (*testExecutor)(nil)

func tensorFloat64Values(value tensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}
