package telemetryops

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
)

func TestCounter(t *testing.T) {
	Convey("Given a counter op with delta=2", t, func() {
		stub := optest.NewStubContext()
		recorder := telemetry.NewInMemory()
		stub.Recorder = recorder

		stub.StepRef = program.Step{
			ID:     "count",
			Op:     "telemetry.counter",
			Config: map[string]any{"name": "tokens_generated", "delta": 2},
		}

		So(Counter{}.Execute(stub), ShouldBeNil)
		So(Counter{}.Execute(stub), ShouldBeNil)

		So(recorder.Counter("tokens_generated"), ShouldEqual, 4)
	})
}

func TestHistogram(t *testing.T) {
	Convey("Given a histogram op driven by a local float value", t, func() {
		stub := optest.NewStubContext()
		recorder := telemetry.NewInMemory()
		stub.Recorder = recorder
		stub.Scope["latency"] = 12.5

		stub.StepRef = program.Step{
			ID: "hist",
			Op: "telemetry.histogram",
			Inputs: map[string]program.ValueRef{
				"value": {Namespace: program.NamespaceLocal, Name: "latency"},
			},
			Config: map[string]any{"name": "decode_latency_ms"},
		}

		So(Histogram{}.Execute(stub), ShouldBeNil)

		samples := recorder.Snapshot()
		So(len(samples), ShouldEqual, 1)
		So(samples[0].Kind, ShouldEqual, "histogram")
		So(samples[0].Name, ShouldEqual, "decode_latency_ms")
		So(samples[0].Value, ShouldEqual, 12.5)
	})
}

func TestScopeStack(t *testing.T) {
	Convey("Given nested scope ops around a counter", t, func() {
		stub := optest.NewStubContext()
		recorder := telemetry.NewInMemory()
		stub.Recorder = recorder

		stub.StepRef = program.Step{
			ID:     "enter",
			Op:     "telemetry.scope.enter",
			Config: map[string]any{"name": "decode"},
		}
		So(EnterScope{}.Execute(stub), ShouldBeNil)

		stub.StepRef = program.Step{
			ID:     "count",
			Op:     "telemetry.counter",
			Config: map[string]any{"name": "tokens", "delta": 1},
		}
		So(Counter{}.Execute(stub), ShouldBeNil)

		stub.StepRef = program.Step{
			ID: "exit",
			Op: "telemetry.scope.exit",
		}
		So(ExitScope{}.Execute(stub), ShouldBeNil)

		samples := recorder.Snapshot()
		So(len(samples), ShouldEqual, 1)
		So(samples[0].Scope, ShouldResemble, []string{"decode"})
	})
}

func TestTraceTensor(t *testing.T) {
	Convey("Given a trace_tensor op on a flat slice", t, func() {
		stub := optest.NewStubContext()
		recorder := telemetry.NewInMemory()
		stub.Recorder = recorder
		stub.Scope["activations"] = []float64{0.1, 0.2, 0.3}

		stub.StepRef = program.Step{
			ID: "trace",
			Op: "telemetry.trace_tensor",
			Inputs: map[string]program.ValueRef{
				"tensor": {Namespace: program.NamespaceLocal, Name: "activations"},
			},
			Config: map[string]any{"name": "attn_layer_3"},
		}

		So(TraceTensor{}.Execute(stub), ShouldBeNil)

		samples := recorder.Snapshot()
		So(len(samples), ShouldEqual, 1)
		So(samples[0].Kind, ShouldEqual, "tensor")
		So(samples[0].Name, ShouldEqual, "attn_layer_3")
		So(samples[0].Values, ShouldResemble, []float64{0.1, 0.2, 0.3})
		So(samples[0].Shape, ShouldResemble, []int{3})
	})
}
