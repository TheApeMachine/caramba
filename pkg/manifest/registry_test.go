package manifest

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type stubOp struct{}

func (stub *stubOp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return stateDict, nil
}

func TestOperationRegistry_Build(t *testing.T) {
	Convey("Given an OperationRegistry", t, func() {
		operationRegistry := NewOperationRegistry()

		Convey("Build", func() {
			Convey("It should build a registered operation", func() {
				operationRegistry.Register("test.stub", func(_ map[string]any) (operation.Operation, error) {
					return &stubOp{}, nil
				})

				built, err := operationRegistry.Build("test.stub", nil)
				So(err, ShouldBeNil)
				So(built, ShouldNotBeNil)

				testDict := state.NewDict()
				result, err := built.Forward(testDict)

				So(err, ShouldBeNil)
				So(result, ShouldNotBeNil)
			})

			Convey("It should return an error for an unknown operation id", func() {
				_, err := operationRegistry.Build("does.not.exist", nil)
				So(err, ShouldNotBeNil)
			})
		})
	})
}

func TestOperationRegistry_Registered(t *testing.T) {
	Convey("Given an OperationRegistry with two constructors", t, func() {
		operationRegistry := NewOperationRegistry()

		operationRegistry.Register("a.op", func(_ map[string]any) (operation.Operation, error) {
			return &stubOp{}, nil
		})

		operationRegistry.Register("b.op", func(_ map[string]any) (operation.Operation, error) {
			return &stubOp{}, nil
		})

		Convey("Registered", func() {
			Convey("It should return both registered identifiers", func() {
				ids := operationRegistry.Registered()
				So(ids, ShouldHaveLength, 2)
				So(ids, ShouldContain, "a.op")
				So(ids, ShouldContain, "b.op")
			})
		})
	})
}

func TestOperationRegistry_GlobalActivationTemplates(t *testing.T) {
	Convey("Given the global manifest operation registry", t, func() {
		activationIDs := []string{
			"activation.relu",
			"activation.gelu",
			"activation.sigmoid",
			"activation.tanh",
			"activation.swiglu",
			"activation.swish",
			"activation.selu",
			"activation.leaky_relu",
		}

		Convey("Build", func() {
			Convey("It should build every activation template operation", func() {
				for _, activationID := range activationIDs {
					built, err := globalRegistry.Build(activationID, nil)

					So(err, ShouldBeNil)
					So(built, ShouldNotBeNil)
				}
			})
		})
	})
}

func BenchmarkOperationRegistry_Build(b *testing.B) {
	operationRegistry := NewOperationRegistry()

	operationRegistry.Register("bench.stub", func(_ map[string]any) (operation.Operation, error) {
		return &stubOp{}, nil
	})

	b.ResetTimer()

	for b.Loop() {
		_, _ = operationRegistry.Build("bench.stub", nil)
	}
}
