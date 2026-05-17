package op

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

type stubOperation struct{}

func (stubOperation) Execute(execContext Context) error { return nil }

func TestRegistry(t *testing.T) {
	Convey("Given a fresh operation registry", t, func() {
		registry := NewRegistry()

		Convey("It should look up registered operations by id", func() {
			So(registry.Register("test.noop", stubOperation{}), ShouldBeNil)
			operation, err := registry.Lookup("test.noop")
			So(err, ShouldBeNil)
			So(operation, ShouldNotBeNil)
		})

		Convey("It should refuse duplicate registrations", func() {
			So(registry.Register("test.noop", stubOperation{}), ShouldBeNil)
			err := registry.Register("test.noop", stubOperation{})
			So(err, ShouldNotBeNil)
		})

		Convey("It should report missing operations by id", func() {
			_, err := registry.Lookup(program.OperationID("missing"))
			So(err, ShouldNotBeNil)
		})

		Convey("IDs should return registered ops in lexical order", func() {
			So(registry.Register("b", stubOperation{}), ShouldBeNil)
			So(registry.Register("a", stubOperation{}), ShouldBeNil)
			ids := registry.IDs()
			So(ids, ShouldResemble, []program.OperationID{"a", "b"})
		})
	})
}
