package drknow

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestNewIdentity(t *testing.T) {
	Convey("Given a call to NewIdentity", t, func() {
		ctx := context.Background()
		identity := NewIdentity(ctx, "test_role", "test system")

		Convey("Then it should be properly initialized", func() {
			So(identity, ShouldNotBeNil)
			So(identity.System, ShouldEqual, "test system")
			So(identity.Role, ShouldEqual, "test_role")
			So(identity.Name, ShouldNotBeEmpty)
			So(identity.Params, ShouldNotBeNil)
			So(identity.conn, ShouldNotBeNil)
			So(identity.Ctx, ShouldEqual, ctx)
		})

		Convey("Then it should have initialized thread with system message", func() {
			So(identity.Params.Thread, ShouldNotBeNil)
			So(len(identity.Params.Thread.Messages), ShouldEqual, 1)
			So(identity.Params.Thread.Messages[0].Role, ShouldEqual, provider.RoleSystem)
			So(identity.Params.Thread.Messages[0].Content, ShouldEqual, "test system")
		})
	})
}

func TestIdentityString(t *testing.T) {
	Convey("Given an Identity instance", t, func() {
		identity := NewIdentity(context.Background(), "test_role", "test system")

		Convey("When converting to string", func() {
			result := identity.String()

			Convey("Then it should include name and role", func() {
				So(result, ShouldContainSubstring, "Name:")
				So(result, ShouldContainSubstring, identity.Name)
				So(result, ShouldContainSubstring, "Role:")
				So(result, ShouldContainSubstring, "test_role")
			})
		})
	})
}

func TestLoad(t *testing.T) {
	Convey("Given an Identity instance", t, func() {
		identity := NewIdentity(context.Background(), "test_role", "test system")

		Convey("When loading non-existent identity", func() {
			success := identity.load()

			Convey("Then it should return false", func() {
				So(success, ShouldBeFalse)
			})
		})
	})
}

func TestCreate(t *testing.T) {
	Convey("Given an Identity instance", t, func() {
		identity := NewIdentity(context.Background(), "test_role", "test system")

		Convey("When creating a new identity", func() {
			oldName := identity.Name
			identity.create()

			Convey("Then it should generate a new name", func() {
				So(identity.Name, ShouldNotEqual, oldName)
			})

			Convey("Then it should initialize params", func() {
				So(identity.Params, ShouldNotBeNil)
			})
		})

		Convey("When creating with invalid data", func() {
			identity.System = ""
			identity.create()

			Convey("Then it should set error", func() {
				So(identity.err, ShouldNotBeNil)
			})
		})
	})
}

func TestSave(t *testing.T) {
	Convey("Given an Identity instance", t, func() {
		identity := NewIdentity(context.Background(), "test_role", "test system")

		Convey("When saving without connection", func() {
			identity.save()

			Convey("Then it should handle gracefully", func() {
				So(identity.err, ShouldBeNil)
			})
		})
	})
}

func TestValidate(t *testing.T) {
	Convey("Given an Identity instance", t, func() {
		identity := NewIdentity(context.Background(), "test_role", "test system")

		Convey("When validating valid identity", func() {
			err := identity.Validate()

			Convey("Then it should pass validation", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When validating with missing system", func() {
			identity.System = ""
			err := identity.Validate()

			Convey("Then it should fail validation", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "system is required")
			})
		})

		Convey("When validating with missing name", func() {
			identity.Name = ""
			err := identity.Validate()

			Convey("Then it should fail validation", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "name is required")
			})
		})

		Convey("When validating with missing role", func() {
			identity.Role = ""
			err := identity.Validate()

			Convey("Then it should fail validation", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "role is required")
			})
		})

		Convey("When validating with missing params", func() {
			identity.Params = nil
			err := identity.Validate()

			Convey("Then it should fail validation", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "params are required")
			})
		})

		Convey("When validating with missing thread", func() {
			identity.Params.Thread = nil
			err := identity.Validate()

			Convey("Then it should fail validation", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "thread is required")
			})
		})
	})
}
