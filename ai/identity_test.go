package ai

import (
	"context"
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/datalake"
	"github.com/theapemachine/caramba/utils"
)

func TestNewIdentity(t *testing.T) {
	Convey("Given NewIdentity function", t, func() {
		ctx := context.Background()

		Convey("When creating a new identity with a role", func() {
			role := "test-role"
			identity := NewIdentity(ctx, role)

			Convey("Should initialize with correct values", func() {
				So(identity, ShouldNotBeNil)
				So(identity.Role, ShouldEqual, role)
				So(identity.Name, ShouldNotBeEmpty)
			})
		})

		Convey("When loading an existing identity", func() {
			role := "existing-role"
			existingIdentity := &Identity{
				Name: "TestBot",
				Role: role,
			}

			// Store the identity in datalake
			data, _ := json.Marshal(existingIdentity)
			datalake.NewConn().Put(ctx, "identities/"+role, data, nil)

			// Try to load it
			loadedIdentity := NewIdentity(ctx, role)

			Convey("Should load existing identity", func() {
				So(loadedIdentity, ShouldNotBeNil)
				So(loadedIdentity.Name, ShouldEqual, existingIdentity.Name)
				So(loadedIdentity.Role, ShouldEqual, existingIdentity.Role)
			})
		})

		Convey("When identity exists but has empty name", func() {
			role := "empty-name-role"
			emptyIdentity := &Identity{
				Name: "",
				Role: role,
			}

			// Store the identity in datalake
			data, _ := json.Marshal(emptyIdentity)
			datalake.NewConn().Put(ctx, "identities/"+role, data, nil)

			// Try to load it
			newIdentity := NewIdentity(ctx, role)

			Convey("Should create new identity", func() {
				So(newIdentity, ShouldNotBeNil)
				So(newIdentity.Name, ShouldNotBeEmpty)
				So(newIdentity.Role, ShouldEqual, role)
			})
		})
	})
}

func TestIdentityString(t *testing.T) {
	Convey("Given Identity String method", t, func() {
		Convey("When converting identity to string", func() {
			identity := &Identity{
				Name: "TestBot",
				Role: "assistant",
			}

			result := identity.String()

			Convey("Should format correctly", func() {
				expected := utils.JoinWith(
					"\n",
					"Name: TestBot",
					"Role: assistant",
				)
				So(result, ShouldEqual, expected)
			})
		})

		Convey("When identity has empty values", func() {
			identity := &Identity{
				Name: "",
				Role: "",
			}

			result := identity.String()

			Convey("Should still format correctly", func() {
				expected := utils.JoinWith(
					"\n",
					"Name: ",
					"Role: ",
				)
				So(result, ShouldEqual, expected)
			})
		})
	})
}
