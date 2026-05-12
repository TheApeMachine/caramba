package config

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewClerkConfig(t *testing.T) {
	Convey("Given NewClerkConfig", t, func() {
		Convey("When viper has no keys it should use defaults", func() {
			viper.Reset()

			clerkConfig := NewClerkConfig()

			So(clerkConfig.Active, ShouldBeTrue)
			So(clerkConfig.SecretKey, ShouldEqual, "")
			So(clerkConfig.AdminSubjectIDs, ShouldBeEmpty)
			So(clerkConfig.PrivilegedOrganizationSlug, ShouldEqual, "")
			So(clerkConfig.RequireAuth, ShouldBeTrue)
		})

		Convey("When viper sets overrides they should appear on the struct", func() {
			viper.Reset()

			viper.Set("clerk.active", true)
			viper.Set("clerk.secret_key", "sk_test_example")

			clerkConfig := NewClerkConfig()

			So(clerkConfig.Active, ShouldBeTrue)
			So(clerkConfig.SecretKey, ShouldEqual, "sk_test_example")
		})

		Convey("When admin_subject_ids is set SubjectHasElevatedAdminPrivileges should match subjects", func() {
			viper.Reset()

			viper.Set("clerk.admin_subject_ids", []string{"user_admin", " user_other "})

			clerkConfig := NewClerkConfig()

			So(clerkConfig.AdminSubjectIDs, ShouldResemble, []string{"user_admin", "user_other"})
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges("user_admin", "", ""),
				ShouldBeTrue,
			)
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges("user_guest", "", ""),
				ShouldBeFalse,
			)
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges("", "", ""),
				ShouldBeFalse,
			)
		})

		Convey("When privileged_organization_slug is set org admins should elevate", func() {
			viper.Reset()

			viper.Set("clerk.privileged_organization_slug", "caramba")

			clerkConfig := NewClerkConfig()

			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges(
					"user_any",
					"caramba",
					"org:admin",
				),
				ShouldBeTrue,
			)
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges(
					"user_any",
					"Caramba",
					"admin",
				),
				ShouldBeTrue,
			)
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges(
					"user_any",
					"caramba",
					"org:member",
				),
				ShouldBeFalse,
			)
			So(
				clerkConfig.SubjectHasElevatedAdminPrivileges(
					"user_any",
					"other-org",
					"org:admin",
				),
				ShouldBeFalse,
			)
		})
	})
}

func BenchmarkNewClerkConfig(b *testing.B) {
	viper.Reset()

	for b.Loop() {
		_ = NewClerkConfig()
	}
}
