package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
)

func TestNewTask(t *testing.T) {
	Convey("Given a task protocol", t, func() {
		initiator := "test-initiator"
		participant := "test-participant"

		Convey("When creating a new task protocol", func() {
			spec := NewTask(initiator, participant)

			Convey("Then it should be properly initialized", func() {
				So(spec, ShouldNotBeNil)
			})

			Convey("When processing an initial question artifact", func() {
				// Initial question from initiator
				question := datura.New(
					datura.WithRole(datura.ArtifactRoleQuestion),
					datura.WithScope(datura.ArtifactScopeAquire),
				)
				question.SetMetaValue("from", initiator)
				question.SetMetaValue("to", participant)

				out, status := spec.Next(question, core.StatusUnknown)

				Convey("Then it should wait for acknowledgment", func() {
					So(status, ShouldEqual, core.StatusWaiting)
					So(out, ShouldNotBeNil)
					So(out.Role(), ShouldEqual, uint32(datura.ArtifactRoleQuestion))
					So(out.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
				})

				Convey("When receiving an acknowledgment", func() {
					// Acknowledgment from participant
					ack := datura.New(
						datura.WithRole(datura.ArtifactRoleAcknowledge),
						datura.WithScope(datura.ArtifactScopeAquire),
					)
					ack.SetMetaValue("from", participant)
					ack.SetMetaValue("to", initiator)

					out, status = spec.Next(ack, core.StatusUnknown)

					Convey("Then it should wait for acknowledgment", func() {
						So(status, ShouldEqual, core.StatusWaiting)
						So(out, ShouldNotBeNil)
						So(out.Role(), ShouldEqual, uint32(datura.ArtifactRoleAcknowledge))
						So(out.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
					})

					// Acknowledgment from initiator
					ack2 := datura.New(
						datura.WithRole(datura.ArtifactRoleAcknowledge),
						datura.WithScope(datura.ArtifactScopeAquire),
					)
					ack2.SetMetaValue("from", initiator)
					ack2.SetMetaValue("to", participant)

					out, status = spec.Next(ack2, core.StatusUnknown)

					Convey("Then it should be busy", func() {
						So(status, ShouldEqual, core.StatusBusy)
						So(out, ShouldNotBeNil)
						So(out.Role(), ShouldEqual, uint32(datura.ArtifactRoleAcknowledge))
						So(out.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
					})

					Convey("When moving to preflight", func() {
						// Question from participant
						preflight := datura.New(
							datura.WithRole(datura.ArtifactRoleQuestion),
							datura.WithScope(datura.ArtifactScopePreflight),
						)
						preflight.SetMetaValue("from", participant)
						preflight.SetMetaValue("to", initiator)

						out, status = spec.Next(preflight, core.StatusUnknown)

						Convey("Then it should wait for parameter acknowledgment", func() {
							So(status, ShouldEqual, core.StatusWaiting)
							So(out, ShouldNotBeNil)
							So(out.Role(), ShouldEqual, uint32(datura.ArtifactRoleQuestion))
							So(out.Scope(), ShouldEqual, uint32(datura.ArtifactScopePreflight))
						})
					})
				})
			})
		})
	})
}

func TestProtocolsMap(t *testing.T) {
	Convey("Given the protocols map", t, func() {
		Convey("When getting the task protocol factory", func() {
			taskProtocol := protocols["task"]

			Convey("Then it should be properly configured", func() {
				So(taskProtocol, ShouldNotBeNil)

				spec := taskProtocol("test-initiator", "test-participant")
				So(spec, ShouldNotBeNil)

				// Test the protocol behavior instead of implementation details
				question := datura.New(
					datura.WithRole(datura.ArtifactRoleQuestion),
					datura.WithScope(datura.ArtifactScopeAquire),
				)
				question.SetMetaValue("from", "test-initiator")
				question.SetMetaValue("to", "test-participant")

				out, status := spec.Next(question, core.StatusUnknown)
				So(status, ShouldEqual, core.StatusWaiting)
				So(out, ShouldNotBeNil)
				So(out.Role(), ShouldEqual, uint32(datura.ArtifactRoleQuestion))
				So(out.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
			})
		})
	})
}
