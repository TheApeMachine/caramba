package ai

import (
	"context"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
)

func TestNewAgentBuilder(t *testing.T) {
	Convey("Given agent builder options", t, func() {
		Convey("When creating a new agent builder with identity", func() {
			builder := NewAgentBuilder(
				WithIdentity("test-agent", "test-role"),
			)

			Convey("Then it should be properly initialized", func() {
				So(builder, ShouldNotBeNil)
				So(builder.ID(), ShouldNotBeEmpty)
			})
		})

		Convey("When creating a new agent builder with tools", func() {
			tools := []string{"tool1", "tool2"}
			builder := NewAgentBuilder(
				WithTools(tools...),
			)

			Convey("Then it should have the tools configured", func() {
				So(builder, ShouldNotBeNil)
				So(builder.tools, ShouldResemble, tools)
			})
		})

		Convey("When creating a new agent builder with params", func() {
			params := core.NewParamsBuilder()
			builder := NewAgentBuilder(
				WithParams(params),
			)

			Convey("Then it should have the params configured", func() {
				So(builder, ShouldNotBeNil)
				So(builder.paramsBuilder, ShouldEqual, params)
			})
		})

		Convey("When creating a new agent builder with context", func() {
			ctx := core.NewContextBuilder()
			builder := NewAgentBuilder(
				WithContext(ctx),
			)

			Convey("Then it should have the context configured", func() {
				So(builder, ShouldNotBeNil)
				So(builder.ctxBuilder, ShouldEqual, ctx)
			})
		})
	})
}

func TestAgentGenerate(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		builder := NewAgentBuilder(
			WithIdentity("test-agent", "test-role"),
			WithCancel(ctx),
		)

		Convey("When generating with a buffer channel", func() {
			buffer := make(chan *datura.Artifact, 1)
			out := builder.Generate(buffer)

			Convey("Then it should follow the protocol flow", func() {
				// Initial task from user to agent
				task := datura.New(
					datura.WithRole(datura.ArtifactRoleUser),
					datura.WithScope(datura.ArtifactScopeUnknown),
					datura.WithPayload([]byte("test")),
					datura.WithMeta("topic", "task"),
				)
				buffer <- task

				// Step 1: Agent (initiator) sends question to provider
				select {
				case result := <-out:
					So(result.Role(), ShouldEqual, uint32(datura.ArtifactRoleQuestion))
					So(result.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
					So(datura.GetMetaValue[string](result, "from"), ShouldEqual, builder.ID())
					So(datura.GetMetaValue[string](result, "to"), ShouldEqual, "provider")
				case <-time.After(200 * time.Millisecond):
					t.Error("timeout waiting for agent question")
				}

				// Step 2: Provider (participant) sends acknowledgment
				providerAck := datura.New(
					datura.WithRole(datura.ArtifactRoleAcknowledge),
					datura.WithScope(datura.ArtifactScopeAquire),
					datura.WithMediatype(datura.MediaTypeTextPlain),
					datura.WithMeta("topic", "task"),
					datura.WithMeta("from", "provider"),
					datura.WithMeta("to", builder.ID()),
				)
				buffer <- providerAck

				// Step 3: Agent (initiator) sends acknowledgment
				select {
				case result := <-out:
					So(result.Role(), ShouldEqual, uint32(datura.ArtifactRoleAcknowledge))
					So(result.Scope(), ShouldEqual, uint32(datura.ArtifactScopeAquire))
					So(datura.GetMetaValue[string](result, "from"), ShouldEqual, builder.ID())
					So(datura.GetMetaValue[string](result, "to"), ShouldEqual, "provider")
				case <-time.After(200 * time.Millisecond):
					t.Error("timeout waiting for agent acknowledgment")
				}

				// Step 4: Provider (participant) sends preflight question
				providerPreflight := datura.New(
					datura.WithRole(datura.ArtifactRoleQuestion),
					datura.WithScope(datura.ArtifactScopePreflight),
					datura.WithMediatype(datura.MediaTypeTextPlain),
					datura.WithMeta("topic", "task"),
					datura.WithMeta("from", "provider"),
					datura.WithMeta("to", builder.ID()),
				)
				buffer <- providerPreflight

				// Step 5: Agent (initiator) should respond with params and context
				select {
				case result := <-out:
					So(result.Role(), ShouldEqual, uint32(datura.ArtifactRoleAcknowledge))
					So(result.Scope(), ShouldEqual, uint32(datura.ArtifactScopeParams))
					So(datura.GetMetaValue[string](result, "from"), ShouldEqual, builder.ID())
					So(datura.GetMetaValue[string](result, "to"), ShouldEqual, "provider")
				case <-time.After(200 * time.Millisecond):
					t.Error("timeout waiting for agent params")
				}

				select {
				case result := <-out:
					So(result.Role(), ShouldEqual, uint32(datura.ArtifactRoleAcknowledge))
					So(result.Scope(), ShouldEqual, uint32(datura.ArtifactScopeContext))
					So(datura.GetMetaValue[string](result, "from"), ShouldEqual, builder.ID())
					So(datura.GetMetaValue[string](result, "to"), ShouldEqual, "provider")
				case <-time.After(200 * time.Millisecond):
					t.Error("timeout waiting for agent context")
				}
			})

			Convey("Then it should handle context cancellation", func() {
				cancel()
				time.Sleep(100 * time.Millisecond)
				So(builder.status, ShouldEqual, core.StatusUnknown)
			})
		})
	})
}
