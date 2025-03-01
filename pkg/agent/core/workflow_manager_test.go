package core

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// MockWorkflow is a mock workflow implementation
type MockWorkflow struct{}

func (w *MockWorkflow) Execute(ctx context.Context, input map[string]any) (map[string]any, error) {
	return map[string]any{"result": "mock result"}, nil
}

func (w *MockWorkflow) Name() string {
	return "MockWorkflow"
}

func (w *MockWorkflow) AddStep(name string, tool Tool, args map[string]any) Workflow {
	return w
}

func (w *MockWorkflow) AddConditionalStep(name string, condition string, tool Tool, args map[string]any) Workflow {
	return w
}

func (w *MockWorkflow) SetErrorHandler(handler func(error) error) Workflow {
	return w
}

func TestNewWorkflowManager(t *testing.T) {
	Convey("Given a need for a workflow manager", t, func() {
		Convey("When creating a new workflow manager", func() {
			manager := NewWorkflowManager()

			Convey("Then it should not be nil", func() {
				So(manager, ShouldNotBeNil)
			})

			Convey("Then it should have a nil workflow", func() {
				So(manager.workflow, ShouldBeNil)
			})
		})
	})
}

func TestSetWorkflowManager(t *testing.T) {
	Convey("Given a workflow manager", t, func() {
		manager := NewWorkflowManager()

		Convey("When setting a workflow", func() {
			mockWorkflow := &MockWorkflow{}
			manager.SetWorkflow(mockWorkflow)

			Convey("Then the workflow should be set", func() {
				So(manager.workflow, ShouldEqual, mockWorkflow)
			})
		})
	})
}
