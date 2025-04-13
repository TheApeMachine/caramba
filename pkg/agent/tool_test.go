package agent

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewAgentTool(t *testing.T) {
	Convey("Given a new agent tool", t, func() {
		tool := NewAgentTool()

		So(tool, ShouldNotBeNil)
	})
}

func TestNewCreateAgentTool(t *testing.T) {
	Convey("Given a create agent tool", t, func() {
		tool := NewCreateAgentTool()

		So(tool, ShouldNotBeNil)
	})
}

func TestUse(t *testing.T) {
	Convey("Given a create agent tool", t, func() {
		tool := NewCreateAgentTool()

		So(tool, ShouldNotBeNil)
	})
}
