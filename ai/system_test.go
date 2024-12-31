package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
)

func TestNewSystem(t *testing.T) {
	Convey("Given a new System initialization", t, func() {
		Convey("When created with structured=true", func() {
			identity := &Identity{
				Role: "assistant",
				Name: "TestBot",
			}

			// Setup viper config for test
			v := viper.New()
			v.Set("prompts.roles.assistant", "assistant")
			v.Set("prompts.instructions.thinking", "Think carefully")
			v.Set("prompts.instructions.schema", "Follow schema")
			v.Set("prompts.system.structured", "Structured {role} with {identity}")
			v.Set("prompts.system.unstructured", "Unstructured {role}")
			viper.Reset()
			viper.MergeConfigMap(v.AllSettings())

			sys := NewSystem(identity, "schema", true)

			Convey("It should have correct structured configuration", func() {
				So(sys.fragments["name"], ShouldEqual, identity.Name)
				So(sys.fragments["role"], ShouldEqual, "assistant")
				So(sys.fragments["prompt"], ShouldEqual, "Structured {role} with {identity}")
				So(sys.fragments["instructions"], ShouldEqual, "Follow schema")
			})
		})

		Convey("When created with structured=false", func() {
			identity := &Identity{
				Role: "assistant",
				Name: "TestBot",
			}

			// Setup viper config for test
			v := viper.New()
			v.Set("prompts.roles.assistant", "assistant")
			v.Set("prompts.instructions.thinking", "Think carefully")
			v.Set("prompts.system.unstructured", "Unstructured {role}")
			viper.Reset()
			viper.MergeConfigMap(v.AllSettings())

			sys := NewSystem(identity, "thinking", false)

			Convey("It should have correct unstructured configuration", func() {
				So(sys.fragments["name"], ShouldEqual, identity.Name)
				So(sys.fragments["role"], ShouldEqual, "assistant")
				So(sys.fragments["prompt"], ShouldEqual, "Unstructured {role}")
				So(sys.fragments["instructions"], ShouldEqual, "Think carefully")
			})
		})
	})
}

func TestSystemString(t *testing.T) {
	Convey("Given a System instance", t, func() {
		Convey("When converting to string with valid data", func() {
			identity := &Identity{
				Role: "assistant",
				Name: "TestBot",
			}

			// Setup viper config for test
			v := viper.New()
			v.Set("prompts.roles.assistant", "assistant")
			v.Set("prompts.instructions.thinking", "Think carefully")
			v.Set("prompts.system.unstructured", "I am a {role} named {name}")
			viper.Reset()
			viper.MergeConfigMap(v.AllSettings())

			sys := NewSystem(identity, "thinking", false)

			Convey("It should format the prompt correctly", func() {
				So(sys.fragments["prompt"], ShouldEqual, "I am a {role} named {name}")
				So(sys.fragments["role"], ShouldEqual, "assistant")
				So(sys.fragments["name"], ShouldEqual, "TestBot")
			})
		})

		Convey("When fragments are missing", func() {
			sys := &System{
				fragments: make(map[string]string),
			}

			Convey("It should have empty fragments", func() {
				So(sys.fragments["name"], ShouldEqual, "")
				So(sys.fragments["role"], ShouldEqual, "")
				So(sys.fragments["prompt"], ShouldEqual, "")
				So(sys.fragments["instructions"], ShouldEqual, "")
			})
		})
	})
}
