package task

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRead(t *testing.T) {
	Convey("Given a new task", t, func() {
		task := NewTask(
			WithMessages(
				NewSystemMessage("Hello, world!"),
				NewUserMessage("test", "Hi, how are you?"),
			),
		)

		buf := bytes.NewBuffer([]byte{})

		Convey("When read", func() {
			n, err := io.Copy(buf, task)

			newTask := NewTask()
			json.Unmarshal(buf.Bytes(), newTask)

			So(err, ShouldBeNil)
			So(n, ShouldNotEqual, 0)
			So(newTask.ID, ShouldEqual, task.ID)
			So(newTask.Metadata, ShouldNotBeNil)
			So(newTask.History, ShouldNotBeNil)
			So(len(newTask.History), ShouldEqual, 2)

			var builder strings.Builder
			for _, part := range newTask.History[0].Parts {
				if part.GetType() == "text" {
					builder.WriteString(part.(*TextPart).Text)
				}
			}

			So(builder.String(), ShouldEqual, "Hello, world!")
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a new task", t, func() {
		task := NewTask(
			WithMessages(
				NewSystemMessage("Hello, world!"),
				NewUserMessage("test", "Hi, how are you?"),
			),
		)

		buf, err := json.Marshal(task)
		So(err, ShouldBeNil)

		Convey("It should be empty", func() {
			So(task.ID, ShouldNotBeNil)
			So(task.Metadata, ShouldNotBeNil)
			So(task.Artifacts, ShouldNotBeNil)

			Convey("When written to", func() {
				newTask := NewTask()
				n, err := io.Copy(newTask, bytes.NewBuffer(buf))
				So(err, ShouldBeNil)
				So(n, ShouldNotEqual, 0)
				So(newTask.History, ShouldNotBeNil)
				So(len(newTask.History), ShouldEqual, 2)

				var builder strings.Builder
				for _, part := range newTask.History[0].Parts {
					if part.GetType() == "text" {
						builder.WriteString(part.(*TextPart).Text)
					}
				}

				So(builder.String(), ShouldEqual, "Hello, world!")
			})
		})
	})
}
