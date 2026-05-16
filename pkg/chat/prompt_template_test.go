package chat

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestPromptTemplate_Apply(test *testing.T) {
	Convey("Given a manifest prompt template", test, func() {
		template, err := newPromptTemplate("user: {{prompt}}\nassistant: ")
		So(err, ShouldBeNil)

		Convey("It should place user text without losing template whitespace", func() {
			So(template.Apply("hello"), ShouldEqual, "user: hello\nassistant: ")
		})
	})
}

func TestNewPromptTemplate(test *testing.T) {
	Convey("Given prompt template text", test, func() {
		Convey("It should accept an empty template", func() {
			template, err := newPromptTemplate("")

			So(err, ShouldBeNil)
			So(template.Apply("hello"), ShouldEqual, "hello")
		})

		Convey("It should require the prompt placeholder", func() {
			_, err := newPromptTemplate("user:")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "{{prompt}}")
		})
	})
}

func BenchmarkPromptTemplate_Apply(benchmark *testing.B) {
	template, err := newPromptTemplate("user: {{prompt}}\nassistant: ")

	if err != nil {
		benchmark.Fatal(err)
	}

	for index := 0; index < benchmark.N; index++ {
		_ = template.Apply("hello")
	}
}
