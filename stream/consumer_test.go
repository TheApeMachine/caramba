package stream

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestNewConsumer(t *testing.T) {
	convey.Convey("Given a call to NewConsumer", t, func() {
		consumer := NewConsumer()

		convey.Convey("Then it should be properly initialized", func() {
			convey.So(consumer, convey.ShouldNotBeNil)
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
			convey.So(consumer.stack, convey.ShouldNotBeNil)
			convey.So(len(consumer.stack), convey.ShouldEqual, 0)
		})
	})
}

func TestPrint(t *testing.T) {
	convey.Convey("Given a Consumer instance", t, func() {
		consumer := NewConsumer()

		convey.Convey("When printing unstructured content", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, false)
				done <- true
			}()

			event := provider.NewEventData()
			event.Text = "plain text"
			stream <- event
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When printing structured content", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, true)
				done <- true
			}()

			event := provider.NewEventData()
			event.Text = `{"key": "value"}`
			stream <- event
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
		})
	})
}

func TestUndetermined(t *testing.T) {
	convey.Convey("Given a Consumer in undetermined state", t, func() {
		consumer := NewConsumer()

		convey.Convey("When encountering a quote", func() {
			consumer.undetermined('"')
			convey.So(consumer.state, convey.ShouldEqual, StateInKey)
		})

		convey.Convey("When encountering a comma", func() {
			consumer.undetermined(',')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})
	})
}

func TestInKey(t *testing.T) {
	convey.Convey("Given a Consumer in inKey state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateInKey

		convey.Convey("When encountering a quote", func() {
			consumer.inKey('"')
			convey.So(consumer.state, convey.ShouldEqual, StateHasKey)
		})

		convey.Convey("When encountering other characters", func() {
			consumer.inKey('a')
			convey.So(consumer.state, convey.ShouldEqual, StateInKey)
		})
	})
}

func TestHasKey(t *testing.T) {
	convey.Convey("Given a Consumer in hasKey state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateHasKey

		convey.Convey("When encountering a colon", func() {
			consumer.hasKey(':')
			convey.So(consumer.state, convey.ShouldEqual, StateHasColon)
		})
	})
}

func TestInValue(t *testing.T) {
	convey.Convey("Given a Consumer in inValue state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateInValue

		convey.Convey("When encountering a quote", func() {
			consumer.inValue('"')
			convey.So(consumer.state, convey.ShouldEqual, StateHasValue)
		})

		convey.Convey("When encountering an escape character", func() {
			consumer.inValue('\\')
			convey.So(consumer.state, convey.ShouldEqual, StateHasEscape)
		})
	})
}

func TestHasValue(t *testing.T) {
	convey.Convey("Given a Consumer in hasValue state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateHasValue

		convey.Convey("When encountering a comma", func() {
			consumer.hasValue(',')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When encountering a closing brace", func() {
			consumer.hasValue('}')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When encountering a closing bracket", func() {
			consumer.hasValue(']')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})
	})
}

func TestHasColon(t *testing.T) {
	convey.Convey("Given a Consumer in hasColon state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateHasColon

		convey.Convey("When encountering a quote", func() {
			consumer.hasColon('"')
			convey.So(consumer.state, convey.ShouldEqual, StateInValue)
		})

		convey.Convey("When encountering an opening brace", func() {
			consumer.hasColon('{')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(len(consumer.stack), convey.ShouldEqual, 1)
			convey.So(consumer.stack[0], convey.ShouldEqual, StateInObject)
		})

		convey.Convey("When encountering an opening bracket", func() {
			consumer.hasColon('[')
			convey.So(consumer.state, convey.ShouldEqual, StateInArrayItem)
			convey.So(len(consumer.stack), convey.ShouldEqual, 1)
			convey.So(consumer.stack[0], convey.ShouldEqual, StateInArray)
		})
	})
}

func TestHasEscape(t *testing.T) {
	convey.Convey("Given a Consumer in hasEscape state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateHasEscape

		convey.Convey("When encountering a quote", func() {
			consumer.hasEscape('"')
			convey.So(consumer.state, convey.ShouldEqual, StateInValue)
		})
	})
}

func TestInArray(t *testing.T) {
	convey.Convey("Given a Consumer in inArray state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateInArray

		convey.Convey("When encountering a quote", func() {
			consumer.inArray('"')
			convey.So(consumer.state, convey.ShouldEqual, StateInArrayItem)
		})

		convey.Convey("When encountering an opening bracket", func() {
			consumer.inArray('[')
			convey.So(len(consumer.stack), convey.ShouldEqual, 1)
			convey.So(consumer.stack[0], convey.ShouldEqual, StateInArray)
		})

		convey.Convey("When encountering a closing bracket", func() {
			consumer.inArray(']')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When encountering a comma", func() {
			consumer.inArray(',')
			convey.So(consumer.state, convey.ShouldEqual, StateInArray)
		})
	})
}

func TestInArrayItem(t *testing.T) {
	convey.Convey("Given a Consumer in inArrayItem state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateInArrayItem

		convey.Convey("When encountering a closing bracket", func() {
			consumer.inArrayItem(']')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When encountering other characters", func() {
			consumer.inArrayItem('a')
			convey.So(consumer.state, convey.ShouldEqual, StateInArrayItem)
		})
	})
}

func TestInObject(t *testing.T) {
	convey.Convey("Given a Consumer in inObject state", t, func() {
		consumer := NewConsumer()
		consumer.state = StateInObject

		convey.Convey("When encountering a closing brace", func() {
			consumer.inObject('}')
			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
		})

		convey.Convey("When encountering other characters", func() {
			consumer.inObject('a')
			convey.So(consumer.state, convey.ShouldEqual, StateInObject)
		})
	})
}

func TestPrintIndent(t *testing.T) {
	convey.Convey("Given a Consumer with various indent levels", t, func() {
		consumer := NewConsumer()

		convey.Convey("When indent is 0", func() {
			consumer.indent = 0
			consumer.printIndent()
			// Visual verification only as output goes to stdout
		})

		convey.Convey("When indent is positive", func() {
			consumer.indent = 2
			consumer.printIndent()
			// Visual verification only as output goes to stdout
		})
	})
}
