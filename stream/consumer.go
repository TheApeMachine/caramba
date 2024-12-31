package stream

import (
	"fmt"

	"github.com/theapemachine/caramba/provider"
)

type State uint

const (
	StateUndetermined State = iota
	StateInKey
	StateInValue
	StateHasKey
	StateHasValue
	StateHasColon
	StateHasEscape
	StateInArray
	StateInArrayItem
	StateInObject
)

/*
Consumer takes an input event stream and dynamically parses streaming JSON chunks in real-time.
This allows for a human-readable format to be displayed to a user, while a streaming model is
generating a response for instant feedback, while still being able to accumulate the chunks
anywhere else in the application to be used as a fully structured format.
*/
type Consumer struct {
	state  State
	indent int
	stack  []State
}

/*
NewConsumer intializes a ready-to-go Consumer and returns a pointer reference to it.
It can take in a stream and will drain it, while simultaniously printing the
human-readable output.
*/
func NewConsumer() *Consumer {
	return &Consumer{indent: 0, stack: make([]State, 0)}
}

/*
Print the incoming stream while at the same time consuming it. This uses a relatively
simple state-machine to parse the structured JSON format, stripping away and ignoring
all structural characters, and just printing the keys and value inside. It respects
the original nesting levels, which makes for a very well structured output, with
significant noise-reduction.
*/
func (consumer *Consumer) Print(stream <-chan provider.Event, structured bool) {
	if !structured {
		for chunk := range stream {
			fmt.Print(chunk.Text)
		}
		return
	}

	for chunk := range stream {
		for _, char := range chunk.Text {
			switch consumer.state {
			case StateUndetermined:
				consumer.undetermined(char)
			case StateInKey:
				consumer.inKey(char)
			case StateInValue:
				consumer.inValue(char)
			case StateHasKey:
				consumer.hasKey(char)
			case StateHasValue:
				consumer.hasValue(char)
			case StateHasColon:
				consumer.hasColon(char)
			case StateHasEscape:
				consumer.hasEscape(char)
			case StateInArray:
				consumer.inArray(char)
			case StateInArrayItem:
				consumer.inArrayItem(char)
			case StateInObject:
				consumer.inObject(char)
			}
		}
	}

	// Reset state when stream ends
	consumer.state = StateUndetermined
	consumer.indent = 0
	consumer.stack = consumer.stack[:0]
}

/*
undetermined describes a state where we do not directly know where
we are within the structure, or if we are even in any structure yet.
*/
func (consumer *Consumer) undetermined(char rune) {
	switch char {
	case '"':
		consumer.state = StateInKey
	case ',':
		fmt.Print("\n")
		consumer.printIndent()
		consumer.state = StateUndetermined
	}
}

/*
inKey tells us that we are currently somewhere after an opening quote of
a key, and have not seen the closing quote yet.
*/
func (consumer *Consumer) inKey(char rune) {
	switch char {
	case '"':
		consumer.state = StateHasKey
	default:
		fmt.Print(string(char))
	}
}

/*
hasKey tell us that we have seen the closing quote, and that we have successfully
captured the key part of a key/value pair, but we have not seen enough yet to determine
what the following state will be.
*/
func (consumer *Consumer) hasKey(char rune) {
	switch char {
	case ':':
		fmt.Print(": ")
		consumer.state = StateHasColon
	default:
		fmt.Print(string(char))
	}
}

/*
inValue means we have encountered the opening quote of a string value, and have not seen
the closing quote yet.
*/
func (consumer *Consumer) inValue(char rune) {
	switch char {
	case '"':
		consumer.state = StateHasValue
	case '\\':
		consumer.state = StateHasEscape
	default:
		fmt.Print(string(char))
	}
}

/*
hasValue means we have seen the closing quote of a string value, and have successfully
captured the value part of the key/value pair.
*/
func (consumer *Consumer) hasValue(char rune) {
	switch char {
	case ',':
		fmt.Print("\n")
		consumer.printIndent()
		consumer.state = StateUndetermined
	case '}', ']':
		fmt.Print("\n")
		if len(consumer.stack) > 0 {
			consumer.indent--
			consumer.stack = consumer.stack[:len(consumer.stack)-1]
			consumer.printIndent()
			fmt.Print(string(char))
		} else {
			consumer.state = StateUndetermined
			consumer.indent = 0
			consumer.stack = consumer.stack[:0]
		}
	default:
		fmt.Print(string(char))
	}
}

/*
hasColon works almost like a waiting station of some kind, which allows
us to consume potential whitespace, or other non-printable characters,
until we see something that is more interesting again to move us forwards.
*/
func (consumer *Consumer) hasColon(char rune) {
	switch char {
	case '"':
		consumer.state = StateInValue
	case '{':
		fmt.Print("\n")
		consumer.indent++
		consumer.stack = append(consumer.stack, StateInObject)
		consumer.printIndent()
		consumer.state = StateUndetermined
	case '[':
		fmt.Print("\n")
		consumer.indent++
		consumer.stack = append(consumer.stack, StateInArray)
		consumer.printIndent()
		fmt.Print("- ")
		consumer.state = StateInArrayItem
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) hasEscape(char rune) {
	switch char {
	case '"':
		fmt.Print(string(char))
		consumer.state = StateInValue
	}
}

func (consumer *Consumer) inArray(char rune) {
	switch char {
	case '"':
		consumer.state = StateInArrayItem
	case '[':
		fmt.Print("[")
		consumer.indent++
		consumer.stack = append(consumer.stack, StateInArray)
	case ']':
		fmt.Print("]")
		if len(consumer.stack) > 0 {
			consumer.indent--
			consumer.stack = consumer.stack[:len(consumer.stack)-1]
		}
		consumer.state = StateUndetermined
	case ',':
		fmt.Print("\n")
		consumer.printIndent()
		fmt.Print("- ")
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) inArrayItem(char rune) {
	switch char {
	case ']':
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) inObject(char rune) {
	switch char {
	case '}':
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) printIndent() {
	for i := 0; i < consumer.indent; i++ {
		fmt.Print("  ")
	}
}
