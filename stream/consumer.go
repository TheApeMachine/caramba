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

type Consumer struct {
	state  State
	indent int
	stack  []State // To track nesting levels
}

func NewConsumer() *Consumer {
	return &Consumer{indent: 0, stack: make([]State, 0)}
}

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
}

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

func (consumer *Consumer) inKey(char rune) {
	switch char {
	case '"':
		consumer.state = StateHasKey
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) hasKey(char rune) {
	switch char {
	case ':':
		fmt.Print(": ")
		consumer.state = StateHasColon
	default:
		fmt.Print(string(char))
	}
}

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
		}
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

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
