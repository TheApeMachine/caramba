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
)

/*
Consumer is a specialized logging type designed to handle streaming, chunked JSON strings.
It strips away the structural elements of the JSON while maintaining indentation levels,
resulting in human-readable output.
*/
type Consumer struct {
	state  State
	indent int
	color  func(string) string
}

func NewConsumer() *Consumer {
	return &Consumer{}
}

func (consumer *Consumer) Print(stream <-chan provider.Event) {
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
		consumer.state = StateHasColon
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) inValue(char rune) {
	switch char {
	case '"':
		fmt.Print("\n")
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
		consumer.state = StateUndetermined
	case '}':
		fmt.Print("\n")
		consumer.indent--
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

func (consumer *Consumer) hasColon(char rune) {
	switch char {
	case '"':
		consumer.state = StateInValue
	case ' ':
		consumer.state = StateHasColon
	case ',':
		fmt.Print("\n")
		consumer.state = StateUndetermined
	case '{':
		fmt.Print("\n")
		consumer.indent++
		consumer.state = StateUndetermined
	case '}':
		fmt.Print("\n")
		consumer.indent--
		consumer.state = StateUndetermined
	case '[':
		fmt.Print("\n")
		consumer.indent++
		fmt.Print("- ")
		consumer.state = StateInArray
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
	case ']':
		consumer.state = StateUndetermined
	case ',':
		fmt.Print("\n- ")
	case '{':
		fmt.Print("\n")
		consumer.indent++
	case '}':
		fmt.Print("\n")
		consumer.indent--
	default:
		fmt.Print(string(char))
	}
}
