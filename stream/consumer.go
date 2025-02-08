package stream

import (
	"fmt"

	"github.com/theapemachine/caramba/provider"
)

/*
State represents the current parsing state of the Consumer's JSON state machine.
It tracks the position and context within the JSON structure being processed,
enabling the consumer to properly format and handle nested structures.
*/
type State uint

/*
Consumer state constants define the possible states during JSON parsing.
These states form a state machine that tracks position and context while
processing JSON data:

	StateUndetermined: Initial state or between structural elements
	                  Used when the parser is looking for the next meaningful token

	StateInKey: Currently parsing a JSON object key
	           Active between opening and closing quotes of a key

	StateInValue: Currently parsing a JSON value
	             Active while processing string values between quotes

	StateHasKey: Just finished parsing a complete key
	            Waiting for a colon separator

	StateHasValue: Just finished parsing a complete value
	              Looking for comma or end of container

	StateHasColon: Found a colon separator after a key
	              Waiting for the start of a value

	StateHasEscape: Processing an escape sequence in a string
	               Handles special characters like \", \\, \n, etc.

	StateInArray: Currently within a JSON array
	            Processes array elements and formatting

	StateInArrayItem: Currently parsing an array item
	                Handles individual elements within arrays

	StateInObject: Currently within a JSON object
	             Processes object members and nested structures
*/
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
Consumer implements a streaming JSON parser that formats data in real-time.
It uses a state machine to track its position within the JSON structure,
enabling proper formatting and indentation of nested structures while
maintaining a clean, human-readable output format.

The Consumer is particularly useful for:
- Processing streaming JSON responses from LLMs
- Converting structured JSON into human-readable format
- Maintaining proper indentation in nested structures
- Handling both simple and complex JSON structures
- Supporting real-time output formatting
*/
type Consumer struct {
	state  State   // Current state in the parsing state machine
	indent int     // Current indentation level for nested structures
	stack  []State // Stack of states for handling nested structures
}

/*
NewConsumer creates and returns a new Consumer instance configured for
JSON stream processing.

The consumer starts with:
- Zero indentation level
- Empty state stack
- StateUndetermined initial state

Returns:

	*Consumer: A new Consumer instance ready for processing JSON streams
*/
func NewConsumer() *Consumer {
	return &Consumer{indent: 0, stack: make([]State, 0)}
}

/*
Print processes and formats an incoming event stream. It handles both structured
and unstructured content, providing appropriate formatting for each type.

For unstructured content:
- Directly outputs event text
- Preserves original formatting
- Filters empty events

For structured (JSON) content:
- Uses state machine for parsing
- Maintains proper indentation
- Formats nested structures
- Handles escape sequences
- Provides clean, readable output

Parameters:

	stream: Input channel of provider Events to process
	structured: Boolean flag indicating if content should be treated as JSON
*/
func (consumer *Consumer) Print(stream <-chan *provider.Event, structured bool) {
	if !structured {
		for chunk := range stream {
			if (chunk.Type == provider.EventChunk || chunk.Type == provider.EventStop) && chunk.Text != "" {
				fmt.Print(chunk.Text)
			}
		}
		return
	}

	for chunk := range stream {
		if chunk.Text != "" {
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

	// Reset state when stream ends
	consumer.state = StateUndetermined
	consumer.indent = 0
	consumer.stack = consumer.stack[:0]
}

/*
undetermined handles the initial state and transitions between structural elements.
This state serves as a decision point for determining the next parsing context.

State transitions:
- On '"': Moves to StateInKey (start of object key)
- On ',': Maintains state but adds newline and indentation
- Other characters: Maintains current state

Parameters:

	char: Current character being processed
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
inKey processes characters while parsing a JSON object key.
Accumulates key characters until the closing quote is found.

State transitions:
- On '"': Moves to StateHasKey (end of key)
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
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
hasKey handles the state after a complete key has been parsed.
Primarily looks for the colon separator that follows a key.

State transitions:
- On ':': Moves to StateHasColon
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
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
inValue processes characters while parsing a JSON string value.
Handles escape sequences and accumulates value characters.

State transitions:
- On '"': Moves to StateHasValue (end of value)
- On '\': Moves to StateHasEscape (start of escape sequence)
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
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
hasValue handles the state after a complete value has been parsed.
Determines what comes next in the JSON structure.

State transitions:
- On ',': Returns to StateUndetermined for next key-value pair
- On '}' or ']': Handles end of current container
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
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
hasColon processes what follows a colon separator in a key-value pair.
Determines the type and structure of the upcoming value.

State transitions:
- On '"': Moves to StateInValue (start of string value)
- On '{': Starts new object, increases indent
- On '[': Starts new array, increases indent
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
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

/*
hasEscape handles escape sequences within string values.
Processes special characters that follow a backslash.

State transitions:
- On '"': Returns to StateInValue with escaped quote
- Other characters: Handles other escape sequences

Parameters:

	char: Current character being processed
*/
func (consumer *Consumer) hasEscape(char rune) {
	switch char {
	case '"':
		fmt.Print(string(char))
		consumer.state = StateInValue
	}
}

/*
inArray processes characters while within a JSON array.
Handles array elements and maintains proper formatting.

State transitions:
- On '"': Moves to StateInArrayItem (start of string item)
- On '[': Starts nested array
- On ']': Ends current array
- On ',': Prepares for next array item
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
*/
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

/*
inArrayItem processes characters while parsing an array item.
Handles individual elements within arrays.

State transitions:
- On ']': Moves to StateUndetermined (end of array)
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
*/
func (consumer *Consumer) inArrayItem(char rune) {
	switch char {
	case ']':
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

/*
inObject processes characters while within a JSON object.
Handles object members and maintains proper structure.

State transitions:
- On '}': Moves to StateUndetermined (end of object)
- Other characters: Maintains state and outputs character

Parameters:

	char: Current character being processed
*/
func (consumer *Consumer) inObject(char rune) {
	switch char {
	case '}':
		consumer.state = StateUndetermined
	default:
		fmt.Print(string(char))
	}
}

/*
printIndent outputs the current level of indentation using spaces.
This method ensures consistent formatting of nested structures by:
- Using two spaces per indentation level
- Applying indentation at the start of new lines
- Maintaining visual hierarchy in the output
*/
func (consumer *Consumer) printIndent() {
	for i := 0; i < consumer.indent; i++ {
		fmt.Print("  ")
	}
}
