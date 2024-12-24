package utils

import (
	"fmt"
	"strings"
)

const (
	colorReset = "\033[0m"
	colorKey   = "\033[36m" // Cyan for keys
	colorValue = "\033[0m"  // Default for values
)

/*
Consumer is a specialized logging type designed to handle streaming, chunked JSON strings.
It strips away the structural elements of the JSON while maintaining indentation levels,
resulting in human-readable output.
*/
type Consumer struct {
	indent       int
	inKey        bool
	hasKey       bool
	inValue      bool
	hasValue     bool
	inString     bool
	currentValue string
	currentKey   string
	contextStack []rune
}

func NewConsumer() *Consumer {
	return &Consumer{
		indent:       0, // Initialize indentation to 0
		contextStack: []rune{},
	}
}

func (consumer *Consumer) Print(stream <-chan string) {
	var inCodeBlock bool
	var codeBlockContent string
	var codeBlockDepth int

	for chunk := range stream {
		for _, char := range chunk {
			// Handle code blocks
			if inCodeBlock {
				if char == '{' {
					codeBlockDepth++
				} else if char == '}' {
					codeBlockDepth--
				}

				if codeBlockDepth == 0 {
					inCodeBlock = false
					// Print the collected code block content
					fmt.Print(Highlight(codeBlockContent))
					fmt.Println()
					fmt.Print(strings.Repeat("  ", consumer.indent))
				} else {
					codeBlockContent += string(char)
				}
				continue
			}

			switch char {
			case '{', '[':
				consumer.indent++
				fmt.Println()
				fmt.Print(strings.Repeat("  ", consumer.indent))
				consumer.resetFlags()
			case '}', ']':
				consumer.indent--
				if consumer.hasValue || consumer.hasKey {
					fmt.Println()
				}
				fmt.Print(strings.Repeat("  ", consumer.indent))
				consumer.resetFlags()
			case '"':
				consumer.handleQuote()
			case ':':
				if consumer.hasKey {
					// Check if we're entering a code block
					if consumer.currentKey == "javascript" || consumer.currentKey == "cypher" {
						inCodeBlock = true
						codeBlockDepth = 0
						codeBlockContent = ""
					}
					fmt.Print(": ")
					consumer.inValue = true
					consumer.hasKey = false
				}
			case ',':
				if consumer.hasValue || consumer.hasKey {
					fmt.Println()
					fmt.Print(strings.Repeat("  ", consumer.indent))
					consumer.resetFlags()
				}
			case ' ', '\t', '\n', '\r':
				if consumer.inValue && consumer.hasValue {
					fmt.Print(string(char))
				}
			default:
				if consumer.inValue {
					fmt.Print(Highlight(string(char)))
					consumer.hasValue = true
					consumer.currentValue += string(char)
				} else if consumer.inKey {
					fmt.Print(Blue(string(char)))
					consumer.hasKey = true
					consumer.currentKey += string(char)
				}
			}
		}
	}
	fmt.Println()
}

func (consumer *Consumer) handleOpenBracket(char rune) {
	fmt.Println(string(char))
	consumer.indent++
	consumer.contextStack = append(consumer.contextStack, char)
	fmt.Print(strings.Repeat("\t", consumer.indent))
}

func (consumer *Consumer) handleCloseBracket() {
	consumer.indent--
	if len(consumer.contextStack) > 0 {
		consumer.contextStack = consumer.contextStack[:len(consumer.contextStack)-1]
	}
	fmt.Println()
	fmt.Print(strings.Repeat("\t", consumer.indent))
	consumer.resetFlags()
}

func (consumer *Consumer) handleQuote() {
	if !consumer.inKey && !consumer.hasKey && !consumer.inValue && !consumer.hasValue {
		consumer.inKey = true
	} else if consumer.inKey {
		consumer.hasKey = true
	} else if consumer.hasKey && !consumer.inValue {
		consumer.inValue = true
	}
}

func (consumer *Consumer) handleComma() {
	if consumer.hasValue {
		consumer.resetFlags()
		fmt.Println(",")
		fmt.Print(strings.Repeat("\t", consumer.indent))
	}
}

func (consumer *Consumer) handleColon() {
	if consumer.hasKey {
		fmt.Print(": ")
		consumer.inValue = true
	}
}

func (consumer *Consumer) resetFlags() {
	consumer.inKey = false
	consumer.hasKey = false
	consumer.inValue = false
	consumer.hasValue = false
	consumer.currentValue = ""
	consumer.currentKey = ""
}
