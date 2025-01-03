package utils

import (
	"encoding/json"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/goombaio/namegenerator"
	"github.com/invopop/jsonschema"
	"github.com/neurosnap/sentences/english"
)

/*
JoinWith concatenates strings with a specified delimiter.
It's a convenience wrapper around strings.Join that provides a more readable interface.
*/
func JoinWith(delim string, args ...string) string {
	return strings.Join(args, delim)
}

/*
ReplaceWith performs template-style string replacement using {placeholder} syntax.
Each replacement is defined by a pair of strings where the first element is the
placeholder name and the second is its replacement value.
*/
func ReplaceWith(template string, args [][]string) string {
	for _, arg := range args {
		template = strings.ReplaceAll(template, "{"+arg[0]+"}", arg[1])
	}

	return template
}

/*
NewID generates a new UUID string.
It uses Google's UUID implementation to ensure uniqueness.
*/
func NewID() string {
	return uuid.New().String()
}

var existingNames = make([]string, 0)

/*
NewName generates a unique, readable name.
It maintains a list of previously generated names to ensure uniqueness
within the current session.
*/
func NewName() string {
	newName := namegenerator.NewNameGenerator(time.Now().UnixNano()).Generate()

	for _, name := range existingNames {
		if name == newName {
			return NewName()
		}
	}

	existingNames = append(existingNames, newName)
	return newName
}

/*
ExtractJSONBlocks finds and parses JSON objects from a string.
It specifically looks for JSON content within markdown-style code blocks
that are marked with the 'json' language identifier. This is particularly
useful when processing structured outputs from AI tools.
*/
func ExtractJSONBlocks(s string) []map[string]interface{} {
	// Extract blocks marked with json language identifier
	codeBlocks := ExtractCodeBlocks(s)

	var results []map[string]interface{}
	for _, blocks := range codeBlocks["json"] {
		if block := ParseJSON(blocks); block != nil {
			results = append(results, block)
		}
	}

	return results
}

/*
ExtractCodeBlocks parses markdown-style code blocks from a string.
Returns a map where keys are language identifiers and values are slices
of code blocks for that language.
*/
func ExtractCodeBlocks(s string) map[string][]string {
	// Match code blocks with language identifiers
	re := regexp.MustCompile("```([a-zA-Z0-9]+)\n([\\s\\S]*?)```")
	matches := re.FindAllStringSubmatch(s, -1)

	codeBlocks := make(map[string][]string)
	for _, match := range matches {
		if len(match) >= 3 {
			language := match[1]
			code := strings.TrimSpace(match[2])
			codeBlocks[language] = append(codeBlocks[language], code)
		}
	}

	return codeBlocks
}

/*
ParseJSON safely converts a JSON string into a map.
Returns nil if the input is not valid JSON, making it safe for parsing
potentially invalid input.
*/
func ParseJSON(s string) map[string]interface{} {
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(s), &result); err == nil {
		return result
	}
	return nil
}

/*
GenerateSchema creates a JSON schema for any type that implements jsonschema struct tags.
It uses the jsonschema reflector to generate a complete schema, with additional properties
disabled and direct type definitions (no references).
*/
func GenerateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var v T
	return reflector.Reflect(v)
}

/*
QuickWrap encapsulates content in XML-style tags for better context separation.
This is particularly useful when preparing content for LLM processing, as it helps
maintain clear boundaries between different content sections.
*/
func QuickWrap(tag, content string, indent int) string {
	return JoinWith("\n",
		Indent("<"+tag+">", indent),
		Indent(Reflow(content), indent+1),
		Indent("</"+tag+">", indent),
	)
}

/*
QuickWrapWithAttributes is a variant of QuickWrap that allows for additional attributes to be added to the tag.
*/
func QuickWrapWithAttributes(tag, content string, indent int, attributes map[string]string) string {
	attr := ""
	for key, value := range attributes {
		attr += " " + key + "=\"" + value + "\""
	}

	return JoinWith("\n",
		Indent("<"+tag+attr+">", indent),
		Indent(Reflow(content), indent+1),
		Indent("</"+tag+">", indent),
	)
}

/*
Indent adds a specified number of spaces to the beginning of a string.
This is useful for formatting output, especially when dealing with
structured data that needs to be visually separated.
*/
func Indent(content string, indent int) string {
	return strings.Repeat("\t", indent) + content
}

/*
Reflow restructures the sentences in the message, so that each sentence
ends with a newline, and each paragraph is separated by an empty line.
It also tries to keep each sentence under 80 characters, and if it's longer,
it will split it into multiple sentences, using a smart algorithm, to make sure
we do not end up with just one or two words on the new line, but a somewhat
balanced distribution of words.
*/
func Reflow(message string) (reflowed string) {
	// Split message into paragraphs based on empty lines
	paragraphs := SplitIntoParagraphs(message)

	var reflowedParagraphs []string
	for _, para := range paragraphs {
		// Split paragraph into sentences with improved regex
		sentences := SplitIntoSentences(para)

		// Wrap each sentence and collect lines
		var wrappedLines []string
		for _, sentence := range sentences {
			wrapped := WrapText(sentence, 80)
			wrappedLines = append(wrappedLines, wrapped...)
		}

		// Join sentences in the paragraph with newlines
		reflowedParagraphs = append(reflowedParagraphs, strings.Join(wrappedLines, "\n"))
	}

	// Join paragraphs with empty lines
	return strings.Join(reflowedParagraphs, "\n\n")
}

/*
Substitute replaces all the placeholders in the message with the actual values.
*/
func Substitute(message string, fragments map[string]string, indent int) string {
	for key, value := range fragments {
		message = Indent(strings.ReplaceAll(message, "{{"+key+"}}", value), indent)
	}

	return message
}

/*
StripXML strips all XML tags the agent might have added to the message. It uses
a regex to find all the tags and remove them.
*/
func StripXML(message string) string {
	pattern := regexp.MustCompile(`<[^>]*>`)
	return pattern.ReplaceAllString(message, "")
}

/*
SplitIntoParagraphs breaks text into logical paragraphs based on blank lines.
It trims whitespace and removes empty paragraphs from the result.
*/
func SplitIntoParagraphs(content string) []string {
	// Split on two or more newline characters
	paragraphs := regexp.MustCompile(`\n{2,}`).Split(content, -1)
	var result []string
	for _, para := range paragraphs {
		trimmed := strings.TrimSpace(para)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

/*
SplitIntoSentences divides text into individual sentences while preserving
punctuation. It uses the neurosnap/sentences package which implements the Punkt algorithm
for unsupervised multilingual sentence boundary detection.
*/
func SplitIntoSentences(para string) []string {
	// Initialize tokenizer with English training data
	tokenizer, err := english.NewSentenceTokenizer(nil)
	if err != nil {
		return []string{para}
	}

	// Tokenize the text into sentences
	sentences := tokenizer.Tokenize(para)

	// Convert to string slice
	var result []string
	for _, s := range sentences {
		if s.Text != "" {
			result = append(result, strings.TrimSpace(s.Text))
		}
	}

	return result
}

/*
WrapText formats text to fit within a specified character width.
It preserves whole words where possible and handles cases where
individual words exceed the specified width by breaking them.
*/
func WrapText(text string, width int) []string {
	words := strings.Fields(text)
	var wrappedLines []string
	var currentLine strings.Builder
	var currentLength int

	for _, word := range words {
		wordLength := len(word)
		if currentLength+wordLength+1 > width {
			// If the word itself is longer than width, break it
			if wordLength > width {
				// Simple hyphenation or break without hyphen
				for len(word) > width {
					wrappedLines = append(wrappedLines, word[:width])
					word = word[width:]
				}
				wrappedLines = append(wrappedLines, word)
			} else {
				// Add the current line and start a new one with this word
				wrappedLines = append(wrappedLines, currentLine.String())
				currentLine.Reset()
				currentLine.WriteString(word)
				currentLength = wordLength
			}
		} else {
			if currentLine.Len() > 0 {
				currentLine.WriteString(" ")
				currentLength += 1
			}
			currentLine.WriteString(word)
			currentLength += wordLength
		}
	}
	// Add any remaining words
	if currentLine.Len() > 0 {
		wrappedLines = append(wrappedLines, currentLine.String())
	}
	return wrappedLines
}
