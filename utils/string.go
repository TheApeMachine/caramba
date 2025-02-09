package utils

import (
	"encoding/json"
	"regexp"
	"strings"
	"time"

	"slices"

	"github.com/google/uuid"
	"github.com/goombaio/namegenerator"
	"github.com/invopop/jsonschema"
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

	if slices.Contains(existingNames, newName) {
		return NewName()
	}

	existingNames = append(existingNames, newName)
	return newName
}

/*
ExtractCommands looks for any commands wrapped in <> and extracts both the main
commands, and any parameter key/value pairs.
*/
func ExtractCommands(s string) (string, map[string]string) {
	pattern := regexp.MustCompile(`<([^>]+)>`)
	matches := pattern.FindStringSubmatch(s)

	if len(matches) < 2 {
		return "", nil
	}

	command := matches[1]

	pattern = regexp.MustCompile(`(\w+)=([^>]+)`)
	matches = pattern.FindStringSubmatch(command)

	if len(matches) < 3 {
		return "", nil
	}

	parameters := make(map[string]string)
	for i := 2; i < len(matches); i += 2 {
		parameters[matches[i]] = matches[i+1]
	}

	return command, parameters
}

/*
ExtractJSONBlocks finds and parses JSON objects from a string.
It specifically looks for JSON content within markdown-style code blocks
that are marked with the 'json' language identifier. This is particularly
useful when processing structured outputs from AI tools.
*/
func ExtractJSONBlocks(s string) []map[string]interface{} {
	// Extract blocks marked with json language identifier
	re := regexp.MustCompile("```json\\s*\\n([\\s\\S]*?)```")
	matches := re.FindAllStringSubmatch(s, -1)

	var results []map[string]interface{}
	for _, match := range matches {
		if len(match) >= 2 {
			if block := ParseJSON(strings.TrimSpace(match[1])); block != nil {
				results = append(results, block)
			}
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
	// Updated regex: allow for Windows (\r\n) or Unix (\n) line breaks
	re := regexp.MustCompile("```([a-zA-Z0-9]+)[\\r\\n]+([\\s\\S]*?)```")
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
func GenerateSchema[T any]() string {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var v T
	schema := reflector.Reflect(v)

	buf, err := schema.MarshalJSON()

	if err != nil {
		return ""
	}

	return string(buf)
}

/*
QuickWrap encapsulates content in XML-style tags for better context separation.
This is particularly useful when preparing content for LLM processing, as it helps
maintain clear boundaries between different content sections.
*/
func QuickWrap(tag, content string, indent int) string {
	return Indent(JoinWith("\n",
		"",
		"["+tag+"]",
		Indent(content, indent+1),
		"[/"+tag+"]",
	), indent)
}

/*
Indent adds a specified number of spaces to the beginning of a string.
This is useful for formatting output, especially when dealing with
structured data that needs to be visually separated.
*/
func Indent(content string, indent int) string {
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		lines[i] = strings.Repeat("  ", indent) + line
	}
	return strings.Join(lines, "\n")
}
