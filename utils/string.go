package utils

import (
	"encoding/json"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/goombaio/namegenerator"
	"github.com/invopop/jsonschema"
	"github.com/theapemachine/errnie"
)

func JoinWith(delim string, args ...string) string {
	return strings.Join(args, delim)
}

func ReplaceWith(template string, args [][]string) string {
	for _, arg := range args {
		template = strings.ReplaceAll(template, "{"+arg[0]+"}", arg[1])
	}

	return template
}

func NewID() string {
	return uuid.New().String()
}

var existingNames = make([]string, 0)

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
This is useful in the process of Structured Outputs for AI tools.
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
ExtractCodeBlocks extracts Markdown code blocks from a string,
and returns a map of language to code block.
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
ParseJSON safely parses a JSON string into a map
*/
func ParseJSON(s string) map[string]interface{} {
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(s), &result); err == nil {
		return result
	}
	return nil
}

/*
GenerateSchema is a generic function that generates the JSON schema for
an object that has jsonschema struct tags
*/
func GenerateSchema[T any]() string {
	var instance T
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(jsonschema.Reflect(&instance), "", "  ")
	}))
}

/*
QuickWrap wraps a string into faux-XML section tags, which helps the
LLM to have more clarity on a large, shared context.
*/
func QuickWrap(tag, content string) string {
	return JoinWith("\n",
		"<"+tag+">",
		"  "+content,
		"</"+tag+">",
	)
}
