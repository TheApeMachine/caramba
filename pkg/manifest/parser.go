package manifest

import (
	"maps"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

var substitutionPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

/*
Parser loads a manifest file, resolves !!include directives and ${var.path} interpolations,
and returns the decoded document.

Include paths use dot-separated segments relative to the project root, e.g.

	!!include architecture.block.dba  →  <root>/architecture/block/dba.yml
*/
type Parser struct {
	root string
	vars map[string]any
}

/*
NewParser creates a Parser anchored at projectRoot for path resolution.
*/
func NewParser(projectRoot string) *Parser {
	return &Parser{root: projectRoot, vars: make(map[string]any)}
}

/*
Parse loads path (relative to project root), merges variables, resolves includes,
and substitutes ${identifier.path} placeholders.
*/
func (parser *Parser) Parse(relativePath string) (map[string]any, error) {
	absolutePath := filepath.Join(parser.root, relativePath)

	documentNode, err := parser.loadYAMLNode(absolutePath)

	if err != nil {
		return nil, err
	}

	rawDocument, err := parser.nodeToAny(documentNode)

	if err != nil {
		return nil, err
	}

	if document, ok := rawDocument.(map[string]any); ok {
		if variablesField, present := document["variables"]; present {
			err = parser.mergeVars(variablesField)

			if err != nil {
				return nil, err
			}
		}
	}

	resolved, err := parser.resolveNode(documentNode)

	if err != nil {
		return nil, err
	}

	document, ok := resolved.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("manifest: root must be a mapping, got %T", resolved)
	}

	return document, nil
}

/*
loadYAMLNode reads a file and parses it into a yaml.Node tree.
*/
func (parser *Parser) loadYAMLNode(absolutePath string) (*yaml.Node, error) {
	fileBytes, err := os.ReadFile(absolutePath)

	if err != nil {
		return nil, fmt.Errorf("manifest: cannot read %s: %w", absolutePath, err)
	}

	var document yaml.Node

	err = yaml.Unmarshal(fileBytes, &document)

	if err != nil {
		return nil, fmt.Errorf("manifest: cannot parse %s: %w", absolutePath, err)
	}

	if document.Kind == yaml.DocumentNode && len(document.Content) > 0 {
		return document.Content[0], nil
	}

	return &document, nil
}

/*
nodeToAny converts a yaml.Node to Go values without resolving includes or substitutions.
*/
func (parser *Parser) nodeToAny(yamlNode *yaml.Node) (any, error) {
	switch yamlNode.Kind {
	case yaml.MappingNode:
		mapping := make(map[string]any, len(yamlNode.Content)/2)

		for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
			fieldName := yamlNode.Content[pairIndex].Value
			fieldValue, err := parser.nodeToAny(yamlNode.Content[pairIndex+1])

			if err != nil {
				return nil, err
			}

			mapping[fieldName] = fieldValue
		}

		return mapping, nil

	case yaml.SequenceNode:
		sequence := make([]any, len(yamlNode.Content))

		for entryIndex, child := range yamlNode.Content {
			element, err := parser.nodeToAny(child)

			if err != nil {
				return nil, err
			}

			sequence[entryIndex] = element
		}

		return sequence, nil

	default:
		var decoded any

		err := yamlNode.Decode(&decoded)

		if err != nil {
			return nil, err
		}

		return decoded, nil
	}
}

/*
resolveNode walks the yaml.Node tree, handling !!include tags and ${} interpolation.
*/
func (parser *Parser) resolveNode(yamlNode *yaml.Node) (any, error) {
	if yamlNode.Kind == yaml.ScalarNode && strings.HasSuffix(yamlNode.Tag, "include") {
		return parser.loadIncludeTarget(yamlNode.Value)
	}

	switch yamlNode.Kind {
	case yaml.MappingNode:
		mapping := make(map[string]any, len(yamlNode.Content)/2)

		for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
			fieldName := yamlNode.Content[pairIndex].Value
			fieldValue, err := parser.resolveNode(yamlNode.Content[pairIndex+1])

			if err != nil {
				return nil, err
			}

			mapping[fieldName] = fieldValue
		}

		return mapping, nil

	case yaml.SequenceNode:
		sequence := make([]any, len(yamlNode.Content))

		for entryIndex, child := range yamlNode.Content {
			element, err := parser.resolveNode(child)

			if err != nil {
				return nil, err
			}

			sequence[entryIndex] = element
		}

		return sequence, nil

	default:
		var decoded any

		err := yamlNode.Decode(&decoded)

		if err != nil {
			return nil, err
		}

		text, ok := decoded.(string)

		if !ok {
			return decoded, nil
		}

		return parser.interpolateString(text)
	}
}

/*
loadIncludeTarget loads a !!include dot-path as YAML and resolves it fully.
*/
func (parser *Parser) loadIncludeTarget(dotPath string) (any, error) {
	relativePath := strings.ReplaceAll(dotPath, ".", string(filepath.Separator)) + ".yml"
	includedPath := filepath.Join(parser.root, relativePath)

	childNode, err := parser.loadYAMLNode(includedPath)

	if err != nil {
		return nil, err
	}

	return parser.resolveNode(childNode)
}

/*
interpolateString replaces ${var.path} using parser.vars; missing keys are an error.
*/
func (parser *Parser) interpolateString(scalar string) (any, error) {
	matches := substitutionPattern.FindAllStringSubmatchIndex(scalar, -1)

	if len(matches) == 0 {
		return scalar, nil
	}

	var builder strings.Builder
	writeOffset := 0

	for _, matchIndexes := range matches {
		builder.WriteString(scalar[writeOffset:matchIndexes[0]])

		placeholderKey := strings.TrimSpace(scalar[matchIndexes[2]:matchIndexes[3]])
		replacement, found := parser.lookupSubstitutionValue(placeholderKey)

		if !found {
			return nil, fmt.Errorf("manifest: undefined variable reference ${%s}", placeholderKey)
		}

		builder.WriteString(replacement)
		writeOffset = matchIndexes[1]
	}

	builder.WriteString(scalar[writeOffset:])

	return builder.String(), nil
}

/*
lookupSubstitutionValue resolves a dot path against parser.vars.
*/
func (parser *Parser) lookupSubstitutionValue(dotPath string) (string, bool) {
	segments := strings.SplitN(dotPath, ".", 2)
	raw, ok := parser.vars[segments[0]]

	if !ok {
		return "", false
	}

	if len(segments) == 1 {
		return fmt.Sprintf("%v", raw), true
	}

	nested, ok := raw.(map[string]any)

	if !ok {
		return "", false
	}

	child := Parser{root: parser.root, vars: nested}

	return child.lookupSubstitutionValue(segments[1])
}

/*
mergeVars stores the variables block into parser.vars.
*/
func (parser *Parser) mergeVars(variablesField any) error {
	rawMap, ok := variablesField.(map[string]any)

	if !ok {
		return fmt.Errorf("manifest: variables must be a mapping, got %T", variablesField)
	}

	maps.Copy(parser.vars, rawMap)

	return nil
}
