package manifest

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

var substitutionPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

/*
Parser loads a manifest file, resolves include directives and ${var.path}
interpolations, and returns the decoded document.

Include paths use dot-separated segments relative to the project root:

	!!include architecture.block.dba  →  <root>/architecture/block/dba.yml
	!include  architecture.block.dba  →  same (single or double bang both work)

Parameterised includes pass a scoped variable namespace to the included file:

	path: architecture.block.dba
	variables:
	  d_model: 512

Inside the included file, those variables are reachable as ${include.d_model}.
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
resolveNode walks the yaml.Node tree, handling include directives and ${} interpolation.

Two include forms are recognised:

 1. Scalar with tag ending in "include":
    !!include architecture.block.dba
    !include  architecture.block.dba

 2. Mapping with an "include" key (parameterised form):
    include: architecture.block.dba
    variables:
    d_model: 512
*/
func (parser *Parser) resolveNode(yamlNode *yaml.Node) (any, error) {
	if yamlNode.Kind == yaml.ScalarNode && strings.HasSuffix(yamlNode.Tag, "include") {
		return parser.loadIncludeTarget(yamlNode.Value, nil)
	}

	if yamlNode.Kind == yaml.MappingNode {
		if dotPath, vars, ok := parser.extractIncludeMapping(yamlNode); ok {
			return parser.loadIncludeTarget(dotPath, vars)
		}
		if count, indexVar, templateNode, ok := parser.extractRepeatMapping(yamlNode); ok {
			return parser.expandRepeat(count, indexVar, templateNode)
		}
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
		var sequence []any

		for _, child := range yamlNode.Content {
			element, err := parser.resolveNode(child)

			if err != nil {
				return nil, err
			}

			if child.Kind == yaml.MappingNode {
				if _, _, _, ok := parser.extractRepeatMapping(child); ok {
					if expSeq, ok := element.([]any); ok {
						sequence = append(sequence, expSeq...)
						continue
					}
				}
			}

			sequence = append(sequence, element)
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
extractIncludeMapping detects a parameterised include mapping of the form:

	include: some.dot.path
	variables:
	  key: value

Returns the dot-path, the resolved variables map, and true when found.
*/
func (parser *Parser) extractIncludeMapping(
	yamlNode *yaml.Node,
) (dotPath string, vars map[string]any, ok bool) {
	rawMap := make(map[string]*yaml.Node, len(yamlNode.Content)/2)

	for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
		key := yamlNode.Content[pairIndex].Value
		rawMap[key] = yamlNode.Content[pairIndex+1]
	}

	pathNode, hasInclude := rawMap["include"]

	if !hasInclude || pathNode.Kind != yaml.ScalarNode {
		return "", nil, false
	}

	dotPath = pathNode.Value
	vars = make(map[string]any)

	varsNode, hasVars := rawMap["variables"]

	if !hasVars {
		return dotPath, vars, true
	}

	resolved, err := parser.resolveNode(varsNode)

	if err != nil {
		return "", nil, false
	}

	varMap, ok := resolved.(map[string]any)

	if !ok {
		return "", nil, false
	}

	maps.Copy(vars, varMap)

	return dotPath, vars, true
}

/*
extractRepeatMapping detects a repeat mapping of the form:

	repeat: 16
	index: i
	template:
	  - id: node_${i}

Returns the count, index variable name, template node, and true when found.
*/
func (parser *Parser) extractRepeatMapping(
	yamlNode *yaml.Node,
) (count int, indexVar string, templateNode *yaml.Node, ok bool) {
	rawMap := make(map[string]*yaml.Node, len(yamlNode.Content)/2)

	for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
		key := yamlNode.Content[pairIndex].Value
		rawMap[key] = yamlNode.Content[pairIndex+1]
	}

	repeatNode, hasRepeat := rawMap["repeat"]

	if !hasRepeat {
		return 0, "", nil, false
	}

	templateNode, hasTemplate := rawMap["template"]

	if !hasTemplate {
		return 0, "", nil, false
	}

	resolvedCount, err := parser.resolveNode(repeatNode)

	if err != nil {
		return 0, "", nil, false
	}

	switch v := resolvedCount.(type) {
	case int:
		count = v
	case float64:
		count = int(v)
	case string:
		fmt.Sscanf(v, "%d", &count)
	}

	indexNode, hasIndex := rawMap["index"]

	if hasIndex && indexNode.Kind == yaml.ScalarNode {
		indexVar = indexNode.Value
	} else {
		indexVar = "i"
	}

	return count, indexVar, templateNode, true
}

/*
expandRepeat evaluates the template node 'count' times, injecting the current
iteration index into the parser variables under 'indexVar'.
*/
func (parser *Parser) expandRepeat(count int, indexVar string, templateNode *yaml.Node) (any, error) {
	var result []any

	for i := 0; i < count; i++ {
		child := &Parser{
			root: parser.root,
			vars: make(map[string]any, len(parser.vars)+2),
		}

		maps.Copy(child.vars, parser.vars)
		child.vars[indexVar] = i
		child.vars["next_"+indexVar] = i + 1

		resolved, err := child.resolveNode(templateNode)

		if err != nil {
			return nil, err
		}

		if seq, ok := resolved.([]any); ok {
			result = append(result, seq...)
		} else {
			result = append(result, resolved)
		}
	}

	return result, nil
}

/*
loadIncludeTarget loads a dot-path as YAML and resolves it with a child parser
that inherits current vars plus an "include" namespace for any passed variables.
*/
func (parser *Parser) loadIncludeTarget(dotPath string, includeVars map[string]any) (any, error) {
	relativePath := strings.ReplaceAll(dotPath, ".", string(filepath.Separator)) + ".yml"
	includedPath := filepath.Join(parser.root, relativePath)

	childNode, err := parser.loadYAMLNode(includedPath)

	if err != nil {
		return nil, err
	}

	child := &Parser{
		root: parser.root,
		vars: make(map[string]any, len(parser.vars)+1),
	}

	maps.Copy(child.vars, parser.vars)

	if len(includeVars) > 0 {
		child.vars["include"] = includeVars
	}

	return child.resolveNode(childNode)
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

/*
ParseBytes parses a manifest from raw bytes instead of a file.
*/
func (parser *Parser) ParseBytes(data []byte) (map[string]any, error) {
	var documentNode yaml.Node
	if err := yaml.Unmarshal(data, &documentNode); err != nil {
		return nil, err
	}

	rawDocument, err := parser.nodeToAny(&documentNode)
	if err != nil {
		return nil, err
	}

	if document, ok := rawDocument.(map[string]any); ok {
		if variablesField, present := document["variables"]; present {
			if err := parser.mergeVars(variablesField); err != nil {
				return nil, err
			}
		}
		return document, nil
	}

	return nil, fmt.Errorf("manifest: root must be a mapping")
}
