package manifest

import (
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

var substitutionPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

const (
	defaultMaxIncludes   = 128
	defaultMaxRepeat     = 4096
	defaultMaxExpansions = 65536
)

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
	root          string
	fileSystem    fs.FS
	maxIncludes   int
	maxRepeat     int
	maxExpansions int
}

/*
NewParser creates a Parser anchored at projectRoot for path resolution.
*/
func NewParser(projectRoot string) *Parser {
	root, err := filepath.Abs(projectRoot)

	if err != nil {
		root = projectRoot
	}

	return &Parser{
		root:          filepath.Clean(root),
		maxIncludes:   defaultMaxIncludes,
		maxRepeat:     defaultMaxRepeat,
		maxExpansions: defaultMaxExpansions,
	}
}

/*
WithFS switches the parser to read every file (including includes) from
the provided fs.FS instead of the operating system. The fs.FS is treated
as the root of all manifest resolution — caller-side sub-rooting (e.g.
fs.Sub(embedded, "template")) handles any prefix selection. Returns the
same parser to allow chaining.

Use this when manifests live in embed.FS (e.g. pkg/asset/template/...) so
that include directives can resolve sibling files inside the embedded
tree the same way they would on disk.
*/
func (parser *Parser) WithFS(fileSystem fs.FS) *Parser {
	parser.fileSystem = fileSystem
	parser.root = "."

	return parser
}

type parseContext struct {
	parser     *Parser
	vars       map[string]any
	stack      []string
	includes   int
	expansions int
}

func (parser *Parser) newContext() *parseContext {
	return &parseContext{
		parser: parser,
		vars:   make(map[string]any),
	}
}

/*
Parse loads path (relative to project root), merges variables, resolves includes,
and substitutes ${identifier.path} placeholders.
*/
func (parser *Parser) Parse(relativePath string) (map[string]any, error) {
	ctx := parser.newContext()
	absolutePath, err := parser.resolveRootPath(relativePath)

	if err != nil {
		return nil, err
	}

	documentNode, err := parser.loadYAMLNode(absolutePath)

	if err != nil {
		return nil, err
	}

	return ctx.resolveDocument(documentNode)
}

func (parser *Parser) resolveRootPath(relativePath string) (string, error) {
	if parser.fileSystem != nil {
		return parser.resolveFSPath(relativePath)
	}

	candidate := relativePath

	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(parser.root, candidate)
	}

	absolutePath, err := filepath.Abs(candidate)

	if err != nil {
		return "", err
	}

	cleanPath := filepath.Clean(absolutePath)
	relativeToRoot, err := filepath.Rel(parser.root, cleanPath)

	if err != nil {
		return "", err
	}

	if relativeToRoot == ".." || strings.HasPrefix(relativeToRoot, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("manifest: path %q escapes parser root %q", relativePath, parser.root)
	}

	return cleanPath, nil
}

/*
resolveFSPath resolves a manifest path against the parser's fs.FS root.
fs.FS uses forward slashes and disallows leading slashes or ".." escapes,
so we do all path math on the slash form, anchor to parser.root, and
reject anything that would climb above the root.
*/
func (parser *Parser) resolveFSPath(relativePath string) (string, error) {
	slashPath := filepath.ToSlash(relativePath)
	candidate := path.Clean(slashPath)

	if path.IsAbs(candidate) || strings.HasPrefix(candidate, "/") {
		return "", fmt.Errorf("manifest: absolute path %q not allowed with fs.FS", relativePath)
	}

	if parser.root != "" && parser.root != "." {
		candidate = path.Join(parser.root, candidate)
	}

	candidate = path.Clean(candidate)

	if candidate == ".." || strings.HasPrefix(candidate, "../") {
		return "", fmt.Errorf("manifest: path %q escapes parser root %q", relativePath, parser.root)
	}

	return candidate, nil
}

func (ctx *parseContext) resolveDocument(documentNode *yaml.Node) (map[string]any, error) {
	rawDocument, err := ctx.nodeToAny(documentNode)

	if err != nil {
		return nil, err
	}

	if document, ok := rawDocument.(map[string]any); ok {
		if variablesField, present := document["variables"]; present {
			err = ctx.mergeVars(variablesField)

			if err != nil {
				return nil, err
			}
		}
	}

	resolved, err := ctx.resolveNode(documentNode)

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
loadYAMLNode reads a file and parses it into a yaml.Node tree. When the
parser was built with WithFS, reads go through that fs.FS; otherwise the
operating system is used. The path encoding follows whichever source is
active (forward-slash for fs.FS, OS-native for os.ReadFile).
*/
func (parser *Parser) loadYAMLNode(absolutePath string) (*yaml.Node, error) {
	var (
		fileBytes []byte
		err       error
	)

	if parser.fileSystem != nil {
		fileBytes, err = fs.ReadFile(parser.fileSystem, absolutePath)
	} else {
		fileBytes, err = os.ReadFile(absolutePath)
	}

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
func (ctx *parseContext) nodeToAny(yamlNode *yaml.Node) (any, error) {
	switch yamlNode.Kind {
	case yaml.MappingNode:
		mapping := make(map[string]any, len(yamlNode.Content)/2)

		for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
			fieldName := yamlNode.Content[pairIndex].Value
			fieldValue, err := ctx.nodeToAny(yamlNode.Content[pairIndex+1])

			if err != nil {
				return nil, err
			}

			mapping[fieldName] = fieldValue
		}

		return mapping, nil

	case yaml.SequenceNode:
		sequence := make([]any, len(yamlNode.Content))

		for entryIndex, child := range yamlNode.Content {
			element, err := ctx.nodeToAny(child)

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
func (ctx *parseContext) resolveNode(yamlNode *yaml.Node) (any, error) {
	if err := ctx.recordExpansion(); err != nil {
		return nil, err
	}

	if yamlNode.Kind == yaml.ScalarNode && strings.HasSuffix(yamlNode.Tag, "include") {
		return ctx.loadIncludeTarget(yamlNode.Value, nil)
	}

	if yamlNode.Kind == yaml.MappingNode {
		if specNode, ok := ctx.extractSafeTensorsTopology(yamlNode); ok {
			return ctx.resolveSafeTensorsTopology(specNode)
		}

		if dotPath, vars, ok, err := ctx.extractIncludeMapping(yamlNode); ok || err != nil {
			if err != nil {
				return nil, err
			}

			return ctx.loadIncludeTarget(dotPath, vars)
		}
		if count, indexVar, offset, templateNode, ok, err := ctx.extractRepeatMapping(yamlNode); ok || err != nil {
			if err != nil {
				return nil, err
			}

			return ctx.expandRepeat(count, indexVar, offset, templateNode)
		}
	}

	switch yamlNode.Kind {
	case yaml.MappingNode:
		mapping := make(map[string]any, len(yamlNode.Content)/2)

		for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
			fieldName := yamlNode.Content[pairIndex].Value
			fieldValue, err := ctx.resolveNode(yamlNode.Content[pairIndex+1])

			if err != nil {
				return nil, err
			}

			mapping[fieldName] = fieldValue
		}

		return mapping, nil

	case yaml.SequenceNode:
		var sequence []any

		for _, child := range yamlNode.Content {
			element, err := ctx.resolveNode(child)

			if err != nil {
				return nil, err
			}

			if child.Kind == yaml.MappingNode {
				if _, _, _, _, ok, _ := ctx.extractRepeatMapping(child); ok {
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

		return ctx.interpolateString(text)
	}
}

func (ctx *parseContext) recordExpansion() error {
	ctx.expansions++

	if ctx.expansions > ctx.parser.maxExpansions {
		return fmt.Errorf("manifest: expansion limit exceeded (%d)", ctx.parser.maxExpansions)
	}

	return nil
}

/*
extractIncludeMapping detects a parameterised include mapping of the form:

	include: some.dot.path
	variables:
	  key: value

Returns the dot-path, the resolved variables map, and true when found.
*/
func (ctx *parseContext) extractIncludeMapping(
	yamlNode *yaml.Node,
) (dotPath string, vars map[string]any, ok bool, err error) {
	rawMap := make(map[string]*yaml.Node, len(yamlNode.Content)/2)

	for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
		key := yamlNode.Content[pairIndex].Value
		rawMap[key] = yamlNode.Content[pairIndex+1]
	}

	pathNode, hasInclude := rawMap["include"]

	if !hasInclude || pathNode.Kind != yaml.ScalarNode {
		return "", nil, false, nil
	}

	dotPath = pathNode.Value
	vars = make(map[string]any)

	varsNode, hasVars := rawMap["variables"]

	if !hasVars {
		return dotPath, vars, true, nil
	}

	resolved, err := ctx.resolveNode(varsNode)

	if err != nil {
		return "", nil, false, err
	}

	varMap, ok := resolved.(map[string]any)

	if !ok {
		return "", nil, false, fmt.Errorf("manifest: include variables must be a mapping, got %T", resolved)
	}

	maps.Copy(vars, varMap)

	return dotPath, vars, true, nil
}

/*
extractRepeatMapping detects a repeat mapping of the form:

	repeat: 16
	index: i
	template:
	  - id: node_${i}

Returns the count, index variable name, template node, and true when found.
*/
func (ctx *parseContext) extractRepeatMapping(
	yamlNode *yaml.Node,
) (
	count int,
	indexVar string,
	offset int,
	templateNode *yaml.Node,
	ok bool,
	err error,
) {
	rawMap := make(map[string]*yaml.Node, len(yamlNode.Content)/2)

	for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
		key := yamlNode.Content[pairIndex].Value
		rawMap[key] = yamlNode.Content[pairIndex+1]
	}

	repeatNode, hasRepeat := rawMap["repeat"]

	if !hasRepeat {
		return 0, "", 0, nil, false, nil
	}

	templateNode, hasTemplate := rawMap["template"]

	if !hasTemplate {
		return 0, "", 0, nil, false, fmt.Errorf("manifest: repeat block requires template")
	}

	resolvedCount, err := ctx.resolveNode(repeatNode)

	if err != nil {
		return 0, "", 0, nil, false, err
	}

	count, err = parseRepeatCount(resolvedCount)

	if err != nil {
		return 0, "", 0, nil, false, err
	}

	if count > ctx.parser.maxRepeat {
		return 0, "", 0, nil, false, fmt.Errorf("manifest: repeat count %d exceeds limit %d", count, ctx.parser.maxRepeat)
	}

	indexNode, hasIndex := rawMap["index"]

	if hasIndex && indexNode.Kind == yaml.ScalarNode {
		indexVar = indexNode.Value
	} else {
		indexVar = "i"
	}

	offsetNode, hasOffset := rawMap["offset"]

	if hasOffset {
		resolvedOffset, err := ctx.resolveNode(offsetNode)

		if err != nil {
			return 0, "", 0, nil, false, err
		}

		offset, err = parseRepeatCount(resolvedOffset)

		if err != nil {
			return 0, "", 0, nil, false, fmt.Errorf("manifest: repeat offset: %w", err)
		}
	}

	return count, indexVar, offset, templateNode, true, nil
}

func parseRepeatCount(value any) (int, error) {
	switch cast := value.(type) {
	case int:
		return cast, nil
	case float64:
		if cast != float64(int(cast)) {
			return 0, fmt.Errorf("manifest: repeat count must be an integer, got %v", cast)
		}

		return int(cast), nil
	case string:
		parsed, err := strconv.Atoi(cast)

		if err != nil {
			return 0, fmt.Errorf("manifest: repeat count must be an integer, got %q", cast)
		}

		return parsed, nil
	default:
		return 0, fmt.Errorf("manifest: repeat count must be an integer, got %T", value)
	}
}

/*
expandRepeat evaluates the template node 'count' times, injecting the current
iteration index into the parser variables under 'indexVar'.
*/
func (ctx *parseContext) expandRepeat(
	count int,
	indexVar string,
	offset int,
	templateNode *yaml.Node,
) (any, error) {
	if count < 0 {
		return nil, fmt.Errorf("manifest: repeat count must be non-negative")
	}

	var result []any

	for i := 0; i < count; i++ {
		child := &parseContext{
			parser:     ctx.parser,
			vars:       make(map[string]any, len(ctx.vars)+2),
			stack:      ctx.stack,
			includes:   ctx.includes,
			expansions: ctx.expansions,
		}

		maps.Copy(child.vars, ctx.vars)
		child.vars[indexVar] = i
		child.vars["next_"+indexVar] = i + 1
		child.vars["offset_"+indexVar] = i + offset
		child.vars["next_offset_"+indexVar] = i + offset + 1

		resolved, err := child.resolveNode(templateNode)
		ctx.expansions = child.expansions

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
func (ctx *parseContext) loadIncludeTarget(dotPath string, includeVars map[string]any) (any, error) {
	ctx.includes++

	if ctx.includes > ctx.parser.maxIncludes {
		return nil, fmt.Errorf("manifest: include limit exceeded (%d)", ctx.parser.maxIncludes)
	}

	relativePath := strings.ReplaceAll(dotPath, ".", string(filepath.Separator)) + ".yml"
	includedPath, err := ctx.parser.resolveRootPath(relativePath)

	if err != nil {
		return nil, err
	}

	for _, stackedPath := range ctx.stack {
		if stackedPath == includedPath {
			return nil, fmt.Errorf("manifest: include cycle detected at %s", includedPath)
		}
	}

	childNode, err := ctx.parser.loadYAMLNode(includedPath)

	if err != nil {
		return nil, err
	}

	child := &parseContext{
		parser:     ctx.parser,
		vars:       make(map[string]any, len(ctx.vars)+1),
		stack:      append(append([]string(nil), ctx.stack...), includedPath),
		includes:   ctx.includes,
		expansions: ctx.expansions,
	}

	maps.Copy(child.vars, ctx.vars)

	if len(includeVars) > 0 {
		child.vars["include"] = includeVars
	}

	resolved, err := child.resolveNode(childNode)
	ctx.includes = child.includes
	ctx.expansions = child.expansions

	return resolved, err
}

/*
interpolateString replaces ${var.path} using parser.vars; missing keys are an error.

When the entire scalar is a single placeholder (e.g. just
"${generation.sequence_length}" with no surrounding text) the
resolved variable's native type is preserved — ints stay ints, floats
stay floats, lists stay lists. Mixed strings ("steps=${steps}") fall
through to fmt-style stringification because the result must concat
into a single string.
*/
func (ctx *parseContext) interpolateString(scalar string) (any, error) {
	matches := substitutionPattern.FindAllStringSubmatchIndex(scalar, -1)

	if len(matches) == 0 {
		return scalar, nil
	}

	if len(matches) == 1 && matches[0][0] == 0 && matches[0][1] == len(scalar) {
		placeholderKey := strings.TrimSpace(scalar[matches[0][2]:matches[0][3]])
		raw, found := ctx.lookupSubstitutionRaw(placeholderKey)

		if !found {
			return nil, fmt.Errorf("manifest: undefined variable reference ${%s}", placeholderKey)
		}

		return raw, nil
	}

	var builder strings.Builder
	writeOffset := 0

	for _, matchIndexes := range matches {
		builder.WriteString(scalar[writeOffset:matchIndexes[0]])

		placeholderKey := strings.TrimSpace(scalar[matchIndexes[2]:matchIndexes[3]])
		replacement, found := ctx.lookupSubstitutionValue(placeholderKey)

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
lookupSubstitutionRaw resolves a dot path against parser.vars and
returns the underlying value with its native type intact. Used for
single-placeholder scalars where preserving int/float/list types
matters (default_shape, count, dimensions, etc.).
*/
func (ctx *parseContext) lookupSubstitutionRaw(dotPath string) (any, bool) {
	segments := strings.SplitN(dotPath, ".", 2)
	raw, ok := ctx.vars[segments[0]]

	if !ok {
		return nil, false
	}

	if len(segments) == 1 {
		return raw, true
	}

	nested, ok := raw.(map[string]any)

	if !ok {
		return nil, false
	}

	child := &parseContext{parser: ctx.parser, vars: nested}

	return child.lookupSubstitutionRaw(segments[1])
}

/*
lookupSubstitutionValue resolves a dot path against parser.vars.
*/
func (ctx *parseContext) lookupSubstitutionValue(dotPath string) (string, bool) {
	segments := strings.SplitN(dotPath, ".", 2)
	raw, ok := ctx.vars[segments[0]]

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

	child := &parseContext{parser: ctx.parser, vars: nested}

	return child.lookupSubstitutionValue(segments[1])
}

/*
mergeVars stores the variables block into parser.vars.
*/
func (ctx *parseContext) mergeVars(variablesField any) error {
	rawMap, ok := variablesField.(map[string]any)

	if !ok {
		return fmt.Errorf("manifest: variables must be a mapping, got %T", variablesField)
	}

	maps.Copy(ctx.vars, rawMap)

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

	if documentNode.Kind == yaml.DocumentNode && len(documentNode.Content) > 0 {
		return parser.newContext().resolveDocument(documentNode.Content[0])
	}

	return parser.newContext().resolveDocument(&documentNode)
}
