package manifest

import (
	"fmt"
)

/*
Compiler turns a parsed manifest document into an executable Graph.
Single entry point: project root and manifest path produce a Graph ready to Execute.
*/
type Compiler struct {
	parser   *Parser
	registry *OperationRegistry
}

/*
NewCompiler creates a Compiler anchored at projectRoot using the global operation registry.
*/
func NewCompiler(projectRoot string) *Compiler {
	return NewCompilerWithRegistry(projectRoot, globalRegistry)
}

/*
NewCompilerWithRegistry creates a Compiler with an explicit OperationRegistry.
*/
func NewCompilerWithRegistry(projectRoot string, operationRegistry *OperationRegistry) *Compiler {
	return &Compiler{
		parser:   NewParser(projectRoot),
		registry: operationRegistry,
	}
}

/*
Compile parses the manifest at path, resolves includes and variables,
and builds an executable Graph from the topology section.
*/
func (compiler *Compiler) Compile(path string) (*Graph, error) {
	document, err := compiler.parser.Parse(path)

	if err != nil {
		return nil, err
	}

	systemBlock, err := compiler.requireMap(document, "system")

	if err != nil {
		return nil, err
	}

	topology, err := compiler.requireMap(systemBlock, "topology")

	if err != nil {
		return nil, err
	}

	return compiler.buildGraph(topology)
}

/*
buildGraph constructs a Graph from a decoded topology map.
*/
func (compiler *Compiler) buildGraph(topology map[string]any) (*Graph, error) {
	graph := newGraph()

	nodesField, err := compiler.requireSequence(topology, "nodes")

	if err != nil {
		return nil, err
	}

	for _, rawEntry := range nodesField {
		nodeMap, ok := rawEntry.(map[string]any)

		if !ok {
			return nil, fmt.Errorf("compiler: node entry must be a mapping")
		}

		node, err := compiler.buildNode(nodeMap)

		if err != nil {
			return nil, err
		}

		graph.addNode(node)
	}

	err = graph.rebuildEdgesFromNodes()

	if err != nil {
		return nil, err
	}

	return graph, nil
}

/*
buildNode instantiates a single Node from its manifest map.
*/
func (compiler *Compiler) buildNode(manifest map[string]any) (*Node, error) {
	nodeID, err := compiler.requireString(manifest, "id")

	if err != nil {
		return nil, err
	}

	opID, err := compiler.requireString(manifest, "op")

	if err != nil {
		return nil, err
	}

	config, err := compiler.optionalConfigMap(manifest["config"])

	if err != nil {
		return nil, fmt.Errorf("compiler: node %q: %w", nodeID, err)
	}

	operationInstance, err := compiler.registry.Build(opID, config)

	if err != nil {
		return nil, fmt.Errorf("compiler: node %q: %w", nodeID, err)
	}

	inPorts, err := compiler.stringListFromField(manifest["in"])

	if err != nil {
		return nil, fmt.Errorf("compiler: node %q in: %w", nodeID, err)
	}

	outPorts, err := compiler.stringListFromField(manifest["out"])

	if err != nil {
		return nil, fmt.Errorf("compiler: node %q out: %w", nodeID, err)
	}

	return &Node{
		ID:  nodeID,
		Op:  operationInstance,
		In:  inPorts,
		Out: outPorts,
	}, nil
}

/*
requireMap returns mapping[key] as a map or an error.
*/
func (compiler *Compiler) requireMap(mapping map[string]any, key string) (map[string]any, error) {
	value, ok := mapping[key]

	if !ok {
		return nil, fmt.Errorf("compiler: missing required key %q", key)
	}

	out, ok := value.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("compiler: key %q must be a mapping, got %T", key, value)
	}

	return out, nil
}

/*
requireString returns mapping[key] as a string or an error.
*/
func (compiler *Compiler) requireString(mapping map[string]any, key string) (string, error) {
	value, ok := mapping[key]

	if !ok {
		return "", fmt.Errorf("compiler: missing required key %q", key)
	}

	text, ok := value.(string)

	if !ok {
		return "", fmt.Errorf("compiler: key %q must be a string, got %T", key, value)
	}

	return text, nil
}

/*
requireSequence returns mapping[key] as a slice or an error.
*/
func (compiler *Compiler) requireSequence(mapping map[string]any, key string) ([]any, error) {
	value, ok := mapping[key]

	if !ok {
		return nil, fmt.Errorf("compiler: missing required key %q", key)
	}

	sequence, ok := value.([]any)

	if !ok {
		return nil, fmt.Errorf("compiler: key %q must be a sequence, got %T", key, value)
	}

	return sequence, nil
}

/*
optionalConfigMap normalizes a config field to a map; nil input yields an empty map.
*/
func (compiler *Compiler) optionalConfigMap(configField any) (map[string]any, error) {
	if configField == nil {
		return map[string]any{}, nil
	}

	configMap, ok := configField.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("config must be a mapping, got %T", configField)
	}

	return configMap, nil
}

/*
stringListFromField decodes a YAML list of strings; nil yields nil.
*/
func (compiler *Compiler) stringListFromField(field any) ([]string, error) {
	if field == nil {
		return nil, nil
	}

	rawList, ok := field.([]any)

	if !ok {
		return nil, fmt.Errorf("must be a sequence, got %T", field)
	}

	out := make([]string, 0, len(rawList))

	for index, item := range rawList {
		text, ok := item.(string)

		if !ok {
			return nil, fmt.Errorf("element %d must be a string, got %T", index, item)
		}

		out = append(out, text)
	}

	return out, nil
}
