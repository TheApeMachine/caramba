package backend

import (
	"fmt"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
resolveTargets picks the IR nodes the backend should execute and
returns them in declaration order along with the runtime-facing
output names. The mapping comes from GraphModule.Config["outputs"]
when present (a map of runtime-name → IR-node-id); otherwise the
single sink node is exposed as "primary".
*/
func resolveTargets(
	module program.GraphModule, irGraph *ir.Graph,
) ([]*ir.Node, []string, error) {
	mapping, err := outputMapping(module)

	if err != nil {
		return nil, nil, err
	}

	if len(mapping) == 0 {
		sinks := irGraph.Sinks()

		if len(sinks) == 0 {
			return nil, nil, fmt.Errorf("manifest has no executable sinks")
		}

		return []*ir.Node{sinks[0]}, []string{"primary"}, nil
	}

	index, err := irGraph.Index()

	if err != nil {
		return nil, nil, err
	}

	names := orderedKeys(mapping)
	targets := make([]*ir.Node, len(names))

	for position, outputName := range names {
		nodeID := mapping[outputName]
		node := index.Node(nodeID)

		if node == nil {
			return nil, nil, fmt.Errorf(
				"output %q references unknown IR node %q",
				outputName,
				nodeID,
			)
		}

		targets[position] = node
	}

	return targets, names, nil
}

func outputMapping(module program.GraphModule) (map[string]string, error) {
	raw, ok := module.Config["outputs"]

	if !ok {
		return nil, nil
	}

	switch typed := raw.(type) {
	case map[string]string:
		out := make(map[string]string, len(typed))

		for key, value := range typed {
			out[key] = value
		}

		return out, nil
	case map[string]any:
		out := make(map[string]string, len(typed))

		for key, value := range typed {
			text, ok := value.(string)

			if !ok {
				return nil, fmt.Errorf("outputs.%s: must be a string IR node id, got %T", key, value)
			}

			out[key] = text
		}

		return out, nil
	}

	return nil, fmt.Errorf("outputs config must be a mapping, got %T", raw)
}

func orderedKeys(mapping map[string]string) []string {
	keys := make([]string, 0, len(mapping))

	for key := range mapping {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	return keys
}
