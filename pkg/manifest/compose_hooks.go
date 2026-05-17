package manifest

/*
Hooks that pkg/model/compose uses to build manifest.Graph values
from outside the manifest package. The Graph type's fields are
intentionally unexported (they back the YAML compiler), so external
package compose cannot append nodes or mark inputs directly. These
thin wrappers expose just what compose needs without widening the
Graph surface for unrelated callers.

Keeping the wrappers in the manifest package preserves the single
construction point for Graph values — anyone reading manifest.Graph
sees that nodes and inputs only ever come through addNode / the
internal externalInputs map, whether the source is a YAML file or
a compose pattern walk.
*/

/*
NewComposedGraph returns an empty Graph ready for incremental node
appends from compose patterns. Identical wire shape to the YAML
compiler's newGraph.
*/
func NewComposedGraph() *Graph {
	return newGraph()
}

/*
AddComposedNode appends a node produced by a compose pattern. Uses
the same validation as the YAML compiler's addNode — id uniqueness,
non-nil node, single output binding.
*/
func AddComposedNode(graph *Graph, node *Node) error {
	if err := graph.addNode(node); err != nil {
		return err
	}

	return nil
}

/*
MarkComposedInput marks `binding` as an external graph input so
ExternalInputs() returns it. compose calls this for each Hints.Inputs
entry before walking patterns.
*/
func MarkComposedInput(graph *Graph, binding string) {
	graph.externalInputs[binding] = true
}
