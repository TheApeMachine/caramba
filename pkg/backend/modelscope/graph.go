package modelscope

/*
NodeData holds arbitrary key-value metadata attached to a graph node or edge.
This mirrors the modelscope frontend's NodeData type exactly.
*/
type NodeData map[string]any

/*
Node represents a single layer or tensor in the model graph.
*/
type Node struct {
	ID    int        `json:"id"`
	Edges []int      `json:"edges"`
	Data  []NodeData `json:"data"`
}

/*
Edge represents a directed data-flow connection between two nodes.
*/
type Edge struct {
	Source string     `json:"source"`
	Target string     `json:"target"`
	ID     int        `json:"id"`
	Data   []NodeData `json:"data"`
}

/*
GraphData is the wire format the modelscope frontend consumes directly.
*/
type GraphData struct {
	Nodes    map[string]*Node `json:"nodes"`
	Edges    map[string]*Edge `json:"edges"`
	Settings Settings         `json:"settings"`
}

/*
Settings mirrors the modelscope Graph settings struct.
*/
type Settings struct {
	Epoch       string `json:"epoch"`
	EpochFormat string `json:"epochFormat"`
	Source      string `json:"source"`
	Target      string `json:"target"`
}

/*
Builder constructs a GraphData incrementally, assigning stable integer IDs.
*/
type Builder struct {
	graph      GraphData
	nodeCount  int
	edgeCount  int
}

/*
NewBuilder creates a Builder with sensible defaults for model inspection.
*/
func NewBuilder() *Builder {
	return &Builder{
		graph: GraphData{
			Nodes: make(map[string]*Node),
			Edges: make(map[string]*Edge),
			Settings: Settings{
				Epoch:       "layer",
				EpochFormat: "",
				Source:      "source",
				Target:      "target",
			},
		},
	}
}

/*
AddNode adds or updates a named node with additional metadata.
*/
func (builder *Builder) AddNode(name string, data NodeData) *Node {
	node, exists := builder.graph.Nodes[name]

	if !exists {
		node = &Node{ID: builder.nodeCount, Edges: []int{}}
		builder.nodeCount++
		builder.graph.Nodes[name] = node
	}

	if data != nil {
		node.Data = append(node.Data, data)
	}

	return node
}

/*
AddEdge adds a directed edge between source and target nodes, creating them
if absent, and registers the cross-references required by the renderer.
*/
func (builder *Builder) AddEdge(source, target string, data NodeData) {
	if source == target {
		builder.AddNode(source, data)
		return
	}

	src := builder.AddNode(source, nil)
	tgt := builder.AddNode(target, nil)

	key := source + "<>" + target
	edge, exists := builder.graph.Edges[key]

	if !exists {
		edge = &Edge{
			Source: source,
			Target: target,
			ID:     builder.edgeCount,
			Data:   []NodeData{},
		}
		builder.edgeCount++
		builder.graph.Edges[key] = edge
		src.Edges = append(src.Edges, tgt.ID)
		tgt.Edges = append(tgt.Edges, src.ID)
	}

	if data != nil {
		edge.Data = append(edge.Data, data)
	}
}

/*
Build returns the completed GraphData.
*/
func (builder *Builder) Build() GraphData {
	return builder.graph
}
