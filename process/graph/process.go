package graph

/*
Process represents a hyper-graph-based thought or reasoning process.
*/
type Hypergraph struct {
	Nodes      []HyperNode   `json:"nodes" jsonschema:"description=Nodes in the hypergraph,required"`
	HyperEdges []HyperEdge   `json:"hyper_edges" jsonschema:"description=Edges connecting multiple nodes,required"`
	Clusters   []NodeCluster `json:"clusters" jsonschema:"description=Emergent groupings of nodes,required"`
}

type HyperNode struct {
	ID         string                 `json:"id" jsonschema:"required,description=Unique identifier for the node"`
	Content    interface{}            `json:"content" jsonschema:"description=The content or value of the node"`
	Properties map[string]interface{} `json:"properties" jsonschema:"description=Additional properties of the node"`
	Dimension  int                    `json:"dimension" jsonschema:"description=Dimensionality of the node"`
	Weight     float64                `json:"weight" jsonschema:"description=Importance or strength of the node"`
	Activation float64                `json:"activation" jsonschema:"description=Current activation level"`
}

type HyperEdge struct {
	ID       string   `json:"id" jsonschema:"required,description=Unique identifier for the edge"`
	NodeIDs  []string `json:"node_ids" jsonschema:"required,description=IDs of connected nodes"`
	Type     string   `json:"type" jsonschema:"description=Type of relationship"`
	Strength float64  `json:"strength" jsonschema:"description=Strength of the connection"`
	Context  string   `json:"context" jsonschema:"description=Context of the relationship"`
}

type NodeCluster struct {
	ID        string    `json:"id" jsonschema:"required,description=Unique identifier for the cluster"`
	NodeIDs   []string  `json:"node_ids" jsonschema:"required,description=IDs of nodes in the cluster"`
	Centroid  []float64 `json:"centroid" jsonschema:"description=Center point of the cluster"`
	Coherence float64   `json:"coherence" jsonschema:"description=Measure of cluster coherence"`
	Label     string    `json:"label" jsonschema:"description=Descriptive label for the cluster"`
}
