package process

// import (
// 	"time"

// 	"github.com/theapemachine/amsh/utils"
// )

// // Memory represents the memory process that observes and stores information
// type Memory struct {
// 	Observations []Observation `json:"observations" jsonschema:"required,title=Observations,description=Information observed from agent conversations"`
// 	Connections  []Connection  `json:"connections" jsonschema:"required,title=Connections,description=Relationships between pieces of information"`
// 	Context      Context       `json:"context" jsonschema:"required,title=Context,description=Current context of the conversation"`
// 	Graph        Graph         `json:"graph" jsonschema:"required,title=Graph,description=Relational memories to remember using the graph database"`
// }

// func (memory *Memory) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "memory", utils.GenerateSchema[Memory]())
// }

// type Graph struct {
// 	Nodes  []Node `json:"nodes" jsonschema:"required,title=Nodes,description=Nodes in the graph"`
// 	Edges  []Edge `json:"edges" jsonschema:"required,title=Edges,description=Edges in the graph"`
// 	Cypher string `json:"cypher" jsonschema:"required,title=Cypher,description=Cypher query to run on the graph database"`
// }

// type Node struct {
// 	Id     string   `json:"id" jsonschema:"required,title=Id,description=Id of the node"`
// 	Labels []string `json:"labels" jsonschema:"required,title=Labels,description=Labels of the node"`
// }

// type Edge struct {
// 	Source       string `json:"source" jsonschema:"required,title=Source,description=Source node of the edge"`
// 	Target       string `json:"target" jsonschema:"required,title=Target,description=Target node of the edge"`
// 	Relationship string `json:"relationship" jsonschema:"required,title=Relationship,description=Relationship between the source and target nodes"`
// }

// // Observation represents a piece of information worth remembering
// type Observation struct {
// 	Content    string         `json:"content" jsonschema:"required,title=Content,description=The actual information being stored"`
// 	Source     string         `json:"source" jsonschema:"required,title=Source,description=Where or who the information came from"`
// 	Importance float64        `json:"importance" jsonschema:"required,title=Importance,description=How important this information is (0-1)"`
// 	Tags       []string       `json:"tags" jsonschema:"required,title=Tags,description=Categorical tags for the information"`
// 	Metadata   map[string]any `json:"metadata" jsonschema:"title=Metadata,description=Additional contextual information"`
// 	Timestamp  time.Time      `json:"timestamp" jsonschema:"required,title=Timestamp,description=When this was observed"`
// }

// // Connection represents a relationship between observations
// type ObservedConnection struct {
// 	SourceID      string    `json:"source_id" jsonschema:"required,title=SourceID,description=ID of the source observation"`
// 	TargetID      string    `json:"target_id" jsonschema:"required,title=TargetID,description=ID of the target observation"`
// 	Relationship  string    `json:"relationship" jsonschema:"required,title=Relationship,description=Type of relationship between the observations"`
// 	Strength      float64   `json:"strength" jsonschema:"required,title=Strength,description=Strength of the relationship (0-1)"`
// 	LastActivated time.Time `json:"last_activated" jsonschema:"required,title=LastActivated,description=When this connection was last reinforced"`
// }

// // Context represents the current state of conversation
// type Context struct {
// 	ActiveTopics []string               `json:"active_topics" jsonschema:"required,title=ActiveTopics,description=Currently discussed topics"`
// 	Participants []string               `json:"participants" jsonschema:"required,title=Participants,description=Active participants in the conversation"`
// 	State        map[string]interface{} `json:"state" jsonschema:"required,title=State,description=Current state of the conversation"`
// 	TimeFrame    TimeFrame              `json:"time_frame" jsonschema:"required,title=TimeFrame,description=Temporal context of the conversation"`
// }

// // TimeFrame represents temporal context
// type TimeFrame struct {
// 	Start    time.Time `json:"start" jsonschema:"required;title=Start;description=When this context began"`
// 	Duration string    `json:"duration" jsonschema:"required,title=Duration,description=How long this context has been active"`
// 	Phase    string    `json:"phase" jsonschema:"required,title=Phase,description=Current phase of the conversation"`
// }
