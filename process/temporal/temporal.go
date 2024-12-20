package temporal

import (
	"time"

	"github.com/theapemachine/amsh/utils"
)

/*
Process represents the evolution of thoughts over time.
*/
type Process struct {
	Timeline       []TimePoint     `json:"timeline" jsonschema:"title=Timeline,description=Sequence of thought states,required"`
	CausalChains   []CausalChain   `json:"causal_chains" jsonschema:"title=CausalChains,description=Cause-effect relationships over time,required"`
	EvolutionRules []EvolutionRule `json:"evolution_rules" jsonschema:"title=EvolutionRules,description=Patterns of state change,required"`
}

func NewProcess() *Process {
	return &Process{}
}

type CausalChain struct {
	ID       string     `json:"id" jsonschema:"title=ID,description=Unique identifier for causal chain,required"`
	EventIDs []string   `json:"event_ids" jsonschema:"title=EventIDs,description=IDs of events in chain,required"`
	Strength float64    `json:"strength" jsonschema:"title=Strength,description=Causal relationship strength,required"`
	Evidence []Evidence `json:"evidence" jsonschema:"title=Evidence,description=Supporting evidence,required"`
}

type Evidence struct {
	Type        string  `json:"type" jsonschema:"title=Type,description=Type of evidence,required"`
	Description string  `json:"description" jsonschema:"title=Description,description=Evidence description,required"`
	Confidence  float64 `json:"confidence" jsonschema:"title=Confidence,description=Confidence level,required"`
	Source      string  `json:"source" jsonschema:"title=Source,description=Evidence source,required"`
}

type TimePoint struct {
	Time   time.Time              `json:"time" jsonschema:"title=Time,description=Point in time,required"`
	State  map[string]interface{} `json:"state" jsonschema:"title=State,description=System state,required"`
	Delta  map[string]float64     `json:"delta" jsonschema:"title=Delta,description=State changes,required"`
	Events []Event                `json:"events" jsonschema:"title=Events,description=Events at this time,required"`
}

type Event struct {
	ID        string                 `json:"id" jsonschema:"title=ID,description=Unique identifier for event,required"`
	Type      string                 `json:"type" jsonschema:"title=Type,description=Type of event,required"`
	Data      map[string]interface{} `json:"data" jsonschema:"title=Data,description=Event data,required"`
	Timestamp time.Time              `json:"timestamp" jsonschema:"title=Timestamp,description=Event time,required"`
}

type EvolutionRule struct {
	ID          string    `json:"id" jsonschema:"title=ID,description=Unique identifier for the evolution rule,required"`
	Condition   Predicate `json:"condition" jsonschema:"title=Condition,description=Condition for the evolution rule,required"`
	Action      Transform `json:"action" jsonschema:"title=Action,description=Action to be taken,required"`
	Priority    int       `json:"priority" jsonschema:"title=Priority,description=Priority of the evolution rule,required"`
	Reliability float64   `json:"reliability" jsonschema:"title=Reliability,description=Reliability of the evolution rule,required"`
}

type Predicate struct {
	Type      string                 `json:"type" jsonschema:"title=Type,description=Type of the predicate,required"`
	Params    map[string]interface{} `json:"params" jsonschema:"title=Params,description=Parameters for the predicate,required"`
	Threshold float64                `json:"threshold" jsonschema:"title=Threshold,description=Threshold for the predicate,required"`
}

type Transform struct {
	Type      string                 `json:"type" jsonschema:"title=Type,description=Type of the transformation,required"`
	Params    map[string]interface{} `json:"params" jsonschema:"title=Params,description=Parameters for the transformation,required"`
	Magnitude float64                `json:"magnitude" jsonschema:"title=Magnitude,description=Magnitude of the transformation,required"`
}

func (ta *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
