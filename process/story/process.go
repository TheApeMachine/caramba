package story

import "github.com/theapemachine/amsh/utils"

/*
Process creates a story from abstract insights.
*/
type Process struct {
	Themes      []Theme      `json:"themes" jsonschema:"required,description=Themes of the story"`
	Sequences   []Sequence   `json:"sequences" jsonschema:"required,description=Ordered sequences of story elements"`
	Connections []Connection `json:"connections" jsonschema:"required,description=Relationships between elements"`
	Analogies   []Analogy    `json:"analogies" jsonschema:"required,description=Analogies between elements"`
}

type Theme struct {
	ID          string   `json:"id" jsonschema:"required,description=Unique identifier for the theme"`
	Description string   `json:"description" jsonschema:"required,description=Description of the theme"`
	Elements    []string `json:"elements" jsonschema:"required,description=Elements related to the theme"`
}

type Sequence struct {
	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the sequence"`
	Description  string   `json:"description" jsonschema:"required,description=Sequence description"`
	OrderedSteps []string `json:"ordered_steps" jsonschema:"required,description=Steps in correct order"`
	Duration     string   `json:"duration" jsonschema:"required,description=Expected duration of sequence"`
	Dependencies []string `json:"dependencies" jsonschema:"required,description=What this sequence depends on"`
}

type Connection struct {
	ID          string  `json:"id" jsonschema:"required,description=Unique identifier for the connection"`
	FromID      string  `json:"from_id" jsonschema:"required,description=Source element ID"`
	ToID        string  `json:"to_id" jsonschema:"required,description=Target element ID"`
	Type        string  `json:"type" jsonschema:"required,description=Type of connection"`
	Description string  `json:"description" jsonschema:"required,description=Nature of the connection"`
	Strength    float64 `json:"strength" jsonschema:"required,description=Connection strength"`
}

type Analogy struct {
	ID            string  `json:"id" jsonschema:"required,description=Unique identifier for the analogy"`
	SourceConcept string  `json:"source_concept" jsonschema:"required,description=The familiar concept used in the analogy"`
	TargetConcept string  `json:"target_concept" jsonschema:"required,description=The abstract concept being explained"`
	Explanation   string  `json:"explanation" jsonschema:"required,description=Human-friendly explanation"`
	Strength      float64 `json:"strength" jsonschema:"required,description=How well the analogy fits"`
}

type AnalogyMap struct {
	ID              string   `json:"id" jsonschema:"required,description=Unique identifier for the analogy map"`
	AbstractConcept string   `json:"abstract_concept" jsonschema:"required,description=The complex concept being mapped"`
	FamiliarConcept string   `json:"familiar_concept" jsonschema:"required,description=The relatable concept"`
	Description     string   `json:"description" jsonschema:"required,description=How the mapping works"`
	Limitations     []string `json:"limitations" jsonschema:"required,description=Where the analogy breaks down"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
