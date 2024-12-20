package context

import "github.com/theapemachine/amsh/utils"

/*
Process grounds abstract insights in user's specific situation.
*/
type Process struct {
	Relevance       []RelevanceMap   `json:"relevance" jsonschema:"required,description=How insights apply to context"`
	Constraints     []Constraint     `json:"constraints" jsonschema:"required,description=Contextual limitations"`
	Opportunities   []Opportunity    `json:"opportunities" jsonschema:"required,description=Context-specific possibilities"`
	Recommendations []Recommendation `json:"recommendations" jsonschema:"required,description=Context-aware suggestions"`
}

type RelevanceMap struct {
	InsightID     string  `json:"insight_id" jsonschema:"required,description=ID of the abstract insight"`
	Applicability float64 `json:"applicability" jsonschema:"required,description=How relevant the insight is"`
	Adaptation    string  `json:"adaptation" jsonschema:"required,description=How to adapt the insight"`
	Context       string  `json:"context" jsonschema:"required,description=Specific contextual factors"`
}

type Constraint struct {
	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the constraint"`
	Type         string   `json:"type" jsonschema:"required,description=Type of constraint (technical, resource, time, etc)"`
	Description  string   `json:"description" jsonschema:"required,description=Human-readable description of the constraint"`
	Severity     float64  `json:"severity" jsonschema:"required,description=How limiting this constraint is"`
	Flexibility  float64  `json:"flexibility" jsonschema:"required,description=How negotiable this constraint is"`
	DeepInsights []string `json:"deep_insights" jsonschema:"required,description=Links to relevant deep analysis insights"`
}

type Opportunity struct {
	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the opportunity"`
	Type         string   `json:"type" jsonschema:"required,description=Type of opportunity (innovation, optimization, etc)"`
	Description  string   `json:"description" jsonschema:"required,description=Human-readable description of the opportunity"`
	Potential    float64  `json:"potential" jsonschema:"required,description=Estimated value of the opportunity"`
	Feasibility  float64  `json:"feasibility" jsonschema:"required,description=How achievable this opportunity is"`
	Requirements []string `json:"requirements" jsonschema:"required,description=What's needed to realize this opportunity"`
}

type Recommendation struct {
	ID            string   `json:"id" jsonschema:"required,description=Unique identifier for the recommendation"`
	Priority      int      `json:"priority" jsonschema:"required,description=Importance ranking"`
	Description   string   `json:"description" jsonschema:"required,description=Human-readable recommendation"`
	Rationale     string   `json:"rationale" jsonschema:"required,description=Why this is recommended"`
	Prerequisites []string `json:"prerequisites" jsonschema:"required,description=What needs to be in place first"`
	Impact        float64  `json:"impact" jsonschema:"required,description=Expected effect of this recommendation"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
