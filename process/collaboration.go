package process

import (
	"github.com/theapemachine/caramba/utils"
)

// CollaborationState tracks the state of collaboration
type CollaborationState string

const (
	CollaborationStateAnalyzing  CollaborationState = "analyzing"
	CollaborationStateForming    CollaborationState = "forming"
	CollaborationStateDiscussing CollaborationState = "discussing"
	CollaborationStateDeciding   CollaborationState = "deciding"
	CollaborationStateCompleted  CollaborationState = "completed"
)

// Collaboration defines the structure for team collaboration
type Collaboration struct {
	State         CollaborationState `json:"state" jsonschema:"description=Current state of collaboration,enum=analyzing,forming,discussing,deciding,completed"`
	Task          Task               `json:"task" jsonschema:"description=The task being collaborated on"`
	Analysis      TaskAnalysis       `json:"analysis" jsonschema:"description=Analysis of the task requirements"`
	TeamFormation TeamFormation      `json:"team_formation" jsonschema:"description=Team formation details"`
	Discussion    []Message          `json:"discussion" jsonschema:"description=Ongoing discussion messages"`
	Decisions     []Decision         `json:"decisions" jsonschema:"description=Decisions made during collaboration"`
}

// Task represents the task being worked on
type Task struct {
	Description string `json:"description" jsonschema:"description=Description of the task"`
	StartTime   string `json:"start_time" jsonschema:"description=When the task started"`
	Deadline    string `json:"deadline,omitempty" jsonschema:"description=Optional deadline for the task"`
	Priority    int    `json:"priority" jsonschema:"description=Task priority (1-5)"`
}

// TaskAnalysis represents the AI's analysis of task requirements
type TaskAnalysis struct {
	Requirements []string `json:"requirements" jsonschema:"description=Identified task requirements"`
	Skills       []string `json:"skills" jsonschema:"description=Required skills for the task"`
	Constraints  []string `json:"constraints" jsonschema:"description=Identified constraints"`
	Complexity   int      `json:"complexity" jsonschema:"description=Estimated task complexity (1-5)"`
}

// TeamFormation represents the team structure
type TeamFormation struct {
	Roles        []Role   `json:"roles" jsonschema:"description=Required roles for the team"`
	Size         int      `json:"size" jsonschema:"description=Optimal team size"`
	Structure    string   `json:"structure" jsonschema:"description=Team structure (flat, hierarchical, etc.)"`
	Capabilities []string `json:"capabilities" jsonschema:"description=Required team capabilities"`
}

// Role represents a required team role
type Role struct {
	Title       string   `json:"title" jsonschema:"description=Role title"`
	Skills      []string `json:"skills" jsonschema:"description=Required skills for this role"`
	Description string   `json:"description" jsonschema:"description=Role description"`
	Importance  int      `json:"importance" jsonschema:"description=Role importance (1-5)"`
}

func (p *Collaboration) Name() string {
	return "collaboration"
}

func (p *Collaboration) Description() string {
	return "Defines the structure for team collaboration and formation"
}

func (p *Collaboration) Schema() interface{} {
	return utils.GenerateSchema[Collaboration]()
}
