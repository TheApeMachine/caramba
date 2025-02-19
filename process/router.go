package process

import (
	"github.com/theapemachine/caramba/utils"
)

// Router defines the structure for team routing and collaboration
type Router struct {
	State      string       `json:"state" jsonschema:"description=Current state of the router,enum=new,analyzing,creating_teams,processing,completed,error"`
	Teams      []Team       `json:"teams" jsonschema:"description=The teams created for this task"`
	Error      string       `json:"error,omitempty" jsonschema:"description=Error message if something went wrong"`
	Discussion []Discussion `json:"discussion" jsonschema:"description=Ongoing team discussions"`
	Decisions  []Decision   `json:"decisions" jsonschema:"description=Team decisions made"`
	Milestones []Milestone  `json:"milestones" jsonschema:"description=Progress milestones"`
}

// Team represents a formed team
type Team struct {
	ID          string   `json:"id" jsonschema:"description=Team identifier"`
	Name        string   `json:"name" jsonschema:"description=Team name"`
	Description string   `json:"description" jsonschema:"description=Team description"`
	Members     []string `json:"members" jsonschema:"description=Team member IDs"`
	Roles       []string `json:"roles" jsonschema:"description=Required roles"`
	Status      string   `json:"status" jsonschema:"description=Current team status"`
}

// Discussion represents team discussions
type Discussion struct {
	ID        string    `json:"id" jsonschema:"description=Discussion identifier"`
	Topic     string    `json:"topic" jsonschema:"description=Discussion topic"`
	Messages  []Message `json:"messages" jsonschema:"description=Discussion messages"`
	Status    string    `json:"status" jsonschema:"description=Discussion status"`
	StartTime string    `json:"start_time" jsonschema:"description=When discussion started"`
	EndTime   string    `json:"end_time,omitempty" jsonschema:"description=When discussion ended"`
}

// Message represents a discussion message
type Message struct {
	AgentID   string `json:"agent_id" jsonschema:"description=ID of the agent sending the message"`
	Content   string `json:"content" jsonschema:"description=Message content"`
	Timestamp string `json:"timestamp" jsonschema:"description=When message was sent"`
	Type      string `json:"type" jsonschema:"description=Type of message"`
}

// Decision represents a team decision
type Decision struct {
	ID          string `json:"id" jsonschema:"description=Decision identifier"`
	Topic       string `json:"topic" jsonschema:"description=Decision topic"`
	Description string `json:"description" jsonschema:"description=Decision description"`
	Rationale   string `json:"rationale" jsonschema:"description=Reasoning behind the decision"`
	Votes       []Vote `json:"votes" jsonschema:"description=Team member votes"`
	Status      string `json:"status" jsonschema:"description=Decision status"`
	Timestamp   string `json:"timestamp" jsonschema:"description=When decision was made"`
}

// Vote represents a team member's vote
type Vote struct {
	AgentID   string `json:"agent_id" jsonschema:"description=ID of the voting agent"`
	Vote      string `json:"vote" jsonschema:"description=The actual vote"`
	Rationale string `json:"rationale" jsonschema:"description=Reasoning for the vote"`
}

// Milestone represents a team progress milestone
type Milestone struct {
	ID          string `json:"id" jsonschema:"description=Milestone identifier"`
	Description string `json:"description" jsonschema:"description=Milestone description"`
	Status      string `json:"status" jsonschema:"description=Milestone status"`
	Timestamp   string `json:"timestamp" jsonschema:"description=When milestone was reached"`
}

func NewRouter() *Router {
	return &Router{}
}

func (p *Router) Name() string {
	return "router"
}

func (p *Router) Description() string {
	return "Routes requests to appropriate teams and coordinates responses"
}

func (p *Router) Schema() interface{} {
	return utils.GenerateSchema[Router]()
}
