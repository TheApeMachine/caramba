package persona

import (
	"github.com/theapemachine/amsh/utils"
)

type Teamlead struct {
	Executor     string        `json:"executor" jsonschema:"title=Executor,description=The executor to use for the team,enum=competition,enum=collaboration,enum=discussion,required"`
	Strategy     string        `json:"strategy" jsonschema:"title=Strategy,description=The recruitment strategy to use,enum=specialist,enum=generalist,enum=hybrid,required"`
	Agents       []Agent       `json:"agents" jsonschema:"title=Agents,description=The agents to use for the team,required"`
	Interactions []Interaction `json:"interactions,omitempty" jsonschema:"title=Interactions,description=How agents should interact during execution"`
}

type Agent struct {
	Name         string   `json:"name" jsonschema:"title=Name,description=The name of the agent,required"`
	Role         string   `json:"role" jsonschema:"title=Role,description=The role of the agent,enum=researcher,enum=developer,enum=analyst,enum=coordinator,required"`
	Workloads    []string `json:"workloads" jsonschema:"title=Workloads,description=The workloads to assign to the agent,required"`
	SystemPrompt string   `json:"system_prompt" jsonschema:"title=System Prompt,description=A detailed system prompt to use for the agent,required"`
	Dependencies []string `json:"dependencies,omitempty" jsonschema:"title=Dependencies,description=Other agents this agent depends on"`
}

type Interaction struct {
	Type              string   `json:"type" jsonschema:"title=Type,description=A short descriptive name for the interaction,required"`
	Agents            []string `json:"agents" jsonschema:"title=Agents,description=Agents involved in this interaction,required"`
	ProcessInParallel bool     `json:"process_in_parallel" jsonschema:"title=Process In Parallel,description=Whether the agents can process this in parallel or should process one after another"`
	Prompt            string   `json:"prompt" jsonschema:"title=Prompt,description=The prompt to use for the interaction,required"`
}

func (teamlead *Teamlead) GenerateSchema() string {
	return utils.GenerateSchema[Teamlead]()
}
