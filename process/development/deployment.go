package development

import "github.com/theapemachine/amsh/utils"

type Deployment struct {
	Environments       []Environment `json:"environments" jsonschema:"description=Deployment environments (e.g., production, staging)"`
	DeploymentStrategy string        `json:"deployment_strategy" jsonschema:"description=Strategy for deploying the system, e.g., blue-green, rolling deployment"`
}

type Environment struct {
	Name          string            `json:"name" jsonschema:"description=Environment name"`
	Configuration map[string]string `json:"configuration" jsonschema:"description=Key-value pairs of configuration settings"`
}

func (deployment *Deployment) GenerateSchema() string {
	return utils.GenerateSchema[Deployment]()
}
