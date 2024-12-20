package development

import "github.com/theapemachine/amsh/utils"

/*
Architecture describes the architecture of the system.
*/
type Architecture struct {
	HighLevelArchitecture  HighLevelArchitecture  `json:"high_level" jsonschema:"title=High-Level Architecture,description=A high-level overview of the architecture,required"`
	StructuralArchitecture StructuralArchitecture `json:"structural" jsonschema:"title=Structural Architecture,description=Structural diagrams of the architecture,required"`
	BehavioralArchitecture BehavioralArchitecture `json:"behavioral" jsonschema:"title=Behavioral Architecture,description=Behavioral diagrams of the architecture,required"`
}

type HighLevelArchitecture struct {
	UseCaseDiagram      string `json:"use_case_diagram" jsonschema:"title=Use Case Diagram,description=The PlantUML use case diagram that describes the use cases that are to be implemented,required"`
	RequirementsDiagram string `json:"requirements_diagram" jsonschema:"title=Requirements Diagram,description=The PlantUML diagram that captures system requirements and dependencies,required"`
	DeploymentDiagram   string `json:"deployment_diagram" jsonschema:"title=Deployment Diagram,description=The PlantUML deployment diagram that details how components are distributed across environments,required"`
}

type StructuralArchitecture struct {
	ComponentDiagram string `json:"component_diagram" jsonschema:"title=Component Diagram,description=The PlantUML component diagram describing system components and dependencies,required"`
	ClassDiagram     string `json:"class_diagram" jsonschema:"title=Class Diagram,description=The PlantUML class diagram outlining classes, relationships, and structure,required"`
}

type BehavioralArchitecture struct {
	SequenceDiagram string `json:"sequence_diagram" jsonschema:"title=Sequence Diagram,description=The PlantUML sequence diagram describing the lifecycle and interaction order of system calls,required"`
	ActivityDiagram string `json:"activity_diagram" jsonschema:"title=Activity Diagram,description=The PlantUML activity diagram showing the flow of activities or control logic,required"`
	StateDiagram    string `json:"state_diagram" jsonschema:"title=State Diagram,description=The PlantUML state diagram that outlines the states and transitions within the system,required"`
}

func (architecture *Architecture) GenerateSchema() string {
	return utils.GenerateSchema[Architecture]()
}
