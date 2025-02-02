package examples

type ExampleAgent interface {
	Run()
}

func NewExampleAgent(exampleAgentType ExampleAgent) ExampleAgent {
	return exampleAgentType
}
