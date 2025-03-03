package process

/*
Workflow is a collection of processes that are executed in order.
This allows for a single agent to execute a more complex task by
breaking it down into smaller, more manageable processes.

For example, this is used for a planner to start with a high-level plan,
and then verify if the agent executes each step correctly.

Given that Workflow implements the StructuredOutput interface, it can
be used as a process in itself.
*/
type Workflow struct {
	iteration int
	processes []StructuredOutput
}

func NewWorkflow(processes []StructuredOutput) *Workflow {
	return &Workflow{
		processes: processes,
	}
}

func (workflow *Workflow) Name() string {
	// Return the name of the current index, or the last one if the iteration is out of bounds
	if workflow.iteration >= len(workflow.processes) {
		return workflow.processes[len(workflow.processes)-1].Name()
	}

	return workflow.processes[workflow.iteration].Name()
}

func (workflow *Workflow) Description() string {
	// Return the description of the current index, or the last one if the iteration is out of bounds
	if workflow.iteration >= len(workflow.processes) {
		return workflow.processes[len(workflow.processes)-1].Description()
	}

	return workflow.processes[workflow.iteration].Description()
}

func (workflow *Workflow) Schema() any {
	// Return the schema of the current index, or the last one if the iteration is out of bounds
	if workflow.iteration >= len(workflow.processes) {
		return workflow.processes[len(workflow.processes)-1].Schema()
	}

	schema := workflow.processes[workflow.iteration].Schema()
	workflow.iteration++
	return schema
}

func (workflow *Workflow) String() string {
	// Return the string of the current index, or the last one if the iteration is out of bounds
	if workflow.iteration >= len(workflow.processes) {
		return workflow.processes[len(workflow.processes)-1].String()
	}

	return workflow.processes[workflow.iteration].String()
}
