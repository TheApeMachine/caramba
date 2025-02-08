package agent

/*
AgentStatus represents the current operational state of an agent.
It is used to track whether an agent is currently processing a request
or is available for new tasks.
*/
type AgentStatus uint

/*
Agent status constants define the possible states an agent can be in:

	AgentStatusIdle: The agent is not currently processing any requests
	AgentStatusBusy: The agent is actively processing a request
*/
const (
	AgentStatusIdle = iota // Agent is available for new tasks
	AgentStatusBusy        // Agent is currently processing a task
)
