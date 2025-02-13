package persona

import (
	"github.com/theapemachine/caramba/utils"
)

type Agent struct {
	ToolRequest []string   `json:"tool_request" jsonschema:"description=Request access to a tool,enum=browser,enum=terminal"`
	Iterate     bool       `json:"iterate" jsonschema:"description=Whether you need to iterate"`
	ScratchPad  ScratchPad `json:"scratchpad" jsonschema:"description=Your scratchpad persisting between interations"`
	Messages    []Message  `json:"messages" jsonschema:"description=Any messages you want to send"`
	Inspect     string     `json:"inspect" jsonschema:"description=Allows you to inspect the system around you,enum=system,enum=agents,enum=topics"`
	Subscribe   string     `json:"subscribe" jsonschema:"description=Allows you to subscribe to a topic or create one if it does not exist"`
	FinalAnswer string     `json:"final_answer" jsonschema:"description=Your final answer"`
}

type ScratchPad struct {
	Thoughts []Thought `json:"thoughts" jsonschema:"description=Your thoughts"`
}

type Thought struct {
	Root     string   `json:"root" jsonschema:"description=A root thought"`
	Branches []Branch `json:"branches" jsonschema:"description=Thought branches related to the root thought"`
}

type Branch struct {
	Leaf string `json:"leaf" jsonschema:"description=A leaf thought"`
}

type Message struct {
	From    string `json:"from"`
	To      string `json:"to" jsonschema:"description=The agent or group you want to send the message to or leave empty to broadcast"`
	ReplyTo string `json:"reply_to" jsonschema:"description=Used if the reply needs to go somewhere else than back to you"`
	Subject string `json:"subject" jsonschema:"description=The subject of your message"`
	Body    string `json:"body" jsonschema:"description=The body of your message"`
}

func (agent *Agent) GenerateSchema() interface{} {
	return utils.GenerateGenericSchema[Agent]()
}

func (agent *Agent) Name() string {
	return "Agent"
}

func (agent *Agent) Description() string {
	return "An advanced AI Autonomous Agent"
}
