package ai

import (
	"bufio"
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type AgentData struct {
	Context *Context `json:"context"`
}

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	*AgentData
	buffer *bufio.ReadWriter
	dec    *json.Decoder
	enc    *json.Encoder
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("NewAgent")

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	context := NewContext()

	agent := &Agent{
		AgentData: &AgentData{
			Context: context,
		},
		buffer: buffer,
		dec:    json.NewDecoder(buffer),
		enc:    json.NewEncoder(buffer),
	}

	return agent
}

/*
Read implements io.Reader for Agent.

It reads from the internal context.
*/
func (agent *Agent) Read(p []byte) (n int, err error) {
	errnie.Debug("Agent.Read")
	return agent.Context.Read(p)
}

/*
Write implements io.Writer for Agent.

It writes to the internal context.
*/
func (agent *Agent) Write(p []byte) (n int, err error) {
	errnie.Debug("Agent.Write", "p", string(p))
	return agent.Context.Write(p)
}

/*
Close implements io.Closer for Agent.

It closes the internal context.
*/
func (agent *Agent) Close() error {
	errnie.Debug("Agent.Close")

	// Close context
	if agent.Context != nil {
		agent.Context.Close()
	}

	agent.AgentData = nil
	return nil
}
