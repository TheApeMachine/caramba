package ai

import (
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
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("NewAgent")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	agent := &Agent{
		AgentData: &AgentData{
			Context: NewContext(),
		},
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the agent data to JSON for reading
	agent.enc.Encode(agent.AgentData)

	return agent
}

/*
Read implements io.Reader for Agent.

It reads from the internal context.
*/
func (agent *Agent) Read(p []byte) (n int, err error) {
	errnie.Debug("Agent.Read")

	if agent.out.Len() == 0 {
		if err = errnie.NewErrIO(agent.enc.Encode(agent.AgentData)); err != nil {
			return 0, err
		}
	}

	return agent.out.Read(p)
}

/*
Write implements io.Writer for Agent.

It writes to the internal context.
*/
func (agent *Agent) Write(p []byte) (n int, err error) {
	errnie.Debug("Agent.Write")

	// Reset the output buffer whenever we write new data
	if agent.out.Len() > 0 {
		agent.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = agent.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf AgentData
	if decErr := agent.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		agent.Context = buf.Context

		// Re-encode to the output buffer for subsequent reads
		if encErr := agent.enc.Encode(agent.AgentData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close implements io.Closer for Agent.

It closes the internal context.
*/
func (agent *Agent) Close() error {
	errnie.Debug("Agent.Close")
	return agent.Context.Close()
}
