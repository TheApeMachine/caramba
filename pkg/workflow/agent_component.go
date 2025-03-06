package workflow

import (
	"encoding/json"
	"io"

	"github.com/davecgh/go-spew/spew"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type AgentComponent struct {
	*core.JSONComponent
	agent *ai.Agent
}

func NewAgentComponent(agent *ai.Agent, inputStream, outputStream *PipeStream) *AgentComponent {
	return &AgentComponent{
		JSONComponent: core.NewJSONComponent(inputStream),
		agent:         agent,
	}
}

func (ac *AgentComponent) Read(p []byte) (int, error) {
	errnie.Debug("Reading from agent component")

	var v any
	if err := ac.ReadJSON(&v); err != nil {
		return 0, err
	}

	data, err := json.Marshal(v)
	if err != nil {
		return 0, err
	}

	spew.Dump("agent read data", data)

	if len(p) < len(data) {
		errnie.Error("Short buffer")
		return 0, io.ErrShortBuffer
	}

	return copy(p, data), nil
}

func (ac *AgentComponent) Write(p []byte) (int, error) {
	errnie.Debug("Writing to agent component", "len", len(p))
	errnie.Debug(string(p))

	var incoming core.Message
	if err := json.Unmarshal(p, &incoming); err != nil {
		return 0, err
	}

	ac.agent.Context.Messages = append(ac.agent.Context.Messages, &incoming)

	if err := ac.WriteJSON(ac.agent.Context); err != nil {
		return 0, err
	}

	return len(p), nil
}

func (ac *AgentComponent) Close() error {
	return ac.JSONComponent.Close()
}
