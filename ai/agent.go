package ai

import (
	"os"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/process"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/errnie"
)

type AgentState uint

const (
	AgentStateIdle AgentState = iota
	AgentStateProcessing
	AgentStateCompleted
	AgentStateIterating
)

type AgentInterface interface {
	Stream(input *datura.Artifact) chan *datura.Artifact
	Process(input *datura.Artifact) chan *datura.Artifact
	SetProcess(processor process.Interface) *Agent
	GetProvider() provider.Interface
	GetState() AgentState
	SetState(state AgentState)
	SendMessage(from, to, content string, data interface{})
	IsIterating() bool
	SetIterating(iterating bool)
}

/*
Agent defines the Agent.
*/
type Agent struct {
	Identity  *Identity                `json:"identity"`
	Params    *provider.ProviderParams `json:"params"`
	Provider  provider.Interface       `json:"provider"`
	Processor process.Interface        `json:"processor"`
	Agents    map[string][]*Agent      `json:"agents"`
	Parent    *Agent                   `json:"parent"`
	Level     int                      `json:"level"`
	Teams     map[string][]*Agent      `json:"teams"`
	State     AgentState
	Messages  chan *datura.Artifact
}

func NewAgent(identity *Identity, agentTools []provider.Tool) *Agent {
	errnie.Info("🤖 "+identity.Name, "agent", "new")

	return &Agent{
		Identity: identity,
		Params: &provider.ProviderParams{
			Messages: []provider.Message{
				{
					Role: "system",
					Content: strings.Join(
						[]string{
							viper.GetViper().GetString("prompts.system"),
							identity.String(),
						},
						"\n\n",
					),
				},
			},
			Tools: agentTools,
		},
		Provider: provider.NewOpenAI(os.Getenv("OPENAI_API_KEY")),
		Agents:   make(map[string][]*Agent),
		Teams:    make(map[string][]*Agent),
		Level:    0,
		State:    AgentStateIdle,
		Messages: make(chan *datura.Artifact, 100),
	}
}

func (agent *Agent) AddContext(context string) {
	agent.Params.Messages = append(agent.Params.Messages, provider.Message{
		Role:    "assistant",
		Content: context,
	})
}

func (agent *Agent) SendMessage(artifact *datura.Artifact) {
	agent.Messages <- artifact
}

func (agent *Agent) SetProcess(processor process.Interface) *Agent {
	agent.Processor = processor
	return agent
}

func (agent *Agent) AddAgent(role string, identity map[string]interface{}, tools []provider.Tool) *Agent {
	errnie.Info("👾 "+agent.Identity.Name, "agent", "addAgent")

	newIdentity := NewIdentityFromMap(identity)
	newAgent := NewAgent(newIdentity, tools)

	if _, exists := agent.Agents[role]; !exists {
		agent.Agents[role] = make([]*Agent, 0)
	}
	agent.Agents[role] = append(agent.Agents[role], newAgent)
	return newAgent
}

func (agent *Agent) GetAgents() map[string][]*Agent {
	return agent.Agents
}

func (agent *Agent) GetAgentsByRole(role string) []*Agent {
	if agents, exists := agent.Agents[role]; exists {
		return agents
	}
	return []*Agent{}
}
