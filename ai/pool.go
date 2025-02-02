package ai

import "sync"

var poolInstance *AgentPool
var once sync.Once

type AgentPool struct {
	agents map[string]*Agent
}

func NewAgentPool() *AgentPool {
	once.Do(func() {
		poolInstance = &AgentPool{
			agents: make(map[string]*Agent),
		}
	})
	return poolInstance
}

func (p *AgentPool) AddAgent(agent *Agent) {
	p.agents[agent.Name] = agent
}

func (p *AgentPool) GetAgent(name string) *Agent {
	return p.agents[name]
}
