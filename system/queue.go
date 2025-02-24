package system

import (
	"encoding/json"
	"sync"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

var once sync.Once
var instance *Queue

type Queue struct {
	mu     sync.RWMutex
	agents map[string]*ai.Agent
	topics map[string][]*ai.Agent
}

func NewQueue() *Queue {
	once.Do(func() {
		instance = &Queue{
			agents: make(map[string]*ai.Agent),
			topics: make(map[string][]*ai.Agent),
		}
	})

	return instance
}

func (q *Queue) AddAgent(agent *ai.Agent) {
	q.mu.Lock()
	defer q.mu.Unlock()

	errnie.Info("🌟 "+agent.Identity.Name, "queue", "addAgent")
	q.agents[agent.Identity.ID] = agent
}

func (q *Queue) GetAgent(id string) *ai.Agent {
	q.mu.RLock()
	defer q.mu.RUnlock()

	return q.agents[id]
}

func (q *Queue) RemoveAgent(id string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	delete(q.agents, id)
}

func (q *Queue) SendMessage(artifact *datura.Artifact) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	errnie.Info("🌟 *Queue.SendMessage")

	decrypted, err := utils.DecryptPayload(artifact)

	if errnie.Error(err) != nil {
		return
	}

	var payload map[string]any
	err = json.Unmarshal(decrypted, &payload)

	if errnie.Error(err) != nil {
		return
	}

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRoleMessage:
		errnie.Info("📨 *Queue.SendMessage", "role", "message")
		if agent, ok := q.agents[payload["from"].(string)]; ok {
			agent.SendMessage(artifact)
		}

	case datura.ArtifactRoleTopic:
		errnie.Info("📨 *Queue.SendMessage", "role", "topic")

		if agents, ok := q.topics[payload["topic"].(string)]; ok {
			for _, agent := range agents {
				agent.SendMessage(artifact)
			}
		}

	case datura.ArtifactRoleBroadcast:
		errnie.Info("📨 *Queue.SendMessage", "role", "broadcast")

		for _, agent := range q.agents {
			agent.SendMessage(artifact)
		}
	}
}

func (q *Queue) Subscribe(topic string, agent *ai.Agent) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if _, ok := q.topics[topic]; !ok {
		q.topics[topic] = make([]*ai.Agent, 0)
	}

	for _, a := range q.topics[topic] {
		if a.Identity.ID == agent.Identity.ID {
			return
		}
	}

	q.topics[topic] = append(q.topics[topic], agent)
}

func (q *Queue) Unsubscribe(topic string, agent *ai.Agent) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if agents, ok := q.topics[topic]; ok {
		for i, a := range agents {
			if a.Identity.ID == agent.Identity.ID {
				q.topics[topic] = append(agents[:i], agents[i+1:]...)
				break
			}
		}
	}
}

func (q *Queue) GetTopicSubscribers(topic string) []*ai.Agent {
	q.mu.RLock()
	defer q.mu.RUnlock()

	if agents, ok := q.topics[topic]; ok {
		return agents
	}
	return []*ai.Agent{}
}

func (q *Queue) GetAllAgents() map[string]*ai.Agent {
	q.mu.RLock()
	defer q.mu.RUnlock()

	return q.agents
}

func (q *Queue) GetAllTopics() map[string][]*ai.Agent {
	q.mu.RLock()
	defer q.mu.RUnlock()

	return q.topics
}
