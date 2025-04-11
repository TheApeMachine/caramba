package catalog

import "sync"

var once sync.Once
var catalog *Catalog

type Catalog struct {
	Agents []Agent
}

type Agent struct {
	Name        string
	Description string
	URL         string
}

func NewCatalog() *Catalog {
	once.Do(func() {
		catalog = &Catalog{}
	})

	return catalog
}

func (catalog *Catalog) AddAgent(agent *Agent) {
	catalog.Agents = append(catalog.Agents, *agent)
}

func (catalog *Catalog) GetAgent(name string) *Agent {
	for _, agent := range catalog.Agents {
		if agent.Name == name {
			return &agent
		}
	}

	return nil
}

func (catalog *Catalog) GetAgents() []Agent {
	return catalog.Agents
}
