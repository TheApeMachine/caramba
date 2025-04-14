package resources

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/catalog"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// AgentResourceManager implements ResourceManager to expose A2A agents as MCP resources
type AgentResourceManager struct {
	catalog       *catalog.Catalog
	subscriptions map[string][]*Subscription
	subManager    *SubscriptionManager
	mu            sync.RWMutex
}

// NewAgentResourceManager creates a new resource manager for A2A agents
func NewAgentResourceManager(catalog *catalog.Catalog) *AgentResourceManager {
	return &AgentResourceManager{
		catalog:       catalog,
		subscriptions: make(map[string][]*Subscription),
		subManager:    NewSubscriptionManager(),
	}
}

// List returns all available agent resources
func (m *AgentResourceManager) List(ctx context.Context) ([]Resource, []ResourceTemplate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var resources []Resource

	// Get all agents from the catalog
	agents := m.catalog.GetAgents()

	// Convert each agent card to a resource
	for _, agentCard := range agents {
		uri := fmt.Sprintf("agent://%s", agentCard.Name)

		// Create a resource for the agent
		resource := Resource{
			URI:         uri,
			Name:        agentCard.Name,
			Description: agentCard.Description,
			MimeType:    "application/json",
			Type:        TextResource,
		}

		resources = append(resources, resource)
	}

	// Add a template for dynamic agent access
	templates := []ResourceTemplate{
		{
			URITemplate: "agent://{name}",
			Name:        "A2A Agent",
			Description: "Access A2A agent resources by name",
			MimeType:    "application/json",
			Type:        TextResource,
			Variables: []TemplateVariable{
				{
					Name:        "name",
					Description: "The name of the agent",
					Required:    true,
				},
			},
		},
	}

	return resources, templates, nil
}

// Read reads the content of an agent resource
func (m *AgentResourceManager) Read(ctx context.Context, uri string) ([]ResourceContent, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Parse the URI to extract the agent name
	if !strings.HasPrefix(uri, "agent://") {
		return nil, fmt.Errorf("invalid agent URI format: %s", uri)
	}

	agentName := strings.TrimPrefix(uri, "agent://")
	if agentName == "" {
		return nil, fmt.Errorf("missing agent name in URI: %s", uri)
	}

	// Get the agent from the catalog
	agentCard := m.catalog.GetAgent(agentName)
	if agentCard == nil {
		return nil, fmt.Errorf("agent not found: %s", agentName)
	}

	// Convert the agent card to JSON
	cardJSON, err := json.MarshalIndent(agentCard, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to serialize agent card: %w", err)
	}

	// Create a resource content with the agent card JSON
	content := ResourceContent{
		URI:      uri,
		MimeType: "application/json",
		Text:     string(cardJSON),
	}

	return []ResourceContent{content}, nil
}

// Subscribe subscribes to agent resource updates
func (m *AgentResourceManager) Subscribe(ctx context.Context, uri string) error {
	// Parse the URI to extract the agent name
	if !strings.HasPrefix(uri, "agent://") {
		return fmt.Errorf("invalid agent URI format: %s", uri)
	}

	agentName := strings.TrimPrefix(uri, "agent://")
	if agentName == "" {
		return fmt.Errorf("missing agent name in URI: %s", uri)
	}

	// Create a new subscription
	_, err := m.subManager.Subscribe(ctx, uri)
	if err != nil {
		return fmt.Errorf("failed to create subscription: %w", err)
	}

	errnie.Debug(fmt.Sprintf("Subscribed to agent resource: %s", uri))

	return nil
}

// Unsubscribe unsubscribes from agent resource updates
func (m *AgentResourceManager) Unsubscribe(ctx context.Context, uri string) error {
	subs := m.subManager.GetSubscriptions(uri)
	for _, sub := range subs {
		m.subManager.Unsubscribe(uri, sub)
	}

	errnie.Debug(fmt.Sprintf("Unsubscribed from agent resource: %s", uri))

	return nil
}

// NotifyUpdate notifies subscribers of an agent update
func (m *AgentResourceManager) NotifyUpdate(agentName string) {
	// Create the URI for the agent
	uri := fmt.Sprintf("agent://%s", agentName)

	// Get the agent from the catalog
	agentCard := m.catalog.GetAgent(agentName)
	if agentCard == nil {
		errnie.New(errnie.WithMessage(fmt.Sprintf("agent not found for notification: %s", agentName)))
		return
	}

	// Convert the agent card to JSON
	cardJSON, err := json.MarshalIndent(agentCard, "", "  ")
	if err != nil {
		errnie.New(errnie.WithError(err))
		return
	}

	// Create a resource content with the agent card JSON
	content := ResourceContent{
		URI:      uri,
		MimeType: "application/json",
		Text:     string(cardJSON),
	}

	// Notify subscribers
	m.subManager.Notify(uri, content)

	errnie.Debug(fmt.Sprintf("Notified subscribers of agent update: %s", agentName))
}
