package resources

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
)

// DefaultManager is a basic implementation of ResourceManager
type DefaultManager struct {
	resources  []Resource
	templates  []ResourceTemplate
	subManager *SubscriptionManager
	mu         sync.RWMutex
	// Add a map to store active subscriptions
	activeSubscriptions map[string][]*Subscription
}

// NewDefaultManager creates a new DefaultManager instance
func NewDefaultManager() *DefaultManager {
	m := &DefaultManager{
		resources:           make([]Resource, 0),
		templates:           make([]ResourceTemplate, 0),
		subManager:          NewSubscriptionManager(),
		activeSubscriptions: make(map[string][]*Subscription),
	}

	// Add some test resources
	m.AddResource(Resource{
		URI:         "file:///test/example.txt",
		Name:        "Example Text File",
		Description: "A sample text file for testing",
		MimeType:    "text/plain",
		Type:        TextResource,
	})

	m.AddResource(Resource{
		URI:         "file:///test/image.png",
		Name:        "Example Image",
		Description: "A sample image file for testing",
		MimeType:    "image/png",
		Type:        BinaryResource,
	})

	// Add test templates
	m.AddTemplate(ResourceTemplate{
		URITemplate: "file:///test/{category}/{name}",
		Name:        "Dynamic File Template",
		Description: "A template for accessing files by category and name",
		MimeType:    "application/octet-stream",
		Type:        BinaryResource,
		Variables: []TemplateVariable{
			{
				Name:        "category",
				Description: "The category of the file",
				Required:    true,
			},
			{
				Name:        "name",
				Description: "The name of the file",
				Required:    true,
			},
		},
	})

	m.AddTemplate(ResourceTemplate{
		URITemplate: "file:///docs/{version}/{page}",
		Name:        "Documentation Template",
		Description: "A template for accessing documentation pages",
		MimeType:    "text/html",
		Type:        TextResource,
		Variables: []TemplateVariable{
			{
				Name:        "version",
				Description: "The documentation version",
				Required:    true,
			},
			{
				Name:        "page",
				Description: "The documentation page",
				Required:    true,
			},
		},
	})

	m.AddTemplate(ResourceTemplate{
		URITemplate: "file:///api/{version}/{endpoint}",
		Name:        "API Endpoint Template",
		Description: "A template for accessing API endpoints",
		MimeType:    "application/json",
		Type:        TextResource,
		Variables: []TemplateVariable{
			{
				Name:        "version",
				Description: "The API version",
				Required:    true,
			},
			{
				Name:        "endpoint",
				Description: "The API endpoint",
				Required:    true,
			},
		},
	})

	return m
}

// List returns all available resources and templates
func (m *DefaultManager) List(ctx context.Context) ([]Resource, []ResourceTemplate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	resources := make([]Resource, len(m.resources))
	copy(resources, m.resources)

	templates := make([]ResourceTemplate, len(m.templates))
	copy(templates, m.templates)

	return resources, templates, nil
}

// Read reads the content of a resource
func (m *DefaultManager) Read(ctx context.Context, uri string) ([]ResourceContent, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// First, check if the URI matches any templates
	matchedTemplate, templateVars := m.findMatchingTemplate(uri)

	// If a template was matched, create a dynamic resource
	if matchedTemplate != nil {
		return m.generateTemplateContent(uri, matchedTemplate, templateVars), nil
	}

	// If no template matched, look for a static resource
	resource := m.findResourceByURI(uri)
	if resource == nil {
		return nil, fmt.Errorf("resource not found: %s", uri)
	}

	// For now, return a placeholder content based on the resource type
	content := ResourceContent{
		URI:      uri,
		MimeType: resource.MimeType,
	}

	if resource.Type == TextResource {
		content.Text = fmt.Sprintf("Content of resource: %s", resource.Name)
	} else {
		// For binary resources, return a small base64-encoded placeholder
		content.Blob = "SGVsbG8gV29ybGQ=" // "Hello World" in base64
	}

	return []ResourceContent{content}, nil
}

// findMatchingTemplate finds a template that matches the given URI
func (m *DefaultManager) findMatchingTemplate(uri string) (*ResourceTemplate, map[string]string) {
	for _, t := range m.templates {
		// Try to match the URI against the template
		vars, err := matchTemplate(t.URITemplate, uri)
		if err == nil {
			return &t, vars
		}
	}
	return nil, nil
}

// findResourceByURI finds a resource by its URI
func (m *DefaultManager) findResourceByURI(uri string) *Resource {
	for _, r := range m.resources {
		if r.URI == uri {
			return &r
		}
	}
	return nil
}

// generateTemplateContent generates content for a template
func (m *DefaultManager) generateTemplateContent(uri string, template *ResourceTemplate, vars map[string]string) []ResourceContent {
	content := ResourceContent{
		URI:      uri,
		MimeType: template.MimeType,
	}

	if template.Type == TextResource {
		// For text resources, include the template variables in the content
		varInfo := "Template Variables:\n"
		for name, value := range vars {
			varInfo += fmt.Sprintf("- %s: %s\n", name, value)
		}
		content.Text = varInfo
	} else {
		// For binary resources, return a placeholder
		content.Blob = "SGVsbG8gV29ybGQ=" // "Hello World" in base64
	}

	return []ResourceContent{content}
}

// matchTemplate tries to match a URI against a template and extract variables
func matchTemplate(template, uri string) (map[string]string, error) {
	// Extract variables from the template
	variables := ParseTemplateVariables(template)
	if len(variables) == 0 {
		return nil, fmt.Errorf("template has no variables")
	}

	// Convert template to a regex pattern
	pattern := template
	for _, v := range variables {
		pattern = strings.Replace(pattern, "{"+v.Name+"}", "([^/]+)", 1)
	}
	pattern = "^" + pattern + "$"

	// Create a regex to match the URI
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid template pattern: %w", err)
	}

	// Match the URI against the pattern
	matches := re.FindStringSubmatch(uri)
	if matches == nil {
		return nil, fmt.Errorf("URI does not match template")
	}

	// Extract variable values
	result := make(map[string]string)
	for i, v := range variables {
		if i+1 < len(matches) {
			result[v.Name] = matches[i+1]
		}
	}

	return result, nil
}

// Subscribe subscribes to resource updates
func (m *DefaultManager) Subscribe(ctx context.Context, uri string) error {
	sub, err := m.subManager.Subscribe(ctx, uri)
	if err != nil {
		return err
	}

	// Store the subscription
	m.mu.Lock()
	m.activeSubscriptions[uri] = append(m.activeSubscriptions[uri], sub)
	m.mu.Unlock()

	return nil
}

// Unsubscribe unsubscribes from resource updates
func (m *DefaultManager) Unsubscribe(ctx context.Context, uri string) error {
	subs := m.subManager.GetSubscriptions(uri)
	for _, sub := range subs {
		m.subManager.Unsubscribe(uri, sub)
	}
	return nil
}

// NotifySubscribers notifies all subscribers of a resource update
func (m *DefaultManager) NotifySubscribers(uri string, content ResourceContent) {
	m.subManager.Notify(uri, content)
}

// Close closes all subscriptions
func (m *DefaultManager) Close() {
	m.subManager.Close()
}

// AddResource adds a new resource to the manager
func (m *DefaultManager) AddResource(resource Resource) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.resources = append(m.resources, resource)
}

// AddTemplate adds a new resource template to the manager
func (m *DefaultManager) AddTemplate(template ResourceTemplate) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.templates = append(m.templates, template)
}
