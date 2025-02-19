package provider

import "github.com/theapemachine/caramba/datura"

type Interface interface {
	Stream(input *datura.Artifact) chan *datura.Artifact
}

/*
ProviderParams defines the parameters for a provider.
*/
type ProviderParams struct {
	Messages           []Message           `json:"messages"`
	StructuredResponse *StructuredResponse `json:"schema,omitempty"`
	Tools              []Tool              `json:"tools,omitempty"`
}

/*
Message defines a message for a provider.
*/
type Message struct {
	Role       string                 `json:"role"`
	Content    string                 `json:"content"`
	Name       string                 `json:"name,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

/*
StructuredResponse defines a structured response for a provider.
*/
type StructuredResponse struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Schema      interface{} `json:"schema"`
}

/*
Parameter defines the parameters for a tool.
*/
type Parameter struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	Required   []string               `json:"required"`
}

/*
Tool defines a tool for a provider.
*/
type Tool struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Parameters  Parameter `json:"parameters"`
}
