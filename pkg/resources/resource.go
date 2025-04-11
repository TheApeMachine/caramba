package resources

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"mime"
	"path/filepath"
	"strings"
)

// ResourceType represents the type of resource content
type ResourceType string

const (
	// TextResource represents a text-based resource
	TextResource ResourceType = "text"
	// BinaryResource represents a binary resource
	BinaryResource ResourceType = "binary"
)

// Resource represents a single resource in the MCP system
type Resource struct {
	URI         string       `json:"uri"`
	Name        string       `json:"name"`
	Description string       `json:"description,omitempty"`
	MimeType    string       `json:"mimeType,omitempty"`
	Type        ResourceType `json:"type"`
}

// ResourceTemplate represents a URI template for dynamic resources
type ResourceTemplate struct {
	URITemplate string             `json:"uriTemplate"`
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	MimeType    string             `json:"mimeType,omitempty"`
	Type        ResourceType       `json:"type"`
	Variables   []TemplateVariable `json:"variables,omitempty"`
}

// ResourceContent represents the content of a resource
type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"` // base64 encoded
}

// ResourceManager handles resource operations
type ResourceManager interface {
	// List returns all available resources
	List(ctx context.Context) ([]Resource, []ResourceTemplate, error)

	// Read reads the content of a resource
	Read(ctx context.Context, uri string) ([]ResourceContent, error)

	// Subscribe subscribes to resource updates
	Subscribe(ctx context.Context, uri string) error

	// Unsubscribe unsubscribes from resource updates
	Unsubscribe(ctx context.Context, uri string) error
}

// NewResource creates a new Resource instance
func NewResource(uri, name string, opts ...ResourceOption) *Resource {
	r := &Resource{
		URI:  uri,
		Name: name,
	}

	// Set default MIME type based on file extension
	if ext := filepath.Ext(uri); ext != "" {
		if mimeType := mime.TypeByExtension(ext); mimeType != "" {
			r.MimeType = mimeType
		}
	}

	// Determine resource type based on MIME type
	if strings.HasPrefix(r.MimeType, "text/") || r.MimeType == "" {
		r.Type = TextResource
	} else {
		r.Type = BinaryResource
	}

	// Apply options
	for _, opt := range opts {
		opt(r)
	}

	return r
}

// ResourceOption is a function that modifies a Resource
type ResourceOption func(*Resource)

// WithDescription sets the resource description
func WithDescription(description string) ResourceOption {
	return func(r *Resource) {
		r.Description = description
	}
}

// WithMimeType sets the resource MIME type
func WithMimeType(mimeType string) ResourceOption {
	return func(r *Resource) {
		r.MimeType = mimeType
		r.Type = BinaryResource
	}
}

// NewResourceContent creates a new ResourceContent instance
func NewResourceContent(uri string, content io.Reader, mimeType string) (*ResourceContent, error) {
	rc := &ResourceContent{
		URI:      uri,
		MimeType: mimeType,
	}

	// Read content
	data, err := io.ReadAll(content)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	// Determine if content is text or binary
	if strings.HasPrefix(mimeType, "text/") || mimeType == "" {
		rc.Text = string(data)
	} else {
		rc.Blob = base64.StdEncoding.EncodeToString(data)
	}

	return rc, nil
}

// NewResourceTemplate creates a new ResourceTemplate instance
func NewResourceTemplate(uriTemplate, name string, opts ...ResourceTemplateOption) *ResourceTemplate {
	t := &ResourceTemplate{
		URITemplate: uriTemplate,
		Name:        name,
	}

	// Set default MIME type
	t.MimeType = "application/octet-stream"
	t.Type = BinaryResource

	// Parse variables from the template
	t.Variables = ParseTemplateVariables(uriTemplate)

	// Apply options
	for _, opt := range opts {
		opt(t)
	}

	return t
}

// ResourceTemplateOption is a function that modifies a ResourceTemplate
type ResourceTemplateOption func(*ResourceTemplate)

// WithTemplateDescription sets the template description
func WithTemplateDescription(description string) ResourceTemplateOption {
	return func(t *ResourceTemplate) {
		t.Description = description
	}
}

// WithTemplateMimeType sets the template MIME type
func WithTemplateMimeType(mimeType string) ResourceTemplateOption {
	return func(t *ResourceTemplate) {
		t.MimeType = mimeType
		if strings.HasPrefix(mimeType, "text/") {
			t.Type = TextResource
		} else {
			t.Type = BinaryResource
		}
	}
}

// WithTemplateVariable adds a variable to the template
func WithTemplateVariable(name, description string, required bool) ResourceTemplateOption {
	return func(t *ResourceTemplate) {
		// Check if the variable already exists
		for i, v := range t.Variables {
			if v.Name == name {
				t.Variables[i].Description = description
				t.Variables[i].Required = required
				return
			}
		}

		// Add the variable if it doesn't exist
		t.Variables = append(t.Variables, TemplateVariable{
			Name:        name,
			Description: description,
			Required:    required,
		})
	}
}
