package task

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Role represents the role of a message sender
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleAgent     Role = "agent"
	RoleTool      Role = "tool"
	RoleDeveloper Role = "developer"
)

// String returns the string representation of the role
func (r Role) String() string {
	return string(r)
}

// Message represents a message in the A2A protocol
type Message struct {
	Role     Role                   `json:"role"`
	Parts    []Part                 `json:"parts"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Custom UnmarshalJSON for Message to handle polymorphic Part types
func (m *Message) UnmarshalJSON(data []byte) error {
	// 1. Define an alias type to avoid recursion
	type messageAlias Message
	var alias messageAlias

	// 2. Unmarshal into a temporary struct that captures Parts as json.RawMessage
	var raw struct {
		Role     Role                   `json:"role"`
		Parts    []json.RawMessage      `json:"parts"` // Capture Parts as raw messages
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal message base structure: %w", err)
	}

	// 3. Assign non-Parts fields from raw struct to alias
	alias.Role = raw.Role
	alias.Metadata = raw.Metadata
	alias.Parts = make([]Part, 0, len(raw.Parts)) // Initialize the Parts slice

	// 4. Iterate through raw parts and unmarshal into concrete types
	for i, rawPart := range raw.Parts {
		// 4a. Peek at the "type" field
		var partType struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawPart, &partType); err != nil {
			// Allow parts that might not have a 'type' field if they are simple strings?
			// Let's assume for now parts MUST be objects with a 'type' field based on the error.
			return fmt.Errorf("failed to unmarshal part type at index %d: %w", i, err)
		}

		// 4b. Unmarshal based on type
		var actualPart Part
		switch partType.Type {
		case "text":
			var p TextPart
			if err := json.Unmarshal(rawPart, &p); err != nil {
				return fmt.Errorf("failed to unmarshal TextPart at index %d: %w", i, err)
			}
			actualPart = &p // Use pointer to match constructor functions
		case "file":
			var p FilePart
			if err := json.Unmarshal(rawPart, &p); err != nil {
				return fmt.Errorf("failed to unmarshal FilePart at index %d: %w", i, err)
			}
			actualPart = &p
		case "data":
			var p DataPart
			if err := json.Unmarshal(rawPart, &p); err != nil {
				return fmt.Errorf("failed to unmarshal DataPart at index %d: %w", i, err)
			}
			actualPart = &p
		default:
			// Potentially handle unknown types gracefully? For now, error out.
			return fmt.Errorf("unknown part type '%s' at index %d", partType.Type, i)
		}
		alias.Parts = append(alias.Parts, actualPart)
	}

	// 5. Assign the populated alias back to the original Message pointer
	*m = Message(alias)
	return nil
}

func (m Message) String() string {
	var builder strings.Builder

	for _, part := range m.Parts {
		if part.GetType() == "text" {
			builder.WriteString(part.(*TextPart).Text)
		}
	}

	return builder.String()
}

// Part represents a message part in the A2A protocol
type Part interface {
	GetType() string
	GetMetadata() map[string]interface{}
}

// TextPart represents a text message part
type TextPart struct {
	Type     string                 `json:"type"`
	Text     string                 `json:"text"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

func (p TextPart) GetType() string                     { return p.Type }
func (p TextPart) GetMetadata() map[string]interface{} { return p.Metadata }

// FilePart represents a file message part
type FilePart struct {
	Type     string                 `json:"type"`
	File     FileContent            `json:"file"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

func (p FilePart) GetType() string                     { return p.Type }
func (p FilePart) GetMetadata() map[string]interface{} { return p.Metadata }

// DataPart represents a data message part
type DataPart struct {
	Type     string                 `json:"type"`
	Data     map[string]interface{} `json:"data"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

func (p DataPart) GetType() string                     { return p.Type }
func (p DataPart) GetMetadata() map[string]interface{} { return p.Metadata }

// FileContent represents the content of a file
type FileContent struct {
	Name     string `json:"name,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Bytes    string `json:"bytes,omitempty"`
	URI      string `json:"uri,omitempty"`
}

// NewSystemMessage creates a new system message
func NewSystemMessage(content string) Message {
	return Message{
		Role: RoleSystem,
		Parts: []Part{
			&TextPart{
				Type: "text",
				Text: content,
			},
		},
	}
}

// NewUserMessage creates a new user message
func NewUserMessage(name, content string) Message {
	return Message{
		Role: RoleUser,
		Parts: []Part{
			&TextPart{
				Type: "text",
				Text: content,
			},
		},
		Metadata: map[string]interface{}{
			"name": name,
		},
	}
}

// NewAssistantMessage creates a new assistant message
func NewAssistantMessage(content string) Message {
	return Message{
		Role: RoleAgent,
		Parts: []Part{
			&TextPart{
				Type: "text",
				Text: content,
			},
		},
	}
}

// NewToolMessage creates a new tool message
func NewToolMessage(content string, toolName string) Message {
	return Message{
		Role: RoleTool,
		Parts: []Part{
			&TextPart{
				Type: "text",
				Text: content,
			},
		},
		Metadata: map[string]interface{}{
			"tool_name": toolName,
		},
	}
}
