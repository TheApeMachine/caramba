package task

type MessageRole string

const (
	MessageRoleUser  MessageRole = "user"
	MessageRoleAgent MessageRole = "agent"
)

type Message struct {
	Role     MessageRole    `json:"role"`
	Parts    []MessagePart  `json:"parts"`
	Metadata map[string]any `json:"metadata"`
}

type Part struct {
	Text string `json:"text"`
}

type TextPart struct {
	Type     string         `json:"type"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

func (t TextPart) GetType() string {
	return t.Type
}

type FilePart struct {
	Type     string         `json:"type"`
	File     File           `json:"file"`
	Metadata map[string]any `json:"metadata"`
}

func (f FilePart) GetType() string {
	return f.Type
}

type File struct {
	Name     string `json:"name"`
	MimeType string `json:"mimeType"`
	Bytes    string `json:"bytes"` //base64 encoded content
	URI      string `json:"uri"`
}

type DataPart struct {
	Type     string         `json:"type"`
	Data     map[string]any `json:"data"`
	Metadata map[string]any `json:"metadata"`
}

func (d DataPart) GetType() string {
	return d.Type
}

// MessagePart represents a part of a message that can be text, file, or data
type MessagePart interface {
	GetType() string
}
