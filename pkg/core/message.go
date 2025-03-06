package core

// Define custom error, compatible with errors.Is
var ErrNoContent = &ErrorNoContent{}

type ErrorNoContent struct{}

func (e *ErrorNoContent) Error() string {
	return "no content provided"
}

type Message struct {
	Role    string `json:"role"`
	Name    string `json:"name,omitempty"`
	Content string `json:"content"`
}

func NewMessage(role string, name string, content string) *Message {
	return &Message{
		Role:    role,
		Name:    name,
		Content: content,
	}
}
