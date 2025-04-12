package task

type MessageRole string

const (
	MessageRoleUnknown   MessageRole = "unknown"
	MessageRoleSystem    MessageRole = "system"
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleTool      MessageRole = "tool"
	MessageRoleAgent     MessageRole = "agent"
)

func (role MessageRole) String() string {
	return string(role)
}

type Part struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type Message struct {
	Role  MessageRole `json:"role"`
	Parts []Part      `json:"parts"`
}

func NewMessage(role MessageRole, parts []Part) *Message {
	return &Message{
		Role:  role,
		Parts: parts,
	}
}

func NewUserMessage(name, message string) Message {
	return Message{
		Role: MessageRoleUser,
		Parts: []Part{
			{
				Type: "text",
				Text: message,
			},
		},
	}
}
